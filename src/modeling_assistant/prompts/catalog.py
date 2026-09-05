from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from modeling_assistant.memory.archive import archive_summary
from modeling_assistant.data.paper_template import map_craft_section_to_template

logger = logging.getLogger(__name__)


def _compact_column_dict(col) -> dict:
    """单列紧凑摘要：只保留建模思路需要的结构信息，样例值最多 3 个。"""
    return {
        "name": col.name,
        "dtype": col.dtype,
        "missing_rate": col.missing_rate,
        "min": col.min,
        "max": col.max,
        "mean": col.mean,
        "std": col.std,
        "unique_count": col.unique_count,
        "sample_values": (col.sample_values or [])[:3],
        "parse_hint": col.parse_hint,
    }


def _column_relevance_score(col, focus_text: str, position: int) -> int:
    """Deterministic relevance score used by the Mathematician's slim data view."""
    name = str(getattr(col, "name", "") or "").strip().lower()
    focus = focus_text.lower()
    score = max(6 - position, 0)  # retain a little source-order/schema signal
    if name and name in focus:
        score += 100
    priority_terms = (
        "target", "label", "outcome", "result", "score", "time", "date", "year",
        "id", "目标", "结果", "得分", "时间", "日期", "年份", "编号", "经度", "纬度",
    )
    if any(term in name for term in priority_terms):
        score += 20
    # A column explicitly mentioned by a parse hint is structurally non-trivial.
    if getattr(col, "parse_hint", ""):
        score += 5
    return score


def _compact_profile_dict(
    profile,
    *,
    max_columns: int | None = None,
    focus_text: str = "",
    basename_paths: bool = False,
) -> dict:
    """数据画像的紧凑摘要：按文件保留边界，只含行列结构信息。

    V12 修复：原始数据（sample_head、全量相关性矩阵、全量样例）不进 prompt。
    LLM 只需要知道"每个文件是什么、有哪些列、列的类型/范围/缺失情况"，
    具体数值由 Coder 生成的代码在运行时读取。
    """
    source_files = list(getattr(profile, "file_summaries", None) or [])
    if not source_files:
        # 旧状态兜底：合并画像退化为单文件摘要
        source_files = [
            type("LegacyFileSummary", (), {
                "path": profile.file_paths[0] if profile.file_paths else "",
                "rows": profile.total_rows,
                "cols": profile.total_cols,
                "issues": list(profile.issues or []),
                "columns": list(profile.columns or []),
            })()
        ]

    # Mathematician uses a bounded view. Other nodes keep the existing complete
    # compact schema by leaving max_columns=None.
    per_file_limits: list[int | None] = [None] * len(source_files)
    if max_columns is not None:
        total_budget = max(max_columns, len(source_files))
        base, remainder = divmod(total_budget, len(source_files))
        per_file_limits = [base + (1 if i < remainder else 0) for i in range(len(source_files))]

    files: list[dict] = []
    shown_total = 0
    for file_index, fs in enumerate(source_files):
        columns = list(fs.columns or [])
        limit = per_file_limits[file_index]
        if limit is not None and len(columns) > limit:
            ranked = sorted(
                enumerate(columns),
                key=lambda pair: (
                    -_column_relevance_score(pair[1], focus_text, pair[0]),
                    pair[0],
                ),
            )[:limit]
            selected_indexes = {index for index, _col in ranked}
            columns = [col for index, col in enumerate(columns) if index in selected_indexes]
        shown_total += len(columns)
        files.append(
            {
                "path": Path(str(fs.path)).name if basename_paths else fs.path,
                "rows": fs.rows,
                "cols": fs.cols,
                "issues": list(fs.issues or []),
                "columns": [_compact_column_dict(c) for c in columns],
                "shown_columns": len(columns),
                "columns_truncated": max(int(fs.cols or len(fs.columns)) - len(columns), 0),
            }
        )
    return {
        "total_rows": profile.total_rows,
        "total_cols": profile.total_cols,
        "shown_columns": shown_total,
        "columns_truncated": max(int(profile.total_cols or 0) - shown_total, 0),
        "issues": list(profile.issues or []),
        "files": files,
    }


@dataclass(frozen=True, slots=True)
class PromptContext:
    static_ltm: BaseModel | None = None
    dynamic_ltm: BaseModel | None = None
    archive: list[BaseModel] | None = None
    empirical: BaseModel | None = None
    control: BaseModel | None = None
    artifacts: BaseModel | None = None
    exemplars: BaseModel | None = None
    extra: dict[str, Any] | None = None

    def to_template_vars(self) -> dict[str, str]:
        extra = self.extra or {}
        # 从 control 中提取 top_k_plans 单独渲染，方便 Realist 模板引用
        top_k_plans_json = "[]"
        if self.control is not None:
            ctrl_data = self.control.model_dump(mode="json")
            # Realist 盲评：不暴露 Mathematician 的自评分、生成顺序偏好或既有 verdict，
            # 避免锚定。只传方案内容与稳定 id。
            blinded_plans = []
            for plan in ctrl_data.get("top_k_plans", []):
                blinded_plans.append({
                    key: plan.get(key)
                    for key in (
                        "id", "title", "description", "strategy_type", "input_data",
                        "assumptions", "mathematical_object", "parameter_estimation",
                        "solution_method", "expected_outputs", "validation_method",
                        "failure_conditions",
                    )
                })
            top_k_plans_json = json.dumps(
                blinded_plans,
                ensure_ascii=False,
                indent=2,
            )
        # 提取阈值用于 prompt 模板
        feasibility_threshold = "60"
        innovation_threshold = "60"
        branch_from_version = "无"
        rebrainstorm_feedback_json = "[]"
        coder_error_log_json = "[]"
        coder_error_count = "0"
        last_result_review_issues_json = "[]"
        # Meta-Router 专用变量默认值
        modeling_revision_count = "0"
        modeling_revision_budget = "4"
        modeling_revision_remaining = "4"
        selected_plan_json = "{}"
        if self.control is not None:
            ctrl_data = self.control.model_dump()
            feasibility_threshold = str(ctrl_data.get("feasibility_threshold", 60))
            innovation_threshold = str(ctrl_data.get("innovation_threshold", 60))
            branch_from_version = str(ctrl_data.get("branch_from_version") or "无")
            rebrainstorm_feedback_json = json.dumps(
                ctrl_data.get("rebrainstorm_feedback", []),
                ensure_ascii=False,
                indent=2,
            )
            coder_error_log_json = json.dumps(
                ctrl_data.get("coder_error_log", []),
                ensure_ascii=False,
                indent=2,
            )
            coder_error_count = str(ctrl_data.get("coder_error_count", 0))
            # V10 修复：注入 last_result_review_issues，让 Architect 区分 Coder 执行失败
            # 和 ResultReviewer 拒绝两类失败，针对性调整模型设计
            last_result_review_issues_json = json.dumps(
                ctrl_data.get("last_result_review_issues", []),
                ensure_ascii=False,
                indent=2,
            )
            # Meta-Router 专用：预算状态
            modeling_revision_count = str(ctrl_data.get("modeling_revision_count", 0))
            modeling_revision_budget = str(ctrl_data.get("modeling_revision_budget", 4))
            modeling_revision_remaining = str(
                ctrl_data.get("modeling_revision_budget", 4)
                - ctrl_data.get("modeling_revision_count", 0)
            )
            # Meta-Router 专用：当前选中的方案
            selected_plan_id = ctrl_data.get("selected_plan_id", "")
            top_k_plans_list = ctrl_data.get("top_k_plans", [])
            selected_plan_json = json.dumps(
                next((p for p in top_k_plans_list if p.get("id") == selected_plan_id), {}),
                ensure_ascii=False,
                indent=2,
            )
        # V14 小题循环：前小题 LTM 摘要 + 结果路径，供当前小题建模/实现/成稿引用
        sub_question_context_json = "[]"
        result_output_filename = "output.csv"
        # V17 结果注册表与章节绑定（Writer 成稿时只允许引用权威结果）
        result_manifest_json = "[]"
        section_result_binding_json = "{}"
        # V17 图表注册表（Writer 只允许引用 manifest 中已生成的图）
        figure_manifest_json = "{}"
        # V18 承重图：结论→承重依赖的显式连接，按角色切片注入
        load_bearing_active = "false"
        load_bearing_map_json = "{}"
        verification_contract_json = "{}"
        conclusion_inventory_json = "[]"
        load_bearing_gaps_json = "[]"
        if self.artifacts is not None:
            fm = getattr(self.artifacts, "figure_manifest", None)
            if isinstance(fm, BaseModel):
                fm = fm.model_dump(mode="json")
            if isinstance(fm, dict):
                figure_manifest_json = json.dumps(fm, ensure_ascii=False, indent=2)
            lbm = getattr(self.artifacts, "load_bearing_map", None)
            if isinstance(lbm, BaseModel):
                lbm = lbm.model_dump(mode="json")
            if isinstance(lbm, dict) and lbm:
                load_bearing_active = "true"
                load_bearing_map_json = json.dumps(lbm, ensure_ascii=False, indent=2)
                verification_contract_json = json.dumps(
                    lbm.get("contract") or {}, ensure_ascii=False, indent=2
                )
                conclusion_inventory_json = json.dumps(
                    lbm.get("conclusions") or [], ensure_ascii=False, indent=2
                )
                gaps: list[str] = []
                gaps.extend(lbm.get("root_gaps") or [])
                gaps.extend(lbm.get("anchor_gaps") or [])
                gaps.extend(lbm.get("shape_risks") or [])
                load_bearing_gaps_json = json.dumps(gaps, ensure_ascii=False, indent=2)
        if self.control is not None:
            ctrl_data = self.control.model_dump(mode="json")
            questions = ctrl_data.get("sub_questions", []) or []
            idx = ctrl_data.get("current_sub_question_index", 0) or 0
            if questions:
                result_output_filename = f"q{idx + 1}.csv"
            result_manifest_json = json.dumps(
                ctrl_data.get("results_manifest", []) or [],
                ensure_ascii=False,
                indent=2,
            )
            n = len(questions) if questions else 1
            section_result_binding_json = json.dumps(
                {f"{4 + i}_problem{i}.tex": i - 1 for i in range(1, n + 1)},
                ensure_ascii=False,
                indent=2,
            )
            prev_ltms = []
            for i, ltm in enumerate(ctrl_data.get("sub_ltms", []) or []):
                if i >= idx:
                    continue
                prev_ltms.append(
                    {
                        "index": i,
                        "objective": ltm.get("objective", ""),
                        "assumptions": list(ltm.get("assumptions", []) or []),
                        "equations": list(ltm.get("equations", []) or []),
                    }
                )
            prev_results = [
                r
                for r in (ctrl_data.get("sub_results", []) or [])
                if r.get("index", 0) < idx
            ]
            sub_question_context_json = json.dumps(
                {
                    "current_index": idx,
                    "current_text": questions[idx] if idx < len(questions) else "",
                    "total": len(questions),
                    "previous_sub_ltms": prev_ltms,
                    "previous_sub_results": prev_results,
                },
                ensure_ascii=False,
                indent=2,
            )
        # archive 摘要视图（轻量，仅含版本号与变更说明）
        archive_summary_json = json.dumps(
            archive_summary(list(self.archive or [])),
            ensure_ascii=False,
            indent=2,
        )
        # 单独渲染数据画像，方便各 prompt 模板直接引用
        data_profile_json = "{}"
        data_file_paths_json = "[]"
        data_columns_json = "[]"
        data_findings_json = "[]"
        data_intelligence_json = "[]"
        # V11 修复：机器生成的字符串列解析建议，供 clarifier/coder 直接引用
        data_parse_hints_json = "[]"
        profile = None
        if self.static_ltm is not None:
            profile = getattr(self.static_ltm, "data_profile", None)
            # V11.4 修复：LangGraph checkpoint 反序列化时，data_profile 可能变成 dict
            # （大表如距离矩阵触发 PydanticSerializationUnexpectedValue，导致 msgpack 回退为 dict）
            # 这里做防御性转换，确保 profile 是 DataProfile 对象
            if isinstance(profile, dict):
                from modeling_assistant.schemas.state import DataProfile
                try:
                    profile = DataProfile.model_validate(profile)
                except Exception:
                    profile = None
            if profile is not None:
                # V12 修复：只注入紧凑摘要，绝不把 sample_head/全量相关性矩阵/全量样例放进 prompt
                data_profile_json = json.dumps(
                    _compact_profile_dict(profile),
                    ensure_ascii=False,
                    indent=2,
                )
                data_file_paths_json = json.dumps(
                    profile.file_paths,
                    ensure_ascii=False,
                    indent=2,
                )
                # V16 修复：列名按文件分组注入，并附带样例值。
                # 旧实现把多附件所有列扁平合并（如距离矩阵 99 个数字列名混着中文列名），
                # Coder 无法判断「哪个文件有哪些列」，导致反复臆造列名被 AST 校验打回。
                file_columns = []
                for fs in profile.file_summaries:
                    file_columns.append(
                        {
                            "file": Path(fs.path).name,
                            "columns": [
                                {
                                    "name": col.name,
                                    "dtype": col.dtype,
                                    "sample_values": (col.sample_values or [])[:3],
                                }
                                for col in fs.columns
                            ],
                        }
                    )
                data_columns_json = json.dumps(
                    file_columns,
                    ensure_ascii=False,
                    indent=2,
                )
                # 提取非空的 parse_hint
                data_parse_hints_json = json.dumps(
                    [
                        {"column": col.name, "dtype": col.dtype, "parse_hint": col.parse_hint}
                        for col in profile.columns
                        if col.parse_hint
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            # 数据认知更新（执行阶段发现的、对原始 schema 的补充认知）
            data_findings_json = json.dumps(
                getattr(self.static_ltm, "data_findings", []) or [],
                ensure_ascii=False,
                indent=2,
            )
            # V12 新增：LLM 数据理解分析师提炼的"解题所需信息"
            if isinstance(self.static_ltm, dict):
                data_intelligence = self.static_ltm.get("data_intelligence", []) or []
            else:
                data_intelligence = getattr(self.static_ltm, "data_intelligence", []) or []
            data_intelligence_json = json.dumps(
                data_intelligence,
                ensure_ascii=False,
                indent=2,
            )
        # V12 修复：static_ltm_json 不再携带原始 data_profile（sample_head/相关性矩阵等），
        # 只带 data_profile_summary（紧凑摘要）+ 其余字段
        static_ltm_json = "{}"
        if self.static_ltm is not None:
            if isinstance(self.static_ltm, BaseModel):
                static_data = self.static_ltm.model_dump(mode="json", exclude={"data_profile"})
            else:
                static_data = dict(self.static_ltm)
                static_data.pop("data_profile", None)
            if profile is not None:
                static_data["data_profile_summary"] = _compact_profile_dict(profile)
            static_ltm_json = json.dumps(static_data, ensure_ascii=False, indent=2)

        # Mathematician 专用精简视图：只携带发散决策所需的信息。完整原始数据、
        # 全量列、长文献元数据和重复的数据洞察继续留在通用上下文中供其他节点使用。
        mathematician_static_ltm_json = "{}"
        mathematician_sub_question_context_json = "{}"
        mathematician_data_findings_json = "[]"
        mathematician_data_intelligence_json = "[]"
        mathematician_empirical_refuted_json = "[]"
        mathematician_empirical_open_questions_json = "[]"
        mathematician_empirical_run_index_json = "[]"
        mathematician_dynamic_ltm_json = "{}"
        mathematician_archive_summary_json = json.dumps(
            archive_summary(list(self.archive or []))[-10:],
            ensure_ascii=False,
            indent=2,
        )
        mathematician_rebrainstorm_feedback_json = "[]"
        if self.dynamic_ltm is not None:
            dynamic_for_math = self.dynamic_ltm.model_dump(mode="json")
            mathematician_dynamic_ltm_json = json.dumps(
                {
                    "assumptions": (dynamic_for_math.get("assumptions", []) or [])[:12],
                    "nomenclature": dict(
                        list((dynamic_for_math.get("nomenclature", {}) or {}).items())[:40]
                    ),
                    "equations": (dynamic_for_math.get("equations", []) or [])[:16],
                    "objective": str(dynamic_for_math.get("objective", ""))[:1000],
                    "solution_outline": str(dynamic_for_math.get("solution_outline", ""))[:2000],
                    "identifiability_checks": (
                        dynamic_for_math.get("identifiability_checks", []) or []
                    )[:10],
                    "constant_relevance": dict(
                        list((dynamic_for_math.get("constant_relevance", {}) or {}).items())[:30]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        if self.control is not None:
            feedback_for_math = self.control.model_dump(mode="json").get(
                "rebrainstorm_feedback", []
            ) or []
            mathematician_rebrainstorm_feedback_json = json.dumps(
                [str(item)[:500] for item in feedback_for_math[-8:]],
                ensure_ascii=False,
                indent=2,
            )
        if self.static_ltm is not None:
            static_obj = self.static_ltm
            raw_problem = getattr(static_obj, "raw_problem", "") or ""
            problem_understanding = getattr(static_obj, "problem_understanding", "") or ""
            current_question = ""
            ctrl_data_for_math: dict[str, Any] = {}
            if self.control is not None:
                ctrl_data_for_math = self.control.model_dump(mode="json")
                questions = ctrl_data_for_math.get("sub_questions", []) or []
                current_index = int(ctrl_data_for_math.get("current_sub_question_index", 0) or 0)
                if current_index < len(questions):
                    current_question = str(questions[current_index])

            intelligence = list(getattr(static_obj, "data_intelligence", []) or [])
            findings = list(getattr(static_obj, "data_findings", []) or [])
            focus_text = "\n".join(
                [current_question, problem_understanding, *map(str, intelligence)]
            )
            literature = list(getattr(static_obj, "literature", []) or [])[:5]
            facts = list(getattr(static_obj, "problem_facts", []) or [])
            math_static = {
                "current_problem": current_question or raw_problem,
                "global_problem_understanding": problem_understanding,
                "data_attachments": [
                    Path(str(path)).name
                    for path in (getattr(static_obj, "data_attachments", []) or [])
                ],
                "data_schema": getattr(static_obj, "data_schema", {}) or {},
                "problem_facts": [
                    fact.model_dump(mode="json") if isinstance(fact, BaseModel) else fact
                    for fact in facts
                ],
                "fact_role_mapping": getattr(static_obj, "fact_role_mapping", {}) or {},
                "literature": [
                    {
                        "title": getattr(item, "title", ""),
                        "summary": (getattr(item, "summary", "") or "")[:240],
                    }
                    for item in literature
                ],
            }
            if profile is not None:
                math_static["data_profile_summary"] = _compact_profile_dict(
                    profile,
                    max_columns=max(int(extra.get("mathematician_max_columns", 32)), 1),
                    focus_text=focus_text,
                    basename_paths=True,
                )
            mathematician_static_ltm_json = json.dumps(
                math_static, ensure_ascii=False, indent=2
            )
            mathematician_data_findings_json = json.dumps(
                [str(item)[:400] for item in findings[-10:]],
                ensure_ascii=False,
                indent=2,
            )
            mathematician_data_intelligence_json = json.dumps(
                [str(item)[:500] for item in intelligence[:12]],
                ensure_ascii=False,
                indent=2,
            )

            previous_ltms = []
            current_index = int(ctrl_data_for_math.get("current_sub_question_index", 0) or 0)
            for index, ltm in enumerate(ctrl_data_for_math.get("sub_ltms", []) or []):
                if index >= current_index:
                    continue
                nomenclature = ltm.get("nomenclature", {}) or {}
                previous_ltms.append(
                    {
                        "index": index,
                        "objective": ltm.get("objective", ""),
                        "solution_outline": str(ltm.get("solution_outline", ""))[:600],
                        "exported_symbols": dict(list(nomenclature.items())[:12]),
                    }
                )
            previous_results = []
            for result in ctrl_data_for_math.get("sub_results", []) or []:
                if int(result.get("index", 0) or 0) >= current_index:
                    continue
                previous_results.append(
                    {
                        "index": result.get("index", 0),
                        "title": result.get("title", ""),
                        "status": result.get("status", ""),
                        "result_files": [
                            Path(str(path)).name for path in (result.get("result_paths", []) or [])
                        ],
                    }
                )
            mathematician_sub_question_context_json = json.dumps(
                {
                    "current_index": current_index,
                    "current_text": current_question,
                    "total": len(ctrl_data_for_math.get("sub_questions", []) or []),
                    "previous_interfaces": previous_ltms,
                    "previous_results": previous_results,
                },
                ensure_ascii=False,
                indent=2,
            )

        if self.empirical is not None:
            emp_data_for_math = self.empirical.model_dump(mode="json")
            mathematician_empirical_refuted_json = json.dumps(
                [str(item)[:400] for item in (emp_data_for_math.get("refuted_assumptions", []) or [])[-10:]],
                ensure_ascii=False,
                indent=2,
            )
            mathematician_empirical_open_questions_json = json.dumps(
                [str(item)[:400] for item in (emp_data_for_math.get("open_questions", []) or [])[-8:]],
                ensure_ascii=False,
                indent=2,
            )
            mathematician_empirical_run_index_json = json.dumps(
                (emp_data_for_math.get("run_index", []) or [])[-6:],
                ensure_ascii=False,
                indent=2,
            )
        # V12 新增：Architect 声明的结果契约（Architect → Coder → ResultReviewer 共用）
        result_contract_json = "{}"
        if self.artifacts is not None:
            rc = getattr(self.artifacts, "result_contract", None)
            if isinstance(rc, BaseModel):
                result_contract_json = rc.model_dump_json(indent=2)
            elif rc is not None:
                result_contract_json = json.dumps(rc, ensure_ascii=False, indent=2)
        # ── empirical 层注入（默认只注入 L2 摘要，L3 原始日志按需查询）──
        empirical_refuted_json = "[]"
        empirical_open_questions_json = "[]"
        empirical_run_index_json = "[]"
        empirical_findings_summary_json = "[]"
        drawer_observations_json = "[]"
        if self.empirical is not None:
            emp_data = self.empirical.model_dump(mode="json")
            empirical_refuted_json = json.dumps(
                emp_data.get("refuted_assumptions", []),
                ensure_ascii=False,
                indent=2,
            )
            empirical_open_questions_json = json.dumps(
                emp_data.get("open_questions", []),
                ensure_ascii=False,
                indent=2,
            )
            empirical_run_index_json = json.dumps(
                emp_data.get("run_index", []),
                ensure_ascii=False,
                indent=2,
            )
            # findings 摘要视图：只含 assumption_tested/verdict/confidence/evidence，不含 suggested_fix 等冗余字段
            empirical_findings_summary_json = json.dumps(
                [
                    {
                        "assumption_tested": f.get("assumption_tested"),
                        "verdict": f.get("verdict"),
                        "confidence": f.get("confidence"),
                        "evidence": f.get("evidence"),
                    }
                    for f in emp_data.get("findings", [])
                ],
                ensure_ascii=False,
                indent=2,
            )
            # Drawer 视觉观察单独提取，供 Reflection 做二次确认
            drawer_observations_json = json.dumps(
                [
                    {
                        "assumption_tested": f.get("assumption_tested"),
                        "evidence": f.get("evidence"),
                        "verdict": f.get("verdict"),
                        "confidence": f.get("confidence"),
                    }
                    for f in emp_data.get("findings", [])
                    if f.get("source_node") == "drawer"
                ],
                ensure_ascii=False,
                indent=2,
            )
        # ── 动态 LTM 拆解字段（供 reflection/meta_router 模板单独引用）──
        dynamic_ltm_assumptions_json = "[]"
        dynamic_ltm_equations_json = "[]"
        dynamic_ltm_objective_json = '""'
        dynamic_ltm_solution_outline_json = '""'
        if self.dynamic_ltm is not None:
            dltm_data = self.dynamic_ltm.model_dump(mode="json")
            dynamic_ltm_assumptions_json = json.dumps(
                dltm_data.get("assumptions", []),
                ensure_ascii=False,
                indent=2,
            )
            dynamic_ltm_equations_json = json.dumps(
                dltm_data.get("equations", []),
                ensure_ascii=False,
                indent=2,
            )
            # Meta-Router 专用：建模目标与方案概要
            dynamic_ltm_objective_json = json.dumps(
                dltm_data.get("objective", ""),
                ensure_ascii=False,
            )
            dynamic_ltm_solution_outline_json = json.dumps(
                dltm_data.get("solution_outline", ""),
                ensure_ascii=False,
            )
        # recent_stdout 默认空，由 reflection_node 在调用前通过 extra 注入
        recent_stdout = str(extra.get("recent_stdout", ""))[:2000]
        # recent_stderr 默认空，由 coder_node 自修复循环通过 extra 注入
        recent_stderr = str(extra.get("recent_stderr", ""))[:2000]
        # V10 修复：result_preview 默认空，由 writer_node 注入真实 CSV 内容预览
        result_preview = str(extra.get("result_preview", ""))[:5000]
        # V11 修复：problem_facts 机器提取的题目常量列表，供 clarifier/coder 引用
        problem_facts_json = "[]"
        if self.static_ltm is not None:
            facts = getattr(self.static_ltm, "problem_facts", []) or []
            problem_facts_json = json.dumps(
                [
                    {
                        "value": f.value,
                        "unit": f.unit,
                        "context": f.context,
                        "role_hint": f.role_hint,
                    }
                    for f in facts
                ],
                ensure_ascii=False,
                indent=2,
            )
        # ── Exemplar 表达知识注入（默认全空，active=False 时模板显示为空块）──
        exemplar_active = "false"
        exemplar_structure_json = "[]"
        exemplar_chart_json = "[]"
        exemplar_writing_json = "{}"
        exemplar_highlights_json = "[]"
        exemplar_quotes_json = "[]"
        style_profile_json = "{}"
        craft_derivation_json = "[]"
        craft_algorithm_json = "[]"
        craft_interpretation_json = "[]"
        craft_writing_json = "[]"
        craft_signature_moves_json = "[]"
        craft_figure_placement_json = "[]"
        craft_section_focus_json = "[]"
        craft_argument_flow_json = "{}"
        if self.exemplars is not None:
            from modeling_assistant.schemas.state import (
                ExemplarPaper,
                GlobalStyleProfile,
                TypeStyleGuide,
            )

            ctx = self.exemplars
            ctx_data = ctx.model_dump(mode="json") if isinstance(ctx, BaseModel) else {}
            if ctx_data.get("active"):
                exemplar_active = "true"
            injection = ctx_data.get("injection", {}) or {}
            guide_raw = ctx_data.get("guide")
            cards_raw = ctx_data.get("cards", []) or []
            profile_raw = ctx_data.get("profile")
            guide = None
            if isinstance(guide_raw, dict):
                try:
                    guide = TypeStyleGuide.model_validate(guide_raw)
                except Exception:
                    guide = None
            cards: list[ExemplarPaper] = []
            for raw in cards_raw:
                if isinstance(raw, dict):
                    try:
                        cards.append(ExemplarPaper.model_validate(raw))
                    except Exception:
                        continue
            profile = None
            if isinstance(profile_raw, dict):
                try:
                    profile = GlobalStyleProfile.model_validate(profile_raw)
                except Exception:
                    profile = None

            # 结构（强注入）：共性骨架 + 变体
            if injection.get("structure", True) and guide is not None:
                structure_entries = [
                    {"section": s} for s in (guide.common_structure + guide.structure_variants)
                ]
                exemplar_structure_json = json.dumps(
                    structure_entries, ensure_ascii=False, indent=2
                )
            # 图表（中注入）：指南推荐 + 卡片图表
            if injection.get("chart", True):
                chart_entries: list[dict] = []
                if guide is not None:
                    chart_entries.extend(
                        {"figure_type": f, "purpose": ""} for f in guide.recommended_figures
                    )
                for card in cards:
                    for fig in card.figures:
                        chart_entries.append(
                            {
                                "figure_type": fig.figure_type,
                                "purpose": fig.purpose,
                                "style_notes": fig.style_notes,
                            }
                        )
                exemplar_chart_json = json.dumps(
                    chart_entries[:20], ensure_ascii=False, indent=2
                )
            # 文风（中/弱注入，可被 Dropout 关闭）：指南基线 + 卡片文风
            if injection.get("writing", True):
                writing_entries: dict[str, str] = {}
                if guide is not None:
                    writing_entries.update(guide.writing_baseline)
                for card in cards:
                    writing_entries.update(card.writing_style)
                    if card.summary_style:
                        writing_entries.setdefault("summary_style", card.summary_style)
                exemplar_writing_json = json.dumps(
                    writing_entries, ensure_ascii=False, indent=2
                )
            # 亮点与短摘录（受 highlight 注入层控制；dropout 可关闭，防同质化）
            if injection.get("highlight", True):
                exemplar_highlights_json = json.dumps(
                    [
                        {"card": c.id, "highlight": h}
                        for c in cards
                        for h in c.highlights
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                exemplar_quotes_json = json.dumps(
                    [
                        {"card": c.id, "quote": q}
                        for c in cards
                        for q in c.quotes
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            if profile is not None:
                style_profile_json = profile.model_dump_json(indent=2)
            # ── 行文技艺参考（craft 层，与 writing 注入开关同步）──
            craft_raw = ctx_data.get("craft")
            if isinstance(craft_raw, dict) and injection.get("writing", True):
                from modeling_assistant.schemas.craft import CraftGuide

                try:
                    craft = CraftGuide.model_validate(craft_raw)
                    craft_derivation_json = json.dumps(
                        [d.model_dump() for d in craft.derivation_common],
                        ensure_ascii=False,
                        indent=2,
                    )
                    craft_algorithm_json = json.dumps(
                        [a.model_dump() for a in craft.algorithm_common],
                        ensure_ascii=False,
                        indent=2,
                    )
                    craft_interpretation_json = json.dumps(
                        [i.model_dump() for i in craft.interpretation_common],
                        ensure_ascii=False,
                        indent=2,
                    )
                    craft_writing_json = json.dumps(
                        [w.model_dump() for w in craft.writing_common],
                        ensure_ascii=False,
                        indent=2,
                    )
                    craft_signature_moves_json = json.dumps(
                        [m.model_dump() for m in craft.signature_moves_common],
                        ensure_ascii=False,
                        indent=2,
                    )
                    craft_figure_placement_json = json.dumps(
                        [f.model_dump() for f in craft.figure_placement_common],
                        ensure_ascii=False,
                        indent=2,
                    )
                    craft_section_focus_json = json.dumps(
                        [
                            {
                                **s.model_dump(),
                                # 与国赛 LaTeX 模板章节文件绑定：
                                # Writer 按此把写作重点落到对应模板章节
                                "template_file": map_craft_section_to_template(s.section),
                            }
                            for s in craft.section_focus_common
                        ],
                        ensure_ascii=False,
                        indent=2,
                    )
                    if craft.argument_flow_common is not None:
                        craft_argument_flow_json = craft.argument_flow_common.model_dump_json(
                            indent=2
                        )
                except Exception:
                    pass
        # ── V15 方法知识库注入（按节点/题型切片，只影响领域判断）──
        # 开关来自 AgentRuntime.render_prompt 注入的 extra（默认开启）；
        # 关闭时所有知识变量为空、active=false，渲染行为与旧版本完全一致。
        method_knowledge_active = "false"
        problem_type = "unknown"
        model_selection_knowledge = ""
        type_knowledge = ""
        assumption_knowledge = ""
        coding_knowledge = ""
        chart_knowledge = ""
        writing_knowledge = ""
        method_knowledge_enabled = str(
            extra.get("method_knowledge_enabled", True)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if method_knowledge_enabled:
            raw_problem = ""
            problem_understanding = ""
            if self.static_ltm is not None:
                raw_problem = getattr(self.static_ltm, "raw_problem", "") or ""
                problem_understanding = (
                    getattr(self.static_ltm, "problem_understanding", "") or ""
                )
            try:
                from modeling_assistant.data.method_knowledge import build_knowledge_payload
                from modeling_assistant.memory.exemplar_search import (
                    PROBLEM_TYPES,
                    judge_problem_type,
                )

                judged_type, _conf = judge_problem_type(
                    raw_problem,
                    problem_understanding=problem_understanding,
                )
                # V22：题型已经 HITL 确认时，以确认值为准（覆盖自动判定）
                control_type = ""
                if self.control is not None:
                    control_type = str(getattr(self.control, "problem_type", "") or "")
                if control_type in PROBLEM_TYPES:
                    judged_type = control_type
                # 空题面 + 无破题理解时判定为 unknown，避免误注入某个具体题型知识
                elif not (raw_problem or "").strip() and not (problem_understanding or "").strip():
                    judged_type = "unknown"
                knowledge = build_knowledge_payload(judged_type)
                method_knowledge_active = knowledge["method_knowledge_active"]
                problem_type = knowledge["problem_type"]
                model_selection_knowledge = knowledge["model_selection_knowledge"]
                type_knowledge = knowledge["type_knowledge"]
                assumption_knowledge = knowledge["assumption_knowledge"]
                coding_knowledge = knowledge["coding_knowledge"]
                chart_knowledge = knowledge["chart_knowledge"]
                writing_knowledge = knowledge["writing_knowledge"]
            except Exception as exc:
                logger.warning("方法知识库注入失败（降级为空知识）: %s", exc)
        return {
            "static_ltm_json": static_ltm_json,
            "mathematician_static_ltm_json": mathematician_static_ltm_json,
            "mathematician_sub_question_context_json": mathematician_sub_question_context_json,
            "mathematician_data_findings_json": mathematician_data_findings_json,
            "mathematician_data_intelligence_json": mathematician_data_intelligence_json,
            "mathematician_empirical_refuted_json": mathematician_empirical_refuted_json,
            "mathematician_empirical_open_questions_json": mathematician_empirical_open_questions_json,
            "mathematician_empirical_run_index_json": mathematician_empirical_run_index_json,
            "mathematician_dynamic_ltm_json": mathematician_dynamic_ltm_json,
            "mathematician_archive_summary_json": mathematician_archive_summary_json,
            "mathematician_rebrainstorm_feedback_json": mathematician_rebrainstorm_feedback_json,
            "dynamic_ltm_json": _json_model(self.dynamic_ltm),
            "archive_json": _json_list(self.archive or []),
            "archive_summary_json": archive_summary_json,
            "control_json": _json_model(self.control),
            "artifacts_json": _json_model(self.artifacts),
            "top_k_plans_json": top_k_plans_json,
            "feasibility_threshold": feasibility_threshold,
            "innovation_threshold": innovation_threshold,
            "branch_from_version": branch_from_version,
            "rebrainstorm_feedback_json": rebrainstorm_feedback_json,
            "coder_error_log_json": coder_error_log_json,
            "coder_error_count": coder_error_count,
            # V10 修复：注入 ResultReviewer 拒绝原因，供 architect 模板针对性调整
            "last_result_review_issues_json": last_result_review_issues_json,
            "data_profile_json": data_profile_json,
            "data_file_paths_json": data_file_paths_json,
            "data_columns_json": data_columns_json,
            "data_findings_json": data_findings_json,
            "data_intelligence_json": data_intelligence_json,
            "sub_question_context_json": sub_question_context_json,
            "result_output_filename": result_output_filename,
            # V17 结果注册表与章节绑定
            "result_manifest_json": result_manifest_json,
            "section_result_binding_json": section_result_binding_json,
            # V17 图表注册表
            "figure_manifest_json": figure_manifest_json,
            # V18 承重图
            "load_bearing_active": load_bearing_active,
            "load_bearing_map_json": load_bearing_map_json,
            "verification_contract_json": verification_contract_json,
            "conclusion_inventory_json": conclusion_inventory_json,
            "load_bearing_gaps_json": load_bearing_gaps_json,
            # V11 修复：机器生成的字符串列解析建议
            "data_parse_hints_json": data_parse_hints_json,
            # V12 修复：结果契约
            "result_contract_json": result_contract_json,
            # V11 修复：机器提取的题目常量
            "problem_facts_json": problem_facts_json,
            # empirical 层
            "empirical_refuted_json": empirical_refuted_json,
            "empirical_open_questions_json": empirical_open_questions_json,
            "empirical_run_index_json": empirical_run_index_json,
            "empirical_findings_summary_json": empirical_findings_summary_json,
            "drawer_observations_json": drawer_observations_json,
            # 动态 LTM 拆解
            "dynamic_ltm_assumptions_json": dynamic_ltm_assumptions_json,
            "dynamic_ltm_equations_json": dynamic_ltm_equations_json,
            # Meta-Router 专用
            "dynamic_ltm_objective_json": dynamic_ltm_objective_json,
            "dynamic_ltm_solution_outline_json": dynamic_ltm_solution_outline_json,
            "selected_plan_json": selected_plan_json,
            "modeling_revision_count": modeling_revision_count,
            "modeling_revision_budget": modeling_revision_budget,
            "modeling_revision_remaining": modeling_revision_remaining,
            # Reflection 专用
            "recent_stdout": recent_stdout,
            # Coder 自修复专用
            "recent_stderr": recent_stderr,
            # Writer 专用：真实结果文件预览（V10 修复）
            "result_preview": result_preview,
            # Writer 完整性警告默认值（writer_node 会通过 extra 覆盖）
            "integrity_warnings": str(extra.get("integrity_warnings", "无（所有关键产物完整）")),
            # Exemplar 表达知识
            "exemplar_active": exemplar_active,
            "exemplar_structure_json": exemplar_structure_json,
            "exemplar_chart_json": exemplar_chart_json,
            "exemplar_writing_json": exemplar_writing_json,
            "exemplar_highlights_json": exemplar_highlights_json,
            "exemplar_quotes_json": exemplar_quotes_json,
            "style_profile_json": style_profile_json,
            "craft_derivation_json": craft_derivation_json,
            "craft_algorithm_json": craft_algorithm_json,
            "craft_interpretation_json": craft_interpretation_json,
            "craft_writing_json": craft_writing_json,
            "craft_signature_moves_json": craft_signature_moves_json,
            "craft_figure_placement_json": craft_figure_placement_json,
            "craft_section_focus_json": craft_section_focus_json,
            "craft_argument_flow_json": craft_argument_flow_json,
            # V15 方法知识库
            "method_knowledge_active": method_knowledge_active,
            "problem_type": problem_type,
            "model_selection_knowledge": model_selection_knowledge,
            "type_knowledge": type_knowledge,
            "assumption_knowledge": assumption_knowledge,
            "coding_knowledge": coding_knowledge,
            "chart_knowledge": chart_knowledge,
            "writing_knowledge": writing_knowledge,
            # V15 终审 LLM 审查：论文全文（final_reviewer_node 注入，缺失时为空）
            "paper_text": str(extra.get("paper_text", "")),
            # V15 论文修订反馈（writer_node 注入，首次撰写时为空）
            "paper_revision_feedback": str(extra.get("paper_revision_feedback", "")),
            # V15 论文模板（AgentRuntime.render_prompt 注入；直接渲染 PromptContext 时默认关闭）
            "paper_template_active": str(extra.get("paper_template_active", "false")),
            "paper_template_structure": str(extra.get("paper_template_structure", "[]")),
            **{key: str(value) for key, value in extra.items() if key not in ("recent_stdout", "recent_stderr", "result_preview")},
        }


class PromptCatalog:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        self.prompt_dir = prompt_dir or Path(__file__).parent / "templates"

    def render(self, name: str, context: PromptContext) -> str:
        template_path = self.prompt_dir / f"{name}.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.format(**context.to_template_vars())


def _json_model(model: BaseModel | None) -> str:
    if model is None:
        return "{}"
    return model.model_dump_json(indent=2)


def _json_list(items: list[BaseModel]) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        indent=2,
    )
