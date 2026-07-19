from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from modeling_assistant.memory.archive import archive_summary


@dataclass(frozen=True, slots=True)
class PromptContext:
    static_ltm: BaseModel | None = None
    dynamic_ltm: BaseModel | None = None
    archive: list[BaseModel] | None = None
    empirical: BaseModel | None = None
    control: BaseModel | None = None
    artifacts: BaseModel | None = None
    extra: dict[str, Any] | None = None

    def to_template_vars(self) -> dict[str, str]:
        extra = self.extra or {}
        # 从 control 中提取 top_k_plans 单独渲染，方便 Realist 模板引用
        top_k_plans_json = "[]"
        if self.control is not None:
            ctrl_data = self.control.model_dump(mode="json")
            top_k_plans_json = json.dumps(
                ctrl_data.get("top_k_plans", []),
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
        # V11 修复：机器生成的字符串列解析建议，供 clarifier/coder 直接引用
        data_parse_hints_json = "[]"
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
                data_profile_json = profile.model_dump_json(indent=2)
                data_file_paths_json = json.dumps(
                    profile.file_paths,
                    ensure_ascii=False,
                    indent=2,
                )
                data_columns_json = json.dumps(
                    [
                        {"name": col.name, "dtype": col.dtype}
                        for col in profile.columns
                    ],
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
        # ── 动态 LTM 拆解字段（供 reflection 模板单独引用）──
        dynamic_ltm_assumptions_json = "[]"
        dynamic_ltm_equations_json = "[]"
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
        return {
            "static_ltm_json": _json_model(self.static_ltm),
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
            # V11 修复：机器生成的字符串列解析建议
            "data_parse_hints_json": data_parse_hints_json,
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
            # Reflection 专用
            "recent_stdout": recent_stdout,
            # Coder 自修复专用
            "recent_stderr": recent_stderr,
            # Writer 专用：真实结果文件预览（V10 修复）
            "result_preview": result_preview,
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
