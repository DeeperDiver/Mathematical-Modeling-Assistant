from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from langgraph.types import interrupt

from modeling_assistant.agents.runtime import AgentRuntime, get_default_runtime
from modeling_assistant.agents.searcher import SearchQuery
from modeling_assistant.data.facts import extract_facts_from_problem
from modeling_assistant.memory.archive import make_snapshot
from modeling_assistant.memory.validation import validate_dynamic_ltm
from modeling_assistant.recording.process_log import (
    ProcessLogEntry,
    archive_prompt,
    make_entry,
    write_log_line,
)
from modeling_assistant.schemas.responses import (
    AnalystResponse,
    ArchitectResponse,
    ClarifierResponse,
    CoderResponse,
    DataIntelligenceResponse,
    DrawerResponse,
    FinalReviewerResponse,
    LoadBearingAnalysisResponse,
    MathematicianResponse,
    MetaRouterResponse,
    MilestoneReviewer1Response,
    RealistResponse,
    ReflectionResponse,
    ResultContract,
    WriterResponse,
)
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    AuthoritativeResult,
    ControlState,
    DynamicLTM,
    EmpiricalFinding,
    EmpiricalLayer,
    GraphState,
    LiteratureItem,
    PlanCandidate,
    REFUTED_CONFIDENCE_THRESHOLD,
    StaticLTM,
    SubQuestionResult,
)
from modeling_assistant.validation.assumption_tags import classify_assumptions

logger = logging.getLogger(__name__)


def _runtime(runtime: AgentRuntime | None) -> AgentRuntime:
    return runtime or get_default_runtime()


def _prompt_audit(
    name: str,
    state: GraphState,
    runtime: AgentRuntime | None,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    """渲染 prompt 并返回 (渲染后的 prompt, audit dict)。"""
    resolved_runtime = _runtime(runtime)
    rendered = resolved_runtime.render_prompt(name, state, extra=extra)
    return rendered, {name: rendered}


def _emit_process(
    runtime: AgentRuntime | None,
    control: ControlState,
    state: GraphState,
    stage: str,
    event: str,
    summary: str,
    details: dict[str, Any] | None = None,
    prompt_text: str | None = None,
    prompt_tag: str | None = None,
) -> ProcessLogEntry:
    """生成一条运行过程记录：先落盘 JSONL（崩溃不丢），再返回供追加到 state。

    建模阶段可传入 prompt_text 存档「模型当时看到了什么」。
    """
    seq = len(state.get("process_log") or []) + 1
    entry = make_entry(control, stage, event, summary, details, seq=seq)
    if runtime is not None:
        try:
            write_log_line(runtime.settings.output_dir, entry)
            if prompt_text:
                tag = prompt_tag or f"{stage}_{seq}"
                prompt_path = archive_prompt(
                    runtime.settings.output_dir, stage, tag, prompt_text
                )
                if prompt_path is not None:
                    entry.details["prompt_file"] = str(prompt_path)
        except Exception as exc:
            logger.warning("运行过程记录落盘失败: %s", exc)
    return entry


def _control(state: GraphState) -> ControlState:
    return state.get("control", ControlState()).model_copy(deep=True)


def _static_ltm(state: GraphState) -> StaticLTM:
    return state.get("static_ltm", StaticLTM()).model_copy(deep=True)


def _dynamic_ltm(state: GraphState) -> DynamicLTM:
    return state.get("dynamic_ltm", DynamicLTM()).model_copy(deep=True)


def _artifacts(state: GraphState) -> ArtifactBundle:
    return state.get("artifacts", ArtifactBundle()).model_copy(deep=True)


def _load_bearing_summary(state: GraphState) -> dict:
    """供 HITL 载荷使用的承重图摘要。"""
    artifacts = _artifacts(state)
    m = artifacts.load_bearing_map
    if m is None:
        return {"present": False}
    return {
        "present": True,
        "analysis_incomplete": m.analysis_incomplete,
        "root_gaps": list(m.root_gaps),
        "anchor_gaps": list(m.anchor_gaps),
        "shape_risks": list(m.shape_risks),
    }


def _empirical(state: GraphState) -> EmpiricalLayer:
    return state.get("empirical", EmpiricalLayer()).model_copy(deep=True)


def _current_sub_question(control: ControlState) -> str:
    """当前小题的题面文本。"""
    questions = control.sub_questions or []
    idx = control.current_sub_question_index
    if 0 <= idx < len(questions):
        return questions[idx]
    return ""


def _record_sub_question_result(
    state: GraphState,
    control: ControlState,
    dynamic_ltm: DynamicLTM,
    artifacts: ArtifactBundle,
    status: str = "passed",
    feedback: list[str] | None = None,
) -> None:
    """把当前小题的 LTM、结果、图表写入 sub_results / sub_ltms。"""
    idx = control.current_sub_question_index
    archive = state.get("ltm_archive", [])
    version = archive[-1].version if archive else ""
    entry = SubQuestionResult(
        index=idx,
        title=_current_sub_question(control),
        ltm_version=version,
        result_paths=list(artifacts.result_paths or []),
        figure_paths=list(artifacts.figure_paths or []),
        status=status,
        feedback=list(feedback or []),
    )
    for i, old in enumerate(control.sub_results):
        if old.index == idx:
            control.sub_results[i] = entry
            break
    else:
        control.sub_results.append(entry)
    while len(control.sub_ltms) <= idx:
        control.sub_ltms.append(DynamicLTM())
    control.sub_ltms[idx] = dynamic_ltm.model_copy(deep=True)


def _result_metrics_snapshot(paths: list[str]) -> dict[str, Any]:
    """轻量指标快照：结果文件的行数/列数（V17 P1，不含图质量检测）。"""
    metrics: dict[str, Any] = {}
    try:
        import pandas as pd

        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            try:
                df = (
                    pd.read_csv(path)
                    if suffix == ".csv"
                    else pd.read_excel(path)
                )
                metrics[path.name] = {
                    "rows": int(len(df)),
                    "cols": int(len(df.columns)),
                }
            except Exception:
                continue
    except Exception:
        pass
    return metrics


def _finalize_authoritative_result(
    state: GraphState,
    control: ControlState,
    artifacts: ArtifactBundle,
    status: Literal["passed", "degraded"] = "passed",
    feedback: list[str] | None = None,
) -> ControlState:
    """把当前小题的验收结果锁定为权威结果，写入 results_manifest。

    V17 结果注册表：只保留当前小题的结果文件（q{i}.csv），避免累积的
    旧路径混入；单题模式（无小题清单）下使用 artifacts.result_paths 全量。
    """
    idx = control.current_sub_question_index
    if control.sub_questions:
        q_fname = f"q{idx + 1}.csv"
        result_paths = [p for p in artifacts.result_paths if p.endswith(q_fname)]
        if not result_paths:
            result_paths = list(artifacts.result_paths)
    else:
        result_paths = list(artifacts.result_paths)
    run_id = f"run_{max(control.coder_run_count - 1, 0)}"
    entry = AuthoritativeResult(
        index=idx,
        title=_current_sub_question(control),
        result_paths=result_paths,
        figure_paths=list(artifacts.figure_paths or []),
        contract=(
            artifacts.result_contract.model_copy(deep=True)
            if artifacts.result_contract is not None
            else None
        ),
        metrics=_result_metrics_snapshot(result_paths),
        run_id=run_id,
        status=status,
        feedback=list(feedback or []),
    )
    for i, old in enumerate(control.results_manifest):
        if old.index == idx:
            control.results_manifest[i] = entry
            break
    else:
        control.results_manifest.append(entry)
    control.results_manifest.sort(key=lambda e: e.index)
    logger.info(
        "Result Manifest 锁定：小题 %d（%s，run=%s，文件=%s）",
        idx + 1,
        status,
        run_id,
        result_paths,
    )
    return control


def _truncate_manifest(control: ControlState, keep_up_to: int) -> ControlState:
    """跨小题回滚时截断结果注册表：保留 index < keep_up_to 的条目。"""
    control.results_manifest = [
        e for e in control.results_manifest if e.index < keep_up_to
    ]
    return control


def _register_figure_manifest(
    state: GraphState,
    artifacts: ArtifactBundle,
    real_figures: list[str],
    run_tag: str,
) -> None:
    """V17：把实际生成的图按 figures_plan 的 plan_id 登记到图表注册表。

    文件名与 plan_id 匹配（figures/{plan_id}.png）；未匹配的图不登记，
    由 paper_check 的 C5 以「未登记图表注册表」警告提示。
    """
    plan_ids = {p.id for p in _artifacts(state).figures_plan}
    manifest: dict[str, dict[str, Any]] = {}
    for path in real_figures:
        stem = Path(path).stem
        if stem in plan_ids:
            manifest[stem] = {
                "path": path,
                "run_id": run_tag,
                "status": "generated",
            }
    if manifest:
        artifacts.figure_manifest = manifest
        logger.info("图表注册表登记 %d 张：%s", len(manifest), list(manifest))


def _advance_sub_question(
    state: GraphState,
    control: ControlState,
    dynamic_ltm: DynamicLTM,
    artifacts: ArtifactBundle,
    status: str = "passed",
    feedback: list[str] | None = None,
) -> tuple[ControlState, bool]:
    """验收通过（或 HITL 接受降级）后推进到下一小题。

    返回 (更新后的 control, 是否还有下一小题)。同时重置当前小题作用域
    的预算、反馈与标志位，让下一题从干净的建模状态开始。
    """
    _record_sub_question_result(
        state, control, dynamic_ltm, artifacts, status=status, feedback=feedback
    )
    next_idx = control.current_sub_question_index + 1
    has_next = next_idx < len(control.sub_questions or [])
    control.current_sub_question_index = next_idx
    control.sub_question_attempts = 0
    control.sub_question_feedback = []
    control.modeling_revision_count = 0
    control.modeling_revision_budget = control.sub_question_budget
    control.debate_round = 0
    control.top_k_plans = []
    control.plan_pool_ids = []
    control.selected_plan_id = None
    control.need_rebrainstorm = False
    control.trigger_clarifier_revision = False
    control.meta_decision = ""
    control.meta_direction_hint = ""
    control.meta_reasoning = ""
    control.coder_error_log = []
    control.last_result_review_issues = []
    control.coder_rollback_target = "architect"
    control.implementation_architecture_reviewed = False
    control.implementation_auto = False
    control.rollback_to_version = None
    control.rollback_source = "none"
    return control, has_next


# ═══════════════════════════════════════════════════════════════════
# 阶段一：输入与全局信息初始化
# ═══════════════════════════════════════════════════════════════════

def problem_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    control.max_debate_rounds = resolved_runtime.settings.max_debate_rounds
    control.innovation_threshold = resolved_runtime.settings.innovation_threshold
    control.feasibility_threshold = resolved_runtime.settings.feasibility_threshold
    control.innovation_weight = resolved_runtime.settings.innovation_weight
    control.feasibility_weight = resolved_runtime.settings.feasibility_weight
    control.phase = "problem_loaded"
    entry = _emit_process(
        runtime, control, state, "problem", "run_started",
        "建模流程启动",
        {
            "llm_model": resolved_runtime.settings.llm_model,
            "problem": (state.get("static_ltm", StaticLTM()).raw_problem or "")[:120],
            "data_attachments": list(state.get("static_ltm", StaticLTM()).data_attachments or []),
        },
    )
    return {"control": control, "process_log": [entry]}


def fact_extractor_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V11 三层防线第一层：纯机器提取题目数值常量。

    在 problem_node 之后、analyst_node 之前运行。
    用正则从 raw_problem 提取所有 (数值, 单位, 上下文) 三元组，
    写入 static_ltm.problem_facts，作为后续所有节点的"真理基准"。

    特点：
    - 纯代码，不调用 LLM，零成本、零幻觉
    - 不可被 LLM 改写（StaticLTM 字段语义为不可变）
    - 后续 Clarifier/Coder 必须引用这些值，否则触发第二层/第三层校验告警
    """
    static_ltm = _static_ltm(state)
    control = _control(state)

    if not static_ltm.raw_problem:
        logger.warning("fact_extractor_node: raw_problem 为空，跳过提取")
        control.phase = "facts_extracted"
        return {"static_ltm": static_ltm, "control": control}

    # 纯机器提取
    # V11.4：传入 data_profile.columns 用于 classify_fact 双重判据识别 data_range
    columns = (
        static_ltm.data_profile.columns
        if static_ltm.data_profile and static_ltm.data_profile.columns
        else None
    )
    facts = extract_facts_from_problem(static_ltm.raw_problem, columns=columns)
    static_ltm.problem_facts = facts

    if facts:
        logger.info(
            "fact_extractor_node: 提取到 %d 个数值常量，示例：%s",
            len(facts),
            [(f.value, f.unit, f.category, f.context[:30]) for f in facts[:3]],
        )
    else:
        logger.info("fact_extractor_node: 未提取到带单位的数值常量")

    control.phase = "facts_extracted"
    return {"static_ltm": static_ltm, "control": control}


_SUB_QUESTION_PATTERNS = [
    re.compile(r"问题\s*(\d+)"),
    re.compile(r"第\s*(\d+)\s*题"),
    re.compile(r"第\s*([一二三四五六七八九十]+)\s*题"),
    re.compile(r"[（(]\s*(\d+)\s*[）)]"),
    re.compile(r"(?:^|\n)\s*([一二三四五六七八九十]+)[、.]"),
]


def split_sub_questions(text: str) -> list[str]:
    """从题面自动拆分小题（问题1/第1题/(1)/一、等），无分隔时整题作为一个小题。"""
    if not text or not text.strip():
        return []
    hits: list[tuple[int, str]] = []
    for pattern in _SUB_QUESTION_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group(0).strip()))
    if not hits:
        return [text.strip()]
    hits.sort(key=lambda x: x[0])
    dedup: list[tuple[int, str]] = []
    for pos, label in hits:
        if not dedup or pos != dedup[-1][0]:
            dedup.append((pos, label))
    segments: list[str] = []
    for k, (pos, _label) in enumerate(dedup):
        end = dedup[k + 1][0] if k + 1 < len(dedup) else len(text)
        segment = text[pos:end].strip()
        if k == 0 and pos > 0:
            # 题面前言（如“请回答以下问题”）并入第一小题作为上下文
            segment = f"{text[:pos].strip()}\n{segment}".strip()
        segments.append(segment)
    return segments


def _parse_edit_sub_questions(edited: str) -> list[str]:
    """解析 HITL edit 输入：以分号/换行分隔的小题清单。"""
    parts = re.split(r"[；;]\s*|\n+", edited or "")
    return [p.strip() for p in parts if p.strip()]


def split_sub_questions_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V14 新增：小题自动拆分 + HITL 确认。

    拆分结果展示给人类确认；approve 直接放行，edit <分号分隔清单> 修正后重新确认。
    """
    control = _control(state)
    static_ltm = _static_ltm(state)

    if control.sub_questions and control.sub_questions_confirmed:
        control.phase = "sub_questions_confirmed"
        return {"control": control}

    if not control.sub_questions:
        control.sub_questions = split_sub_questions(static_ltm.raw_problem)

    control.hitl_required = True
    control.hitl_stage = "sub_question_split"
    decision = interrupt({
        "stage": "sub_question_split",
        "message": "已自动拆分小题，请确认清单（或修正后重新确认）。",
        "hint": (
            "输入 'approve' 确认拆分；"
            "或 'edit 问题1：…；问题2：…' 用分号分隔的清单替换。"
        ),
        "sub_questions": list(control.sub_questions),
    })
    action = _parse_hitl_decision(decision)
    control.hitl_required = False
    control.hitl_stage = "none"
    if action["type"] == "edit":
        edited = _parse_edit_sub_questions(action.get("version") or "")
        if edited:
            control.sub_questions = edited
            control.sub_questions_confirmed = False
            control.phase = "sub_question_split_edited"
        else:
            control.phase = "sub_question_split_edited"
    else:
        control.sub_questions_confirmed = True
        control.phase = "sub_questions_confirmed"
    entry = _emit_process(
        runtime, control, state, "split_sub_questions", "sub_questions_split",
        (
            "人类修正小题清单"
            if action["type"] == "edit"
            else "人类确认小题清单"
        ),
        {
            "decision": action["type"],
            "sub_questions": list(control.sub_questions),
        },
    )
    return {"control": control, "process_log": [entry]}


def analyst_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    static_ltm = _static_ltm(state)

    system_prompt, audit = _prompt_audit("analyst", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "analyst", state, AnalystResponse, system_prompt=system_prompt
        )
        static_ltm.problem_understanding = response.problem_understanding
        static_ltm.data_schema = response.data_schema
    except Exception as exc:
        logger.error("Analyst LLM 调用失败: %s", exc)
        if static_ltm.raw_problem and not static_ltm.problem_understanding:
            static_ltm.problem_understanding = (
                "围绕赛题目标、数据可得性、约束条件和评价指标建立结构化理解。"
            )

    control = _control(state)
    control.phase = "static_ltm_initialized"
    return {"static_ltm": static_ltm, "control": control, "prompt_audit": audit}


def searcher_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    static_ltm = _static_ltm(state)

    if not static_ltm.literature:
        # 基于 Analyst 的破题思路 + 原始问题提取检索关键词
        search_context = static_ltm.problem_understanding or static_ltm.raw_problem
        try:
            raw = resolved_runtime.invoke(
                "searcher",
                state,
                system_prompt=(
                    "你是一个学术检索专家。根据以下破题思路与原始问题，提取 3-5 个核心检索关键词。\n"
                    "V19 要求：ArXiv 只索引英文元数据，必须把中文概念翻译成英文专业术语"
                    "（如：NIPT→non-invasive prenatal testing；Y 染色体浓度→fetal Y chromosome DNA fraction；"
                    "孕周→gestational age；染色体非整倍体→chromosome aneuploidy / trisomy）。\n"
                    "用逗号分隔，只输出英文关键词，不要其他内容。\n\n"
                    f"破题思路：{search_context}\n\n"
                    f"原始问题：{static_ltm.raw_problem}"
                ),
            )
            keywords = [k.strip() for k in raw.split(",") if k.strip()]
        except Exception:
            keywords = []

        query = SearchQuery(
            keywords=keywords,
            problem_statement=search_context,
            max_results=5,
        )
        try:
            results = resolved_runtime.searcher.search(query)
        except Exception as exc:
            logger.warning("检索失败，使用占位结果: %s", exc)
            from modeling_assistant.agents.searcher import StubSearcher
            results = StubSearcher().search(query)

        # 质量校验：去重、过滤占位、关键词相关性过滤
        from modeling_assistant.agents.searcher import validate_search_results
        validated = validate_search_results(results, keywords, min_relevance_keywords=1)
        if not validated:
            logger.warning(
                "检索结果全部未通过质量校验（关键词: %s），保留原始结果。",
                keywords,
            )
            validated = results
        elif len(validated) < len(results):
            logger.info(
                "质量校验过滤了 %d/%d 条结果。",
                len(results) - len(validated),
                len(results),
            )

        static_ltm.literature = [
            LiteratureItem(
                title=r.title,
                authors=r.authors,
                source=r.source,
                summary=r.summary,
                url=r.url,
            )
            for r in validated
        ]

    control = _control(state)
    control.phase = "literature_collected"
    # V17：把检索到的参考文献（标题/作者）记录到运行过程日志
    entry = _emit_process(
        runtime, control, state, "searcher", "literature_retrieved",
        f"检索到 {len(static_ltm.literature)} 条参考文献",
        {
            "count": len(static_ltm.literature),
            "keywords": keywords,
            "literature": [
                {
                    "title": item.title,
                    "authors": item.authors,
                    "source": item.source,
                }
                for item in static_ltm.literature
            ],
        },
    )
    return {"static_ltm": static_ltm, "control": control, "process_log": [entry]}


def data_analyst_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V12 新增：数据理解分析师。

    在 data_profile 之后、建模节点之前运行。LLM 只拿到紧凑数据概要
    （行列结构、类型、缺失率、少量样例），提炼"解题思路所需的信息"，
    写入 static_ltm.data_intelligence：
    - 每个文件/表是什么、关键列、分组结构
    - 哪些文件与题目直接相关、如何关联
    - 数据层面的风险与 Coder 运行时应注意事项

    原始数据（sample_head / 全量相关性矩阵 / 全量样例）不进任何 prompt，
    具体数值只由 Coder 的代码在运行时读取。
    失败不阻塞流程：保留空 intelligence，下游照常运行。
    """
    resolved_runtime = _runtime(runtime)
    static_ltm = _static_ltm(state)

    system_prompt, audit = _prompt_audit("data_analyst", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "data_analyst", state, DataIntelligenceResponse, system_prompt=system_prompt
        )
        insights = [i.strip() for i in response.insights if i and i.strip()]
        if insights:
            static_ltm.data_intelligence = insights
            logger.info("DataAnalyst 提炼 %d 条数据情报", len(insights))
    except Exception as exc:
        logger.error("DataAnalyst LLM 调用失败（不阻塞流程）: %s", exc)

    control = _control(state)
    control.phase = "data_intelligence_extracted"
    return {"static_ltm": static_ltm, "control": control, "prompt_audit": audit}


def exemplar_loader_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """加载优秀论文表达知识（Exemplar Learning System）。

    在 searcher 之后、mathematician 之前运行：判定题型并交 HITL 确认，
    再用确认后的题型检索 L1 卡片、L2 题型指南与 L3 全局偏好，
    组装 ExemplarContext 供下游 prompt 注入。

    设计原则：
    - 无知识库或相关性低于阈值 → 返回 inactive 的 ExemplarContext，
      现有建模/路由/验证逻辑完全不变。
    - 题型判定（如 evaluation 这类易误判题型）必须经人类确认，避免误判
      导致方法知识与示例库注入错误方向。
    - 图表风格/亮点卡按 style_dropout_rate 概率随机关闭（防同质化）；
      句子级句法规则（writing 层）稳定注入，不参与 dropout。
    """
    resolved_runtime = _runtime(runtime)
    static_ltm = _static_ltm(state)
    control = _control(state)
    from modeling_assistant.memory.exemplar_search import (
        PROBLEM_TYPES,
        judge_problem_type,
        load_exemplar_context,
    )

    # 1) 题型判定 + HITL 确认（已确认过的复用，不重复打断）
    if not control.problem_type:
        ptype, confidence = judge_problem_type(
            static_ltm.raw_problem,
            runtime=resolved_runtime,
            problem_understanding=static_ltm.problem_understanding,
        )
        control.problem_type = ptype
        control.problem_type_confidence = confidence
        control.hitl_required = True
        control.hitl_stage = "problem_type"
        decision = interrupt({
            "stage": "problem_type",
            "message": (
                f"已判定题型为「{ptype}」（置信度 {confidence:.2f}）。"
                "题型决定方法知识、示例库与后续建模方向，请确认或修正。"
            ),
            "hint": (
                "输入 'approve' 确认题型；"
                "输入 'set optimization|physics|forecasting|evaluation|data_mining' "
                "覆盖题型。"
            ),
            "problem_type": ptype,
            "confidence": confidence,
        })
        control.hitl_required = False
        control.hitl_stage = "none"
        action = _parse_hitl_decision(decision)
        if action["type"] == "set":
            override = str(action.get("version") or "").strip()
            if override in PROBLEM_TYPES:
                control.problem_type = override
                control.problem_type_confidence = 1.0
                logger.info("题型经人类修正为：%s", override)
            else:
                logger.warning(
                    "人类输入的题型无效，保留判定值 %s：%s",
                    control.problem_type,
                    override,
                )

    # 2) 按确认后的题型加载示例知识
    context = load_exemplar_context(
        resolved_runtime.settings,
        static_ltm.raw_problem,
        runtime=resolved_runtime,
        problem_understanding=static_ltm.problem_understanding,
        problem_type=control.problem_type,
    )
    # 注入强度分级（数值作为各层注入概率）；额外 dropout 只作用于图表风格与
    # 亮点/短摘录（易同质化部分），句法规则（writing 层）不参与 dropout。
    if context.active:
        for key, strength in resolved_runtime.settings.style_injection.items():
            if strength <= 0:
                context.injection[key] = False
            elif strength < 1.0 and random.random() > strength:
                context.injection[key] = False
        for layer in ("chart", "highlight"):
            if context.injection.get(layer, False) and random.random() < resolved_runtime.settings.style_dropout_rate:
                context.injection[layer] = False
                logger.info(
                    "Exemplar %s 注入被 Dropout 关闭（rate=%.2f）",
                    layer,
                    resolved_runtime.settings.style_dropout_rate,
                )
    return {"exemplars": context, "control": control}


# ═══════════════════════════════════════════════════════════════════
# 阶段二：建模核心 —— "先发散，后剪枝"
# ═══════════════════════════════════════════════════════════════════

def _update_plan_pool(control: ControlState, w_inn: float, w_fea: float) -> None:
    """V23：保留综合评分最高的前 N 个 keep 方案作为「方案池」。

    供架构 HITL 呈现各方案实现路径，由人类测试后定夺采用哪一个。
    """
    kept = [p for p in control.top_k_plans if p.verdict == "keep"]
    ranked = sorted(kept, key=lambda p: p.total_score(w_inn, w_fea), reverse=True)
    control.plan_pool_ids = [p.id for p in ranked[: control.plan_pool_size]]


def _apply_plan_gate(plan: PlanCandidate, threshold: int) -> None:
    """应用证据优先硬门槛；创新性永远不作为淘汰条件。"""
    hard_scores = (
        plan.problem_fit_score,
        plan.data_assumption_score,
        plan.mathematical_correctness_score,
        plan.computability_score,
    )
    if any(hard_scores):
        if min(hard_scores) < threshold:
            plan.verdict = "kill"
        elif plan.verifiability_score < threshold:
            plan.verdict = "reject"
        else:
            plan.verdict = "keep"
    else:
        # 兼容旧模型/旧测试响应：仅以可行性为硬门槛。
        plan.verdict = "kill" if plan.feasibility_score < threshold else "keep"


def mathematician_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """发散与创新：头脑风暴 Top-K 候选方案。

    分支重建（Goal.md）：由 Mathematician 主动判断是否需要从 LTM Archive
    中的某个历史版本提取灵感。系统只在 LLM 明确请求时才执行分支重建。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    archive = state.get("ltm_archive", [])

    # 消费并重置 rebrainstorm 标志，避免 Milestone Reviewer 1 打回后循环
    control.need_rebrainstorm = False

    # 重置分支请求状态：每次调用都是一次新的主动决策
    control.branch_from_version = None

    # Meta-Router 决策已消费：重置 trigger_clarifier_revision 和 meta_decision，
    # 让下次 reflection 能重新调用 Meta-Router（否则 already_triggered=True 跳过）
    if control.meta_decision:
        control.trigger_clarifier_revision = False
        control.meta_decision = ""

    control.debate_round += 1
    control.phase = "model_brainstorming"

    system_prompt, audit = _prompt_audit("mathematician", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "mathematician", state, MathematicianResponse, system_prompt=system_prompt
        )
        # 处理 LLM 请求的按需 Archive 详情查询
        if response.requested_version and archive:
            requested = response.requested_version
            from modeling_assistant.memory.archive import checkout_snapshot
            try:
                full_ltm = checkout_snapshot(archive, requested)
                supplement = (
                    f"\n\n--- 版本 {requested} 的完整动态 LTM（按需补充）---\n"
                    f"{full_ltm.model_dump_json(indent=2)}\n"
                    f"请基于以上完整信息重新生成方案。"
                )
                system_prompt += supplement
                response = resolved_runtime.invoke_structured(
                    "mathematician", state, MathematicianResponse, system_prompt=system_prompt
                )
            except ValueError:
                logger.warning("Mathematician 请求了不存在的版本: %s", requested)

        # 处理 LLM 请求的按需执行证据查询（与 requested_version 对称的拉模式）
        if response.requested_evidence_run_id:
            run_id = response.requested_evidence_run_id
            log_path = Path(resolved_runtime.output_path("logs", f"{run_id}.log"))
            if log_path.exists():
                try:
                    log_content = log_path.read_text(encoding="utf-8")[:3000]
                    supplement = (
                        f"\n\n--- 执行日志 {run_id}（按需补充）---\n"
                        f"{log_content}\n"
                        f"请基于以上完整执行证据重新生成方案。"
                    )
                    system_prompt += supplement
                    response = resolved_runtime.invoke_structured(
                        "mathematician", state, MathematicianResponse, system_prompt=system_prompt
                    )
                except Exception as exc:
                    logger.warning("读取执行日志 %s 失败: %s", run_id, exc)
            else:
                logger.warning("Mathematician 请求了不存在的执行日志: %s", run_id)
        # 处理 LLM 主动提出的分支重建请求
        if response.branch_requested and archive:
            requested_version = response.branch_from_version
            if requested_version and any(s.version == requested_version for s in archive):
                control.branch_from_version = requested_version
            else:
                control.branch_from_version = archive[-1].version
            logger.info(
                "Mathematician 主动请求分支重建到 %s，原因：%s",
                control.branch_from_version,
                response.branch_reason,
            )

        source_version = archive[-1].version if archive else None
        strategy_types = ("baseline", "primary", "challenge", "alternative")
        control.top_k_plans = [
            PlanCandidate(
                id=(
                    "candidate_"
                    + hashlib.sha256(
                        f"{control.debate_round}|{i}|{plan.get('title', '')}".encode("utf-8")
                    ).hexdigest()[:6]
                ),
                title=plan.get("title", "未命名方案"),
                description=plan.get("description", ""),
                strategy_type=plan.get("strategy_type") or strategy_types[i],
                input_data=plan.get("input_data", []),
                assumptions=plan.get("assumptions", []),
                mathematical_object=plan.get("mathematical_object", ""),
                parameter_estimation=plan.get("parameter_estimation", ""),
                solution_method=plan.get("solution_method", ""),
                expected_outputs=plan.get("expected_outputs", []),
                validation_method=plan.get("validation_method", ""),
                failure_conditions=plan.get("failure_conditions", []),
                # Mathematician 不再自评；由 Realist 独立赋值。
                innovation_score=0,
                feasibility_score=0,
                source_snapshot_version=control.branch_from_version or source_version,
            )
            for i, plan in enumerate(response.plans)
        ]
    except Exception as exc:
        logger.error("Mathematician LLM 调用失败: %s", exc)
        # fallback: 保留旧方案或生成默认方案
        if not control.top_k_plans:
            source_version = archive[-1].version if archive else None
            control.top_k_plans = [
                PlanCandidate(
                    id="plan_fallback",
                    title="默认方案",
                    description="LLM 调用失败，使用默认建模方案。",
                    innovation_score=60,
                    feasibility_score=60,
                    source_snapshot_version=control.branch_from_version or source_version,
                )
            ]

    # V17 运行过程记录：发散留痕（候选方案/评分/分支重建）
    plans_summary = [
        {
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "strategy_type": p.strategy_type,
        }
        for p in control.top_k_plans
    ]
    entry = _emit_process(
        runtime, control, state, "mathematician", "plans_generated",
        f"第 {control.debate_round} 轮发散：生成 {len(plans_summary)} 个候选方案",
        {
            "debate_round": control.debate_round,
            "plans": plans_summary,
            "branch_from_version": control.branch_from_version,
            "rebrainstorm_feedback": list(control.rebrainstorm_feedback),
        },
        prompt_text=system_prompt,
        prompt_tag=f"round{control.debate_round}",
    )
    return {"control": control, "prompt_audit": audit, "process_log": [entry]}


def realist_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """挑刺与剪枝：从数据、算力、常识三维度评估每个方案。

    - 题目匹配/数据假设/数学正确性/可计算性未过硬门槛 → kill
    - 可验证性未过硬门槛 → reject（打回补全）
    - 创新性仅作加分项，不作为淘汰条件
    - 其余 → keep
    - 若全部被剪枝 → need_rebrainstorm = True，路由回 Mathematician
    - 否则选综合评分最高的 keep 方案为 selected_plan_id
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)

    if not control.top_k_plans:
        control.need_rebrainstorm = True
        control.phase = "plan_scored"
        entry = _emit_process(
            runtime, control, state, "realist", "plans_scored",
            "无候选方案可评审，标记重新发散",
            {"selected_plan_id": None, "need_rebrainstorm": True},
        )
        return {"control": control, "process_log": [entry]}

    w_inn = control.innovation_weight
    w_fea = control.feasibility_weight
    plan_by_id = {plan.id: plan for plan in control.top_k_plans}

    system_prompt, audit = _prompt_audit("realist", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "realist", state, RealistResponse, system_prompt=system_prompt
        )
        # 用 LLM 的 per-plan 评估更新 top_k_plans
        for evaln in response.plan_evaluations:
            plan = plan_by_id.get(evaln.plan_id)
            if plan:
                plan.innovation_score = evaln.innovation_score
                plan.feasibility_score = evaln.feasibility_score
                plan.problem_fit_score = evaln.problem_fit_score
                plan.data_assumption_score = evaln.data_assumption_score
                plan.mathematical_correctness_score = evaln.mathematical_correctness_score
                plan.verifiability_score = evaln.verifiability_score
                plan.computability_score = evaln.computability_score
                plan.fatal_risks = list(evaln.fatal_risks)
                plan.review_feedback = evaln.feedback

        # 应用阈值剪枝：覆盖未在 LLM 评估中的方案
        for plan in control.top_k_plans:
            _apply_plan_gate(plan, control.feasibility_threshold)

        kept = [p for p in control.top_k_plans if p.verdict == "keep"]
        if kept:
            selected = max(kept, key=lambda p: p.total_score(w_inn, w_fea))
            control.selected_plan_id = selected.id
            control.innovation_score = selected.innovation_score
            control.feasibility_score = selected.feasibility_score
            control.need_rebrainstorm = False
        else:
            # 全部被剪枝 → 需要重新头脑风暴
            control.need_rebrainstorm = True
            control.selected_plan_id = None
            control.innovation_score = 0
            control.feasibility_score = 0
    except Exception as exc:
        logger.error("Realist LLM 调用失败: %s", exc)
        # fallback: 应用阈值剪枝并选最优
        for plan in control.top_k_plans:
            _apply_plan_gate(plan, control.feasibility_threshold)

        kept = [p for p in control.top_k_plans if p.verdict == "keep"]
        if kept:
            selected = max(kept, key=lambda p: p.total_score(w_inn, w_fea))
            control.selected_plan_id = selected.id
            control.innovation_score = selected.innovation_score
            control.feasibility_score = selected.feasibility_score
            control.need_rebrainstorm = False
        else:
            viable = [
                plan
                for plan in control.top_k_plans
                if plan.feasibility_score >= control.feasibility_threshold
            ]
            selected = max(
                viable or control.top_k_plans,
                key=lambda p: p.total_score(w_inn, w_fea),
            )
            control.selected_plan_id = selected.id
            control.innovation_score = selected.innovation_score
            control.feasibility_score = selected.feasibility_score
            control.need_rebrainstorm = not viable

    _update_plan_pool(control, w_inn, w_fea)
    control.phase = "plan_scored"
    # V17 运行过程记录：剪枝评分留痕（verdict/选中方案/阈值）
    entry = _emit_process(
        runtime, control, state, "realist", "plans_scored",
        (
            f"剪枝评分：选中 {control.selected_plan_id}（创新 {control.innovation_score}"
            f"/可行 {control.feasibility_score}，阈值 "
            f"{control.innovation_threshold}/{control.feasibility_threshold}）"
            if control.selected_plan_id
            else "全部方案被剪枝，需要重新发散"
        ),
        {
            "evaluations": [
                {
                    "plan_id": p.id,
                    "innovation_score": p.innovation_score,
                    "feasibility_score": p.feasibility_score,
                    "problem_fit_score": p.problem_fit_score,
                    "data_assumption_score": p.data_assumption_score,
                    "mathematical_correctness_score": p.mathematical_correctness_score,
                    "verifiability_score": p.verifiability_score,
                    "computability_score": p.computability_score,
                    "verdict": p.verdict,
                }
                for p in control.top_k_plans
            ],
            "innovation_threshold": control.innovation_threshold,
            "feasibility_threshold": control.feasibility_threshold,
            "selected_plan_id": control.selected_plan_id,
            "innovation_score": control.innovation_score,
            "feasibility_score": control.feasibility_score,
            "need_rebrainstorm": control.need_rebrainstorm,
        },
        prompt_text=system_prompt,
        prompt_tag=f"round{control.debate_round}",
    )
    return {"control": control, "prompt_audit": audit, "process_log": [entry]}


def arbiter_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    archive = state.get("ltm_archive", [])

    # 如果没有历史版本，直接放行
    if not archive:
        control.phase = "plan_arbitrated"
        entry = _emit_process(
            runtime, control, state, "arbiter", "arbitration",
            "无历史版本可仲裁，直接放行进入 Clarifier",
            {"action": "approve", "rollback_to_version": None},
        )
        return {"control": control, "process_log": [entry]}

    system_prompt, audit = _prompt_audit("arbiter", state, runtime)
    try:
        from modeling_assistant.schemas.responses import ArbiterResponse

        response = resolved_runtime.invoke_structured(
            "arbiter", state, ArbiterResponse, system_prompt=system_prompt
        )
        # 处理 LLM 请求的按需 Archive 详情查询
        if response.requested_version and archive:
            requested = response.requested_version
            from modeling_assistant.memory.archive import checkout_snapshot
            try:
                full_ltm = checkout_snapshot(archive, requested)
                supplement = (
                    f"\n\n--- 版本 {requested} 的完整动态 LTM（按需补充）---\n"
                    f"{full_ltm.model_dump_json(indent=2)}\n"
                    f"请基于以上完整信息重新对比和决策。"
                )
                system_prompt += supplement
                response = resolved_runtime.invoke_structured(
                    "arbiter", state, ArbiterResponse, system_prompt=system_prompt
                )
            except ValueError:
                logger.warning("Arbiter 请求了不存在的版本: %s", requested)
        if response.action == "rollback" and response.rollback_version:
            control.rollback_to_version = response.rollback_version
            control.phase = "rollback_recommended"
            control.hitl_required = True
            control.hitl_stage = "arbitration"
        else:
            control.phase = "plan_arbitrated"
    except Exception as exc:
        logger.error("Arbiter LLM 调用失败: %s", exc)
        # fallback: 如果辩论轮数过多且有历史版本，回滚到最后一个
        if control.debate_round > control.max_debate_rounds:
            control.rollback_to_version = archive[-1].version
            control.phase = "rollback_recommended"
            control.hitl_required = True
            control.hitl_stage = "arbitration"
        else:
            control.phase = "plan_arbitrated"

    # V17 运行过程记录：仲裁留痕
    entry = _emit_process(
        runtime, control, state, "arbiter", "arbitration",
        (
            f"建议回滚到 {control.rollback_to_version}"
            if control.rollback_to_version
            else "仲裁通过，放行进入 Clarifier"
        ),
        {
            "rollback_to_version": control.rollback_to_version,
            "hitl_required": control.hitl_required,
            "phase": control.phase,
        },
        prompt_text=system_prompt,
    )
    return {"control": control, "prompt_audit": audit, "process_log": [entry]}


# ═══════════════════════════════════════════════════════════════════
# 阶段三：方案具体化与 LTM 快照管理
# ═══════════════════════════════════════════════════════════════════

def clarifier_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """总结胜出方案，注入 LTM，进行符号查重与公式闭环校验。

    - 写入新 LTM 前先归档旧 LTM 到 Archive
    - major_bump：若 objective 发生根本性变化 → v2.0，否则 v1.x
    - 校验符号闭环；失败则在 prompt_audit 记录错误

    V11 修复：在写入 dynamic_ltm 之前，调用第二层常量校验。
    检查 assumptions/equations 中的数值是否与 problem_facts 一致。
    如果出现冲突（如 3 m/s 被写成 1 m/s），记录到 audit 与 coder_error_log，
    让下游节点能看到常量偏差，但不阻塞流程（避免死循环）。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    old_dynamic_ltm = _dynamic_ltm(state)
    static_ltm = _static_ltm(state)
    archive = state.get("ltm_archive", [])

    system_prompt, audit = _prompt_audit("clarifier", state, runtime)
    commit_summary = ""
    try:
        response = resolved_runtime.invoke_structured(
            "clarifier", state, ClarifierResponse, system_prompt=system_prompt
        )
        commit_summary = response.commit_summary
        new_dynamic_ltm = DynamicLTM(
            assumptions=response.assumptions,
            nomenclature=response.nomenclature,
            equations=response.equations,
            objective=response.objective,
            solution_outline=response.solution_outline,
            identifiability_checks=response.identifiability_checks,
            constant_relevance=response.constant_relevance,
        )
    except Exception as exc:
        logger.error("Clarifier LLM 调用失败: %s", exc)
        # fallback: 基于选中的方案构建 LTM
        selected_plan = next(
            (plan for plan in control.top_k_plans if plan.id == control.selected_plan_id),
            None,
        )
        plan_title = selected_plan.title if selected_plan else "待定方案"
        plan_description = selected_plan.description if selected_plan else "等待进一步澄清。"
        new_dynamic_ltm = DynamicLTM(
            assumptions=[
                "所有下游节点只能依据当前动态 LTM 中的设定工作。",
                "若数据或算力约束冲突，优先触发回滚或返回 Architect。",
            ],
            nomenclature={
                "S_inn": "创新性评分",
                "S_fea": "可行性评分",
                "Score_total": "综合评分",
            },
            equations=["Score_total = 0.5 * S_inn + 0.5 * S_fea"],
            objective=f"细化并执行：{plan_title}",
            solution_outline=plan_description,
            identifiability_checks=["LLM 调用失败，需由人类完成参数可识别性检查。"],
            constant_relevance={},
        )

    # 符号查重校验（公式闭环已移除，见 validation.py）
    validation_errors = validate_dynamic_ltm(new_dynamic_ltm)
    if validation_errors:
        audit["clarifier_validation_errors"] = "; ".join(validation_errors)
        logger.warning("Clarifier LTM 符号查重警告：%s", validation_errors)
        # 移除内部修复循环：符号查重很少失败，即使失败也由 milestone_reviewer_1 审查
        # 原修复循环实测 5 次全部失败，徒耗 LLM 调用

    # V11 修复：第二层常量校验 —— 在写入 dynamic_ltm 之前检查数值一致性
    from modeling_assistant.validation.constants import check_ltm_against_facts
    constant_issues = check_ltm_against_facts(new_dynamic_ltm, static_ltm)
    if constant_issues:
        audit["clarifier_constant_issues"] = "; ".join(constant_issues)
        logger.warning("Clarifier 常量校验告警：%s", constant_issues)
        # 把常量校验告警附加到 rebrainstorm_feedback，让 milestone_reviewer 看到
        # 但不阻塞写入（避免死循环），由下游节点决定是否需要重新 brainstorm
        # 这里把告警放进 coder_error_log 以便 Architect/Coder 能看到
        control.coder_error_log.extend(constant_issues)

    # 归档：snapshot 存 new_dynamic_ltm（提交快照语义，而非"被覆盖的旧版"）
    # 这样 rollback 到 vN 取出的是"vN 这次提交的内容"，而非"vN 之前的内容"。
    # 第一次调用时 old_dynamic_ltm 为空，archive 为空 → v1.0 是首次提交的内容。
    major_change = (
        old_dynamic_ltm.objective != ""
        and new_dynamic_ltm.objective != old_dynamic_ltm.objective
    )
    checkpoint_id = None
    if config and "configurable" in config:
        checkpoint_id = config["configurable"].get("checkpoint_id")
    snapshot = make_snapshot(
        new_dynamic_ltm,
        archive,
        reason="Clarifier committed new Core State Two.",
        commit_summary=commit_summary,
        major_bump=major_change,
        checkpoint_id=checkpoint_id,
    )

    control.phase = "dynamic_ltm_committed"
    control.hitl_required = True
    control.hitl_stage = "architecture"
    # 重置 trigger_clarifier_revision：Clarifier 已完成修正，下游 collect_artifacts
    # 可正常前进到 Writer。否则该标志会一直为 True，导致 collect_artifacts 永久跳过 Writer
    control.trigger_clarifier_revision = False
    # Meta-Router 决策已消费：重置 meta_decision
    control.meta_decision = ""
    # V17 运行过程记录：LTM 提交留痕（供重新评估方案）
    entry = _emit_process(
        runtime, control, state, "clarifier", "ltm_committed",
        f"提交动态 LTM v{snapshot.version}：目标「{new_dynamic_ltm.objective[:60]}」",
        {
            "version": snapshot.version,
            "commit_summary": commit_summary,
            "objective": new_dynamic_ltm.objective,
            "assumptions": list(new_dynamic_ltm.assumptions),
            "nomenclature": dict(new_dynamic_ltm.nomenclature),
            "equations": list(new_dynamic_ltm.equations),
            "solution_outline": new_dynamic_ltm.solution_outline,
            "identifiability_checks": list(new_dynamic_ltm.identifiability_checks),
            "constant_relevance": dict(new_dynamic_ltm.constant_relevance),
            "constant_issues": constant_issues,
            "validation_errors": validation_errors,
        },
        prompt_text=system_prompt,
        prompt_tag=f"v{snapshot.version}",
    )
    return {
        "dynamic_ltm": new_dynamic_ltm,
        "ltm_archive": [snapshot],
        "control": control,
        "prompt_audit": audit,
        "process_log": [entry],
    }


def milestone_reviewer_1_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Milestone Reviewer 1：阶段一自动评审。

    检查 Clarifier 产出的动态 LTM 是否完整、与静态 LTM 一致。
    - approval=True：进入 HITL 1，由人类最终决断。
    - approval=False：携带 feedback 返回 Mathematician 重新发散。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)

    # 硬性校验：只检查非空，完全移除 validate_dynamic_ltm 调用
    # 理由：
    # 1. validate_dynamic_ltm 已降级为软警告（仅符号查重，见 validation.py）
    # 2. Clarifier 已尽力修复，milestone 用同一规则再判只会死循环
    # 3. 真正的符号一致性由 Coder 执行反馈 + LLM 语义审查保证
    hard_issues: list[str] = []
    if not dynamic_ltm.assumptions:
        hard_issues.append("假设列表为空。")
    if not dynamic_ltm.nomenclature:
        hard_issues.append("符号表为空。")
    if not dynamic_ltm.equations:
        hard_issues.append("公式列表为空。")
    if not dynamic_ltm.objective:
        hard_issues.append("目标函数/优化目标为空。")

    if hard_issues:
        control.modeling_revision_count += 1
        control.phase = "milestone_review_1_rejected"
        control.need_rebrainstorm = True
        control.rebrainstorm_feedback.extend(hard_issues)
        entry = _emit_process(
            runtime, control, state, "milestone_reviewer_1", "milestone_review",
            "里程碑评审：硬性校验未通过，打回重新建模",
            {"approval": False, "issues": list(hard_issues)},
        )
        return {"control": control, "process_log": [entry]}

    system_prompt, audit = _prompt_audit("milestone_reviewer_1", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "milestone_reviewer_1", state, MilestoneReviewer1Response, system_prompt=system_prompt
        )
        if not response.approval:
            control.modeling_revision_count += 1
            control.phase = "milestone_review_1_rejected"
            control.need_rebrainstorm = True
            control.rebrainstorm_feedback.extend(response.issues)
            control.rebrainstorm_feedback.append(response.feedback)
        else:
            control.phase = "milestone_review_1_approved"
    except Exception as exc:
        logger.error("Milestone Reviewer 1 LLM 调用失败: %s", exc)
        # LLM 失败时保守放行，避免阻塞人类 HITL
        control.phase = "milestone_review_1_approved"

    # V17 运行过程记录：里程碑评审留痕
    approved = control.phase == "milestone_review_1_approved"
    entry = _emit_process(
        runtime, control, state, "milestone_reviewer_1", "milestone_review",
        "里程碑评审通过，进入架构确认" if approved else "里程碑评审打回，重新发散",
        {
            "approval": approved,
            "issues": list(control.rebrainstorm_feedback)[-5:],
        },
        prompt_text=system_prompt,
    )
    return {"control": control, "prompt_audit": audit, "process_log": [entry]}


def load_bearing_analyzer_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """承重结构分析（V18）：把结论与承重依赖显式连接，生成验证契约。

    在 milestone_reviewer_1 通过后、hitl_architecture 之前运行。输出
    LoadBearingMap 写入 artifacts.load_bearing_map，供人类审核、Architect
    规划、Writer 呈现、final_reviewer 对账。LLM 失败时降级为纯规则保守图，
    并标记 analysis_incomplete，不允许"未分析"静默通过。
    """
    from modeling_assistant.analysis.load_bearing import build_load_bearing_map

    resolved_runtime = _runtime(runtime)
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    static_ltm = _static_ltm(state)
    empirical = _empirical(state)
    artifacts = _artifacts(state)

    archive = state.get("ltm_archive", [])
    ltm_version = archive[-1].version if archive else "v0.0"
    response = None
    audit: dict[str, str] = {}
    system_prompt = ""
    try:
        system_prompt, audit = _prompt_audit("load_bearing_analyzer", state, runtime)
        response = resolved_runtime.invoke_structured(
            "load_bearing_analyzer",
            state,
            LoadBearingAnalysisResponse,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        logger.warning("承重分析 LLM 调用失败，降级为纯规则保守图: %s", exc)

    load_bearing_map = build_load_bearing_map(
        dynamic_ltm,
        static_ltm,
        empirical,
        control,
        response=response,
        ltm_version=ltm_version,
    )
    artifacts.load_bearing_map = load_bearing_map
    control.phase = "load_bearing_analyzed"
    entry = _emit_process(
        runtime,
        control,
        state,
        "load_bearing_analyzer",
        "load_bearing_analyzed",
        (
            f"承重分析完成：{len(load_bearing_map.constructs)} 个构造，"
            f"根缺口 {len(load_bearing_map.root_gaps)}，"
            f"锚点缺口 {len(load_bearing_map.anchor_gaps)}，"
            f"形态风险 {len(load_bearing_map.shape_risks)}"
        ),
        {
            "ltm_version": ltm_version,
            "analysis_incomplete": load_bearing_map.analysis_incomplete,
            "root_gaps": list(load_bearing_map.root_gaps),
            "anchor_gaps": list(load_bearing_map.anchor_gaps),
            "shape_risks": list(load_bearing_map.shape_risks),
            "conclusions": [
                {
                    "id": v.id,
                    "question_ref": v.question_ref,
                    "verdict_shape": v.verdict_shape,
                    "fallback_required": v.fallback_required,
                }
                for v in load_bearing_map.conclusions
            ],
        },
        prompt_text=system_prompt if response is not None else None,
    )
    return {
        "artifacts": artifacts,
        "control": control,
        "prompt_audit": audit,
        "process_log": [entry],
    }


def hitl_plan_selection_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Human 模式下在 Clarifier 固化 LTM 前让人选择四个候选方案。"""
    control = _control(state)
    ranked = sorted(
        control.top_k_plans,
        key=lambda p: p.total_score(control.innovation_weight, control.feasibility_weight),
        reverse=True,
    )
    plans = [
        {
            "id": p.id,
            "title": p.title,
            "strategy_type": p.strategy_type,
            "description": p.description,
            "problem_fit_score": p.problem_fit_score,
            "data_assumption_score": p.data_assumption_score,
            "mathematical_correctness_score": p.mathematical_correctness_score,
            "verifiability_score": p.verifiability_score,
            "computability_score": p.computability_score,
            "innovation_score": p.innovation_score,
            "feasibility_score": p.feasibility_score,
            "verdict": p.verdict,
            "validation_method": p.validation_method,
            "failure_conditions": p.failure_conditions,
            "fatal_risks": p.fatal_risks,
            "review_feedback": p.review_feedback,
        }
        for p in ranked
    ]
    selectable = {p.id for p in ranked if p.verdict == "keep"}
    decision = interrupt({
        "stage": "plan_selection",
        "message": "请在 Clarifier 固化数学模型前审核四个独立候选方案及其致命风险。",
        "hint": (
            f"输入 'approve' 采用当前方案（{control.selected_plan_id or '无'}）；"
            f"'choose <plan_id>' 改选（可选 {sorted(selectable)}）；"
            "'revise <反馈>' 返回 Mathematician 重新生成四个方案。"
        ),
        "plans": plans,
        "selected_plan_id": control.selected_plan_id,
    })
    action = _parse_hitl_decision(decision)
    if action["type"] == "choose":
        chosen = str(action.get("version") or "").strip()
        if chosen in selectable:
            control.selected_plan_id = chosen
            selected = next(p for p in ranked if p.id == chosen)
            control.innovation_score = selected.innovation_score
            control.feasibility_score = selected.feasibility_score
            control.phase = "plan_selection_approved"
        else:
            control.phase = "plan_selection_approved"
            control.rebrainstorm_feedback.append(f"忽略无效方案选择：{chosen}")
    elif action["type"] in {"revise", "reject", "redirect"}:
        feedback = str(action.get("version") or "").strip()
        if feedback:
            control.rebrainstorm_feedback.append(f"人工方案选择反馈：{feedback}")
        control.need_rebrainstorm = True
        control.modeling_revision_count += 1
        control.phase = "plan_selection_rebrainstorm"
    else:
        control.phase = "plan_selection_approved"
    control.hitl_required = False
    control.hitl_stage = "none"
    entry = _emit_process(
        runtime, control, state, "hitl_plan_selection", "plan_selection_hitl",
        f"人工方案选择：{action['type']}，当前方案 {control.selected_plan_id}",
        {"decision": action["type"], "selected_plan_id": control.selected_plan_id},
    )
    return {"control": control, "process_log": [entry]}


def hitl_architecture_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Milestone Reviewer 1：架构确认前的人类审核。

    首次进入时 interrupt() 暂停图执行，等待用户输入。
    V23：展示方案池（Realist 保留的前 N 个 keep 方案）各自的实现路径，
    人类测试后可用 choose <plan_id> 定夺采用哪个方案。
    重点审核对象是 Clarifier 提出的模型假设（可能影响全局走向的关键假设已在
    assumptions 中以【关键】标注；每条假设带【全文】/【问题N】放置标签）。用户输入：
    - 'approve' 采用评分最高方案并放行进入架构设计
    - 'choose <plan_id>' 改选方案池中的其他方案（回 Clarifier 按新方案重新提炼）
    - 'rollback <version>' 回滚到指定 LTM 版本
    - 'revise <假设反馈>' 打回 Clarifier 按反馈修改假设与分类后重新评审
    """
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    plan_by_id = {p.id: p for p in control.top_k_plans}
    pool_ids = control.plan_pool_ids or [
        p.id
        for p in sorted(
            (p for p in control.top_k_plans if p.verdict == "keep"),
            key=lambda p: p.total_score(
                control.innovation_weight, control.feasibility_weight
            ),
            reverse=True,
        )[: control.plan_pool_size]
    ]
    plan_pool = [
        {
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "innovation_score": p.innovation_score,
            "feasibility_score": p.feasibility_score,
            "verdict": p.verdict,
        }
        for pid in pool_ids
        for p in [plan_by_id.get(pid)]
        if p is not None
    ]
    decision = interrupt({
        "stage": "architecture",
        "message": (
            f"请审核当前建模方案：前 {len(plan_pool)} 个候选方案及其实现路径如下"
            "（方案池），可在测试后定夺采用哪一个；同时逐条审核模型假设及其分类："
            "【全文】假设进入全文假设章（3_assumptions.tex，≤6 条）；"
            "【问题N】假设只进入对应小题章节；【关键】假设必须配验证实验。"
        ),
        "hint": (
            f"输入 'approve' 采用评分最高方案（{control.selected_plan_id or '无'}）"
            "进入架构设计；"
            f"'choose <plan_id>' 改选方案池中的其他方案（可选 {[p['id'] for p in plan_pool]}）；"
            "'rollback v1.0' 回滚到指定版本；"
            "'revise <反馈>' 打回 Clarifier 修改假设与分类。"
        ),
        "plan_pool": plan_pool,
        "assumptions": list(dynamic_ltm.assumptions or []),
        "assumption_review": classify_assumptions(dynamic_ltm.assumptions or []),
        "dynamic_ltm": dynamic_ltm.model_dump(),
        "load_bearing_summary": _load_bearing_summary(state),
        "control_summary": {
            "phase": state.get("control", ControlState()).phase,
            "selected_plan_id": state.get("control", ControlState()).selected_plan_id,
            "innovation_score": state.get("control", ControlState()).innovation_score,
            "feasibility_score": state.get("control", ControlState()).feasibility_score,
        },
    })

    action = _parse_hitl_decision(decision)

    if action["type"] == "rollback":
        control.rollback_to_version = action.get("version")
        control.rollback_source = "architecture_hitl"
        control.phase = "hitl_rollback_requested"
    elif action["type"] == "revise":
        # 人类要求修改假设：携带反馈回 Clarifier 重新提炼 LTM（不进建模发散）
        control.hitl_required = False
        control.hitl_stage = "none"
        control.rollback_source = "none"
        control.phase = "architecture_revised"
        feedback = action.get("version") or ""
        if feedback:
            control.rebrainstorm_feedback.append(f"人类架构审核（假设）反馈：{feedback}")
        logger.info(
            "架构 HITL：人类打回假设并要求修改（反馈=%s），回 Clarifier 重新提炼",
            feedback[:120],
        )
    elif action["type"] == "choose":
        chosen = str(action.get("version") or "").strip()
        valid_ids = {p["id"] for p in plan_pool}
        if chosen in valid_ids:
            if chosen != control.selected_plan_id:
                control.selected_plan_id = chosen
                control.phase = "architecture_plan_switched"
                logger.info(
                    "架构 HITL：人类改选方案 %s，回 Clarifier 按新方案重新提炼 LTM",
                    chosen,
                )
            else:
                control.phase = "architecture_approved"
        else:
            logger.warning(
                "架构 HITL：人类选择的方案无效：%s（可选 %s），保留当前方案",
                chosen,
                sorted(valid_ids),
            )
            control.phase = "architecture_approved"
        control.hitl_required = False
        control.hitl_stage = "none"
        control.rollback_source = "none"
    else:
        control.phase = "architecture_approved"
        control.hitl_required = False
        control.hitl_stage = "none"
        control.rollback_source = "none"
    entry = _emit_process(
        runtime, control, state, "hitl_architecture", "architecture_hitl",
        (
            {
                "rollback": f"人类回滚到 {control.rollback_to_version}",
                "revise": "人类打回假设并要求修改，回 Clarifier 重新提炼",
                "choose": (
                    f"人类改选方案 {control.selected_plan_id}，回 Clarifier 重新提炼"
                    if control.phase == "architecture_plan_switched"
                    else "人类确认当前方案，进入架构设计"
                ),
            }.get(
                action["type"],
                "人类批准建模方案（含假设），进入架构设计",
            )
        ),
        {
            "decision": action["type"],
            "version": action.get("version"),
            "feedback": action.get("version"),
            "selected_plan_id": control.selected_plan_id,
            "innovation_score": control.innovation_score,
            "feasibility_score": control.feasibility_score,
        },
    )
    return {"control": control, "process_log": [entry]}


def hitl_arbitration_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Arbiter 回滚建议的人类确认。

    首次进入时 interrupt() 暂停图执行，等待用户输入。
    用户输入 'approve' 接受回滚建议，'reject' 拒绝回滚继续进入 Clarifier。
    """
    control = _control(state)
    archive = state.get("ltm_archive", [])

    decision = interrupt({
        "stage": "arbitration",
        "message": f"Arbiter 建议回滚到版本 {control.rollback_to_version}。",
        "hint": "输入 'approve' 接受回滚，或 'reject' 拒绝回滚继续进入 Clarifier。",
        "rollback_to_version": control.rollback_to_version,
        "archive_versions": [snap.version for snap in archive],
        "control_summary": {
            "phase": control.phase,
            "debate_round": control.debate_round,
            "selected_plan_id": control.selected_plan_id,
            "innovation_score": control.innovation_score,
            "feasibility_score": control.feasibility_score,
        },
    })

    action = _parse_hitl_decision(decision)
    if action["type"] == "approve":
        control.phase = "arbitration_rollback_confirmed"
        control.rollback_source = "arbitration"
        control.hitl_required = False
        control.hitl_stage = "none"
    else:
        control.rollback_to_version = None
        control.rollback_source = "none"
        control.phase = "arbitration_rejected"
        control.hitl_required = False
        control.hitl_stage = "none"
    entry = _emit_process(
        runtime, control, state, "hitl_arbitration", "arbitration_hitl",
        (
            "人类接受回滚建议"
            if action["type"] == "approve"
            else "人类拒绝回滚，继续进入 Clarifier"
        ),
        {
            "decision": action["type"],
            "rollback_to_version": control.rollback_to_version,
        },
    )
    return {"control": control, "process_log": [entry]}


def hitl_modeling_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """建模预算耗尽时的人类介入节点。

    当 modeling_revision_count >= modeling_revision_budget 时触发，
    让人类决断下一步，而非直接产出"待验证"论文。
    V21：主方案结果为退化解（所有样本/分组结果相同、答案列常量、最优值全部
    落在边界值）时必须判 fail，不得以「结果诚实/待验证」为由接受。

    三个选项：
    - accept：接受失败，前进到 collect_artifacts（现行"待验证"降级行为）
    - retry：重置 budget，回 architect 重试（沿用当前 selected_plan，但人类介入后继续）
    - redirect：重置 budget，回 mathematician 重新发散（人类可注入 direction_hint 换方向）
    """
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    artifacts = _artifacts(state)
    result_paths = getattr(artifacts, "result_paths", []) or []

    # 设置 HITL 标志，让 cli.py 主循环识别并处理中断
    control.hitl_required = True
    control.hitl_stage = "modeling"

    decision = interrupt({
        "stage": "modeling",
        "message": (
            f"建模预算已耗尽（{control.modeling_revision_count}/{control.modeling_revision_budget}）。"
            "系统多次尝试未能产出通过验证的结果，请人类决断下一步。"
            "若主方案结果是退化解（所有样本/分组结果相同、答案列常量、"
            "最优值全部落在边界值），必须判 fail，不得以「结果诚实/待验证」为由接受。"
        ),
        "hint": (
            "输入 'accept' 接受失败并产出'待验证'论文（仅限非退化解："
            "结果有区分度但数值待验证）；"
            "输入 'retry' 重置预算并回到 Architect 重试当前方案；"
            "输入 'redirect <方向提示>' 重置预算并回到 Mathematician 重新发散。"
        ),
        "control_summary": {
            "phase": control.phase,
            "budget_used": control.modeling_revision_count,
            "budget_limit": control.modeling_revision_budget,
            "current_sub_question_index": control.current_sub_question_index,
            "current_sub_question": _current_sub_question(control),
            "selected_plan_id": control.selected_plan_id,
            "trigger_clarifier_revision": control.trigger_clarifier_revision,
            "meta_decision": control.meta_decision,
            "meta_direction_hint": control.meta_direction_hint,
        },
        "dynamic_ltm_summary": {
            "objective": dynamic_ltm.objective,
            "assumptions_count": len(dynamic_ltm.assumptions),
            "equations_count": len(dynamic_ltm.equations),
        },
        "result_paths": result_paths,
        "has_backup_results": bool(getattr(artifacts, "result_paths", None)),
    })

    action = _parse_hitl_decision(decision)
    # 重置 HITL 标志（HITL 已执行）
    control.hitl_required = False
    control.hitl_stage = "none"

    if action["type"] == "retry":
        # 重置预算，回 architect 重试当前方案
        control.modeling_revision_count = 0
        control.trigger_clarifier_revision = False
        control.meta_decision = ""
        control.phase = "hitl_modeling_retry"
        logger.info("HITL modeling: 人类选择 retry，重置预算回 Architect 重试")
    elif action["type"] == "redirect":
        # 重置预算，回 mathematician 重新发散
        control.modeling_revision_count = 0
        control.trigger_clarifier_revision = False
        control.need_rebrainstorm = True
        control.rebrainstorm_feedback.append("人类介入：要求重新发散建模方向")
        # 人类可注入方向提示
        hint = action.get("version") or ""
        if hint:
            control.meta_direction_hint = hint
            control.rebrainstorm_feedback.append(f"人类方向提示：{hint}")
        control.meta_decision = ""
        control.phase = "hitl_modeling_redirect"
        logger.info("HITL modeling: 人类选择 redirect，重置预算回 Mathematician 重新发散")
    else:
        # accept：接受失败。小题循环模式下接受当前小题并推进下一题；
        # 单题模式（无小题清单）保留原“待验证论文”语义。
        if control.sub_questions:
            # V17 结果注册表：人类接受降级结果，锁为 degraded 权威结果
            control = _finalize_authoritative_result(
                state, control, artifacts, status="degraded",
                feedback=["HITL modeling：人类接受当前结果（可能带缺陷）"],
            )
            control, _has_next = _advance_sub_question(
                state, control, dynamic_ltm, artifacts,
                status="passed",
                feedback=["HITL modeling：人类接受当前结果（可能带缺陷）"],
            )
            control.phase = "sub_question_passed"
            logger.info(
                "HITL modeling: 人类选择 accept，接受当前小题并推进（下一题=%d）",
                control.current_sub_question_index,
            )
            entry = _emit_process(
                runtime, control, state, "hitl_modeling", "modeling_hitl",
                "建模预算耗尽：人类接受当前小题（降级）并推进",
                {"decision": "accept", "current_sub_question_index": control.current_sub_question_index - 1},
            )
            return {"control": control, "dynamic_ltm": DynamicLTM(), "process_log": [entry]}
        # V17 结果注册表：单题模式降级接受同样锁定（index=0）
        control = _finalize_authoritative_result(
            state, control, artifacts, status="degraded",
            feedback=["HITL modeling：人类接受失败结果（待验证）"],
        )
        control.phase = "hitl_modeling_accepted"
        logger.info("HITL modeling: 人类选择 accept，接受失败产出'待验证'论文")

    entry = _emit_process(
        runtime, control, state, "hitl_modeling", "modeling_hitl",
        {
            "retry": "建模预算耗尽：人类选择 retry，回 Architect 重试",
            "redirect": "建模预算耗尽：人类选择 redirect，回 Mathematician 重新发散",
            "accept": "建模预算耗尽：人类接受现状（单题模式）",
        }.get(action["type"], "建模预算耗尽：人类决策"),
        {
            "decision": action["type"],
            "version": action.get("version"),
            "score": action.get("score"),
        },
    )
    return {"control": control, "process_log": [entry]}


def hitl_final_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Milestone Reviewer 2：终稿审查。

    首次进入时 interrupt() 暂停图执行，等待用户输入。
    重点展示 3_assumptions.tex，供人工检查假设章的放置与闭环。
    用户输入：
    - 'approve' 完成（可附带 'score <0-100>' 把本次评价回写示例库）
    - 'retry' 回到建模阶段重新打磨
    - 'rewrite <反馈>' 回到 Writer 按反馈重写论文（有预算上限，防死循环）
    """
    control = _control(state)
    resolved_runtime = _runtime(runtime)
    paper_dir = Path(resolved_runtime.output_path("paper"))
    decision = interrupt({
        "stage": "final",
        "message": (
            "请审核最终论文，重点检查模型假设章（3_assumptions.tex）："
            "1) 是否只有通俗的全文建模前提；"
            "2) 是否混入参数、模型分布、数据规则等实现细节；"
            "3) 关键假设是否都有敏感性/对照验证。"
        ),
        "hint": (
            "输入 'approve' 完成流程；'retry' 回到建模阶段重新打磨；"
            f"'rewrite <反馈>' 回到 Writer 重写论文（剩余 {max(0, control.paper_revision_budget - control.paper_revision_count)} 次）。"
        ),
        "artifacts_summary": state.get("artifacts", ArtifactBundle()).model_dump(),
        "paper_review_report": control.paper_review_report,
        "assumptions_section": _read_assumptions_section(paper_dir),
    })

    action = _parse_hitl_decision(decision)

    if action["type"] == "retry":
        control.rollback_to_version = action.get("version")
        control.rollback_source = "final_hitl"
        control.phase = "hitl_retry_requested"
    elif action["type"] == "rewrite":
        if control.paper_revision_count >= control.paper_revision_budget:
            # 预算耗尽：不能再回 Writer，等同 approve（完成流程）
            control.phase = "completed"
            control.hitl_required = False
            control.hitl_stage = "none"
            control.rollback_source = "none"
            logger.warning(
                "论文重写预算已耗尽（%d/%d），跳过 rewrite 直接完成",
                control.paper_revision_count,
                control.paper_revision_budget,
            )
        else:
            control.paper_revision_count += 1
            feedback = action.get("version") or ""
            control.paper_revision_feedback.append(feedback or "论文需要修订（未给出具体反馈）")
            control.phase = "paper_rewrite_requested"
            logger.info(
                "HITL 终审要求重写论文（%d/%d）：%s",
                control.paper_revision_count,
                control.paper_revision_budget,
                feedback[:120],
            )
    else:
        control.phase = "completed"
        control.hitl_required = False
        control.hitl_stage = "none"
        control.rollback_source = "none"

    result: dict = {"control": control}
    # Exemplar 反馈回写：用户评分以滑动平均更新卡片/指南质量权重
    score = action.get("score")
    if score is not None:
        resolved_runtime = _runtime(runtime)
        context = state.get("exemplars", ExemplarContext())
        if context.active:
            from modeling_assistant.memory.exemplar_feedback import apply_feedback_to_context

            updated = apply_feedback_to_context(
                context, score, resolved_runtime.settings.feedback_alpha
            )
            result["exemplars"] = updated
            # 持久化回写：更新卡片与指南文件，让反馈跨会话生效
            try:
                from modeling_assistant.data.exemplars import save_card, save_guide

                cards_dir = resolved_runtime.settings.exemplars_dir / "cards"
                for card in updated.cards:
                    save_card(card, cards_dir)
                if updated.guide is not None:
                    save_guide(
                        updated.guide,
                        resolved_runtime.settings.exemplars_dir / "guides",
                    )
            except Exception as exc:
                logger.warning("Exemplar 反馈落盘失败: %s", exc)
            logger.info(
                "Exemplar 反馈回写：score=%.2f，alpha=%.2f",
                score,
                resolved_runtime.settings.feedback_alpha,
            )
    # V17 运行过程记录：终审决策留痕
    entry = _emit_process(
        runtime, control, state, "hitl_final", "final_hitl",
        {
            "approve": "终审通过，流程完成",
            "retry": "终审要求回到建模阶段重做",
            "rewrite": "终审要求重写论文",
        }.get(action["type"], "终审完成"),
        {
            "decision": action["type"],
            "version": action.get("version"),
            "score": score,
            "paper_revision_count": control.paper_revision_count,
        },
    )
    result["process_log"] = [entry]
    return result


def _parse_hitl_decision(decision) -> dict:
    """解析用户输入，支持字符串和字典两种 resume 格式。"""
    score: float | None = None
    if isinstance(decision, dict):
        raw_score = decision.get("score")
        if isinstance(raw_score, (int, float)):
            score = float(raw_score) / 100.0 if raw_score > 1 else float(raw_score)
        return {
            "type": decision.get("action", "approve"),
            "version": decision.get("version"),
            "level": decision.get("level", ""),
            "target": decision.get("target"),
            "score": score,
        }
    text = str(decision).strip().lower()
    score_match = re.search(r"score\s+(\d+(?:\.\d+)?)", text)
    if score_match:
        raw_score = float(score_match.group(1))
        score = raw_score / 100.0 if raw_score > 1 else raw_score
    if text.startswith("rollback"):
        parts = text.split()
        version = parts[1] if len(parts) > 1 else None
        return {"type": "rollback", "version": version, "score": score}
    if text.startswith("retry"):
        parts = text.split()
        version = parts[1] if len(parts) > 1 else None
        return {"type": "retry", "version": version, "score": score}
    if text.startswith("reject"):
        return {"type": "reject", "version": None, "score": score}
    if text.startswith("redirect"):
        parts = text.split(maxsplit=1)
        hint = parts[1] if len(parts) > 1 else ""
        return {"type": "redirect", "version": hint, "score": score}
    if text.startswith("revise"):
        parts = text.split(maxsplit=1)
        feedback = parts[1] if len(parts) > 1 else ""
        return {"type": "revise", "version": feedback, "score": score}
    if text.startswith("rewrite"):
        parts = text.split(maxsplit=1)
        feedback = parts[1] if len(parts) > 1 else ""
        return {"type": "rewrite", "version": feedback, "score": score}
    if text.startswith("auto"):
        return {"type": "auto", "version": None, "score": score}
    if text.startswith("accept"):
        return {"type": "accept", "version": None, "score": score}
    if text.startswith("pass"):
        return {"type": "pass", "version": None, "score": score}
    fail_match = re.match(r"^fail\s+(code|architecture|model)\s*(.*)$", text)
    if fail_match:
        return {
            "type": "fail",
            "level": fail_match.group(1),
            "version": fail_match.group(2).strip(),
            "score": score,
        }
    cross_match = re.match(r"^cross\s+(\d+)\s*(.*)$", text)
    if cross_match:
        return {
            "type": "cross",
            "target": int(cross_match.group(1)) - 1,  # 用户输入 1-based
            "version": cross_match.group(2).strip(),
            "score": score,
        }
    if text.startswith("edit"):
        parts = text.split(maxsplit=1)
        return {"type": "edit", "version": parts[1] if len(parts) > 1 else "", "score": score}
    if text.startswith("choose"):
        parts = text.split(maxsplit=1)
        return {"type": "choose", "version": parts[1] if len(parts) > 1 else "", "score": score}
    if text.startswith("set"):
        parts = text.split(maxsplit=1)
        return {"type": "set", "version": parts[1] if len(parts) > 1 else "", "score": score}
    return {"type": "approve", "version": None, "score": score}


def architect_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    artifacts = _artifacts(state)

    # Meta-Router 决策已消费：重置 trigger_clarifier_revision 和 meta_decision
    if control.meta_decision:
        control.trigger_clarifier_revision = False
        control.meta_decision = ""

    system_prompt, audit = _prompt_audit("architect", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "architect", state, ArchitectResponse, system_prompt=system_prompt
        )
        artifacts.outline = response.outline
        artifacts.pseudocode = response.pseudocode
        # V12 修复：结果契约随 Architect 产物下传，供 Coder/ResultReviewer 使用
        artifacts.result_contract = response.result_contract
        # V13 新增：实现架构（算法摘要 + 图表/表格计划）下传，供人类审核与编程手实现
        artifacts.algorithms_summary = response.algorithms_summary
        artifacts.figures_plan = response.figures_plan
        artifacts.tables_plan = response.tables_plan
    except Exception as exc:
        logger.error("Architect LLM 调用失败: %s", exc)
        dynamic_ltm = _dynamic_ltm(state)
        # 契约置空：fallback 伪代码没有契约，避免陈旧契约误导 ResultReviewer
        artifacts.result_contract = ResultContract()
        artifacts.outline = {
            "摘要": "概述问题、方法、结果和创新点。",
            "问题重述": "严格引用静态 LTM 的问题理解与数据字典。",
            "模型建立": dynamic_ltm.solution_outline or "根据当前动态 LTM 展开模型。",
            "模型求解": "声明输入输出、算法伪代码和复杂度。",
            "结果分析": "组织图表、敏感性分析和误差讨论。",
        }
        artifacts.pseudocode = [
            "load_data(schema)",
            "fit_baseline_model(data, assumptions)",
            "score_candidates(results)",
            "export_figures_and_tables(results)",
        ]

    control.phase = "execution_spec_ready"
    # V17 运行过程记录：架构规划留痕（大纲/算法/图表计划/结果契约）
    entry = _emit_process(
        runtime, control, state, "architect", "architecture_planned",
        f"架构规划完成：大纲 {len(artifacts.outline)} 节，"
        f"图表计划 {len(artifacts.figures_plan)} 张，"
        f"表格计划 {len(artifacts.tables_plan)} 张",
        {
            "outline": dict(artifacts.outline),
            "algorithms_summary": artifacts.algorithms_summary,
            "figures_plan": [
                {
                    "id": f.id,
                    "figure_type": f.figure_type,
                    "kind": f.kind,
                    "caption": f.caption,
                    "section": f.section,
                    "required": f.required,
                }
                for f in artifacts.figures_plan
            ],
            "tables_plan": [
                {"id": t.id, "title": t.title, "section": t.section}
                for t in artifacts.tables_plan
            ],
            "result_contract": (
                artifacts.result_contract.model_dump(mode="json")
                if artifacts.result_contract is not None
                else {}
            ),
        },
        prompt_text=system_prompt,
    )
    return {
        "artifacts": artifacts,
        "control": control,
        "prompt_audit": audit,
        "process_log": [entry],
    }


def hitl_implementation_review_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V13 新增：实现架构人类审核（Human-in-the-loop）。

    在 Architect 之后、编程手之前暂停，把"方案与实现架构说明书"
    （用什么算法、跑多少图表/表格、结果契约长什么样）交给人类审核。
    - approve → 放行给编程手（dispatch_implementation）
    - revise <反馈> → 返回 Architect 按反馈修改
    - rollback <version> → 回滚到指定 LTM 版本
    """
    control = _control(state)
    artifacts = _artifacts(state)

    from modeling_assistant.handoff.spec import build_architecture_spec_md

    spec_md = build_architecture_spec_md(state)
    artifacts.architecture_spec_md = spec_md

    control.hitl_required = True
    control.hitl_stage = "implementation_architecture"

    decision = interrupt({
        "stage": "implementation_architecture",
        "message": "请审核「方案与实现架构说明书」：算法、预期图表/表格、结果契约。",
        "hint": (
            "输入 'approve' 放行给编程手实现；"
            "'revise <反馈>' 返回 Architect 修改；"
            "'rollback <version>' 回滚到指定版本。"
        ),
        "architecture_spec_md": spec_md,
        "artifacts_summary": artifacts.model_dump(),
        "control_summary": {
            "phase": control.phase,
            "selected_plan_id": control.selected_plan_id,
            "innovation_score": control.innovation_score,
            "feasibility_score": control.feasibility_score,
            "modeling_budget_used": control.modeling_revision_count,
            "modeling_budget_limit": control.modeling_revision_budget,
        },
    })

    action = _parse_hitl_decision(decision)
    control.hitl_required = False
    control.hitl_stage = "none"
    if action["type"] == "rollback":
        control.rollback_to_version = action.get("version")
        control.rollback_source = "architecture_hitl"
        control.phase = "hitl_implementation_rollback_requested"
        control.implementation_architecture_reviewed = False
    elif action["type"] == "revise":
        control.phase = "hitl_implementation_revised"
        control.implementation_architecture_reviewed = False
        feedback = action.get("version") or ""
        if feedback:
            control.rebrainstorm_feedback.append(f"人类架构反馈：{feedback}")
    else:
        control.phase = "implementation_architecture_approved"
        control.implementation_architecture_reviewed = True
    return {"control": control, "artifacts": artifacts}


def dispatch_implementation_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V13 新增：架构审核通过后，把编程手任务包写到输出目录。

    任务包（tasks/coder_task.md + coder_task.json）是"拿出来交给编程手"的
    物理载体；内置 Coder 或外部 Codex 实例都从这份规格工作。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    artifacts = _artifacts(state)

    from modeling_assistant.handoff.spec import write_coder_task_package

    recent_stderr = control.coder_error_log[-1] if control.coder_error_log else ""
    md_path, _json_path = write_coder_task_package(
        state,
        resolved_runtime.settings.output_dir,
        recent_stderr=recent_stderr,
    )
    artifacts.coder_task_dir = str(md_path.parent)
    control.phase = "implementation_dispatched"
    logger.info("编程手任务包已生成：%s", md_path)
    return {"artifacts": artifacts, "control": control}


def hitl_implementation_human_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V13 新增：等待人工编程手交付代码（Human-in-the-loop Coding）。

    架构审核通过后，把 coder_task.md/json 写到任务目录并暂停；
    人类编程手在任务目录编写 solution.py（可选 figures.py）后输入 approve，
    系统继续执行与验证。输入 auto 则回退到内置 Coder。
    失败重试时，任务包会带上最近一次 stderr，让人有针对性地修复。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    artifacts = _artifacts(state)

    from modeling_assistant.handoff.spec import write_coder_task_package

    recent_stderr = control.coder_error_log[-1] if control.coder_error_log else ""
    md_path, _json_path = write_coder_task_package(
        state,
        resolved_runtime.settings.output_dir,
        recent_stderr=recent_stderr,
    )
    artifacts.coder_task_dir = str(md_path.parent)

    control.hitl_required = True
    control.hitl_stage = "implementation_human"
    decision = interrupt({
        "stage": "implementation_human",
        "message": (
            f"请人工编程手在以下任务目录编写代码：{md_path.parent}\n"
            "必需交付：solution.py（完整可执行）；可选：figures.py（图表）。\n"
            "任务说明见 coder_task.md，务必遵守其中的建模设定、结果契约与实现约束。"
        ),
        "hint": (
            "输入 'approve' 表示已交付，系统开始执行与验证；"
            "输入 'auto' 交给内置 Coder 自动实现；"
            "输入 'revise <反馈>' 返回 Architect 修改方案。"
        ),
        "task_dir": str(md_path.parent),
        "task_md": str(md_path),
        "architecture_spec_md": (
            artifacts.architecture_spec_md
            or md_path.read_text(encoding="utf-8")[:6000]
        ),
        "artifacts_summary": artifacts.model_dump(),
    })

    action = _parse_hitl_decision(decision)
    control.hitl_required = False
    control.hitl_stage = "none"
    if action["type"] == "auto":
        control.implementation_auto = True
        control.phase = "implementation_auto"
    elif action["type"] == "revise":
        control.phase = "implementation_revised"
        control.implementation_architecture_reviewed = False
        feedback = action.get("version") or ""
        if feedback:
            control.rebrainstorm_feedback.append(f"人类编程反馈：{feedback}")
    else:
        control.phase = "implementation_human_ready"
    return {"control": control, "artifacts": artifacts}


def _result_preview_for_acceptance(artifacts: ArtifactBundle, limit: int = 4000) -> str:
    """为人工验收生成结果文件预览（前几行），不把全量数据塞进 HITL 载荷。"""
    previews: list[str] = []
    for path_str in (artifacts.result_paths or [])[:3]:
        path = Path(path_str)
        if not path.exists():
            previews.append(f"{path.name}: 文件不存在")
            continue
        try:
            import pandas as pd

            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            previews.append(
                f"{path.name}: {df.shape[0]} 行 × {df.shape[1]} 列\n"
                f"{df.head(8).to_string()[:1200]}"
            )
        except Exception as exc:
            previews.append(f"{path.name}: 读取失败 {exc}")
    return "\n\n".join(previews)[:limit]


def _combined_sub_ltms(control: ControlState) -> DynamicLTM:
    """把全部已完成小题的 LTM 合并成 Writer 可用的聚合 LTM。

    V20：假设按原文去重（保序），避免多小题各自写入相同的【全文】假设后
    在 3_assumptions.tex 重复出现。
    """
    ltms = control.sub_ltms or []
    if not ltms:
        return DynamicLTM()
    assumptions: list[str] = []
    nomenclature: dict[str, str] = {}
    equations: list[str] = []
    outlines: list[str] = []
    objectives: list[str] = []
    seen_assumptions: set[str] = set()
    for ltm in ltms:
        for assumption in ltm.assumptions or []:
            if assumption not in seen_assumptions:
                seen_assumptions.add(assumption)
                assumptions.append(assumption)
        nomenclature.update(ltm.nomenclature or {})
        equations.extend(ltm.equations or [])
        if ltm.solution_outline:
            outlines.append(ltm.solution_outline)
        if ltm.objective:
            objectives.append(ltm.objective)
    return DynamicLTM(
        assumptions=assumptions,
        nomenclature=nomenclature,
        equations=equations,
        objective="；".join(objectives),
        solution_outline="\n".join(outlines),
    )


def _read_assumptions_section(paper_dir: Path) -> str:
    """读取论文 3_assumptions.tex 内容供终审 HITL 展示；缺失时安全降级。"""
    candidates = [
        paper_dir / "sections" / "3_assumptions.tex",
        paper_dir / "3_assumptions.tex",
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            return text or "（3_assumptions.tex 为空）"
    return "（未找到 3_assumptions.tex：假设章缺失或尚未生成）"


def sub_question_acceptance_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V14 新增：小题人工验收闸门。

    ResultReviewer 机械校验通过后，由人类决定本小题是否可继续推进：
    - pass → 记录结果并推进下一小题（或进入 Writer）
    - fail code|architecture|model <反馈> → 回到对应层级重做当前小题
    - cross <i> <反馈> → 触发跨小题 HITL（极端情况：前面小题模型可能错了）
    """
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    artifacts = _artifacts(state)

    control.hitl_required = True
    control.hitl_stage = "sub_question_acceptance"
    decision = interrupt({
        "stage": "sub_question_acceptance",
        "message": (
            f"小题 {control.current_sub_question_index + 1} 已通过机械校验，请人工验收。"
        ),
        "hint": (
            "输入 'pass' 通过并推进；"
            "'fail code|architecture|model <反馈>' 回到对应层级；"
            "'cross <小题编号> <反馈>' 声明与前面小题模型冲突。"
        ),
        "sub_question_index": control.current_sub_question_index,
        "sub_question_text": _current_sub_question(control),
        "result_paths": list(artifacts.result_paths or []),
        "figure_paths": list(artifacts.figure_paths or []),
        "result_preview": _result_preview_for_acceptance(artifacts),
        "result_contract": (
            artifacts.result_contract.model_dump(mode="json")
            if artifacts.result_contract is not None
            else {}
        ),
        "review_warnings": (state.get("prompt_audit", {}) or {}).get(
            "result_reviewer_warnings", ""
        ),
        "dynamic_ltm_summary": {
            "objective": dynamic_ltm.objective,
            "assumptions_count": len(dynamic_ltm.assumptions),
            "equations_count": len(dynamic_ltm.equations),
        },
    })

    action = _parse_hitl_decision(decision)
    control.hitl_required = False
    control.hitl_stage = "none"
    feedback = action.get("version") or ""

    if action["type"] == "fail":
        level = action.get("level") or "code"
        control.sub_question_attempts += 1
        if feedback:
            control.sub_question_feedback.append(f"fail {level}: {feedback}")
        if level == "architecture":
            # 回到 Architect 后需要重新审核实现架构
            control.implementation_architecture_reviewed = False
        control.phase = f"sub_question_fail_{level}"
        logger.info(
            "小题 %d 验收失败（%s，第 %d/%d 次）: %s",
            control.current_sub_question_index + 1,
            level,
            control.sub_question_attempts,
            control.sub_question_budget,
            feedback[:100],
        )
    elif action["type"] == "cross":
        control.cross_sub_question_target = action.get("target", -1)
        if feedback:
            control.sub_question_feedback.append(
                f"cross {control.cross_sub_question_target + 1}: {feedback}"
            )
        control.phase = "cross_sub_question_requested"
        logger.info(
            "小题 %d 声明与小题 %d 模型冲突",
            control.current_sub_question_index + 1,
            control.cross_sub_question_target + 1,
        )
    else:
        # pass：记录当前小题成果并推进
        passed_idx = control.current_sub_question_index
        # V17 结果注册表：验收通过即锁定本题唯一权威结果
        control = _finalize_authoritative_result(
            state, control, artifacts, status="passed"
        )
        control, _has_next = _advance_sub_question(
            state, control, dynamic_ltm, artifacts, status="passed"
        )
        control.phase = "sub_question_passed"
        logger.info(
            "小题 %d 验收通过，推进到下一小题（index=%d）",
            passed_idx + 1,
            control.current_sub_question_index,
        )
        entry = _emit_process(
            runtime, control, state, "sub_question_acceptance", "sub_question_accepted",
            f"小题 {passed_idx + 1} 人工验收通过，推进下一小题",
            {
                "decision": "pass",
                "passed_index": passed_idx,
                "result_paths": list(artifacts.result_paths),
            },
        )
        return {"control": control, "dynamic_ltm": DynamicLTM(), "process_log": [entry]}
    entry = _emit_process(
        runtime, control, state, "sub_question_acceptance", "sub_question_rejected",
        (
            f"小题 {control.current_sub_question_index + 1} 验收失败"
            f"（{action.get('level')}，第 {control.sub_question_attempts} 次）"
            if action["type"] == "fail"
            else f"小题 {control.current_sub_question_index + 1} 声明跨小题冲突"
        ),
        {
            "decision": action["type"],
            "level": action.get("level"),
            "target": action.get("target"),
            "feedback": feedback,
            "sub_question_attempts": control.sub_question_attempts,
        },
    )
    return {"control": control, "process_log": [entry]}


def cross_sub_question_hitl_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """V14 新增：跨小题模型冲突 HITL（极端情况，不自动回滚）。

    展示目标小题快照与当前反馈，由人类决定：
    - accept <反馈> → 接受演进，重做当前小题
    - rollback <小题编号> → 回滚到该小题快照之后重做（截断其后的小题成果）
    - continue → 保留现状，把当前小题视为通过并推进
    """
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    artifacts = _artifacts(state)
    target = control.cross_sub_question_target

    target_ltm = {}
    target_result = {}
    if 0 <= target < len(control.sub_ltms):
        target_ltm = control.sub_ltms[target].model_dump(mode="json")
    if 0 <= target < len(control.sub_results):
        target_result = control.sub_results[target].model_dump(mode="json")

    control.hitl_required = True
    control.hitl_stage = "cross_sub_question"
    decision = interrupt({
        "stage": "cross_sub_question",
        "message": (
            f"当前小题（{control.current_sub_question_index + 1}）声明与小题 "
            f"{target + 1} 的模型冲突。这是极端情况，不会自动回滚。"
        ),
        "hint": (
            "输入 'accept <反馈>' 接受演进并重做当前小题；"
            "'rollback <小题编号>' 回滚到该小题快照之后重做；"
            "'continue' 保留现状，把当前小题视为通过并推进。"
        ),
        "target_index": target,
        "target_ltm": target_ltm,
        "target_result": target_result,
        "archive_versions": [s.version for s in state.get("ltm_archive", [])],
        "current_feedback": list(control.sub_question_feedback),
    })

    action = _parse_hitl_decision(decision)
    control.hitl_required = False
    control.hitl_stage = "none"

    if action["type"] == "rollback":
        idx_text = (action.get("version") or "").strip()
        try:
            rollback_idx = int(idx_text) - 1
        except (TypeError, ValueError):
            rollback_idx = target
        if 0 <= rollback_idx < control.current_sub_question_index:
            control.sub_ltms = control.sub_ltms[: rollback_idx + 1]
            control.sub_results = control.sub_results[: rollback_idx + 1]
            # V17 结果注册表：被重做的小题及其后条目一并截断
            control = _truncate_manifest(control, rollback_idx)
            control.current_sub_question_index = rollback_idx
            control.sub_question_attempts = 0
            control.sub_question_feedback = []
            control.modeling_revision_count = 0
            control.modeling_revision_budget = control.sub_question_budget
            control.implementation_architecture_reviewed = False
            control.implementation_auto = False
            control.phase = "cross_rollback_requested"
            logger.info("跨小题回滚：回到小题 %d 重做", rollback_idx + 1)
            entry = _emit_process(
                runtime, control, state, "cross_sub_question", "cross_rollback",
                f"跨小题回滚到小题 {rollback_idx + 1} 重做",
                {"decision": "rollback", "rollback_index": rollback_idx},
            )
            return {"control": control, "dynamic_ltm": DynamicLTM(), "process_log": [entry]}
        control.phase = "cross_rollback_invalid"
        entry = _emit_process(
            runtime, control, state, "cross_sub_question", "cross_rollback_invalid",
            "跨小题回滚目标无效，保留现状",
            {"decision": "rollback", "rollback_index": action.get("version")},
        )
        return {"control": control, "process_log": [entry]}

    if action["type"] == "accept":
        feedback = action.get("version") or ""
        control.sub_question_feedback.append(f"跨小题接受演进：{feedback}")
        control.phase = "cross_accept_evolution"
        logger.info("跨小题冲突：人类接受演进，重做当前小题")
        entry = _emit_process(
            runtime, control, state, "cross_sub_question", "cross_accept_evolution",
            "跨小题冲突：人类接受演进，重做当前小题",
            {"decision": "accept", "feedback": feedback},
        )
        return {"control": control, "process_log": [entry]}

    # continue：保留现状，视为当前小题通过并推进
    # V17 结果注册表：跨小题保留现状同样锁为权威结果
    control = _finalize_authoritative_result(
        state, control, artifacts, status="passed",
        feedback=["跨小题冲突：人类选择保留现状继续"],
    )
    control, _has_next = _advance_sub_question(
        state,
        control,
        dynamic_ltm,
        artifacts,
        status="passed",
        feedback=["跨小题冲突：人类选择保留现状继续"],
    )
    control.phase = "sub_question_passed"
    entry = _emit_process(
        runtime, control, state, "cross_sub_question", "cross_continue",
        "跨小题冲突：人类选择保留现状并推进",
        {"decision": "continue"},
    )
    return {"control": control, "dynamic_ltm": DynamicLTM(), "process_log": [entry]}


# ═══════════════════════════════════════════════════════════════════
# 阶段四：并行执行与自纠错
# ═══════════════════════════════════════════════════════════════════

def drawer_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """可视化工程师：生成并执行绘图代码，产出真实图片。

    V9 增强：与 coder_node 对称，添加自修复循环（最多 2 次重试，不消耗 budget）。
    Drawer 失败的主要原因：禁止库 import（lifelines 等）、列名不存在（如 'sex'）、
    字符串字面量跨行。这些都可以通过把 stderr 回传给 LLM 让其针对性修复。

    V9 修复：代码执行失败时不记录 LLM 想象的"视觉观察"。原逻辑不论代码执行成功与否
    都会把 response.observation 写入 empirical 层，导致 LLM 虚构的"散点呈凸性趋势"
    被当作实证证据污染 reflection/clarifier。现在仅在代码执行成功且产出真实图片时
    才记录视觉观察。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    empirical = _empirical(state)

    MAX_SELF_REPAIR = 2  # 自修复次数上限（不消耗 budget）
    recent_stderr = ""
    artifacts = ArtifactBundle()
    audit: dict[str, str] = {}

    for attempt in range(MAX_SELF_REPAIR + 1):  # 0, 1, 2 = 共 3 次机会
        # 渲染 prompt：自修复时注入 recent_stderr 让 drawer 看到完整错误
        extra = {"recent_stderr": recent_stderr} if recent_stderr else None
        system_prompt, audit = _prompt_audit("drawer", state, runtime, extra=extra)

        try:
            response = resolved_runtime.invoke_structured(
                "drawer", state, DrawerResponse, system_prompt=system_prompt
            )
        except Exception as exc:
            logger.error("Drawer LLM 调用失败: %s", exc)
            artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]
            return {"artifacts": artifacts, "empirical": empirical, "prompt_audit": audit}

        if not response.figure_code:
            # LLM 未返回绘图代码，不可自修复
            artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]
            return {"artifacts": artifacts, "empirical": empirical, "prompt_audit": audit}

        resolved_runtime.write_file("figures", "figures.py", content=response.figure_code)
        # V11.2 修复（Bug 1）：预先创建 figures 目录，避免 LLM 忘记 os.makedirs 时
        # plt.savefig('figures/figure1.png') 因目录不存在而失败
        Path(resolved_runtime.output_path("figures")).mkdir(parents=True, exist_ok=True)
        # 执行绘图代码（run_code 内部会先做预检：ast.parse + 禁止库扫描）
        success, _stdout, stderr = resolved_runtime.run_code(response.figure_code)

        if success:
            figure_dir = Path(resolved_runtime.output_path("figures"))
            real_figures: list[str] = []
            if figure_dir.exists():
                figure_files = sorted(
                    p
                    for p in figure_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
                    and p.name not in {"placeholder.png", "figures.py"}
                )
                real_figures = [str(p) for p in figure_files]
            if real_figures:
                # 成功！产出真实图片
                artifacts.figure_paths = real_figures
                # V17：按 figures_plan 登记图表注册表（plan_id -> 实际文件）
                state_plan = _artifacts(state).figures_plan
                plan_ids = {p.id for p in state_plan}
                manifest: dict[str, dict[str, Any]] = {}
                run_tag = f"drawer_{control.coder_run_count}"
                if response.figure_ids:
                    for fid, path in zip(response.figure_ids, real_figures):
                        manifest[fid] = {
                            "path": path,
                            "run_id": run_tag,
                            "status": "generated",
                        }
                for path in real_figures:
                    stem = Path(path).stem
                    if stem in plan_ids and stem not in manifest:
                        manifest[stem] = {
                            "path": path,
                            "run_id": run_tag,
                            "status": "generated",
                        }
                artifacts.figure_manifest = manifest
                if manifest:
                    logger.info(
                        "Drawer 图表注册表登记 %d 张：%s",
                        len(manifest),
                        list(manifest),
                    )
                # V9 修复：仅在代码执行成功且产出真实图片时才记录视觉观察
                # 避免代码失败时 LLM 虚构的"散点呈凸性趋势"被当作实证证据
                if response.observation and response.observation.strip():
                    existing_count = len(empirical.findings)
                    evidence_text = response.observation.strip()
                    if response.image_stats and response.image_stats.strip():
                        evidence_text = f"{evidence_text} | 统计佐证: {response.image_stats.strip()}"
                    empirical.findings.append(EmpiricalFinding(
                        id=f"finding_drawer_{existing_count + 1}",
                        run_id=f"drawer_{control.coder_run_count}",
                        source_node="drawer",
                        assumption_tested="变量关系形态（视觉观察）",
                        evidence=evidence_text,
                        verdict=response.observation_verdict,
                        confidence=response.observation_confidence,
                    ))
                    from modeling_assistant.schemas.state import _rebuild_empirical_derived_fields
                    _rebuild_empirical_derived_fields(empirical)
                    logger.info(
                        "Drawer 视觉观察（verdict=%s, conf=%.2f）：%s",
                        response.observation_verdict,
                        response.observation_confidence,
                        response.observation.strip()[:100],
                    )
                if attempt > 0:
                    logger.info("Drawer 自修复第 %d 次尝试成功", attempt)
                return {"artifacts": artifacts, "empirical": empirical, "prompt_audit": audit}
            # 代码执行成功但未生成图片，可自修复
            # V11.2 修复（Bug 1）：原提示让 LLM 保存到"当前工作目录"，
            # 但实际检测的是 figures/ 子目录，导致 LLM 困惑。改为明确要求
            # 保存到 figures/ 子目录。
            recent_stderr = (
                f"代码执行成功但未在 figures/ 子目录下生成图片文件。\n"
                f"必须使用 plt.savefig('figures/figure1.png')（注意要带 figures/ 前缀），\n"
                f"并在保存前执行 os.makedirs('figures', exist_ok=True)。\n"
                f"不要使用 plt.savefig('figure1.png')（缺 figures/ 前缀会被检测到根目录），\n"
                f"也不要使用绝对路径或 ./figures/ 前缀。\n"
                f"期望在 {figure_dir} 下找到 .png/.jpg/.pdf 文件。"
            )
        else:
            # 执行失败，记录 stderr 用于自修复
            recent_stderr = stderr
            logger.warning("Drawer 绘图代码执行失败 (attempt %d): %s", attempt, stderr[:200])

        # 尝试自修复
        if attempt < MAX_SELF_REPAIR:
            logger.info(
                "Drawer 自修复尝试 %d/%d: %s",
                attempt + 1, MAX_SELF_REPAIR, recent_stderr[:200],
            )
            continue

        # 自修复耗尽，使用 placeholder
        artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]
        return {"artifacts": artifacts, "empirical": empirical, "prompt_audit": audit}

    # 不应到达此处，但保险起见
    artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]
    return {"artifacts": artifacts, "empirical": empirical, "prompt_audit": audit}


def coder_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """完全屏蔽冗长 Context，只依据动态 LTM 与 Architect 产物编写代码。

    回滚触发机制（Goal.md）：连续失败 3 次 → 按错误类型退回 Architect 或 Clarifier。
    - SyntaxError / ImportError / NameError / TypeError → architect（代码规范问题）
    - ValueError / RuntimeError / 求解失败 → clarifier（设定/公式问题）

    注意：LLM 调用失败、返回空代码、代码执行失败均视为一次失败。

    V8 增强：代码执行前做预检（ast.parse + 禁止库扫描），失败直接自修复重试，
    不消耗 budget。执行失败也进入自修复循环（最多 2 次），把完整 stderr 回传给
    coder 让其针对性修复。自修复仍失败才走原失败路径（消耗 budget）。

    实证证据落盘：每次执行（无论成败）都把 stdout/stderr 落盘到
    outputs/logs/run_{n}.log，供 Reflection 节点按需读取。失败日志
    摘要结构化为「[run_id] summary (log_path)」，避免 500 字符截断丢信息。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)

    # 提取真实数据路径，供代码执行时使用
    static_ltm = _static_ltm(state)
    data_paths = (
        static_ltm.data_profile.file_paths
        if static_ltm.data_profile
        else []
    )

    MAX_SELF_REPAIR = 2  # 自修复次数上限（不消耗 budget）
    recent_stderr = ""
    artifacts = ArtifactBundle()
    audit: dict[str, str] = {}
    external_mode = resolved_runtime.settings.coder_external_mode == "codex"
    human_mode = resolved_runtime.settings.coder_external_mode == "human"
    state_artifacts = _artifacts(state)
    # 人工编程手模式：一次交付一次执行，不做 LLM 自修复；
    # 失败由外层循环把 stderr 带回任务包，让人针对性修复
    repair_limit = 0 if (human_mode and not control.implementation_auto) else MAX_SELF_REPAIR
    if human_mode and not control.implementation_auto:
        task_dir = Path(state_artifacts.coder_task_dir or resolved_runtime.output_path("tasks"))
        if not (task_dir / "solution.py").exists():
            # 未交付 → 与"LLM 未生成代码"同等的空代码失败路径
            control.coder_error_count += 1
            run_id = f"run_{control.coder_run_count}"
            log_path = resolved_runtime.output_path("logs", f"{run_id}_empty.log")
            control.coder_error_log.append(
                f"[{run_id}] 人工编程手未交付 solution.py（任务目录：{task_dir}）。 (日志: {log_path})"
            )
            control.coder_run_count += 1
            control.phase = "code_generation_empty"
            control.coder_rollback_target = "architect"
            artifacts.result_paths = []
            artifacts.clear_result_paths = True
            entry = _emit_process(
                runtime, control, state, "coder", "code_generation_empty",
                "人工编程手未交付 solution.py",
                {"run_id": run_id, "task_dir": str(task_dir)},
            )
            return {
                "artifacts": artifacts,
                "control": control,
                "prompt_audit": audit,
                "process_log": [entry],
            }

    for attempt in range(repair_limit + 1):
        if human_mode and not control.implementation_auto:
            # 人工模式不需要渲染 LLM prompt；直接读取任务目录中的交付物
            system_prompt, audit = "", {}
        else:
            # 渲染 prompt：自修复时注入 recent_stderr 让 coder 看到完整错误
            extra = {"recent_stderr": recent_stderr} if recent_stderr else None
            system_prompt, audit = _prompt_audit("coder", state, runtime, extra=extra)

        try:
            if human_mode and not control.implementation_auto:
                # V13：读取人工编程手交付的 solution.py（可选 figures.py）
                task_dir = Path(state_artifacts.coder_task_dir or resolved_runtime.output_path("tasks"))
                solution_path = task_dir / "solution.py"
                figures_path = task_dir / "figures.py"
                if not solution_path.exists():
                    raise FileNotFoundError(
                        f"人工编程手未交付 solution.py（任务目录：{task_dir}）"
                    )
                code = solution_path.read_text(encoding="utf-8")
                artifacts.human_figure_code_path = (
                    str(figures_path) if figures_path.exists() else ""
                )
                # V16 修复：小题循环下结果文件为 results/q{i}.csv（与任务包约束一致），
                # 旧 hardcode 检查 results/output.csv 导致人工/外部编程手交付被判失败。
                q_fname = (
                    f"q{control.current_sub_question_index + 1}.csv"
                    if control.sub_questions
                    else "output.csv"
                )
                response = CoderResponse(code=code, result_path=f"results/{q_fname}")
                logger.info("人工编程手交付 solution.py（%d 字符）", len(code))
            elif external_mode:
                # V13：把任务包交给本机另一个 Codex 实例实现
                from modeling_assistant.handoff.codex import CodexCoderAdapter
                from modeling_assistant.handoff.spec import write_coder_task_package

                task_dir = Path(state_artifacts.coder_task_dir or resolved_runtime.output_path("tasks"))
                task_dir.mkdir(parents=True, exist_ok=True)
                write_coder_task_package(
                    state,
                    resolved_runtime.settings.output_dir,
                    recent_stderr=recent_stderr,
                )
                adapter = CodexCoderAdapter(resolved_runtime.settings)
                ok, code, log = adapter.generate(task_dir)
                if not ok:
                    raise RuntimeError(f"外部编程手失败: {log}")
                q_fname = (
                    f"q{control.current_sub_question_index + 1}.csv"
                    if control.sub_questions
                    else "output.csv"
                )
                response = CoderResponse(code=code, result_path=f"results/{q_fname}")
                logger.info("外部编程手（Codex）产出 solution.py（%d 字符）", len(code))
            else:
                # V11.4：传入 fallback_parser，处理 LLM 偶发返回纯代码块的情况
                from modeling_assistant.agents.runtime import _coder_fallback_parser
                response = resolved_runtime.invoke_structured(
                    "coder", state, CoderResponse,
                    system_prompt=system_prompt,
                    fallback_parser=_coder_fallback_parser,
                )
        except Exception as exc:
            # LLM 调用失败，不可自修复，直接走失败路径
            logger.error("Coder LLM 调用失败: %s", exc)
            run_id = f"run_{control.coder_run_count}"
            log_path = resolved_runtime.write_file(
                "logs", f"{run_id}_llm_error.log",
                content=f"=== RUN {run_id} (LLM CALL FAILED) ===\n{exc}\n",
            )
            control.coder_run_count += 1
            control.coder_error_count += 1
            control.coder_error_log.append(
                f"[{run_id}] LLM 调用失败: {str(exc)[:200]} (日志: {log_path})"
            )
            control.phase = "code_generation_failed"
            control.coder_rollback_target = "architect"
            # V9 修复：清空旧 result_paths，避免 merge_artifacts_reducer 保留旧值
            # 导致 route_after_coder 误判为成功 → result_reviewer 检查旧文件 → 死循环
            artifacts.result_paths = []
            artifacts.clear_result_paths = True
            entry = _emit_process(
                runtime, control, state, "coder", "code_generation_failed",
                f"Coder LLM 调用失败（run={run_id}）",
                {"run_id": run_id, "error": str(exc)[:200], "log_path": log_path},
            )
            return {
                "artifacts": artifacts,
                "control": control,
                "prompt_audit": audit,
                "process_log": [entry],
            }

        if not response.code:
            # LLM 未返回代码，不可自修复，直接走失败路径
            control.coder_error_count += 1
            run_id = f"run_{control.coder_run_count}"
            log_path = resolved_runtime.output_path("logs", f"{run_id}_empty.log")
            control.coder_error_log.append(f"[{run_id}] Coder 未生成任何代码。 (日志: {log_path})")
            control.coder_run_count += 1
            control.phase = "code_generation_empty"
            control.coder_rollback_target = "architect"
            logger.warning("Coder 未生成代码 (第 %d 次)。", control.coder_error_count)
            # V9 修复：清空旧 result_paths（同上）
            artifacts.result_paths = []
            artifacts.clear_result_paths = True
            entry = _emit_process(
                runtime, control, state, "coder", "code_generation_empty",
                f"Coder 未生成任何代码（第 {control.coder_error_count} 次）",
                {"run_id": run_id, "log_path": log_path},
            )
            return {
                "artifacts": artifacts,
                "control": control,
                "prompt_audit": audit,
                "process_log": [entry],
            }

        # V11 修复：第三层常量校验 —— 在执行代码前做静态扫描
        # 如果发现关键常量缺失或列名错误，直接进入自修复循环，不消耗 budget
        from modeling_assistant.validation.constants import check_code_against_facts
        constant_issues = check_code_against_facts(
            response.code, static_ltm, artifacts, _dynamic_ltm(state)
        )
        if constant_issues:
            run_id = f"run_{control.coder_run_count}"
            issues_text = "\n".join(constant_issues)
            logger.warning(
                "Coder 常量校验失败 (attempt %d, run_id=%s): %s",
                attempt, run_id, issues_text[:200],
            )
            # 把校验问题作为 stderr 回传给 Coder 自修复
            recent_stderr = (
                f"【V11 常量校验失败】\n{issues_text}\n"
                f"请根据 problem_facts 列表修正代码中的数值常量，"
                f"或根据 data_columns_json 修正列名访问。"
            )
            control.coder_error_log.append(f"[{run_id}_precheck] 常量校验: {issues_text[:200]}")
            if attempt < MAX_SELF_REPAIR:
                continue
            # V11.2 修复（Bug 3）：自修复耗尽时写 precheck 日志文件，
            # 让 reflection_node 能找到日志并消费 budget，避免死循环。
            # 原逻辑只写内存 coder_error_log，reflection 找不到日志文件而跳过，
            # budget 不增加，route_after_reflection 看到 result_paths 空 + budget 未耗尽
            # 会无限回退到 architect。
            precheck_log_path = resolved_runtime.write_file(
                "logs", f"{run_id}_precheck.log",
                content=(
                    f"=== RUN {run_id} (PRECHECK FAILED) ===\n"
                    f"=== CONSTANT ISSUES ===\n{issues_text}\n"
                    f"=== CODE ===\n{response.code}\n"
                ),
            )
            control.coder_error_count += 1
            control.coder_run_count += 1
            # 使用独立 phase，与执行失败区分，便于调试
            control.phase = "code_precheck_failed"
            control.coder_rollback_target = "architect"
            control.coder_error_log.append(
                f"[{run_id}] 常量校验失败 (日志: {precheck_log_path})"
            )
            artifacts.result_paths = []
            artifacts.clear_result_paths = True
            entry = _emit_process(
                runtime, control, state, "coder", "code_precheck_failed",
                f"Coder 常量/列名校验失败（run={run_id}）",
                {
                    "run_id": run_id,
                    "constant_issues": list(constant_issues),
                    "log_path": precheck_log_path,
                },
            )
            return {
                "artifacts": artifacts,
                "control": control,
                "prompt_audit": audit,
                "process_log": [entry],
            }

        resolved_runtime.write_file("results", "model.py", content=response.code)

        # 执行代码（run_code 内部会先做预检：ast.parse + 禁止库扫描）
        success, stdout, stderr = resolved_runtime.run_code(
            response.code, data_paths=data_paths
        )

        if success:
            result_path_str = response.result_path or "results/output.csv"
            expected_path = Path(result_path_str)
            if not expected_path.is_absolute():
                expected_path = resolved_runtime.settings.output_dir / expected_path
            if expected_path.exists():
                run_id = f"run_{control.coder_run_count}"
                # V14 小题循环：结果统一归一到 results/q{i}.csv，避免各小题互相覆盖
                sub_result_path = expected_path
                if control.sub_questions:
                    sub_result_path = (
                        resolved_runtime.settings.output_dir
                        / "results"
                        / f"q{control.current_sub_question_index + 1}.csv"
                    )
                    sub_result_path.parent.mkdir(parents=True, exist_ok=True)
                    if sub_result_path != expected_path and not sub_result_path.exists():
                        shutil.copy2(expected_path, sub_result_path)
                        logger.info("小题结果已归一到 %s", sub_result_path)
                # V10 修复：备份成功结果文件，避免被后续失败的 Coder 覆盖
                # 当 ResultReviewer 拒绝时，原 output.csv 会被清空（V9 行为），但磁盘上的
                # output_run_N.csv 备份保留。writer_node 在 result_paths 为空时可扫描备份
                # 目录加载最新的成功结果，让论文基于真实数值而非降级到"待验证"。
                try:
                    backup_path = expected_path.parent / f"output_{run_id}.csv"
                    shutil.copy2(expected_path, backup_path)
                    logger.info("Coder 成功结果已备份至 %s", backup_path)
                except Exception as exc:
                    logger.warning("备份结果文件失败 %s: %s", expected_path, exc)
                log_path = resolved_runtime.write_file(
                    "logs", f"{run_id}.log",
                    content=(
                        f"=== RUN {run_id} (SUCCESS) ===\n"
                        f"=== STDOUT ===\n{stdout}\n"
                        f"=== STDERR ===\n{stderr}\n"
                        f"=== RESULT ===\n{expected_path}\n"
                    ),
                )
                control.coder_run_count += 1
                control.coder_error_count = 0
                control.coder_error_log = []
                control.phase = "code_executed_successfully"
                artifacts.result_paths = [str(sub_result_path)]
                # V13：人工编程手模式下，若交付了 figures.py，执行并收集真实图表
                if human_mode and artifacts.human_figure_code_path:
                    fig_path = Path(artifacts.human_figure_code_path)
                    if fig_path.exists():
                        Path(resolved_runtime.output_path("figures")).mkdir(parents=True, exist_ok=True)
                        fig_ok, _fig_stdout, fig_err = resolved_runtime.run_code(
                            fig_path.read_text(encoding="utf-8")
                        )
                        if fig_ok:
                            figure_dir = Path(resolved_runtime.output_path("figures"))
                            real_figures = [
                                str(p)
                                for p in figure_dir.iterdir()
                                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
                                and p.name not in {"placeholder.png", "figures.py"}
                            ]
                            if real_figures:
                                artifacts.figure_paths = real_figures
                                # V17：人工交付的图同样登记图表注册表
                                _register_figure_manifest(
                                    state,
                                    artifacts,
                                    real_figures,
                                    run_tag=f"human_{control.coder_run_count}",
                                )
                        else:
                            logger.warning("人工 figures.py 执行失败: %s", fig_err[:200])
                if attempt > 0:
                    logger.info("Coder 自修复第 %d 次尝试成功", attempt)
                entry = _emit_process(
                    runtime, control, state, "coder", "code_executed",
                    f"Coder 执行成功（run={run_id}）→ {sub_result_path}",
                    {
                        "run_id": run_id,
                        "result_file": str(sub_result_path),
                        "result_paths": list(artifacts.result_paths),
                    },
                )
                return {
                    "artifacts": artifacts,
                    "control": control,
                    "prompt_audit": audit,
                    "process_log": [entry],
                }
            # 结果文件缺失，可自修复
            run_id = f"run_{control.coder_run_count}"
            log_path = resolved_runtime.write_file(
                "logs", f"{run_id}_missing.log",
                content=(
                    f"=== RUN {run_id} (RESULT MISSING) ===\n"
                    f"=== STDOUT ===\n{stdout}\n"
                    f"=== STDERR ===\n{stderr}\n"
                    f"expected: {expected_path}\n"
                ),
            )
            control.coder_run_count += 1
            summary = _extract_error_summary(stderr or f"结果文件缺失：{expected_path}")
            control.coder_error_log.append(f"[{run_id}] {summary} (日志: {log_path})")
            recent_stderr = f"代码执行成功但未找到结果文件：{expected_path}\n请检查 RESULT_PATH 是否正确指向 MODELING_OUTPUT_DIR/results/output.csv"
        else:
            # 执行失败，记录日志（不消耗 budget，用于自修复）
            run_id = f"run_{control.coder_run_count}"
            log_path = resolved_runtime.write_file(
                "logs", f"{run_id}_error.log",
                content=(
                    f"=== RUN {run_id} (FAILED) ===\n"
                    f"=== STDERR ===\n{stderr}\n"
                    f"=== STDOUT ===\n{stdout}\n"
                ),
            )
            control.coder_run_count += 1
            summary = _extract_error_summary(stderr)
            control.coder_error_log.append(f"[{run_id}] {summary} (日志: {log_path})")
            recent_stderr = stderr

        # 尝试自修复
        if attempt < repair_limit:
            logger.info(
                "Coder 自修复尝试 %d/%d (run_id=%s): %s",
                attempt + 1, MAX_SELF_REPAIR, run_id, recent_stderr[:200],
            )
            continue

        # 自修复耗尽，走原失败路径（消耗 budget）
        control.coder_error_count += 1
        control.phase = "code_execution_failed"
        control.coder_rollback_target = _classify_coder_error(recent_stderr)
        logger.warning(
            "Coder 代码执行失败 (第 %d 次, 自修复耗尽), 回滚目标=%s: %s",
            control.coder_error_count,
            control.coder_rollback_target,
            recent_stderr[:200],
        )
        # V9 修复：清空旧 result_paths（同上）
        artifacts.result_paths = []
        artifacts.clear_result_paths = True
        entry = _emit_process(
            runtime, control, state, "coder", "code_execution_failed",
            f"Coder 代码执行失败（第 {control.coder_error_count} 次，自修复耗尽），"
            f"回滚目标={control.coder_rollback_target}",
            {
                "run_id": run_id,
                "rollback_target": control.coder_rollback_target,
                "error_summary": _extract_error_summary(recent_stderr),
                "log_path": log_path,
            },
        )
        return {
            "artifacts": artifacts,
            "control": control,
            "prompt_audit": audit,
            "process_log": [entry],
        }

    # 不应到达此处，但保险起见
    artifacts.result_paths = []
    artifacts.clear_result_paths = True
    return {"artifacts": artifacts, "control": control, "prompt_audit": audit}


def _invoke_meta_router(
    state: GraphState,
    runtime: AgentRuntime | None,
    resolved_runtime: AgentRuntime,
    refuted_findings: list,
) -> MetaRouterResponse | None:
    """中枢 LLM（Meta-Router）：Reflection 发现 refuted 后判断下一步走向。

    基于 Reflection 的反馈和全局失败历史，决策回哪个节点修正：
    - rediscover → Mathematician（重新发散，换建模范式）
    - refine_assumptions → Clarifier（局部修正假设）
    - adjust_architecture → Architect（调整模型设计）
    - accept_failure → collect_artifacts（接受失败，Writer 标注待验证）

    失败时返回 None，调用方回退到原逻辑（默认回 Clarifier）。
    """
    try:
        refuted_findings_json = json.dumps(
            [
                {
                    "assumption_tested": f.assumption_tested,
                    "evidence": f.evidence,
                    "verdict": f.verdict,
                    "confidence": f.confidence,
                    "suggested_fix": f.suggested_fix,
                }
                for f in refuted_findings
            ],
            ensure_ascii=False,
            indent=2,
        )
        system_prompt, _audit = _prompt_audit(
            "meta_router", state, runtime,
            extra={"refuted_findings_json": refuted_findings_json},
        )
        decision = resolved_runtime.invoke_structured(
            "meta_router", state, MetaRouterResponse, system_prompt=system_prompt
        )
        return decision if isinstance(decision, MetaRouterResponse) else None
    except Exception as exc:
        logger.warning("Meta-Router LLM 调用失败，回退到原逻辑: %s", exc)
        return None


def reflection_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Coder 成功后的反思节点：从执行输出提炼实证发现。

    核心职责：
    - 读取最近一次 Coder 成功执行的日志（outputs/logs/run_{n}.log）
    - 调用 LLM 提炼为 1-3 条结构化 EmpiricalFinding
    - 不修改 dynamic_ltm，只写 empirical 层（保持定稿语义纯净）
    - 若产生高置信度 refuted 发现且修正预算未耗尽，设置 trigger_clarifier_revision

    设计意图：打破「定稿=真相」假设，把执行产物转化为可被下游读取的实证发现，
    形成「假设—验证—修正」闭环。Clarifier 决定是否吸收，而非自动污染定稿。

    与 ResultReviewer 的协作：ResultReviewer 可能已经基于机械检验设置了
    trigger_clarifier_revision。Reflection 节点不会重置该标志，只在「自己新发现
    refuted 且预算仍有剩余」时追加触发。预算耗尽则跳过新触发，避免无限循环。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    empirical = _empirical(state)
    artifacts_in = _artifacts(state)

    # 注意：不重置 trigger_clarifier_revision。ResultReviewer 可能已经设置过。
    # Reflection 只能在「上游未触发修正 + 自己新发现 refuted + 预算有剩余」时追加触发。

    # V6 修复：读取 result_paths，用于判断 coder 是否失败
    result_paths_empty = not artifacts_in.result_paths

    # 读取最近一次 Coder 执行的日志（兼容成功与失败日志）
    # coder_run_count 已在落盘后自增，所以最近一次的 run_id 是 run_{count-1}
    last_run_idx = max(control.coder_run_count - 1, 0)
    run_id = f"run_{last_run_idx}"
    # 按优先级查找：成功日志 > 各种失败日志
    log_candidates = [
        (f"{run_id}.log", "success"),           # 成功
        (f"{run_id}_error.log", "failed"),      # 代码执行失败
        (f"{run_id}_missing.log", "result_missing"),  # 结果文件缺失
        (f"{run_id}_empty.log", "empty_code"),  # LLM 未生成代码
        (f"{run_id}_llm_error.log", "llm_failed"),  # LLM 调用失败
        (f"{run_id}_precheck.log", "precheck_failed"),  # V11.2: 常量校验失败
    ]
    log_path: Path | None = None
    execution_status = "unknown"
    for candidate, status in log_candidates:
        candidate_path = Path(resolved_runtime.output_path("logs", candidate))
        if candidate_path.exists():
            log_path = candidate_path
            execution_status = status
            break

    if log_path is None:
        # V11.2 修复（Bug 3 兜底）：找不到任何日志（不应发生，但保险），
        # 必须消费 budget，避免 route_after_reflection 看到 budget 未耗尽而
        # 无限回退到 architect 导致死循环。
        logger.warning(
            "Reflection 找不到任何 Coder 日志（run_id=%s），消费 budget 兜底",
            run_id,
        )
        if control.modeling_revision_count < control.modeling_revision_budget:
            control.modeling_revision_count += 1
        control.phase = "reflection_done"
        return {"control": control, "empirical": empirical}

    try:
        raw_content = log_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Reflection 读取日志失败 %s: %s", log_path, exc)
        control.phase = "reflection_skipped"
        return {"control": control, "empirical": empirical}

    if not raw_content.strip():
        control.phase = "reflection_skipped"
        return {"control": control, "empirical": empirical}

    # 在内容前加状态标记，让 LLM 知道这是失败还是成功执行
    status_marker = f"=== EXECUTION STATUS: {execution_status.upper()} ===\n"
    stdout_content = status_marker + raw_content

    # 通过 extra 注入 recent_stdout，渲染 reflection prompt
    system_prompt, audit = _prompt_audit(
        "reflection", state, runtime, extra={"recent_stdout": stdout_content[:2000]}
    )
    try:
        response = resolved_runtime.invoke_structured(
            "reflection", state, ReflectionResponse, system_prompt=system_prompt
        )

        # 构造 EmpiricalFinding 列表
        existing_count = len(empirical.findings)
        new_findings: list[EmpiricalFinding] = []
        for i, f in enumerate(response.findings):
            new_findings.append(EmpiricalFinding(
                id=f"finding_{existing_count + i + 1}",
                run_id=run_id,
                source_node="reflection",
                assumption_tested=f.assumption_tested,
                evidence=f.evidence,
                verdict=f.verdict,
                confidence=f.confidence,
                suggested_fix=f.suggested_fix,
            ))

        if new_findings:
            empirical.findings.extend(new_findings)
            empirical.run_index.append({
                "run_id": run_id,
                "summary": response.run_summary or f"执行 {run_id}",
                "log_path": str(log_path),
            })
            logger.info(
                "Reflection 提取 %d 条发现（run_id=%s）：%s",
                len(new_findings),
                run_id,
                response.run_summary,
            )

        # 追加触发 Clarifier 修正：仅在「上游未触发 + 自己发现 refuted + 预算有剩余」时
        already_triggered = control.trigger_clarifier_revision
        has_refuted = any(
            f.verdict == "refuted"
            and f.confidence >= REFUTED_CONFIDENCE_THRESHOLD
            for f in new_findings
        )
        if has_refuted and not already_triggered and control.modeling_revision_count < control.modeling_revision_budget:
            # Meta-Router（中枢 LLM）决策：基于全局失败历史判断下一步走向。
            # 不写死条件边，让 LLM 统筹判断回 Mathematician / Clarifier / Architect 还是接受失败。
            # 失败时回退到原逻辑（默认回 Clarifier）。
            refuted_findings = [
                f for f in new_findings
                if f.verdict == "refuted" and f.confidence >= REFUTED_CONFIDENCE_THRESHOLD
            ]
            meta_decision = _invoke_meta_router(
                state, runtime, resolved_runtime, refuted_findings
            )
            if meta_decision is not None:
                control.meta_decision = meta_decision.decision
                control.meta_direction_hint = meta_decision.direction_hint
                control.meta_reasoning = meta_decision.reasoning
                # 消费 budget（无论决策是什么，都算一次修正尝试）
                control.modeling_revision_count += 1
                control.trigger_clarifier_revision = True
                control.phase = "revision_triggered"
                logger.info(
                    "Meta-Router 决策：%s（置信度 %.2f）— %s | direction_hint=%s（预算 %d/%d）",
                    meta_decision.decision,
                    meta_decision.confidence,
                    meta_decision.reasoning[:100],
                    meta_decision.direction_hint[:100],
                    control.modeling_revision_count,
                    control.modeling_revision_budget,
                )
            else:
                # Meta-Router 调用失败，回退到原逻辑（回 Clarifier）
                control.trigger_clarifier_revision = True
                control.modeling_revision_count += 1
                control.phase = "revision_triggered"
                logger.info(
                    "Meta-Router 失败，回退到 Clarifier 修正（预算 %d/%d）",
                    control.modeling_revision_count,
                    control.modeling_revision_budget,
                )
        elif already_triggered:
            # 上游 ResultReviewer 已触发修正，保留其决策
            control.phase = "revision_triggered"
        elif result_paths_empty and not has_refuted:
            # V6 修复（问题 B）：coder 失败（result_paths 空）+ 无 refuted 发现 + 上游未触发修正
            # 消费 1 次 budget，让 route_after_reflection 的 budget 检查能正确反映已用预算，
            # 避免 architect→coder 失败→reflection→回退 architect→coder 失败... 死循环。
            # budget 未耗尽时 +1；budget 已耗尽时不再 +1（route_after_reflection 会强制前进到 writer）。
            if control.modeling_revision_count < control.modeling_revision_budget:
                control.modeling_revision_count += 1
                logger.info(
                    "Coder 失败但无 refuted 发现，消费 budget (%d/%d) 以触发回退重试",
                    control.modeling_revision_count,
                    control.modeling_revision_budget,
                )
            control.phase = "reflection_done"
        else:
            control.phase = "reflection_done"
    except Exception as exc:
        logger.error("Reflection LLM 调用失败: %s", exc)
        control.phase = "reflection_failed"
        # 即使 LLM 失败，也记录 run_index 便于后续按需查询
        empirical.run_index.append({
            "run_id": run_id,
            "summary": f"Reflection 失败: {str(exc)[:100]}",
            "log_path": str(log_path),
        })

    entry = _emit_process(
        runtime, control, state, "reflection", "reflected",
        f"反思完成：提取 {len(empirical.findings)} 条实证发现"
        f"（{execution_status}，run={run_id}）",
        {
            "run_id": run_id,
            "execution_status": execution_status,
            "findings_count": len(empirical.findings),
            "trigger_clarifier_revision": control.trigger_clarifier_revision,
            "meta_decision": control.meta_decision,
        },
        prompt_text=system_prompt,
    )
    return {
        "control": control,
        "empirical": empirical,
        "prompt_audit": audit,
        "process_log": [entry],
    }


def _classify_coder_error(stderr: str) -> str:
    """根据 stderr 内容判定 Coder 回滚目标：architect 或 clarifier。

    - 语法/导入/命名/类型错误 → architect（代码规范问题，需重新设计伪代码）
    - 求解/优化/数学错误 → clarifier（设定/公式问题，需重新建模）
    """
    stderr_lower = stderr.lower()
    # clarifier 触发：求解、优化、数学相关错误
    clarifier_keywords = [
        "valueerror", "runtimeerror", "optimization", "solver",
        "convergence", "infeasible", "singular", "nan", "inf",
        "math", "domain", "division by zero",
    ]
    if any(kw in stderr_lower for kw in clarifier_keywords):
        return "clarifier"
    # architect 触发：语法、导入、命名、类型错误
    architect_keywords = [
        "syntaxerror", "importerror", "module", "nameerror",
        "typeerror", "attributeerror", "keyerror", "indexerror",
    ]
    if any(kw in stderr_lower for kw in architect_keywords):
        return "architect"
    # 默认：architect
    return "architect"


def _extract_error_summary(stderr: str) -> str:
    """从 stderr 提取最后一行错误类型和消息，用于结构化日志摘要。

    不调 LLM，纯规则提取。提取不到时返回首行或「未知错误」。
    """
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return "未知错误"
    # 优先取最后一行以 Error/Exception 结尾的
    error_prefixes = (
        "Error", "ValueError", "RuntimeError", "TypeError",
        "KeyError", "IndexError", "AttributeError", "ImportError",
        "SyntaxError", "Exception",
    )
    for line in reversed(lines):
        if line.startswith(error_prefixes) or "Error:" in line or "Exception:" in line:
            return line[:200]
    return lines[-1][:200]


# ═══════════════════════════════════════════════════════════════════
# 阶段五：最终整合与成稿
# ═══════════════════════════════════════════════════════════════════

def _compile_latex_to_pdf(tex_path: Path, work_dir: Path) -> Path | None:
    """尝试使用 xelatex 或 pdflatex 编译 tex 为 pdf（跑两遍解决交叉引用）。"""
    for compiler in ("xelatex", "pdflatex"):
        if shutil.which(compiler):
            try:
                for _ in range(2):
                    subprocess.run(
                        [compiler, "-interaction=nonstopmode", str(tex_path.name)],
                        cwd=work_dir,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                pdf_path = tex_path.with_suffix(".pdf")
                if pdf_path.exists():
                    return pdf_path
            except subprocess.CalledProcessError as exc:
                logger.warning("%s 编译失败: %s", compiler, exc.stderr[:200])
    return None


def _apply_front_matter(main_tex: str, latex_content: str) -> str:
    """把 writer 输出的「标题/摘要/关键词」替换进模板 main.tex 的封面占位。

    latex_content 格式约定：每行一个字段，前缀分别为 标题：/ 摘要：/ 关键词：。
    """
    title = ""
    abstract = ""
    keywords = ""
    for line in (latex_content or "").splitlines():
        line = line.strip()
        if line.startswith("标题："):
            title = line[len("标题：") :].strip()
        elif line.startswith("摘要："):
            abstract = line[len("摘要：") :].strip()
        elif line.startswith("关键词："):
            keywords = line[len("关键词：") :].strip()
    if title:
        main_tex = re.sub(
            r"\\papertitle\{[^}]*\}",
            # lambda 返回字面文本：re.sub 会解析 replacement 中的反斜杠转义
            # （如 \p 报 bad escape），必须用 lambda 避免。
            lambda _m: f"\\papertitle{{{title}}}",
            main_tex,
            count=1,
        )
    if abstract or keywords:
        main_tex = re.sub(
            r"\\abstractcn\s*\{(.*?)\}\s*\{(.*?)\}",
            lambda _m: f"\\abstractcn{{{abstract}}}{{{keywords}}}",
            main_tex,
            count=1,
            flags=re.DOTALL,
        )
    return main_tex


def writer_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    artifacts_in = _artifacts(state)
    ltm_for_writer = _combined_sub_ltms(control) if control.sub_ltms else dynamic_ltm

    # V15：国赛模板模式 —— 复制模板并按实际子问题数量调整 main.tex 章节
    paper_dir = Path(resolved_runtime.output_path("paper"))
    from modeling_assistant.data.paper_template import copy_template

    template_structure = copy_template(
        resolved_runtime.settings.paper_template_dir,
        paper_dir,
        len(control.sub_questions or []) or 1,
    )

    # V10 修复：当 result_paths 为空时，扫描 results 目录下的 output_run_*.csv 备份
    # 这避免了"Coder 曾经成功产出真实结果，但被后续失败覆盖"导致的降级。
    # 备份是 Coder 成功执行时由 coder_node 写入的（output_run_N.csv），即使后续
    # ResultReviewer 拒绝了当前 result_paths，磁盘上的备份仍保留真实数值。
    using_backup_results = False
    if not artifacts_in.result_paths:
        results_dir = Path(resolved_runtime.output_path("results"))
        if results_dir.exists():
            # 扫描所有 output_run_*.csv 备份，按 run_id 排序取最新（编号最大）的一个
            backup_files = sorted(
                results_dir.glob("output_run_*.csv"),
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
            )
            # 过滤掉空文件（仅含注释行或字节数过小）
            valid_backups = [
                p for p in backup_files
                if p.stat().st_size > 20  # 至少 20 字节，过滤只有注释的空文件
            ]
            if valid_backups:
                latest_backup = valid_backups[-1]
                artifacts_in.result_paths = [str(latest_backup)]
                using_backup_results = True
                logger.info(
                    "Writer 加载历史成功结果备份：%s（当前 result_paths 为空，使用最新备份避免降级）",
                    latest_backup,
                )

    # 前置完整性检查：检测关键产物是否缺失
    integrity_warnings: list[str] = []
    if not ltm_for_writer.objective:
        integrity_warnings.append("动态 LTM 的 objective 为空：建模目标未确定，论文不得编造具体目标与结果。")
    if not ltm_for_writer.assumptions:
        integrity_warnings.append("动态 LTM 的 assumptions 为空：建模假设未确定，论文不得编造假设。")
    if not ltm_for_writer.equations:
        integrity_warnings.append("动态 LTM 的 equations 为空：核心方程未确定，论文不得编造公式。")
    if not artifacts_in.result_paths:
        integrity_warnings.append("result_paths 为空：Coder 未产出任何数值结果。论文中所有数值结果必须标注为「待验证」或「理论推导」，不得声称为已计算的结果。")
    elif using_backup_results:
        # V10 修复：使用备份结果时，标注警告但允许 writer 基于真实数值生成论文
        integrity_warnings.append(
            f"result_paths 来自历史成功备份（{artifacts_in.result_paths[0]}）：当前会话 ResultReviewer 拒绝了最新结果，"
            f"但 Coder 此前成功产出过真实数值。论文可基于该备份结果撰写，但需在论文中标注「结果来自历史执行备份，未经最新验证」。"
        )
    # 只有当 figure_paths 全部是 placeholder 或为空时才警告。
    # 如果含真实图片（非 placeholder），即使历史失败残留了 placeholder 也不警告，
    # 因为 writer 可以引用真实图片。
    real_figures = [p for p in artifacts_in.figure_paths if "placeholder" not in p.lower()]
    if not real_figures:
        integrity_warnings.append("figure_paths 全为占位图或为空：图表未真正生成。论文中不得声称「如图所示」并引用具体图表。")
    # V17：图表计划完整性 —— required 图未生成/未登记 → 警告
    fig_plan = getattr(artifacts_in, "figures_plan", []) or []
    fig_manifest = getattr(artifacts_in, "figure_manifest", {}) or {}
    missing_required = [
        p.id
        for p in fig_plan
        if p.required
        and (
            p.id not in fig_manifest
            or not Path(fig_manifest[p.id].get("path", "")).exists()
        )
    ]
    if missing_required:
        integrity_warnings.append(
            f"图表规划中的必需图未生成或未登记：{missing_required}。"
            "论文不得引用这些图，并应在对应章节标注「图表待补充」。"
        )

    integrity_text = "\n".join(f"- {w}" for w in integrity_warnings) if integrity_warnings else "无（所有关键产物完整）"

    # V10 修复：读取 result_paths 中的 CSV 内容（前 50 行）注入到 writer prompt
    # 让 writer 能直接引用真实数值而非编造。仅在 result_paths 非空时注入。
    result_preview = ""
    if artifacts_in.result_paths:
        try:
            import pandas as pd
            preview_parts: list[str] = []
            for path_str in artifacts_in.result_paths:
                path = Path(path_str)
                if path.exists() and path.suffix.lower() in (".csv", ".xlsx", ".xls"):
                    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
                    preview_lines = []
                    preview_lines.append(f"=== 结果文件 {path} ===")
                    preview_lines.append(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
                    preview_lines.append(f"列名: {list(df.columns)}")
                    preview_lines.append("前 50 行数据：")
                    preview_lines.append(df.head(50).to_string())
                    # 数值列统计摘要
                    numeric_df = df.select_dtypes(include=["number"])
                    if not numeric_df.empty:
                        preview_lines.append("\n数值列统计摘要：")
                        preview_lines.append(numeric_df.describe().to_string())
                    preview_parts.append("\n".join(preview_lines))
            result_preview = "\n\n".join(preview_parts)[:8000]
        except Exception as exc:
            logger.warning("Writer 读取结果文件预览失败: %s", exc)
            result_preview = ""

    extra = {"integrity_warnings": integrity_text}
    if result_preview:
        extra["result_preview"] = result_preview
    if control.paper_revision_feedback:
        extra["paper_revision_feedback"] = "\n".join(
            f"- {fb}" for fb in control.paper_revision_feedback
        )

    writer_state = dict(state)
    if control.sub_ltms:
        writer_state["dynamic_ltm"] = ltm_for_writer
    system_prompt, audit = _prompt_audit(
        "writer", writer_state, runtime, extra=extra
    )
    artifacts = ArtifactBundle()
    # V10 修复：保留 result_paths（含备份路径）传给 writer，让 writer 引用真实数值
    artifacts.result_paths = list(artifacts_in.result_paths)
    response = None
    try:
        response = resolved_runtime.invoke_structured(
            "writer", state, WriterResponse, system_prompt=system_prompt
        )
        latex_path = Path(resolved_runtime.output_path("paper", "main.tex"))
        if template_structure is not None and response.sections:
            # ── 模板模式：写各章节文件 + references.tex，main.tex 保留模板格式 ──
            sections_dir = paper_dir / "sections"
            sections_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in response.sections.items():
                # V16 修复：模板章节位于 paper/sections/ 下（main.tex 用
                # \input{sections/1_restatement} 引用），LLM 返回的键可能是
                # "1_restatement.tex" 或 "sections/1_restatement.tex"，
                # 统一归一到 sections/ 子目录（main.tex/references.tex 除外）。
                if fname.endswith(".tex") and fname not in ("main.tex", "references.tex"):
                    target = sections_dir / fname
                else:
                    target = paper_dir / fname
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            # 标题/摘要/关键词替换 main.tex 封面占位
            if response.latex_content:
                main_tex_text = latex_path.read_text(encoding="utf-8")
                main_tex_text = _apply_front_matter(main_tex_text, response.latex_content)
                latex_path.write_text(main_tex_text, encoding="utf-8")
            paper_full_text = "\n".join(
                p.read_text(encoding="utf-8", errors="replace")
                for p in paper_dir.rglob("*.tex")
            )
            # Exemplar 查重护栏：检测是否整句复制了示例库表达
            from modeling_assistant.validation.originality import check_writer_output

            warnings = check_writer_output(
                paper_full_text,
                state.get("exemplars"),
                n=resolved_runtime.settings.plagiarism_ngram,
                threshold=resolved_runtime.settings.plagiarism_threshold,
            )
            if warnings:
                audit["exemplar_originality_warning"] = "\n".join(warnings)
                logger.warning("Exemplar 查重护栏告警：%s", warnings)
            pdf_path = _compile_latex_to_pdf(latex_path, latex_path.parent)
            if pdf_path:
                artifacts.pdf_path = str(pdf_path)
        elif response.latex_content:
            # ── 旧行为：LLM 输出完整 main.tex ──
            resolved_runtime.write_file("paper", "main.tex", content=response.latex_content)
            from modeling_assistant.validation.originality import check_writer_output

            warnings = check_writer_output(
                response.latex_content,
                state.get("exemplars"),
                n=resolved_runtime.settings.plagiarism_ngram,
                threshold=resolved_runtime.settings.plagiarism_threshold,
            )
            if warnings:
                audit["exemplar_originality_warning"] = "\n".join(warnings)
                logger.warning("Exemplar 查重护栏告警：%s", warnings)
            pdf_path = _compile_latex_to_pdf(latex_path, latex_path.parent)
            if pdf_path:
                artifacts.pdf_path = str(pdf_path)
        artifacts.latex_path = str(latex_path)
    except Exception as exc:
        logger.error("Writer LLM 调用失败: %s", exc)
        artifacts.latex_path = resolved_runtime.output_path("paper", "main.tex")

    control.phase = "latex_drafted"
    control.hitl_required = True
    control.hitl_stage = "final"
    entry = _emit_process(
        runtime, control, state, "writer", "paper_drafted",
        f"论文草稿完成：{artifacts.latex_path or 'N/A'}",
        {
            "latex_path": artifacts.latex_path,
            "pdf_path": artifacts.pdf_path,
            "sections_count": len(getattr(response, "sections", {}) or {}),
            "integrity_warnings": integrity_text,
            "result_paths": list(artifacts.result_paths),
        },
    )
    return {
        "artifacts": artifacts,
        "control": control,
        "prompt_audit": audit,
        "process_log": [entry],
    }


def final_reviewer_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """论文终审：确定性验收（零 LLM）+ LLM 灵活审查（可选，失败降级）。

    V15 升级：原实现只设置 phase。现在：
    1. 确定性检查：章节存在性/一级标题、占位符、内部泄露、图片引用、编译。
       硬错误 → phase=paper_review_failed（报告供 HITL 展示，人可 rewrite 回 Writer）。
    2. 确定性通过后调用 LLM 做语义审查（数值一致性、图表解读、结构、表达），
       LLM 失败或报 fail 均不阻塞流程，最终裁决权在 HITL 终审。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    artifacts = _artifacts(state)
    paper_dir = Path(resolved_runtime.output_path("paper"))

    from modeling_assistant.validation.paper_check import check_paper

    # V17：把 Result Manifest 传给确定性验收，执行论文数字 ↔ 结果文件机器比对
    manifest = [
        entry.model_dump(mode="json")
        for entry in (control.results_manifest or [])
    ] or None
    # V18：执行证据回流后按承重契约验收（root_gaps/anchor_gaps/形态兜底）
    load_bearing_map = None
    if artifacts.load_bearing_map is not None:
        from modeling_assistant.analysis.load_bearing import reconcile_load_bearing_map

        load_bearing_map = reconcile_load_bearing_map(
            artifacts.load_bearing_map,
            state.get("empirical", EmpiricalLayer()),
        )
    deterministic = check_paper(
        paper_dir,
        manifest=manifest,
        results_root=resolved_runtime.settings.output_dir,
        figures_plan=[
            f.model_dump(mode="json") for f in (artifacts.figures_plan or [])
        ]
        or None,
        figure_manifest=dict(artifacts.figure_manifest or {}),
        load_bearing_map=load_bearing_map,
    )
    control.paper_review_report = dict(deterministic)
    audit = {
        "paper_check_report": json.dumps(deterministic, ensure_ascii=False, indent=2)
    }

    if not deterministic["passed"]:
        control.phase = "paper_review_failed"
        logger.warning(
            "论文确定性验收未通过（%d 个硬错误）：%s",
            len(deterministic["issues"]),
            deterministic["issues"][:3],
        )
        entry = _emit_process(
            runtime, control, state, "final_reviewer", "paper_reviewed",
            f"确定性验收未通过（{len(deterministic['issues'])} 个硬错误）",
            {
                "passed": False,
                "phase": control.phase,
                "issues_count": len(deterministic["issues"]),
                "first_issues": deterministic["issues"][:5],
            },
        )
        return {"control": control, "prompt_audit": audit, "process_log": [entry]}

    # 确定性通过 → LLM 灵活审查（失败降级为通过，不阻塞 HITL）
    try:
        from modeling_assistant.data.paper_template import read_paper_text

        paper_text = read_paper_text(paper_dir)
        result_preview = _result_preview_for_acceptance(artifacts)
        system_prompt, llm_audit = _prompt_audit(
            "final_reviewer",
            state,
            runtime,
            extra={
                "paper_text": paper_text,
                "result_preview": result_preview,
            },
        )
        response = resolved_runtime.invoke_structured(
            "final_reviewer",
            state,
            FinalReviewerResponse,
            system_prompt=system_prompt,
        )
        llm_report = {
            "verdict": response.verdict,
            "issues": list(response.issues),
            "suggestions": list(response.suggestions),
            "numerical_consistency": response.numerical_consistency,
            "summary": response.summary,
        }
        control.paper_review_report["llm"] = llm_report
        audit.update(llm_audit)
        control.phase = (
            "paper_review_failed" if response.verdict == "fail" else "paper_review_passed"
        )
        logger.info("终审 LLM 审查：%s — %s", response.verdict, response.summary[:120])
    except Exception as exc:
        logger.warning("终审 LLM 审查失败（降级为通过）: %s", exc)
        control.phase = "paper_review_passed"

    # V17 运行过程记录：终审留痕
    llm_verdict = (control.paper_review_report.get("llm") or {}).get("verdict")
    entry = _emit_process(
        runtime, control, state, "final_reviewer", "paper_reviewed",
        (
            f"终审验收：确定性通过，LLM 审查={llm_verdict or '降级通过'}"
            if llm_verdict
            else "终审验收：确定性通过（LLM 审查降级）"
        ),
        {
            "passed": control.paper_review_report.get("passed"),
            "phase": control.phase,
            "llm_verdict": llm_verdict,
            "issues_count": len(control.paper_review_report.get("issues", [])),
        },
    )
    return {"control": control, "prompt_audit": audit, "process_log": [entry]}
