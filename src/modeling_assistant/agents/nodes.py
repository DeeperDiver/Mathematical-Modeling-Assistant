from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from modeling_assistant.agents.runtime import AgentRuntime, get_default_runtime
from modeling_assistant.agents.searcher import SearchQuery
from modeling_assistant.data.facts import extract_facts_from_problem
from modeling_assistant.memory.archive import make_snapshot
from modeling_assistant.memory.validation import validate_dynamic_ltm
from modeling_assistant.schemas.responses import (
    AnalystResponse,
    ArchitectResponse,
    ClarifierResponse,
    CoderResponse,
    DrawerResponse,
    MathematicianResponse,
    MetaRouterResponse,
    MilestoneReviewer1Response,
    RealistResponse,
    ReflectionResponse,
    WriterResponse,
)
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    EmpiricalFinding,
    EmpiricalLayer,
    GraphState,
    LiteratureItem,
    PlanCandidate,
    REFUTED_CONFIDENCE_THRESHOLD,
    StaticLTM,
)

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


def _control(state: GraphState) -> ControlState:
    return state.get("control", ControlState()).model_copy(deep=True)


def _static_ltm(state: GraphState) -> StaticLTM:
    return state.get("static_ltm", StaticLTM()).model_copy(deep=True)


def _dynamic_ltm(state: GraphState) -> DynamicLTM:
    return state.get("dynamic_ltm", DynamicLTM()).model_copy(deep=True)


def _artifacts(state: GraphState) -> ArtifactBundle:
    return state.get("artifacts", ArtifactBundle()).model_copy(deep=True)


def _empirical(state: GraphState) -> EmpiricalLayer:
    return state.get("empirical", EmpiricalLayer()).model_copy(deep=True)


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
    return {"control": control}


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
                    "你是一个学术检索专家。根据以下破题思路与原始问题，提取 3-5 个核心检索关键词，"
                    "用逗号分隔，只输出关键词，不要其他内容。\n\n"
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
                source=r.source,
                summary=r.summary,
                url=r.url,
            )
            for r in validated
        ]

    control = _control(state)
    control.phase = "literature_collected"
    return {"static_ltm": static_ltm, "control": control}


# ═══════════════════════════════════════════════════════════════════
# 阶段二：建模核心 —— "先发散，后剪枝"
# ═══════════════════════════════════════════════════════════════════

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
        control.top_k_plans = [
            PlanCandidate(
                id=plan.get("id", f"plan_{i}"),
                title=plan.get("title", "未命名方案"),
                description=plan.get("description", ""),
                innovation_score=plan.get("innovation_score", 50),
                feasibility_score=plan.get("feasibility_score", 50),
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

    return {"control": control, "prompt_audit": audit}


def realist_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """挑刺与剪枝：从数据、算力、常识三维度评估每个方案。

    - feasibility < threshold 的方案 → kill（剪枝）
    - innovation < threshold 的方案 → reject（打回修改）
    - 其余 → keep
    - 若全部被剪枝 → need_rebrainstorm = True，路由回 Mathematician
    - 否则选综合评分最高的 keep 方案为 selected_plan_id
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)

    if not control.top_k_plans:
        control.need_rebrainstorm = True
        control.phase = "plan_scored"
        return {"control": control}

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
                plan.verdict = evaln.verdict

        # 应用阈值剪枝：覆盖未在 LLM 评估中的方案
        for plan in control.top_k_plans:
            if plan.feasibility_score < control.feasibility_threshold:
                plan.verdict = "kill"
            elif plan.innovation_score < control.innovation_threshold:
                plan.verdict = "reject"
            else:
                plan.verdict = "keep"

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
            if plan.feasibility_score < control.feasibility_threshold:
                plan.verdict = "kill"
            elif plan.innovation_score < control.innovation_threshold:
                plan.verdict = "reject"
            else:
                plan.verdict = "keep"

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
                if plan.innovation_score >= control.innovation_threshold
                and plan.feasibility_score >= control.feasibility_threshold
            ]
            selected = max(
                viable or control.top_k_plans,
                key=lambda p: p.total_score(w_inn, w_fea),
            )
            control.selected_plan_id = selected.id
            control.innovation_score = selected.innovation_score
            control.feasibility_score = selected.feasibility_score
            control.need_rebrainstorm = not viable

    control.phase = "plan_scored"
    return {"control": control, "prompt_audit": audit}


def arbiter_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    archive = state.get("ltm_archive", [])

    # 如果没有历史版本，直接放行
    if not archive:
        control.phase = "plan_arbitrated"
        return {"control": control}

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

    return {"control": control, "prompt_audit": audit}


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
    return {
        "dynamic_ltm": new_dynamic_ltm,
        "ltm_archive": [snapshot],
        "control": control,
        "prompt_audit": audit,
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
        return {"control": control}

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

    return {"control": control, "prompt_audit": audit}


def hitl_architecture_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Milestone Reviewer 1：架构确认前的人类审核。

    首次进入时 interrupt() 暂停图执行，等待用户输入。
    用户输入 'approve' 继续，'rollback <version>' 回滚。
    """
    decision = interrupt({
        "stage": "architecture",
        "message": "请审核当前建模方案。",
        "hint": "输入 'approve' 放行进入架构设计，或 'rollback v1.0' 回滚到指定版本。",
        "dynamic_ltm": _dynamic_ltm(state).model_dump(),
        "control_summary": {
            "phase": state.get("control", ControlState()).phase,
            "selected_plan_id": state.get("control", ControlState()).selected_plan_id,
            "innovation_score": state.get("control", ControlState()).innovation_score,
            "feasibility_score": state.get("control", ControlState()).feasibility_score,
        },
    })

    control = _control(state)
    action = _parse_hitl_decision(decision)

    if action["type"] == "rollback":
        control.rollback_to_version = action.get("version")
        control.rollback_source = "architecture_hitl"
        control.phase = "hitl_rollback_requested"
    else:
        control.phase = "architecture_approved"
        control.hitl_required = False
        control.hitl_stage = "none"
        control.rollback_source = "none"
    return {"control": control}


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
    return {"control": control}


def hitl_modeling_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """建模预算耗尽时的人类介入节点。

    当 modeling_revision_count >= modeling_revision_budget 时触发，
    让人类决断下一步，而非直接产出"待验证"论文。

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
        ),
        "hint": (
            "输入 'accept' 接受失败并产出'待验证'论文；"
            "输入 'retry' 重置预算并回到 Architect 重试当前方案；"
            "输入 'redirect <方向提示>' 重置预算并回到 Mathematician 重新发散。"
        ),
        "control_summary": {
            "phase": control.phase,
            "budget_used": control.modeling_revision_count,
            "budget_limit": control.modeling_revision_budget,
            "current_sub_problem": control.current_sub_problem,
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
        # accept：接受失败，前进到 collect_artifacts
        control.phase = "hitl_modeling_accepted"
        logger.info("HITL modeling: 人类选择 accept，接受失败产出'待验证'论文")

    return {"control": control}


def hitl_final_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """Milestone Reviewer 2：终稿审查。

    首次进入时 interrupt() 暂停图执行，等待用户输入。
    用户输入 'approve' 完成，'retry' 回到建模阶段重新打磨。
    """
    decision = interrupt({
        "stage": "final",
        "message": "请审核最终论文。",
        "hint": "输入 'approve' 完成流程，或 'retry' 回到建模阶段重新打磨。",
        "artifacts_summary": state.get("artifacts", ArtifactBundle()).model_dump(),
    })

    control = _control(state)
    action = _parse_hitl_decision(decision)

    if action["type"] == "retry":
        control.rollback_to_version = action.get("version")
        control.rollback_source = "final_hitl"
        control.phase = "hitl_retry_requested"
    else:
        control.phase = "completed"
        control.hitl_required = False
        control.hitl_stage = "none"
        control.rollback_source = "none"
    return {"control": control}


def _parse_hitl_decision(decision) -> dict:
    """解析用户输入，支持字符串和字典两种 resume 格式。"""
    if isinstance(decision, dict):
        return {
            "type": decision.get("action", "approve"),
            "version": decision.get("version"),
        }
    text = str(decision).strip().lower()
    if text.startswith("rollback"):
        parts = text.split()
        version = parts[1] if len(parts) > 1 else None
        return {"type": "rollback", "version": version}
    if text.startswith("retry"):
        parts = text.split()
        version = parts[1] if len(parts) > 1 else None
        return {"type": "retry", "version": version}
    if text.startswith("reject"):
        return {"type": "reject"}
    if text.startswith("redirect"):
        parts = text.split(maxsplit=1)
        hint = parts[1] if len(parts) > 1 else ""
        return {"type": "redirect", "version": hint}
    if text.startswith("accept"):
        return {"type": "accept"}
    return {"type": "approve"}


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
    except Exception as exc:
        logger.error("Architect LLM 调用失败: %s", exc)
        dynamic_ltm = _dynamic_ltm(state)
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
    return {"artifacts": artifacts, "control": control, "prompt_audit": audit}


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

    for attempt in range(MAX_SELF_REPAIR + 1):  # 0, 1, 2 = 共 3 次机会
        # 渲染 prompt：自修复时注入 recent_stderr 让 coder 看到完整错误
        extra = {"recent_stderr": recent_stderr} if recent_stderr else None
        system_prompt, audit = _prompt_audit("coder", state, runtime, extra=extra)

        try:
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
            return {"artifacts": artifacts, "control": control, "prompt_audit": audit}

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
            return {"artifacts": artifacts, "control": control, "prompt_audit": audit}

        # V11 修复：第三层常量校验 —— 在执行代码前做静态扫描
        # 如果发现关键常量缺失或列名错误，直接进入自修复循环，不消耗 budget
        from modeling_assistant.validation.constants import check_code_against_facts
        constant_issues = check_code_against_facts(response.code, static_ltm, artifacts)
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
            return {"artifacts": artifacts, "control": control, "prompt_audit": audit}

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
                artifacts.result_paths = [str(expected_path)]
                if attempt > 0:
                    logger.info("Coder 自修复第 %d 次尝试成功", attempt)
                return {"artifacts": artifacts, "control": control, "prompt_audit": audit}
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
        if attempt < MAX_SELF_REPAIR:
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
        return {"artifacts": artifacts, "control": control, "prompt_audit": audit}

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

    return {"control": control, "empirical": empirical, "prompt_audit": audit}


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
    """尝试使用 xelatex 或 pdflatex 编译 tex 为 pdf，返回 pdf 路径或 None。"""
    for compiler in ("xelatex", "pdflatex"):
        if shutil.which(compiler):
            try:
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


def writer_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    dynamic_ltm = _dynamic_ltm(state)
    artifacts_in = _artifacts(state)

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
    if not dynamic_ltm.objective:
        integrity_warnings.append("动态 LTM 的 objective 为空：建模目标未确定，论文不得编造具体目标与结果。")
    if not dynamic_ltm.assumptions:
        integrity_warnings.append("动态 LTM 的 assumptions 为空：建模假设未确定，论文不得编造假设。")
    if not dynamic_ltm.equations:
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

    integrity_text = "\n".join(f"- {w}" for w in integrity_warnings) if integrity_warnings else "无（所有关键产物完整）"

    # V10 修复：读取 result_paths 中的 CSV 内容（前 50 行）注入到 writer prompt
    # 让 writer 能直接引用真实数值而非编造。仅在 result_paths 非空时注入。
    result_preview = ""
    if artifacts_in.result_paths:
        try:
            import pandas as pd
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
                    result_preview = "\n".join(preview_lines)[:5000]  # 截断到 5000 字符
                    break  # 只读第一个结果文件
        except Exception as exc:
            logger.warning("Writer 读取结果文件预览失败: %s", exc)
            result_preview = ""

    extra = {"integrity_warnings": integrity_text}
    if result_preview:
        extra["result_preview"] = result_preview

    system_prompt, audit = _prompt_audit(
        "writer", state, runtime, extra=extra
    )
    artifacts = ArtifactBundle()
    # V10 修复：保留 result_paths（含备份路径）传给 writer，让 writer 引用真实数值
    artifacts.result_paths = list(artifacts_in.result_paths)
    try:
        response = resolved_runtime.invoke_structured(
            "writer", state, WriterResponse, system_prompt=system_prompt
        )
        # 写入真实 LaTeX 文件
        latex_path = Path(resolved_runtime.output_path("paper", "main.tex"))
        if response.latex_content:
            resolved_runtime.write_file("paper", "main.tex", content=response.latex_content)
            # 尝试编译 PDF
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
    return {"artifacts": artifacts, "control": control, "prompt_audit": audit}


def final_reviewer_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    control = _control(state)
    control.phase = "final_review_ready"
    return {"control": control}