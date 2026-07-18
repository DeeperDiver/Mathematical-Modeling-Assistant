from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from modeling_assistant.agents.runtime import AgentRuntime, get_default_runtime
from modeling_assistant.agents.searcher import SearchQuery
from modeling_assistant.memory.archive import make_snapshot
from modeling_assistant.memory.validation import validate_dynamic_ltm
from modeling_assistant.schemas.responses import (
    AnalystResponse,
    ArchitectResponse,
    ClarifierResponse,
    CoderResponse,
    DrawerResponse,
    MathematicianResponse,
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
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    old_dynamic_ltm = _dynamic_ltm(state)
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

    # 符号查重与公式闭环校验
    validation_errors = validate_dynamic_ltm(new_dynamic_ltm)
    if validation_errors:
        audit["clarifier_validation_errors"] = "; ".join(validation_errors)
        logger.warning("Clarifier LTM 校验失败：%s", validation_errors)
        # 尝试一次 LLM 修复
        try:
            repair_prompt = system_prompt + "\n\n以下是符号/公式校验错误，请修正后重新输出：\n" + "\n".join(validation_errors)
            repaired = resolved_runtime.invoke_structured(
                "clarifier", state, ClarifierResponse, system_prompt=repair_prompt
            )
            commit_summary = repaired.commit_summary
            new_dynamic_ltm = DynamicLTM(
                assumptions=repaired.assumptions,
                nomenclature=repaired.nomenclature,
                equations=repaired.equations,
                objective=repaired.objective,
                solution_outline=repaired.solution_outline,
            )
            remaining = validate_dynamic_ltm(new_dynamic_ltm)
            if not remaining:
                audit["clarifier_validation_errors"] = ""
                logger.info("Clarifier LLM 修复后校验通过。")
            else:
                audit["clarifier_validation_errors"] = "; ".join(remaining)
                logger.warning("Clarifier 修复后仍有校验错误，保留 LLM 输出。")
        except Exception as repair_exc:
            logger.error("Clarifier 修复尝试失败: %s", repair_exc)

    # 归档旧 LTM：objective 变化 → major_bump (v2.0)
    major_change = (
        old_dynamic_ltm.objective != ""
        and new_dynamic_ltm.objective != old_dynamic_ltm.objective
    )
    checkpoint_id = None
    if config and "configurable" in config:
        checkpoint_id = config["configurable"].get("checkpoint_id")
    snapshot = make_snapshot(
        old_dynamic_ltm,
        archive,
        reason="Clarifier overwrote Core State Two.",
        commit_summary=commit_summary,
        major_bump=major_change,
        checkpoint_id=checkpoint_id,
    )

    control.phase = "dynamic_ltm_committed"
    control.hitl_required = True
    control.hitl_stage = "architecture"
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

    # 先做一次硬性校验：空字段或符号闭环失败直接拒绝
    hard_issues: list[str] = []
    if not dynamic_ltm.assumptions:
        hard_issues.append("假设列表为空。")
    if not dynamic_ltm.nomenclature:
        hard_issues.append("符号表为空。")
    if not dynamic_ltm.equations:
        hard_issues.append("公式列表为空。")
    if not dynamic_ltm.objective:
        hard_issues.append("目标函数/优化目标为空。")
    validation_errors = validate_dynamic_ltm(dynamic_ltm)
    hard_issues.extend(validation_errors)

    if hard_issues:
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
    return {"type": "approve"}


def architect_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    artifacts = _artifacts(state)

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

    扩展（视觉洞察回流）：绘图成功后，把 LLM 对图像的文字观察写入 empirical 层，
    作为 inconclusive 发现（低置信度，需 Reflection 或后续验证才能定性）。
    这让「画出图后才发现变量关系非线性」的洞察不再丢失。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)
    empirical = _empirical(state)

    system_prompt, audit = _prompt_audit("drawer", state, runtime)
    artifacts = ArtifactBundle()
    try:
        response = resolved_runtime.invoke_structured(
            "drawer", state, DrawerResponse, system_prompt=system_prompt
        )
        # 写入真实文件
        if response.figure_code:
            resolved_runtime.write_file("figures", "figures.py", content=response.figure_code)
            # 执行绘图代码，并扫描生成的图片文件
            success, _stdout, stderr = resolved_runtime.run_code(response.figure_code)
            if success:
                figure_dir = Path(resolved_runtime.output_path("figures"))
                if figure_dir.exists():
                    figure_files = sorted(
                        p
                        for p in figure_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
                        and p.name not in {"placeholder.png", "figures.py"}
                    )
                    artifacts.figure_paths = [str(p) for p in figure_files] or [
                        resolved_runtime.output_path("figures", "placeholder.png")
                    ]
                else:
                    artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]
            else:
                logger.warning("Drawer 绘图代码执行失败: %s", stderr[:200])
                artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]
        else:
            artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]

        # 视觉洞察回流：把 Drawer 对图像的观察写入 empirical 层
        # 使用 LLM 自评的 verdict/confidence，而非硬编码 0.5。
        # 这让「散点明显非线性」这类强信号能直接触发 Clarifier 修正。
        # 若有 image_stats（客观统计量），附加到 evidence 作为 Reflection 二次确认的依据。
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
            # 重新派生 refuted/open_questions
            from modeling_assistant.schemas.state import _rebuild_empirical_derived_fields
            _rebuild_empirical_derived_fields(empirical)
            logger.info(
                "Drawer 视觉观察（verdict=%s, conf=%.2f）：%s",
                response.observation_verdict,
                response.observation_confidence,
                response.observation.strip()[:100],
            )
    except Exception as exc:
        logger.error("Drawer LLM 调用失败: %s", exc)
        artifacts.figure_paths = [resolved_runtime.output_path("figures", "placeholder.png")]

    control.phase = "figures_ready"
    return {"artifacts": artifacts, "control": control, "empirical": empirical, "prompt_audit": audit}


def coder_node(state: GraphState, runtime: AgentRuntime | None = None, config: dict | None = None) -> GraphState:
    """完全屏蔽冗长 Context，只依据动态 LTM 与 Architect 产物编写代码。

    回滚触发机制（Goal.md）：连续失败 3 次 → 按错误类型退回 Architect 或 Clarifier。
    - SyntaxError / ImportError / NameError / TypeError → architect（代码规范问题）
    - ValueError / RuntimeError / 求解失败 → clarifier（设定/公式问题）

    注意：LLM 调用失败、返回空代码、代码执行失败均视为一次失败。

    实证证据落盘：每次执行（无论成败）都把 stdout/stderr 落盘到
    outputs/logs/run_{n}.log，供 Reflection 节点按需读取。失败日志
    摘要结构化为「[run_id] summary (log_path)」，避免 500 字符截断丢信息。
    """
    resolved_runtime = _runtime(runtime)
    control = _control(state)

    system_prompt, audit = _prompt_audit("coder", state, runtime)
    artifacts = ArtifactBundle()
    try:
        response = resolved_runtime.invoke_structured(
            "coder", state, CoderResponse, system_prompt=system_prompt
        )
        if not response.code:
            # LLM 未返回代码，视为一次失败
            control.coder_error_count += 1
            run_id = f"run_{control.coder_run_count}"
            log_path = resolved_runtime.output_path("logs", f"{run_id}_empty.log")
            control.coder_error_log.append(f"[{run_id}] Coder 未生成任何代码。 (日志: {log_path})")
            control.coder_run_count += 1
            control.phase = "code_generation_empty"
            control.coder_rollback_target = "architect"
            logger.warning(
                "Coder 未生成代码 (第 %d 次)。",
                control.coder_error_count,
            )
            artifacts.result_paths = []
        else:
            resolved_runtime.write_file("results", "model.py", content=response.code)

            # 提取真实数据路径，供代码执行时使用
            static_ltm = _static_ltm(state)
            data_paths = (
                static_ltm.data_profile.file_paths
                if static_ltm.data_profile
                else []
            )

            # 实际执行代码，并传入真实数据路径
            success, stdout, stderr = resolved_runtime.run_code(
                response.code, data_paths=data_paths
            )
            if success:
                result_path_str = response.result_path or "results/output.csv"
                expected_path = Path(result_path_str)
                # LLM 返回的 result_path 约定为相对于 output_dir 的相对路径
                if not expected_path.is_absolute():
                    expected_path = resolved_runtime.settings.output_dir / expected_path
                if expected_path.exists():
                    # 落盘 stdout/stderr 供 Reflection 节点读取
                    run_id = f"run_{control.coder_run_count}"
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
                else:
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
                    control.coder_error_count += 1
                    control.coder_error_log.append(
                        f"[{run_id}] 代码执行成功但未找到结果文件：{expected_path} (日志: {log_path})"
                    )
                    control.phase = "code_result_missing"
                    control.coder_rollback_target = "architect"
                    artifacts.result_paths = []
            else:
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
                control.coder_error_count += 1
                # 结构化摘要：[run_id] summary (log_path)，便于按需调取完整日志
                summary = _extract_error_summary(stderr)
                control.coder_error_log.append(f"[{run_id}] {summary} (日志: {log_path})")
                control.phase = "code_execution_failed"
                # 按错误类型判定回滚目标
                control.coder_rollback_target = _classify_coder_error(stderr)
                logger.warning(
                    "Coder 代码执行失败 (第 %d 次), 回滚目标=%s: %s",
                    control.coder_error_count,
                    control.coder_rollback_target,
                    stderr[:200],
                )
                artifacts.result_paths = []
    except Exception as exc:
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
        # LLM 调用失败时不应虚构结果路径，否则 ResultReviewer 会检测到文件不存在
        # 并触发无意义的回退循环。
        artifacts.result_paths = []

    return {"artifacts": artifacts, "control": control, "prompt_audit": audit}


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

    # 注意：不重置 trigger_clarifier_revision。ResultReviewer 可能已经设置过。
    # Reflection 只能在「上游未触发修正 + 自己新发现 refuted + 预算有剩余」时追加触发。

    # 读取最近一次 Coder 成功执行的日志
    # coder_run_count 已在落盘后自增，所以最近一次的 run_id 是 run_{count-1}
    last_run_idx = max(control.coder_run_count - 1, 0)
    run_id = f"run_{last_run_idx}"
    log_path = Path(resolved_runtime.output_path("logs", f"{run_id}.log"))

    if not log_path.exists():
        # 最近一次没有成功日志（可能是失败路径过来的），跳过反思
        control.phase = "reflection_skipped"
        return {"control": control, "empirical": empirical}

    try:
        stdout_content = log_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Reflection 读取日志失败 %s: %s", log_path, exc)
        control.phase = "reflection_skipped"
        return {"control": control, "empirical": empirical}

    if not stdout_content.strip():
        control.phase = "reflection_skipped"
        return {"control": control, "empirical": empirical}

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
        if has_refuted and not already_triggered and control.empirical_revision_count < control.empirical_revision_budget:
            control.trigger_clarifier_revision = True
            control.empirical_revision_count += 1
            control.phase = "revision_triggered"
            logger.info(
                "Reflection 触发 Clarifier 修正（已用 %d/%d 预算）",
                control.empirical_revision_count,
                control.empirical_revision_budget,
            )
        elif already_triggered:
            # 上游 ResultReviewer 已触发修正，保留其决策
            control.phase = "revision_triggered"
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

    system_prompt, audit = _prompt_audit("writer", state, runtime)
    artifacts = ArtifactBundle()
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