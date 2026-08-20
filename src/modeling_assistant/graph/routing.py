from __future__ import annotations

import logging
from typing import Literal

from modeling_assistant.schemas.state import ControlState, GraphState

logger = logging.getLogger(__name__)


def _modeling_budget_exhausted(control: ControlState) -> bool:
    """建模阶段统一预算守卫。

    覆盖所有回到 mathematician/clarifier 的路径，预算耗尽时强制放行到 HITL，
    避免死循环。详见 ControlState.modeling_revision_budget 注释。
    """
    return control.modeling_revision_count >= control.modeling_revision_budget


def route_after_realist(state: GraphState) -> Literal["mathematician", "arbiter", "clarifier"]:
    """Realist 之后的路由：

    Goal.md 要求：当 debate_round > 3 时 Arbiter 才介入。
    因此统一使用 ``debate_round > max_debate_rounds`` 作为 Arbiter 触发条件。

    1. 超过最大轮数（无论分数是否达标）→ arbiter
    2. 全部被剪枝 → mathematician（重新发散）；预算耗尽 → arbiter 仲裁
    3. 分数达标 → clarifier
    4. 分数不达标 → mathematician；预算耗尽 → arbiter 仲裁
    """
    control = state["control"]
    scores_ok = (
        control.innovation_score >= control.innovation_threshold
        and control.feasibility_score >= control.feasibility_threshold
    )
    # 1. 超过最大轮数 → arbiter 检查退化
    if control.debate_round > control.max_debate_rounds:
        return "arbiter"
    # 2. 全部被剪枝 → 回 mathematician（预算耗尽时强制 arbiter 仲裁）
    if control.need_rebrainstorm:
        if _modeling_budget_exhausted(control):
            logger.warning(
                "Modeling budget exhausted (%d/%d) at realist rebrainstorm, forcing arbiter",
                control.modeling_revision_count, control.modeling_revision_budget,
            )
            return "arbiter"
        return "mathematician"
    # 3. 分数达标 → 直接到 clarifier
    if scores_ok:
        return "clarifier"
    # 4. 分数不达标 → 回 mathematician（预算耗尽时强制 arbiter）
    if _modeling_budget_exhausted(control):
        logger.warning(
            "Modeling budget exhausted (%d/%d) at realist low scores, forcing arbiter",
            control.modeling_revision_count, control.modeling_revision_budget,
        )
        return "arbiter"
    return "mathematician"


def route_after_arbiter(state: GraphState) -> Literal["clarifier", "rollback", "hitl_arbitration"]:
    control = state["control"]
    if control.hitl_required and control.hitl_stage == "arbitration":
        return "hitl_arbitration"
    if control.rollback_to_version:
        return "rollback"
    return "clarifier"


def route_after_coder(
    state: GraphState,
) -> Literal["architect", "clarifier", "result_reviewer", "reflection"]:
    """Coder 之后的路由。

    V6.1 修复：所有 coder 失败路径（无论 error_count 多少）都经过 reflection 节点，
    让 reflection_node 消费 budget（coder 失败 + 无 refuted 时 +1），
    然后由 route_after_reflection 决定回退到 coder_rollback_target 或前进到 writer。
    这避免了「architect → coder 失败 3 次 → architect」死循环（budget 不增加）。

    V9 修复：双重保险 —— 优先检查 control.phase，避免依赖 result_paths（可能含旧值
    导致误判）。coder_node 失败时会设置 clear_result_paths=True 让 reducer 清空，
    但为了健壮性，这里也检查 phase。

    - 失败 phase（code_execution_failed/code_generation_failed/code_generation_empty）
      → 进入 reflection 节点，不论 result_paths 状态。
    - 成功 phase + 有结果文件路径 → 进入 result_reviewer 验证。
    - 其他情况 → 进入 reflection。
    """
    control = state["control"]
    artifacts = state.get("artifacts", {})
    result_paths = getattr(artifacts, "result_paths", []) if hasattr(artifacts, "result_paths") else artifacts.get("result_paths", [])

    # V9 修复：失败 phase 强制走 reflection，避免依赖 result_paths（可能含旧值）
    # 失败 phase 由 coder_node 在失败路径设置：
    #   - code_generation_failed: LLM 调用失败
    #   - code_generation_empty: LLM 未返回代码
    #   - code_execution_failed: 自修复耗尽，代码执行失败
    #   - code_precheck_failed: V11.2 新增，常量校验失败自修复耗尽
    if control.phase in (
        "code_generation_failed",
        "code_generation_empty",
        "code_execution_failed",
        "code_precheck_failed",
    ):
        return "reflection"

    # 成功路径：检查 result_paths
    if result_paths:
        return "result_reviewer"

    # 兜底：result_paths 为空但 phase 不在失败列表中（不应发生，但保险起见）
    return "reflection"


def route_after_result_reviewer(
    state: GraphState,
) -> Literal["reflection", "sub_question_acceptance"]:
    """ResultReviewer 之后的路由。

    V9 修复：验证失败也统一走 reflection 节点消费 budget。
    原逻辑直接回退到 architect/clarifier 不消费 budget，导致死循环：
      architect → coder 成功但结果质量差 → result_reviewer 失败 → clarifier
      → milestone_reviewer_1 通过 → architect → coder ... 无限循环，budget 一直为 0。

    - 验证失败 → 进入 reflection 节点消费 budget，由 route_after_reflection 决定回退。
    - 验证通过 → 进入 reflection 节点提取实证发现。
    """
    # V14 小题循环：机械校验通过后进入人工验收闸门；
    # 失败仍走 reflection 消费预算并带回反馈。
    control = state["control"]
    if control.phase == "result_review_passed":
        return "sub_question_acceptance"
    return "reflection"


def route_after_reflection(state: GraphState) -> Literal["clarifier", "collect_artifacts", "architect", "mathematician", "hitl_modeling"]:
    """Reflection 之后的路由：

    - Meta-Router 决策优先：Reflection 发现 refuted 后由中枢 LLM 判断走向
      （rediscover→mathematician / refine_assumptions→clarifier /
       adjust_architecture→architect / accept_failure→collect_artifacts）
    - 无 Meta-Router 决策时回退到原逻辑：回 Clarifier 修正假设
    - 预算耗尽时触发 HITL modeling，让人类决断（accept/retry/redirect），而非直接产出"待验证"论文。
    - V6 修复（问题 B）：coder 失败（result_paths 为空）且未触发 clarifier 修正时，
      不应前进到 writer 生成不完整论文，而应回退到 coder_rollback_target 重试。
      预算耗尽时触发 HITL modeling，由人类决断。
    """
    control = state["control"]
    artifacts = state.get("artifacts", {})
    result_paths = getattr(artifacts, "result_paths", []) if hasattr(artifacts, "result_paths") else artifacts.get("result_paths", [])

    if control.trigger_clarifier_revision and not _modeling_budget_exhausted(control):
        # Meta-Router 决策优先：按中枢 LLM 的判断路由
        meta = control.meta_decision
        if meta == "rediscover":
            logger.info("Meta-Router 决策：回 Mathematician 重新发散")
            return "mathematician"
        elif meta == "adjust_architecture":
            logger.info("Meta-Router 决策：回 Architect 调整模型设计")
            return "architect"
        elif meta == "accept_failure":
            logger.info("Meta-Router 决策：接受失败，前进到 collect_artifacts")
            return "collect_artifacts"
        # meta == "refine_assumptions" 或 meta 为空 → 回 Clarifier（原逻辑）
        return "clarifier"
    if control.trigger_clarifier_revision and _modeling_budget_exhausted(control):
        logger.warning(
            "Modeling budget exhausted (%d/%d) at reflection, triggering HITL modeling",
            control.modeling_revision_count, control.modeling_revision_budget,
        )
        return "hitl_modeling"

    # V6 修复（问题 B）：coder 失败（result_paths 为空）时不应前进到 writer
    if not result_paths:
        if _modeling_budget_exhausted(control):
            logger.warning(
                "Modeling budget exhausted (%d/%d) at reflection with empty result_paths, "
                "triggering HITL modeling (writer will mark results as 待验证 if human accepts)",
                control.modeling_revision_count, control.modeling_revision_budget,
            )
            return "hitl_modeling"
        # 预算未耗尽 → 回退到 coder_rollback_target 重试
        target = control.coder_rollback_target or "architect"
        logger.info(
            "Coder failed with empty result_paths, returning to %s (budget %d/%d)",
            target, control.modeling_revision_count, control.modeling_revision_budget,
        )
        return target

    return "collect_artifacts"


def route_after_architecture_hitl(state: GraphState) -> Literal["rollback", "architect"]:
    control = state["control"]
    if control.rollback_to_version:
        return "rollback"
    return "architect"


def route_after_hitl_modeling(
    state: GraphState,
) -> Literal["collect_artifacts", "architect", "mathematician"]:
    """HITL modeling 之后的路由：

    根据人类决策路由：
    - accept（hitl_modeling_accepted）→ collect_artifacts（产出"待验证"论文）
    - retry（hitl_modeling_retry）→ architect（重置预算后回 Architect 重试当前方案）
    - redirect（hitl_modeling_redirect）→ mathematician（重置预算后回 Mathematician 重新发散）
    """
    control = state["control"]
    phase = control.phase
    if phase == "sub_question_passed":
        # 小题循环：HITL 接受当前小题后推进
        if control.current_sub_question_index < len(control.sub_questions or []):
            return "mathematician"
        return "collect_artifacts"
    if phase == "hitl_modeling_retry":
        logger.info("HITL modeling: 人类选择 retry，回 Architect 重试")
        return "architect"
    if phase == "hitl_modeling_redirect":
        logger.info("HITL modeling: 人类选择 redirect，回 Mathematician 重新发散")
        return "mathematician"
    # accept 或其他 → collect_artifacts
    logger.info("HITL modeling: 人类选择 accept，前进到 collect_artifacts")
    return "collect_artifacts"


def route_after_milestone_reviewer_1(
    state: GraphState,
) -> Literal["mathematician", "hitl_architecture"]:
    control = state["control"]
    if control.need_rebrainstorm:
        # 预算耗尽 → 强制放行到 HITL，让人类决断
        if _modeling_budget_exhausted(control):
            logger.warning(
                "Modeling budget exhausted (%d/%d) at milestone rejection, forcing HITL",
                control.modeling_revision_count, control.modeling_revision_budget,
            )
            return "hitl_architecture"
        return "mathematician"
    return "hitl_architecture"


def route_after_final_review(state: GraphState) -> Literal["rollback", "hitl_final", "writer"]:
    """终审后路由：
    - rollback_to_version（retry）→ 先 rollback checkout，再回建模阶段
    - paper_rewrite_requested（rewrite）→ 回 Writer 按反馈重写论文
    - 其他（approve / 预算耗尽的 rewrite）→ 完成
    """
    control = state["control"]
    if control.rollback_to_version:
        return "rollback"
    if control.phase == "paper_rewrite_requested":
        return "writer"
    return "hitl_final"


def route_after_rollback(state: GraphState) -> Literal["architect", "mathematician"]:
    """根据回滚来源决定下一步：

    - architecture_hitl / arbitration → 回滚到 architect（阶段三）
    - final_hitl → 回滚到 mathematician（阶段二，重新建模）
    """
    control = state["control"]
    if control.rollback_source in ("final_hitl", "cross_sub_question"):
        return "mathematician"
    return "architect"


# ── V14 小题循环路由 ─────────────────────────────────────────────

def route_after_split(state: GraphState) -> Literal["analyst", "split_sub_questions"]:
    """小题确认后进入全局分析；edit 后回到拆分节点重新确认。"""
    control = state["control"]
    if control.sub_questions_confirmed:
        return "analyst"
    return "split_sub_questions"


def route_after_acceptance(
    state: GraphState,
) -> Literal[
    "mathematician",
    "architect",
    "hitl_implementation_human",
    "cross_sub_question_hitl",
    "hitl_modeling",
    "collect_artifacts",
]:
    """小题验收之后的路由。"""
    control = state["control"]
    phase = control.phase
    if phase == "sub_question_fail_code":
        if control.sub_question_attempts >= control.sub_question_budget:
            return "hitl_modeling"
        return "hitl_implementation_human"
    if phase == "sub_question_fail_architecture":
        if control.sub_question_attempts >= control.sub_question_budget:
            return "hitl_modeling"
        return "architect"
    if phase == "sub_question_fail_model":
        if control.sub_question_attempts >= control.sub_question_budget:
            return "hitl_modeling"
        return "mathematician"
    if phase == "cross_sub_question_requested":
        return "cross_sub_question_hitl"
    if phase == "sub_question_passed":
        if control.current_sub_question_index < len(control.sub_questions or []):
            return "mathematician"
        return "collect_artifacts"
    # 兜底：继续建模当前小题
    return "mathematician"


def route_after_cross_sub_question(
    state: GraphState,
) -> Literal["mathematician", "collect_artifacts"]:
    """跨小题 HITL 之后的路由：accept/rollback 重做当前小题，continue 推进。"""
    control = state["control"]
    if control.phase == "sub_question_passed":
        if control.current_sub_question_index < len(control.sub_questions or []):
            return "mathematician"
        return "collect_artifacts"
    return "mathematician"


# ── V13 编程手模式路由（builtin / codex / human）──────────────────

def route_after_architect_external(
    state: GraphState,
) -> Literal["hitl_implementation_review", "dispatch_implementation"]:
    """外部编程手模式下 Architect 之后的路由。

    首次进入先做实现架构人工审核；审核通过后（含后续失败重试）
    直接进入任务包分发，不再重复打断。
    """
    control = state["control"]
    if control.implementation_architecture_reviewed:
        return "dispatch_implementation"
    return "hitl_implementation_review"


def route_after_implementation_review(
    state: GraphState,
) -> Literal["dispatch_implementation", "architect", "rollback"]:
    """实现架构人工审核之后的路由。"""
    control = state["control"]
    if control.rollback_to_version:
        return "rollback"
    if control.phase == "hitl_implementation_revised":
        return "architect"
    return "dispatch_implementation"


def route_after_implementation_human(
    state: GraphState,
) -> Literal["coder", "architect"]:
    """等待人工编程手交付之后的路由。

    - approve/auto → coder（human 模式读取 solution.py；auto 走内置 LLM）
    - revise → architect（按反馈修改方案）
    """
    control = state["control"]
    if control.phase == "implementation_revised":
        return "architect"
    return "coder"
