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
) -> Literal["reflection", "architect", "clarifier"]:
    """ResultReviewer 之后的路由。

    V9 修复：验证失败也统一走 reflection 节点消费 budget。
    原逻辑直接回退到 architect/clarifier 不消费 budget，导致死循环：
      architect → coder 成功但结果质量差 → result_reviewer 失败 → clarifier
      → milestone_reviewer_1 通过 → architect → coder ... 无限循环，budget 一直为 0。

    - 验证失败 → 进入 reflection 节点消费 budget，由 route_after_reflection 决定回退。
    - 验证通过 → 进入 reflection 节点提取实证发现。
    """
    return "reflection"


def route_after_reflection(state: GraphState) -> Literal["clarifier", "collect_artifacts", "architect"]:
    """Reflection 之后的路由：

    - 有高置信度 refuted 发现且修正预算未耗尽 → 回 Clarifier 修正假设
    - 预算耗尽时即使有 refuted 发现也强制放行，避免无限循环。
    - V6 修复（问题 B）：coder 失败（result_paths 为空）且未触发 clarifier 修正时，
      不应前进到 writer 生成不完整论文，而应回退到 coder_rollback_target 重试。
      预算耗尽时才强制前进到 collect_artifacts，由 writer 在 integrity_warnings
      中标注「result_paths 为空，所有数值结果必须标注为待验证」。
    """
    control = state["control"]
    artifacts = state.get("artifacts", {})
    result_paths = getattr(artifacts, "result_paths", []) if hasattr(artifacts, "result_paths") else artifacts.get("result_paths", [])

    if control.trigger_clarifier_revision and not _modeling_budget_exhausted(control):
        return "clarifier"
    if control.trigger_clarifier_revision and _modeling_budget_exhausted(control):
        logger.warning(
            "Modeling budget exhausted (%d/%d) at reflection, forcing collect_artifacts",
            control.modeling_revision_count, control.modeling_revision_budget,
        )
        return "collect_artifacts"

    # V6 修复（问题 B）：coder 失败（result_paths 为空）时不应前进到 writer
    if not result_paths:
        if _modeling_budget_exhausted(control):
            logger.warning(
                "Modeling budget exhausted (%d/%d) at reflection with empty result_paths, "
                "forcing collect_artifacts (writer will mark results as 待验证)",
                control.modeling_revision_count, control.modeling_revision_budget,
            )
            return "collect_artifacts"
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


def route_after_final_review(state: GraphState) -> Literal["rollback", "hitl_final"]:
    """终审后：如果 hitl_final 设置了 rollback_to_version，先 rollback checkout，再回建模阶段。"""
    control = state["control"]
    if control.rollback_to_version:
        return "rollback"
    return "hitl_final"


def route_after_rollback(state: GraphState) -> Literal["architect", "mathematician"]:
    """根据回滚来源决定下一步：

    - architecture_hitl / arbitration → 回滚到 architect（阶段三）
    - final_hitl → 回滚到 mathematician（阶段二，重新建模）
    """
    control = state["control"]
    if control.rollback_source == "final_hitl":
        return "mathematician"
    return "architect"
