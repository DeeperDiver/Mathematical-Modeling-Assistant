from __future__ import annotations

from typing import Literal

from modeling_assistant.schemas.state import GraphState


def route_after_realist(state: GraphState) -> Literal["mathematician", "arbiter", "clarifier"]:
    """Realist 之后的路由：

    Goal.md 要求：当 debate_round > 3 时 Arbiter 才介入。
    因此统一使用 ``debate_round > max_debate_rounds`` 作为 Arbiter 触发条件。

    1. 超过最大轮数（无论分数是否达标）→ arbiter
    2. 全部被剪枝 → mathematician（重新发散）
    3. 分数达标 → clarifier
    4. 分数不达标 → mathematician
    """
    control = state["control"]
    scores_ok = (
        control.innovation_score >= control.innovation_threshold
        and control.feasibility_score >= control.feasibility_threshold
    )
    # 1. 超过最大轮数 → arbiter 检查退化
    if control.debate_round > control.max_debate_rounds:
        return "arbiter"
    # 2. 全部被剪枝 → 必须回 mathematician
    if control.need_rebrainstorm:
        return "mathematician"
    # 3. 分数达标 → 直接到 clarifier
    if scores_ok:
        return "clarifier"
    # 4. 分数不达标 → 回 mathematician
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
) -> Literal["architect", "clarifier", "result_reviewer", "collect_artifacts"]:
    """Coder 之后的路由。

    - 连续失败 3 次 → 按错误类型回退到 architect 或 clarifier。
    - 有结果文件路径 → 进入 result_reviewer 验证。
    - 无结果文件路径 → 降级到 collect_artifacts（保持流程可用性，避免死循环）。
    """
    control = state["control"]
    artifacts = state.get("artifacts", {})
    result_paths = getattr(artifacts, "result_paths", []) if hasattr(artifacts, "result_paths") else artifacts.get("result_paths", [])

    if control.coder_error_count >= 3:
        return control.coder_rollback_target
    if result_paths:
        return "result_reviewer"
    return "collect_artifacts"


def route_after_result_reviewer(
    state: GraphState,
) -> Literal["reflection", "architect", "clarifier"]:
    """ResultReviewer 之后的路由：

    - 验证失败 → 按错误类型回退到 architect 或 clarifier（与原逻辑一致）
    - 验证通过 → 进入 reflection 节点提取实证发现
    """
    control = state["control"]
    if control.phase == "result_review_failed":
        return control.coder_rollback_target
    return "reflection"


def route_after_reflection(state: GraphState) -> Literal["clarifier", "collect_artifacts"]:
    """Reflection 之后的路由：

    - 有高置信度 refuted 发现且修正预算未耗尽 → 回 Clarifier 修正假设
    - 否则 → collect_artifacts（保持原流程进入 Writer）

    预算耗尽时即使有 refuted 发现也强制放行，避免无限循环。
    """
    control = state["control"]
    if control.trigger_clarifier_revision:
        return "clarifier"
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
