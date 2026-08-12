"""HITL 反馈回写：以滑动平均慢更新卡片与指南的质量权重，避免一次反馈带偏知识库。"""

from __future__ import annotations

from modeling_assistant.schemas.state import ExemplarContext


def apply_feedback(
    current_score: float,
    feedback_score: float,
    alpha: float,
) -> float:
    """滑动平均：new = (1-alpha)*old + alpha*feedback。"""
    if not 0.0 <= feedback_score <= 1.0:
        raise ValueError(f"feedback_score 必须在 [0,1] 内，收到 {feedback_score}")
    return round((1.0 - alpha) * current_score + alpha * feedback_score, 3)


def apply_feedback_to_context(
    context: ExemplarContext,
    feedback_score: float,
    alpha: float,
) -> ExemplarContext:
    """把一次人类评分回写到注入包中的卡片与指南（内存中，供节点落盘）。"""
    updated = context.model_copy(deep=True)
    for card in updated.cards:
        card.quality_score = apply_feedback(card.quality_score, feedback_score, alpha)
    if updated.guide is not None:
        updated.guide.quality_score = apply_feedback(
            updated.guide.quality_score, feedback_score, alpha
        )
    return updated
