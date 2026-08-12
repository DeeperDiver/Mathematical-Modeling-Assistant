"""优秀论文检索：题型判定 → 候选筛选 → TF-IDF 相关性 → ExemplarContext。

只注入与当前题型匹配且相关性达标的卡片；未命中时返回 inactive 的
ExemplarContext，调用方保持原有流程不变。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from modeling_assistant.config.settings import AppSettings
from modeling_assistant.data.exemplars import load_cards, load_guides, load_global_profile
from modeling_assistant.schemas.state import (
    ExemplarContext,
    ExemplarPaper,
    GlobalStyleProfile,
    TypeStyleGuide,
)
from modeling_assistant.validation.originality import overlap_ratio

logger = logging.getLogger(__name__)

PROBLEM_TYPES = ("optimization", "physics", "forecasting", "evaluation", "data_mining")

PROBLEM_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "optimization": ("优化", "调度", "规划", "配送", "路径", "排班", "选址", "分配", "库存"),
    "physics": ("物理", "运动", "轨迹", "速度", "加速度", "材料", "干涉", "弹道", "外延"),
    "forecasting": ("预测", "时序", "时间序列", "趋势", "预报", "客流"),
    "evaluation": ("评价", "评估", "评分", "综合", "比较", "层次分析"),
    "data_mining": ("分类", "聚类", "挖掘", "识别", "检测", "特征", "异常"),
}


def judge_problem_type(
    problem_text: str,
    *,
    runtime: Any | None = None,
    problem_understanding: str = "",
) -> tuple[str, float]:
    """规则关键词优先；零命中时用 LLM 兜底，仍失败则保守返回 data_mining。"""
    combined = f"{problem_text}\n{problem_understanding or ''}"
    scores = {
        ptype: sum(1 for kw in kws if kw in combined)
        for ptype, kws in PROBLEM_TYPE_KEYWORDS.items()
    }
    total = sum(scores.values())
    if total > 0:
        best = max(scores, key=scores.get)
        confidence = round(min(1.0, scores[best] / total + 0.05 * scores[best]), 3)
        return best, confidence

    llm_type = _llm_judge(combined, runtime)
    if llm_type:
        return llm_type, 0.6
    return "data_mining", 0.1


def _llm_judge(problem_text: str, runtime: Any) -> str | None:
    if runtime is None or getattr(runtime, "client", None) is None:
        return None
    system_prompt = (
        "你是数学建模赛题分类器。从以下类别中选择最合适的一类："
        "optimization（优化/调度/规划）、physics（物理机理/运动/材料）、"
        "forecasting（预测/时序）、evaluation（评价/决策分析）、data_mining（数据挖掘/识别）。"
        "只输出 JSON：{\"problem_type\": \"...\"}\n\n赛题：\n"
        f"{problem_text[:3000]}"
    )
    try:
        raw = runtime.invoke("exemplar_type_judge", {}, system_prompt=system_prompt)
        from modeling_assistant.agents.runtime import _extract_json

        data = json.loads(_extract_json(raw))
        ptype = str(data.get("problem_type", "")).strip()
        return ptype if ptype in PROBLEM_TYPES else None
    except Exception as exc:
        logger.warning("LLM 题型判定失败，回退默认: %s", exc)
        return None


def _card_text(card: ExemplarPaper) -> str:
    """卡片检索文本：标题 + 结构 + 亮点 + 标签 + 文风，全部为表达特征。"""
    parts = [
        card.title,
        card.problem_type,
        card.contest,
        " ".join(card.structure.keys()),
        " ".join(card.highlights),
        " ".join(card.tags),
        " ".join(f"{k}:{v}" for k, v in card.writing_style.items()),
    ]
    return " ".join(p for p in parts if p)


def _tfidf_scores(query: str, texts: list[str]) -> list[float]:
    """字符 n-gram TF-IDF，适配中英文混排；空语料时返回全 0。"""
    if not texts or not query.strip():
        return [0.0] * len(texts)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        matrix = vectorizer.fit_transform(texts + [query])
        query_vec = matrix[-1]
        docs = matrix[:-1]
        return [float((doc @ query_vec.T).toarray()[0, 0]) for doc in docs]
    except Exception as exc:
        logger.warning("TF-IDF 计算失败，退化为 0 分: %s", exc)
        return [0.0] * len(texts)


def _pick_guide(
    guides: list[TypeStyleGuide],
    problem_type: str,
    contest: str = "",
) -> TypeStyleGuide | None:
    """同题型指南优先取同赛事，其次取无赛事约束的，再取第一份。"""
    candidates = [g for g in guides if g.problem_type == problem_type]
    if not candidates:
        return None
    for g in candidates:
        if g.contest and g.contest == contest:
            return g
    for g in candidates:
        if not g.contest:
            return g
    return candidates[0]


def search_exemplars(
    problem_text: str,
    *,
    settings: AppSettings,
    runtime: Any | None = None,
    problem_understanding: str = "",
    contest: str = "",
) -> ExemplarContext:
    """加载知识库 → 题型判定 → 候选筛选 → 相关性阈值 → 注入包。"""
    exemplars_dir = Path(settings.exemplars_dir)
    profile = load_global_profile(exemplars_dir / "profile.yaml")
    cards = load_cards(exemplars_dir / "cards")
    guides = load_guides(exemplars_dir / "guides")
    if not cards and not guides:
        # L3 全局偏好独立于知识库命中，始终可作为最上层软约束注入
        if _profile_nonempty(profile):
            return ExemplarContext(profile=profile)
        return ExemplarContext()

    problem_type, _conf = judge_problem_type(
        problem_text, runtime=runtime, problem_understanding=problem_understanding
    )
    candidates = [c for c in cards if c.problem_type == problem_type]
    if not candidates:
        return ExemplarContext()

    texts = [_card_text(c) for c in candidates]
    sims = _tfidf_scores(problem_text, texts)
    # 中文短题面下 TF-IDF 余弦偏保守，补充字符 2-gram 重合率并取最大值
    overlaps = [overlap_ratio(problem_text, [t], n=2) for t in texts]
    contest_boost = [1.2 if c.contest and c.contest == contest else 0.8 for c in candidates]
    combined = [
        sim + 0.5 * overlap for sim, overlap in zip(sims, overlaps)
    ]
    scored = sorted(
        zip(candidates, combined, contest_boost),
        key=lambda item: item[1] * item[2],
        reverse=True,
    )

    relevance = max(max(sims), max(overlaps)) if sims else 0.0
    if relevance < settings.exemplar_min_relevance:
        logger.info(
            "Exemplar 相关性 %.3f 低于阈值 %.3f，关闭注入（题型=%s）",
            relevance,
            settings.exemplar_min_relevance,
            problem_type,
        )
        return ExemplarContext()

    top = [c for c, _sim, _boost in scored[: settings.exemplar_top_k]]
    guide = _pick_guide(guides, problem_type, contest)
    context = ExemplarContext(
        active=True,
        guide=guide,
        cards=top,
        profile=profile,
        injection={k: v > 0 for k, v in settings.style_injection.items()},
    )
    logger.info(
        "Exemplar 命中：题型=%s，注入 %d 张卡片，相关性=%.3f，指南=%s，全局偏好=%s",
        problem_type,
        len(top),
        relevance,
        guide.problem_type if guide else "无",
        "有" if profile.notes or profile.figure_preferences or profile.color_palette else "无",
    )
    return context


def _profile_nonempty(profile: GlobalStyleProfile) -> bool:
    """L3 偏好是否有实际内容（区别于默认空对象）。"""
    return bool(
        profile.notes
        or profile.color_palette
        or profile.figure_preferences
        or profile.writing_preferences
    )


def load_exemplar_context(
    settings: AppSettings,
    problem_text: str,
    *,
    runtime: Any | None = None,
    problem_understanding: str = "",
) -> ExemplarContext:
    """供 exemplar_loader_node 调用的入口。"""
    return search_exemplars(
        problem_text,
        settings=settings,
        runtime=runtime,
        problem_understanding=problem_understanding,
    )
