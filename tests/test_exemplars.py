"""Exemplar Learning System 单元测试。

覆盖：数据模型、卡片加载、题型判定、检索命中/未命中/低阈值、
prompt 注入渲染、查重护栏、反馈滑动平均、loader 节点无库降级。
"""

from __future__ import annotations

import json
from pathlib import Path

from modeling_assistant.agents.nodes import exemplar_loader_node
from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config.settings import AppSettings, load_settings
from modeling_assistant.data.exemplars import load_cards, save_card, save_guide
from modeling_assistant.memory.exemplar_feedback import (
    apply_feedback,
    apply_feedback_to_context,
)
from modeling_assistant.memory.exemplar_search import (
    judge_problem_type,
    search_exemplars,
)
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import (
    ControlState,
    ExemplarContext,
    ExemplarFigure,
    ExemplarPaper,
    GlobalStyleProfile,
    StaticLTM,
    TypeStyleGuide,
)
from modeling_assistant.validation.originality import (
    check_originality,
    check_writer_output,
)


# ── 1. 数据模型 ──────────────────────────────────────────────────────────────


def test_exemplar_context_defaults_inactive():
    ctx = ExemplarContext()
    assert ctx.active is False
    assert ctx.cards == []
    assert ctx.guide is None
    assert ctx.injection == {"structure": True, "chart": True, "writing": True}
    # JSON round-trip
    restored = ExemplarContext.model_validate_json(ctx.model_dump_json())
    assert restored.active is False


def test_exemplar_paper_defaults():
    card = ExemplarPaper(id="c1")
    assert card.quality_score == 0.5
    assert card.structure == {}


# ── 2. 卡片加载 ──────────────────────────────────────────────────────────────


def test_load_cards_skips_broken_files(tmp_path):
    good = ExemplarPaper(id="good", problem_type="optimization")
    save_card(good, tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    cards = load_cards(tmp_path)
    assert [c.id for c in cards] == ["good"]


def test_manual_cards_load_from_repo():
    """仓库内 2 张手工示例卡片应可正常加载。"""
    repo_cards = Path(__file__).resolve().parents[1] / "exemplars" / "cards"
    cards = load_cards(repo_cards)
    assert len(cards) >= 2
    assert {c.problem_type for c in cards} >= {"optimization", "physics"}


# ── 3. 题型判定 ──────────────────────────────────────────────────────────────


def test_judge_problem_type_keywords():
    assert judge_problem_type("城市物流配送调度优化问题")[0] == "optimization"
    assert judge_problem_type("烟雾弹运动轨迹与遮蔽时间")[0] == "physics"
    ptype, conf = judge_problem_type("完全没有赛题关键词 abcdefg")
    assert ptype == "data_mining"
    assert conf < 0.5


# ── 4. 检索：命中 / 未命中 / 低于阈值 ────────────────────────────────────────


def _make_kb(tmp_path: Path) -> Path:
    """构造含 optimization + physics 卡片与 optimization 指南的临时知识库。"""
    cards_dir = tmp_path / "cards"
    guides_dir = tmp_path / "guides"
    save_card(
        ExemplarPaper(
            id="opt1",
            title="城市绿色物流配送调度",
            problem_type="optimization",
            contest="华中杯",
            structure={"问题重述": "概述", "模型建立": "定义变量与约束"},
            highlights=["两阶段启发式降低求解时间"],
            tags=["vehicle-routing", "heuristic"],
            writing_style={"tense": "一般现在时"},
        ),
        cards_dir,
    )
    save_card(
        ExemplarPaper(
            id="phy1",
            title="烟雾弹干扰效能建模",
            problem_type="physics",
            contest="国赛",
            structure={"模型假设": "匀速下沉", "模型建立": "运动学方程"},
            highlights=["多层球壳遮蔽概率模型"],
            tags=["ballistics"],
        ),
        cards_dir,
    )
    save_guide(
        TypeStyleGuide(
            problem_type="optimization",
            contest="华中杯",
            common_structure=["问题重述", "模型建立", "模型求解", "结果分析"],
            recommended_figures=["gantt"],
            writing_baseline={"tense": "一般现在时"},
            exemplar_ids=["opt1"],
        ),
        guides_dir,
    )
    (tmp_path / "profile.yaml").write_text(
        "color_palette: ['#4C72B0']\nfigure_preferences:\n  - 优先箱线图\n",
        encoding="utf-8",
    )
    return tmp_path


def test_search_hits_matching_type(tmp_path):
    kb = _make_kb(tmp_path)
    settings = AppSettings(
        exemplars_dir=kb,
        exemplar_min_relevance=0.25,
        exemplar_top_k=2,
    )
    ctx = search_exemplars(
        "城市绿色物流配送调度优化路径规划",
        settings=settings,
        contest="华中杯",
    )
    assert ctx.active is True
    assert [c.id for c in ctx.cards] == ["opt1"]
    assert ctx.guide is not None
    assert ctx.guide.problem_type == "optimization"
    assert "问题重述" in ctx.guide.common_structure
    assert ctx.profile is not None
    assert ctx.profile.color_palette == ["#4C72B0"]


def test_search_no_cards_inactive(tmp_path):
    settings = AppSettings(exemplars_dir=tmp_path)  # 空知识库
    ctx = search_exemplars("城市绿色物流配送调度优化", settings=settings)
    assert ctx.active is False


def test_search_type_mismatch_inactive(tmp_path):
    kb = _make_kb(tmp_path)
    settings = AppSettings(exemplars_dir=kb, exemplar_min_relevance=0.25)
    # 题面是评价类，知识库无评价卡片 → 不注入
    ctx = search_exemplars("供应商综合评价与层次分析", settings=settings)
    assert ctx.active is False


def test_search_type_match_injects_even_with_low_text_overlap(tmp_path):
    """题型命中即注入：表达特征与题面字面重合天然偏低，不应阻断同题型参考。"""
    cards_dir = tmp_path / "cards"
    save_card(
        ExemplarPaper(
            id="low1",
            title="z",
            problem_type="optimization",
            tags=[],
            structure={},
            highlights=[],
        ),
        cards_dir,
    )
    settings = AppSettings(exemplars_dir=tmp_path, exemplar_min_relevance=0.25)
    ctx = search_exemplars("城市绿色物流配送调度优化", settings=settings)
    # 题型命中但卡片文本与题面无共享 n-gram → 仍注入（同题型兜底）
    assert ctx.active is True
    assert [c.id for c in ctx.cards] == ["low1"]


# ── 5. Prompt 注入渲染 ───────────────────────────────────────────────────────


def test_prompt_catalog_injects_exemplar_vars():
    ctx = PromptContext(
        exemplars=ExemplarContext(
            active=True,
            guide=TypeStyleGuide(
                problem_type="optimization",
                common_structure=["问题重述", "模型建立"],
                recommended_figures=["gantt"],
                writing_baseline={"tense": "一般现在时"},
            ),
            cards=[
                ExemplarPaper(
                    id="opt1",
                    title="城市绿色物流配送调度",
                    problem_type="optimization",
                    structure={"问题重述": "概述"},
                    figures=[
                        ExemplarFigure(
                            figure_type="gantt",
                            purpose="展示车辆时间轴",
                            style_notes="不同车辆不同颜色",
                        )
                    ],
                    writing_style={"detail_level": "公式与文字解释并重"},
                    summary_style="背景+方法+结果",
                    highlights=["两阶段启发式"],
                    quotes=["本文针对城市绿色物流配送问题，构建了双目标优化模型。"],
                )
            ],
            profile=GlobalStyleProfile(color_palette=["#4C72B0"]),
            injection={"structure": True, "chart": True, "writing": True},
        )
    )
    catalog = PromptCatalog()
    rendered = catalog.render("writer", ctx)
    assert "exemplar_active=true" in rendered
    assert "问题重述" in rendered
    assert "两阶段启发式" in rendered
    assert "#4C72B0" in rendered


def test_prompt_catalog_inactive_yields_empty_blocks():
    catalog = PromptCatalog()
    rendered = catalog.render("writer", PromptContext(exemplars=ExemplarContext()))
    assert "exemplar_active=false" in rendered
    assert "exemplar_quotes_json" not in rendered  # 空 JSON 不注入示例内容


def test_all_templates_render_with_exemplar_vars():
    """所有引用 exemplar 变量的模板在 active 与 inactive 下都能渲染。"""
    catalog = PromptCatalog()
    for name in ("architect", "drawer", "writer", "milestone_reviewer_1"):
        catalog.render(name, PromptContext(exemplars=ExemplarContext()))
        catalog.render(
            name,
            PromptContext(
                exemplars=ExemplarContext(
                    active=True,
                    guide=TypeStyleGuide(problem_type="optimization"),
                    cards=[ExemplarPaper(id="c1", problem_type="optimization")],
                    profile=GlobalStyleProfile(),
                )
            ),
        )


# ── 6. 查重护栏 ──────────────────────────────────────────────────────────────


def test_check_originality_detects_overlap():
    ref = "本文针对城市绿色物流配送问题，构建了以碳排放与配送成本为双目标的车辆路径优化模型。"
    output = ref + " 其余部分是完全原创的补充内容。"
    report = check_originality(output, [ref], n=8, threshold=0.15)
    assert report["passed"] is False
    assert report["overlap_ratio"] > 0.15


def test_check_originality_passes_for_distinct_text():
    output = "本文提出一种基于时间序列的交通拥堵预测方法，并利用真实数据验证了有效性。"
    report = check_originality(output, ["城市绿色物流配送双目标优化模型"], n=8, threshold=0.15)
    assert report["passed"] is True


def test_check_writer_output_inactive_returns_empty():
    assert check_writer_output("任意文本", None) == []
    assert check_writer_output("任意文本", ExemplarContext()) == []


# ── 7. 反馈滑动平均 ──────────────────────────────────────────────────────────


def test_apply_feedback_sliding_average():
    assert apply_feedback(0.8, 0.5, alpha=0.3) == 0.71
    assert apply_feedback(0.8, 1.0, alpha=0.3) == 0.86


def test_apply_feedback_to_context_updates_cards_and_guide():
    context = ExemplarContext(
        active=True,
        guide=TypeStyleGuide(problem_type="optimization", quality_score=0.5),
        cards=[ExemplarPaper(id="c1", quality_score=0.8)],
    )
    updated = apply_feedback_to_context(context, 0.2, alpha=0.3)
    assert updated.cards[0].quality_score == 0.62
    assert updated.guide.quality_score == 0.41
    # 原对象不变（deep copy）
    assert context.cards[0].quality_score == 0.8


# ── 8. Loader 节点无库降级 ───────────────────────────────────────────────────


def test_exemplar_loader_node_inactive_with_empty_library(tmp_path):
    runtime = AgentRuntime.from_settings(
        load_settings(output_dir=tmp_path / "out", exemplars_dir=tmp_path / "kb")
    )
    result = exemplar_loader_node(
        {
            "static_ltm": StaticLTM(raw_problem="城市绿色物流配送调度优化"),
            "control": ControlState(),
        },
        runtime=runtime,
    )
    assert result["exemplars"].active is False
