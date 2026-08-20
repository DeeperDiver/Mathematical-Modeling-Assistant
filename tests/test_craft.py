"""行文技艺层（Writing Craft Layer）单元测试。"""

from __future__ import annotations

from modeling_assistant.data.craft_aggregate import (
    _normalize_algorithm_type,
    aggregate_craft_guides,
)
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.craft import (
    AlgorithmPattern,
    ArgumentFlow,
    CraftGuide,
    DerivationPattern,
    FigurePlacement,
    InterpretationPattern,
    SectionFocus,
    WritingCraft,
    WritingExample,
    WritingPattern,
)
from modeling_assistant.schemas.state import ExemplarContext


def _make_craft(card_id: str, section: str = "模型建立") -> WritingCraft:
    return WritingCraft(
        card_id=card_id,
        derivation=[
            DerivationPattern(
                section=section,
                trigger="目标函数含非线性项时",
                organization=["符号引入", "前提假设", "逐步推导", "目标函数", "物理解释"],
                notation_usage="符号先定义后使用",
                depth_strategy="关键步骤逐步推导",
                text_formula_ratio="每步配一句解释",
                closing_moves=["解释每项含义"],
                examples=[WritingExample(function="推导动机句", text="为刻画非线性特征，本文从基本关系出发逐步推导。")],
            )
        ],
        algorithm=[
            AlgorithmPattern(
                algorithm_type="遗传算法",
                presentation=["伪代码", "流程图"],
                flow_organization=["输入", "初始化", "迭代", "终止", "输出"],
                complexity_analysis="在算法后给出时间复杂度",
                convergence_justification="通过目标值下降曲线论证",
                result_reporting="按算法步骤对应报告",
                support_figures=["收敛曲线"],
            )
        ],
        interpretation=[
            InterpretationPattern(
                target="参数",
                organization=["先整体意义", "后取值范围"],
                domain_linking="与题目场景挂钩",
                parameter_meaning="说明单位与范围",
                sensitivity_handling="±10% 扰动",
                common_moves=["该参数表示"],
            )
        ],
        writing=[
            WritingPattern(
                function="过渡衔接句",
                examples=[WritingExample(function="过渡衔接句", text="基于上述假设，下面建立相应的数学模型。")],
                usage_notes="用于章节衔接",
            )
        ],
        figure_placements=[
            FigurePlacement(
                figure_type="line",
                section="模型求解",
                argument_role="展示目标值收敛趋势",
                caption_style="说明横纵轴含义",
            )
        ],
        section_focuses=[
            SectionFocus(
                section="模型建立",
                focus="定义变量与目标函数",
                weight=0.3,
                internal_order=["符号", "假设", "推导"],
            )
        ],
        argument_flow=ArgumentFlow(
            steps=["问题重述", "模型假设", "模型建立", "模型求解"],
            transitions=["由问题提出假设"],
        ),
    )


def test_aggregate_craft_guides_groups_by_type():
    crafts = [_make_craft("c1"), _make_craft("c2")]
    guides = aggregate_craft_guides(
        crafts,
        card_types={"c1": "optimization", "c2": "optimization"},
        min_occurrences=2,
    )
    assert len(guides) == 1
    g = guides[0]
    assert g.problem_type == "optimization"
    assert g.exemplar_ids == ["c1", "c2"]
    # 两篇共有的模式进入共性
    assert len(g.derivation_common) == 1
    assert len(g.algorithm_common) == 1
    assert len(g.writing_common) == 1
    assert len(g.figure_placement_common) == 1
    assert len(g.section_focus_common) == 1
    assert g.argument_flow_common is not None


def test_aggregate_requires_min_occurrences():
    crafts = [_make_craft("c1"), _make_craft("c2", section="模型分析")]
    guides = aggregate_craft_guides(
        crafts,
        card_types={"c1": "optimization", "c2": "optimization"},
        min_occurrences=2,
    )
    g = guides[0]
    # 推导章节不同（模型建立 vs 模型分析）→ 不进入共性
    assert len(g.derivation_common) == 0


def test_normalize_algorithm_type_groups_heuristics():
    assert _normalize_algorithm_type("遗传算法") == "启发式"
    assert _normalize_algorithm_type("贪心策略") == "启发式"
    assert _normalize_algorithm_type("动态规划") == "动态规划"
    assert _normalize_algorithm_type("线性规划求解") == "精确求解"
    assert _normalize_algorithm_type("蒙特卡洛仿真") == "随机模拟"


def test_craft_injected_into_templates():
    ctx = ExemplarContext(
        active=True,
        craft=CraftGuide(
            problem_type="optimization",
            writing_common=[
                WritingPattern(
                    function="摘要句子",
                    examples=[WritingExample(function="摘要句子", text="针对该问题，本文提出了相应的建模方案。")],
                )
            ],
            figure_placement_common=[
                FigurePlacement(
                    figure_type="line",
                    section="模型求解",
                    argument_role="展示收敛趋势",
                )
            ],
            section_focus_common=[
                SectionFocus(
                    section="模型建立",
                    focus="定义变量与目标",
                    weight=0.3,
                    internal_order=["符号", "假设"],
                )
            ],
            argument_flow_common=ArgumentFlow(steps=["问题", "模型", "求解"], transitions=["由问题引出模型"]),
        ),
        injection={"structure": True, "chart": True, "writing": True},
    )
    catalog = PromptCatalog()
    assert "行文技艺参考" in catalog.render("writer", PromptContext(exemplars=ctx))
    assert "图片位置规划参考" in catalog.render("drawer", PromptContext(exemplars=ctx))
    assert "正文侧重点与论证链条参考" in catalog.render("architect", PromptContext(exemplars=ctx))


def test_craft_empty_when_inactive():
    catalog = PromptCatalog()
    rendered = catalog.render("writer", PromptContext(exemplars=ExemplarContext()))
    # 默认空值不抛错，也不含技艺内容
    assert "数学推导安排" in rendered  # 标题存在
    assert "craft_derivation_json" not in rendered  # 未注入内容
