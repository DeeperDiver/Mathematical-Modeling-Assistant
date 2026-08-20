"""行文技艺层（Writing Craft Layer）数据模型。

与 Exemplar 表达学习互补：学习正文行文中「数学推导、算法分析、模型解释、
行文语言、图片位置规划、正文侧重点与论证链条」六大技艺是如何组织与展开的。

只记录"怎么讲"（组织顺序、表达动作、功能句型），不记录"讲了什么"
（具体公式、数值、算法参数）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class WritingExample(BaseModel):
    """功能化范例句：去题目化，仅体现句式功能。"""

    function: str  # 句型功能：推导动机句/假设铺垫句/过渡衔接句/结果解读句/结论升华句/局限说明句
    text: str  # 范例句（≤60 字，去题目化）
    note: str = ""  # 这句在行文中起什么作用


class DerivationPattern(BaseModel):
    """数学推导安排。"""

    section: str = ""  # 通常出现在哪个章节
    trigger: str = ""  # 什么情况触发推导
    organization: list[str] = Field(default_factory=list)  # 展开顺序
    notation_usage: str = ""  # 符号使用纪律
    depth_strategy: str = ""  # 详略策略
    text_formula_ratio: str = ""  # 文字与公式配合节奏
    closing_moves: list[str] = Field(default_factory=list)  # 收尾动作
    examples: list[WritingExample] = Field(default_factory=list)


class AlgorithmPattern(BaseModel):
    """算法分析安排。"""

    algorithm_type: str = ""  # 启发式/精确/贪心/DP/随机模拟…
    presentation: list[str] = Field(default_factory=list)  # 伪代码/流程图/步骤列表
    flow_organization: list[str] = Field(default_factory=list)
    complexity_analysis: str = ""
    convergence_justification: str = ""
    result_reporting: str = ""
    support_figures: list[str] = Field(default_factory=list)
    examples: list[WritingExample] = Field(default_factory=list)


class InterpretationPattern(BaseModel):
    """模型解释安排。"""

    target: str = ""  # 模型/参数/边界/结果
    organization: list[str] = Field(default_factory=list)
    domain_linking: str = ""
    parameter_meaning: str = ""
    sensitivity_handling: str = ""
    common_moves: list[str] = Field(default_factory=list)
    examples: list[WritingExample] = Field(default_factory=list)


class WritingPattern(BaseModel):
    """行文语言：功能句型库。"""

    function: str  # 句型功能分类
    examples: list[WritingExample] = Field(default_factory=list)
    usage_notes: str = ""  # 使用时机与注意事项


class FigurePlacement(BaseModel):
    """图片位置规划：图-章节-论证作用映射。"""

    figure_type: str
    section: str  # 放在哪个章节
    argument_role: str  # 支撑什么论证
    caption_style: str = ""  # 图注写法


class SectionFocus(BaseModel):
    """正文侧重点：每节写作重点与篇幅。"""

    section: str
    focus: str  # 写作重点
    weight: float = 0.0  # 篇幅占比（0~1）
    internal_order: list[str] = Field(default_factory=list)  # 节内展开顺序


class ArgumentFlow(BaseModel):
    """全文论证链条。"""

    steps: list[str] = Field(default_factory=list)  # 问题→假设→模型→求解→验证→评价
    transitions: list[str] = Field(default_factory=list)  # 步骤间衔接方式


class WritingCraft(BaseModel):
    """单篇深加工卡：一篇论文的六大行文技艺。"""

    card_id: str
    derivation: list[DerivationPattern] = Field(default_factory=list)
    algorithm: list[AlgorithmPattern] = Field(default_factory=list)
    interpretation: list[InterpretationPattern] = Field(default_factory=list)
    writing: list[WritingPattern] = Field(default_factory=list)
    figure_placements: list[FigurePlacement] = Field(default_factory=list)
    section_focuses: list[SectionFocus] = Field(default_factory=list)
    argument_flow: ArgumentFlow | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CraftGuide(BaseModel):
    """题型级行文技艺指南：同题型多篇共有模式（≥3 篇）。"""

    problem_type: str
    derivation_common: list[DerivationPattern] = Field(default_factory=list)
    algorithm_common: list[AlgorithmPattern] = Field(default_factory=list)
    interpretation_common: list[InterpretationPattern] = Field(default_factory=list)
    writing_common: list[WritingPattern] = Field(default_factory=list)
    figure_placement_common: list[FigurePlacement] = Field(default_factory=list)
    section_focus_common: list[SectionFocus] = Field(default_factory=list)
    argument_flow_common: ArgumentFlow | None = None
    exemplar_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
