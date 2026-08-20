"""LLM 结构化响应模型 —— 每个节点对应一个 Pydantic 模型，用于 JSON 模式解析。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalystResponse(BaseModel):
    problem_understanding: str = ""
    data_schema: dict[str, str] = Field(default_factory=dict)


class PlanEvaluation(BaseModel):
    """Realist 对单个候选方案的评估。"""

    plan_id: str
    innovation_score: int = Field(default=0, ge=0, le=100)
    feasibility_score: int = Field(default=0, ge=0, le=100)
    verdict: Literal["keep", "kill", "reject"] = "keep"
    feedback: str = ""


class MathematicianResponse(BaseModel):
    plans: list[dict] = Field(default_factory=list)
    branch_requested: bool = False
    branch_from_version: str | None = None
    branch_reason: str = ""
    requested_version: str | None = None
    requested_evidence_run_id: str | None = None  # 请求查看某次执行的完整 stdout 日志
    # 每个 plan: {id, title, description, innovation_score, feasibility_score}


class RealistResponse(BaseModel):
    innovation_score: int = Field(default=0, ge=0, le=100)
    feasibility_score: int = Field(default=0, ge=0, le=100)
    selected_plan_id: str = ""
    feedback: str = ""
    plan_evaluations: list[PlanEvaluation] = Field(default_factory=list)


class ClarifierResponse(BaseModel):
    assumptions: list[str] = Field(default_factory=list)
    nomenclature: dict[str, str] = Field(default_factory=dict)
    equations: list[str] = Field(default_factory=list)
    objective: str = ""
    solution_outline: str = ""
    commit_summary: str = ""


class MilestoneReviewer1Response(BaseModel):
    approval: bool = True
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""


class ResultColumnSpec(BaseModel):
    """结果契约中的单列规格。"""

    name: str
    dtype: Literal["int", "float", "category", "text", "datetime"] = "float"
    min: float | None = None
    max: float | None = None
    # 为 True 时，该列在不同样本/分组间必须存在区分度（nunique > 1）
    distinct_required: bool = False
    description: str = ""


class ResultContract(BaseModel):
    """Architect 声明的输出结果契约。

    仅声明"答案应该长什么样"，不含具体数值：
    - allow_single_row=True 表示标量答案（如"有效遮蔽时长"）合法，不应被单行误杀
    - min_rows/max_rows 限定期望行数
    - columns 声明必需列、类型、合理范围与是否要求区分度
    """

    description: str = ""
    allow_single_row: bool = False
    min_rows: int | None = None
    max_rows: int | None = None
    columns: list[ResultColumnSpec] = Field(default_factory=list)
    allow_extra_columns: bool = True


class FigurePlan(BaseModel):
    """Architect 声明的预期图表：类型、用途、数据来源。

    V15 新增 kind 字段，区分数据图与论文必需的非数据图：
    - data：数据驱动图（散点/折线/热力图/箱线图等），由 Drawer 基于真实结果绘制
    - flowchart：技术路线图/求解流程图（非数据图），竞赛论文通常至少需要一张
    - diagram：模型结构/变量关系/指标体系图（非数据图）

    V17 升级：架构阶段必须把每张图规划到「可直接成稿」的完整度——
    图注（caption）、目标章节（section）、内容规格（content_spec）与
    是否论文必需（required），配合 figure_manifest 形成
    「规划 → 生成 → 引用 → 校验」闭环。
    """

    id: str = ""
    figure_type: str = ""
    kind: Literal["data", "flowchart", "diagram"] = "data"
    purpose: str = ""
    data_source: str = ""
    # V17 新增：图注（LaTeX \caption 文本，可直接进论文）
    caption: str = ""
    # V17 新增：目标章节文件（如 5_problem1.tex / 2_analysis.tex）
    section: str = ""
    # V17 新增：内容规格（数据来源列、变量、期望形状/统计量，供 Drawer 精确实现）
    content_spec: str = ""
    # V17 新增：论文必需图；缺失/未引用时校验打回
    required: bool = True


class TablePlan(BaseModel):
    """Architect 声明的预期结果表/附表。"""

    id: str = ""
    title: str = ""
    columns: list[str] = Field(default_factory=list)
    purpose: str = ""
    # V17 新增：目标章节文件、内容规格、是否论文必需
    section: str = ""
    content_spec: str = ""
    required: bool = True


class ArchitectResponse(BaseModel):
    outline: dict[str, str] = Field(default_factory=dict)
    pseudocode: list[str] = Field(default_factory=list)
    # V12 修复：机器可读的结果契约，让 Coder 按契约产出、
    # ResultReviewer 按契约验证，替代通用启发式误杀。
    result_contract: ResultContract = Field(default_factory=ResultContract)
    # V13 新增：实现架构摘要与图表/表格计划。
    # 这些字段用于生成"方案与实现架构说明书"，经人类审核后
    # 打包给编程手（外部 AI）实现。
    algorithms_summary: str = ""
    figures_plan: list[FigurePlan] = Field(default_factory=list)
    tables_plan: list[TablePlan] = Field(default_factory=list)


class DataIntelligenceResponse(BaseModel):
    """数据理解分析师的结构化输出。

    只提炼"解题思路需要知道的信息"，不复述原始数据：
    - 每个文件/表是什么（实体/样本/关系）
    - 行列语义、关键列、分组结构
    - 哪些文件/列与题目相关，哪些可能无关
    - 数据层面的风险（缺失、异构、需要按文件分组等）
    """

    insights: list[str] = Field(default_factory=list)


class CoderResponse(BaseModel):
    code: str = ""
    result_path: str = ""


class DrawerResponse(BaseModel):
    figure_code: str = ""
    figure_paths: list[str] = Field(default_factory=list)
    # V17 新增：本代码块产出的图对应的 figures_plan.id 列表
    # （与 figure_paths 一一对应；缺省时按文件名与 plan_id 匹配降级）
    figure_ids: list[str] = Field(default_factory=list)
    observation: str = ""  # Drawer 对所绘图像的文字观察（视觉洞察回流）
    # 让 Drawer 自评观察的强度，避免硬编码 0.5 导致「散点明显非线性」这类强信号
    # 永远只能进 open_questions 而无法触发 Clarifier 修正。
    observation_verdict: Literal["confirmed", "refuted", "inconclusive"] = "inconclusive"
    observation_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # 图像统计摘要：Drawer 在绘图代码中计算的关键统计量（如散点图 X/Y 范围、
    # 凸性方向、Pearson/Spearman 相关系数），作为 observation 的客观佐证。
    # 让 Reflection 节点能基于客观统计量对视觉观察做二次确认，而非仅依赖
    # Drawer 的主观描述。
    image_stats: str = ""


class ArbiterResponse(BaseModel):
    action: str = "approve"
    rollback_version: str | None = None
    reason: str = ""
    requested_version: str | None = None


class WriterResponse(BaseModel):
    latex_content: str = ""
    # V15：按模板章节输出的分文件内容 {文件名: latex 源码}。
    # 模板模式（paper_template_dir 存在）下优先使用 sections 写各章节文件，
    # main.tex 保留模板格式骨架；sections 为空时回退到旧行为（latex_content 写 main.tex）。
    sections: dict[str, str] = Field(default_factory=dict)


class FinalReviewerResponse(BaseModel):
    """终审 LLM 灵活审查结果（final_reviewer 节点，在确定性检查之后）。"""

    verdict: Literal["pass", "fail"] = "pass"
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    numerical_consistency: str = ""  # 数值一致性结论（与结果文件预览比对）
    summary: str = ""  # 一句话总评


class ReflectionFinding(BaseModel):
    """Reflection 节点提取的单条实证发现。"""

    assumption_tested: str
    evidence: str
    verdict: Literal["confirmed", "refuted", "inconclusive"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_fix: str | None = None


class ReflectionResponse(BaseModel):
    """Reflection 节点的结构化输出。"""

    findings: list[ReflectionFinding] = Field(default_factory=list)
    run_summary: str = ""  # 一句话总结本次执行


class MetaRouterResponse(BaseModel):
    """中枢 LLM（Meta-Router）的结构化输出。

    在 Reflection 发现 refuted 后调用，基于全局失败历史决定下一步走向。
    不写死条件边，让 LLM 统筹判断：回 Mathematician 重新发散、回 Clarifier
    局部修正、回 Architect 调整设计、或接受失败前进到 Writer。
    """

    decision: Literal[
        "rediscover",          # 回 Mathematician 重新发散（换建模范式）
        "refine_assumptions",  # 回 Clarifier 局部修正（同范式内调整假设）
        "adjust_architecture", # 回 Architect 调整（模型设计/伪代码层面）
        "accept_failure",      # 接受失败，前进到 collect_artifacts（Writer 标注待验证）
    ]
    reasoning: str = ""  # 决策理由（便于审计）
    direction_hint: str = ""  # 给下游节点的方向提示（注入 prompt，不强制采纳）
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LoadBearingConstruct(BaseModel):
    """承重分析器 LLM 输出的单条构造语义（验证状态/承重度/缺口由规则层计算）。"""

    construct: str
    construct_type: Literal[
        "metric", "model", "method_library", "parameter",
        "threshold", "abstract_structure", "assumption", "data_item",
    ] = "parameter"
    is_root: bool = False
    physical_anchor: str = ""
    risk_if_wrong: str = ""
    required_experiment: Literal[
        "calibration", "perturbation", "contrast", "cross_check",
        "case_study", "artifact",
    ] = "perturbation"


class LoadBearingConclusion(BaseModel):
    """承重分析器 LLM 输出的单条结论语义（兜底强制规则由规则层补充）。"""

    question_ref: str
    answer_type: Literal["verdict", "numeric", "scheme", "comparison", "ranking"] = "verdict"
    verdict_shape: Literal["all_positive", "all_negative", "mixed", "conditional"] = "mixed"
    construct_refs: list[str] = Field(default_factory=list)
    fallback_required: bool = False
    fallback_spec: str = ""


class LoadBearingAnalysisResponse(BaseModel):
    """承重分析器的结构化输出（语义层；规则层负责验证状态/承重度/缺口/契约）。"""

    constructs: list[LoadBearingConstruct] = Field(default_factory=list)
    conclusions: list[LoadBearingConclusion] = Field(default_factory=list)
    reasoning: str = ""
