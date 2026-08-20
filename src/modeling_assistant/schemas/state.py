from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from modeling_assistant.schemas.craft import CraftGuide

from modeling_assistant.schemas.responses import FigurePlan, ResultContract, TablePlan
from modeling_assistant.recording.process_log import ProcessLogEntry


def overwrite_reducer(_old: Any, new: Any) -> Any:
    """Keep the latest authoritative LTM value."""
    return new


def append_reducer(old: list[Any] | None, new: list[Any] | Any | None) -> list[Any]:
    """Append archive entries while tolerating LangGraph's partial updates."""
    current = list(old or [])
    if new is None:
        return current
    if isinstance(new, list):
        return current + new
    return current + [new]


def merge_dict_reducer(
    old: dict[str, str] | None,
    new: dict[str, str] | None,
) -> dict[str, str]:
    merged = dict(old or {})
    merged.update(new or {})
    return merged


class LiteratureItem(BaseModel):
    title: str
    authors: str = ""
    source: str = ""
    summary: str = ""
    url: str | None = None


class ColumnProfile(BaseModel):
    """单个数据列的画像。"""

    name: str
    dtype: str = ""  # int, float, category, datetime, text 等
    missing_rate: float = 0.0
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    unique_count: int | None = None
    sample_values: list[Any] = Field(default_factory=list)
    # V11 修复：机器生成的字符串列解析建议
    # 对于 dtype=text 的列，根据样例值自动推断解析代码（如 '16W' → str.replace('W','').astype(float)）
    # 该字段由 data_profile_node 自动填充，不可被 LLM 改写
    parse_hint: str = ""


class FileSummary(BaseModel):
    """单个数据文件（或单 sheet）的画像，保留文件边界。"""

    path: str = ""
    rows: int = 0
    cols: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class DataProfile(BaseModel):
    """真实数据附件的机器生成画像。"""

    file_paths: list[str] = Field(default_factory=list)
    total_rows: int = 0
    total_cols: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    sample_head: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    # V12 修复：按文件（保持边界）的独立画像。
    # 多附件/异构表格不再被合并成一张大表后让 LLM 猜语义，
    # prompt 只注入 file_summaries 的紧凑摘要，原始数据不进 prompt。
    file_summaries: list[FileSummary] = Field(default_factory=list)


class ProblemFact(BaseModel):
    """从题目原文机器提取的数值常量。

    V11 三层防线第一层：纯机器提取，不经过 LLM 改写。
    作为后续 LLM 解释的"真理基准"，用于：
    - 第二层：Clarifier 写入 dynamic_ltm 时校验是否引用了这些常量
    - 第三层：Coder 生成代码后扫描代码常量是否与这些值匹配
    """

    value: float
    unit: str = ""  # 如 "m/s"、"m"、"s"、"m/s²"
    context: str = ""  # 提取该数值的原文片段，便于 LLM 消歧
    # LLM 标注的角色（如 "烟幕下沉速度"、"导弹速度"、"有效遮蔽半径"）
    # 由 Analyst 填充，机器不强制；但留空时下游会发警告
    role_hint: str = ""
    # V11.4：fact 语义类型，由 fact_extractor 用双重判据自动填充
    # - physical_param：物理量参数（如 "速度 3 m/s"），代码必须以字面量形式出现
    # - data_range：数据列范围描述（如 "GC 含量正常范围 40%-60%"），
    #   是数据筛选阈值而非建模参数，代码可不写字面量
    # - count：纯计数单位（如 "3 枚"、"5 次"），不参与代码常量校验
    category: Literal["physical_param", "data_range", "count"] = "physical_param"


class StaticLTM(BaseModel):
    raw_problem: str = ""
    data_attachments: list[str] = Field(default_factory=list)
    data_schema: dict[str, str] = Field(default_factory=dict)
    data_profile: DataProfile | None = None
    problem_understanding: str = ""
    literature: list[LiteratureItem] = Field(default_factory=list)
    # 数据认知更新：执行阶段发现的、对原始数据 schema 的补充认知
    # （如「列 X 实际有强时序性」「列 Y 分布显著非正态」）。
    # 原始字段（raw_problem/data_schema/data_profile）保持不可变语义，
    # data_findings 是追加字段，允许 data_profile_node 等节点追加发现。
    data_findings: list[str] = Field(default_factory=list)
    # V11 三层防线第一层：机器提取的题目数值常量
    # 由 fact_extractor_node（纯代码，无 LLM）从 raw_problem 用正则提取，
    # 不可被 LLM 改写。作为整个流程的"真理基准"。
    problem_facts: list[ProblemFact] = Field(default_factory=list)
    # V11 三层防线第一层：LLM 标注后的常量角色映射
    # 由 Analyst 节点填充（基于 problem_facts 的 context 推断每个数值的语义角色）
    # 例如：{"3.0": "烟幕下沉速度 m/s", "300.0": "导弹速度 m/s"}
    fact_role_mapping: dict[str, str] = Field(default_factory=dict)
    # V12 新增：LLM 数据理解分析师提炼的"解题所需信息"。
    # 例如："附件1是反射率光谱：波数0~4000 cm-1，反射率0~100%，4个文件分别对应
    # 10°/15°入射角下的SiC/Si样品，建模需按文件分组拟合薄膜厚度。"
    # 只存语义结论，不存原始数据。
    data_intelligence: list[str] = Field(default_factory=list)


class DynamicLTM(BaseModel):
    assumptions: list[str] = Field(default_factory=list)
    nomenclature: dict[str, str] = Field(default_factory=dict)
    equations: list[str] = Field(default_factory=list)
    objective: str = ""
    solution_outline: str = ""


class LtmSnapshot(BaseModel):
    version: str
    dynamic_ltm: DynamicLTM
    commit_summary: str = ""
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint_id: str | None = None


class PlanCandidate(BaseModel):
    id: str
    title: str
    description: str
    innovation_score: int = Field(default=0, ge=0, le=100)
    feasibility_score: int = Field(default=0, ge=0, le=100)
    source_snapshot_version: str | None = None
    verdict: Literal["keep", "kill", "reject"] = "keep"

    def total_score(self, w_inn: float = 0.5, w_fea: float = 0.5) -> float:
        """综合评分：Score_total = w1 * S_inn + w2 * S_fea。"""
        return w_inn * self.innovation_score + w_fea * self.feasibility_score


class ArtifactBundle(BaseModel):
    outline: dict[str, str] = Field(default_factory=dict)
    pseudocode: list[str] = Field(default_factory=list)
    figure_paths: list[str] = Field(default_factory=list)
    result_paths: list[str] = Field(default_factory=list)
    latex_path: str | None = None
    pdf_path: str | None = None
    # V9 修复：sentinel 标志，节点失败时设为 True，merge_artifacts_reducer 看到后
    # 清空 base.result_paths。这解决了 merge_artifacts_reducer 的"只追加不清空"问题：
    # coder/result_reviewer 失败时返回 result_paths=[]，但 reducer 不会清空旧值，
    # 导致路由错乱（route_after_coder 看到非空 result_paths 误判为成功）。
    clear_result_paths: bool = False
    # V12 修复：Architect 声明的结果契约，ResultReviewer 按契约验证
    result_contract: ResultContract | None = None
    # V13 新增：实现架构（算法摘要 + 图表/表格计划）与编程手任务包
    algorithms_summary: str = ""
    figures_plan: list[FigurePlan] = Field(default_factory=list)
    tables_plan: list[TablePlan] = Field(default_factory=list)
    architecture_spec_md: str = ""
    coder_task_dir: str = ""
    # V13：人工编程手交付的绘图代码路径（可选，不存在则为空）
    human_figure_code_path: str = ""
    # V17 图表注册表：plan_id -> {"path": 实际文件, "run_id": 来源, "status": "generated"}
    # 由 drawer_node 按 figures_plan 落盘登记；Writer 只允许引用这里的图。
    figure_manifest: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # V17：drawer 失败时置 True，清空 base.figure_manifest（与 clear_result_paths 对称）
    clear_figure_manifest: bool = False


def merge_artifacts_reducer(
    old: ArtifactBundle | dict[str, Any] | None,
    new: ArtifactBundle | dict[str, Any] | None,
) -> ArtifactBundle:
    base = ArtifactBundle.model_validate(old or {})
    incoming = ArtifactBundle.model_validate(new or {})

    if incoming.outline:
        base.outline = incoming.outline
    if incoming.pseudocode:
        base.pseudocode = incoming.pseudocode
    # 合并 figure_paths：如果 incoming 含真实图片（非 placeholder），
    # 应移除 base 中的 placeholder，避免历史失败残留污染真实图片列表。
    # 场景：第一次 drawer 失败返回 [placeholder]，第二次 drawer 成功返回 [figure1.png]，
    # 合并后应为 [figure1.png] 而非 [placeholder, figure1.png]。
    incoming_has_real_figure = any(
        "placeholder" not in Path(p).name.lower() for p in incoming.figure_paths
    )
    if incoming_has_real_figure:
        base.figure_paths = [
            p for p in base.figure_paths if "placeholder" not in Path(p).name.lower()
        ]
    for path in incoming.figure_paths:
        if path not in base.figure_paths:
            base.figure_paths.append(path)
    # V9 修复：incoming 显式请求清空 result_paths（coder/result_reviewer 失败时）
    # 时，先清空 base.result_paths，再追加 incoming.result_paths（通常为空）。
    # 这避免了"只追加不清空"导致的路由错乱：
    # coder 失败返回 [] 但 base 保留旧值 → route_after_coder 误判为成功 →
    # result_reviewer 检查旧文件 → 失败回退不消费 budget → 死循环。
    if incoming.clear_result_paths:
        base.result_paths = []
    for path in incoming.result_paths:
        if path not in base.result_paths:
            base.result_paths.append(path)
    if incoming.latex_path:
        base.latex_path = incoming.latex_path
    # V12 修复：契约随 Architect 产物更新；None 表示本轮未声明（保留旧值），
    # 空契约 ResultContract() 表示"本轮明确无契约"（清掉旧契约）。
    if incoming.result_contract is not None:
        base.result_contract = incoming.result_contract
    # V13 新增：架构计划随 Architect 产物覆盖更新
    if incoming.algorithms_summary:
        base.algorithms_summary = incoming.algorithms_summary
    # V17 修复：图表/表格计划按 plan_id 合并（多轮 Architect 各自补充，
    # 不再整表覆盖——否则多小题模式下 Writer 只能看到最后一题的图计划）
    if incoming.figures_plan:
        plan_by_id = {f.id: f for f in base.figures_plan}
        for fig in incoming.figures_plan:
            plan_by_id[fig.id] = fig
        base.figures_plan = list(plan_by_id.values())
    if incoming.tables_plan:
        table_by_id = {t.id: t for t in base.tables_plan}
        for t in incoming.tables_plan:
            table_by_id[t.id] = t
        base.tables_plan = list(table_by_id.values())
    # V17 图表注册表：按 plan_id 覆盖；drawer 失败时显式清空
    if incoming.clear_figure_manifest:
        base.figure_manifest = {}
    for plan_id, entry in (incoming.figure_manifest or {}).items():
        base.figure_manifest[plan_id] = entry
    if incoming.architecture_spec_md:
        base.architecture_spec_md = incoming.architecture_spec_md
    if incoming.coder_task_dir:
        base.coder_task_dir = incoming.coder_task_dir
    if incoming.human_figure_code_path:
        base.human_figure_code_path = incoming.human_figure_code_path
    return base


# ── 实证发现层 ──────────────────────────────────────────────────────
# 独立于 LTM 的执行证据层，用于打破「定稿 = 真相」假设：
# Coder / Drawer / ResultReviewer 产出的实证发现写入此层，
# 下游节点（Mathematician / Clarifier）按需读取，
# 由 Clarifier 决定是否吸收进 dynamic_ltm，而非自动污染定稿。

class EmpiricalFinding(BaseModel):
    """单条实证发现。"""

    id: str
    run_id: str  # 对应 outputs/logs/run_{n}.log；data_profile 阶段用 "data_profile"
    source_node: Literal["coder", "drawer", "result_reviewer", "reflection", "data_profile"]
    assumption_tested: str  # 例如「残差正态性」
    evidence: str  # 例如「Shapiro-Wilk p=0.001」
    verdict: Literal["confirmed", "refuted", "inconclusive"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_fix: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    consumed_by: list[str] = Field(default_factory=list)


class EmpiricalLayer(BaseModel):
    """实证发现层 —— 独立于 LTM，不污染定稿语义。

    - findings：完整发现列表（带去重合并）
    - refuted_assumptions：自动派生，高置信度 refuted 才进入（confidence>=0.7）
    - open_questions：自动派生，低置信度或 inconclusive 的观察
    - run_index：执行日志索引，仅含 run_id/summary/log_path，不含原始日志
    """

    findings: list[EmpiricalFinding] = Field(default_factory=list)
    refuted_assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    run_index: list[dict[str, Any]] = Field(default_factory=list)


# 高置信度 refuted 的阈值
REFUTED_CONFIDENCE_THRESHOLD = 0.7
# 防御性 TTL：findings 与 run_index 的最大保留条数
_EMPIRICAL_FINDINGS_TTL = 20
_EMPIRICAL_RUN_INDEX_TTL = 10


def _rebuild_empirical_derived_fields(layer: EmpiricalLayer) -> None:
    """根据 findings 重新派生 refuted_assumptions 和 open_questions。"""
    layer.refuted_assumptions = [
        f.assumption_tested
        for f in layer.findings
        if f.verdict == "refuted" and f.confidence >= REFUTED_CONFIDENCE_THRESHOLD
    ]
    layer.open_questions = [
        f"{f.assumption_tested}（{f.evidence}）"
        for f in layer.findings
        if f.verdict == "inconclusive" or f.confidence < REFUTED_CONFIDENCE_THRESHOLD
    ]


def merge_empirical_reducer(
    old: EmpiricalLayer | dict[str, Any] | None,
    new: EmpiricalLayer | dict[str, Any] | None,
) -> EmpiricalLayer:
    """合并 empirical 层。

    - 同 assumption_tested 的新发现覆盖旧的（取置信度更高者）
    - run_index 追加不去重（每次执行都是新 run）
    - TTL 限制：findings 保留最近 20 条，run_index 保留最近 10 条
    """
    base = EmpiricalLayer.model_validate(old or {})
    incoming = EmpiricalLayer.model_validate(new or {})

    for finding in incoming.findings:
        existing = next(
            (f for f in base.findings if f.assumption_tested == finding.assumption_tested),
            None,
        )
        if existing:
            # 同主题发现取置信度更高者
            if finding.confidence > existing.confidence:
                base.findings.remove(existing)
                base.findings.append(finding)
        else:
            base.findings.append(finding)

    # TTL：只保留最近 N 条
    if len(base.findings) > _EMPIRICAL_FINDINGS_TTL:
        base.findings = base.findings[-_EMPIRICAL_FINDINGS_TTL:]

    # run_index 追加
    base.run_index.extend(incoming.run_index)
    if len(base.run_index) > _EMPIRICAL_RUN_INDEX_TTL:
        base.run_index = base.run_index[-_EMPIRICAL_RUN_INDEX_TTL:]

    _rebuild_empirical_derived_fields(base)
    return base


class SubQuestionResult(BaseModel):
    """单个小题的验收结果与产物记录。"""

    index: int = 0
    title: str = ""
    ltm_version: str = ""
    result_paths: list[str] = Field(default_factory=list)
    figure_paths: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "passed", "failed"] = "pending"
    feedback: list[str] = Field(default_factory=list)


class AuthoritativeResult(BaseModel):
    """V17 结果注册表条目：小题验收通过时锁定的「唯一权威结果」。

    与 sub_results 的区别：sub_results 只是历史记录；results_manifest 是
    Writer 成稿与 paper_check 数字校验唯一允许引用的数据源，包含指标快照、
    来源 run_id 与验收契约，防止多轮执行产生多组「最优结果」后
    Writer 引用错误文件（如把 q2 的参数写进问题 1 章节）。
    """

    index: int = 0
    title: str = ""
    result_paths: list[str] = Field(default_factory=list)
    figure_paths: list[str] = Field(default_factory=list)
    contract: ResultContract | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""
    status: Literal["passed", "degraded"] = "passed"
    feedback: list[str] = Field(default_factory=list)
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ControlState(BaseModel):
    phase: str = "init"
    debate_round: int = 0
    max_debate_rounds: int = 3
    innovation_threshold: int = 60
    feasibility_threshold: int = 60
    innovation_weight: float = 0.5
    feasibility_weight: float = 0.5
    top_k_plans: list[PlanCandidate] = Field(default_factory=list)
    selected_plan_id: str | None = None
    innovation_score: int = Field(default=0, ge=0, le=100)
    feasibility_score: int = Field(default=0, ge=0, le=100)
    need_rebrainstorm: bool = False
    coder_error_count: int = 0
    coder_error_log: list[str] = Field(default_factory=list)
    coder_rollback_target: Literal["architect", "clarifier"] = "architect"
    rebrainstorm_feedback: list[str] = Field(default_factory=list)
    branch_from_version: str | None = None
    hitl_required: bool = False
    hitl_stage: Literal[
        "architecture",
        "implementation_architecture",
        "implementation_human",
        "sub_question_split",
        "sub_question_acceptance",
        "cross_sub_question",
        "final",
        "arbitration",
        "modeling",
        "none",
    ] = "none"
    rollback_to_version: str | None = None
    rollback_source: Literal["architecture_hitl", "final_hitl", "arbitration", "none"] = "none"
    # ── 实证反思与假设修正 ──
    coder_run_count: int = 0  # Coder 累计执行次数，用于生成 run_id
    trigger_clarifier_revision: bool = False  # Reflection 设置此标志触发回 Clarifier
    empirical_revision_count: int = 0  # 假设被推翻后触发 Clarifier 的已用次数
    empirical_revision_budget: int = 2  # 预算（防止无限循环）
    # ── 建模阶段统一预算 ──
    # 覆盖所有回到 mathematician/clarifier 的路径，单一计数避免叠加浪费：
    #   - need_rebrainstorm（milestone 拒绝 / realist 全剪枝）
    #   - trigger_clarifier_revision（empirical 触发）
    #   - coder_rollback_target=clarifier（求解失败）
    # 预算耗尽后：所有回退路径强制放行到 HITL，由人类决断
    modeling_revision_count: int = 0
    modeling_revision_budget: int = 4  # 统一预算（原 empirical=2 + milestone=2 + realist 兜底）
    # V10 修复：记录 ResultReviewer 最近一次拒绝的具体原因，供 Architect 针对性调整模型设计。
    # 与 coder_error_log 不同：coder_error_log 混合了 Coder 执行失败和 ResultReviewer 拒绝两类信息，
    # 而 last_result_review_issues 专门记录"Coder 成功执行但结果质量不通过"的拒绝原因（如常量列、
    # 边界值、空文件），让 Architect 能区分两类失败并针对性调整：
    # - Coder 执行失败 → 调整伪代码复杂度/依赖
    # - ResultReviewer 拒绝 → 调整模型约束（如避免常量列、避免边界值）
    last_result_review_issues: list[str] = Field(default_factory=list)
    # Meta-Router（中枢 LLM）决策：Reflection 发现 refuted 后，由中枢 LLM 判断
    # 下一步走向（rediscover/refine_assumptions/adjust_architecture/accept_failure）。
    # 空字符串表示未调用 Meta-Router，route_after_reflection 回退到原硬编码逻辑。
    meta_decision: str = ""  # MetaRouterResponse.decision 的值
    meta_direction_hint: str = ""  # 中枢 LLM 给下游节点的方向提示
    meta_reasoning: str = ""  # 中枢 LLM 的决策理由（审计用）
    # ── V13 编程手模式（人工 / Codex / 内置）──
    # 架构说明书已经过人类审核，后续 coder 失败回退时不再重复打断审核
    implementation_architecture_reviewed: bool = False
    # 人类在"等待人工编程手交付"节点选择 auto → 回退到内置 Coder
    implementation_auto: bool = False
    # ── V14 小题循环（Sub-Question Loop）──
    # 自动拆分并确认后的小题清单；每题独立建模、实现与验收
    sub_questions: list[str] = Field(default_factory=list)
    sub_questions_confirmed: bool = False
    current_sub_question_index: int = 0
    # 已完成小题的 LTM（与 LTM Archive 版本一一对应）
    sub_ltms: list[DynamicLTM] = Field(default_factory=list)
    sub_results: list[SubQuestionResult] = Field(default_factory=list)
    # V17 结果注册表：每题验收通过的权威结果（Writer / paper_check 唯一数据源）
    results_manifest: list[AuthoritativeResult] = Field(default_factory=list)
    # 当前小题的失败/重试次数与预算；通过后重置
    sub_question_attempts: int = 0
    sub_question_budget: int = 4
    sub_question_feedback: list[str] = Field(default_factory=list)
    # cross <i> <反馈> 的目标小题编号（0-based），触发跨小题 HITL
    cross_sub_question_target: int = -1
    # ── V15 论文修订与验收（Paper Revision & Review）──
    # HITL 终审支持 rewrite <反馈> 回到 Writer 重写论文（有预算防死循环）；
    # final_reviewer 的验收报告（确定性检查 + LLM 灵活审查）供终审展示。
    paper_revision_count: int = 0
    paper_revision_budget: int = 2
    paper_revision_feedback: list[str] = Field(default_factory=list)
    paper_review_report: dict[str, Any] = Field(default_factory=dict)


# ──────────────── 优秀论文表达学习层（Exemplar Learning System） ────────────────
# 离线提炼的 L1 单篇卡片 / L2 题型指南 / L3 全局偏好，运行时检索后注入
# Architect / Drawer / Writer / Reviewer 的 prompt。只影响「怎么说」，
# 不影响「算什么」：公式、数值与方法仍只走 LTM + Coder 验证链。


class ExemplarFigure(BaseModel):
    """单张示例图表的风格描述。"""

    figure_type: str  # boxplot / scatter / heatmap / pareto / convergence / gantt ...
    purpose: str = ""  # 这张图回答什么问题
    style_notes: str = ""  # 配色、标注、字号、坐标轴习惯
    example_path: str = ""  # 示例图文件路径（可选）


class ExemplarPaper(BaseModel):
    """L1 单篇论文卡片：由摄取器从（题目, 优秀论文）对提炼。"""

    id: str
    title: str = ""
    source_path: str = ""  # 原文路径
    problem_type: str = ""  # 题型标签：optimization/physics/forecasting/evaluation/data_mining
    contest: str = ""  # 赛事语境：国赛/美赛/华中杯/...
    year: int | None = None
    structure: dict[str, str] = Field(default_factory=dict)  # {章节名: 目的/写法}
    section_notes: list[str] = Field(default_factory=list)  # 每节写法要点
    figures: list[ExemplarFigure] = Field(default_factory=list)
    writing_style: dict[str, str] = Field(default_factory=dict)  # 文风特征
    summary_style: str = ""  # 摘要写法套路
    highlights: list[str] = Field(default_factory=list)  # 个性亮点（只进 L1，不进 L2）
    pitfalls: list[str] = Field(default_factory=list)  # 雷区
    quotes: list[str] = Field(default_factory=list)  # 短摘录，单条 ≤ 80 字（受查重护栏约束）
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)  # 反馈回写权重
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TypeStyleGuide(BaseModel):
    """L2 题型风格指南：由同组多篇卡片聚合，多篇共有才进共性字段。"""

    problem_type: str
    contest: str = ""
    common_structure: list[str] = Field(default_factory=list)  # 共性骨架
    structure_variants: list[str] = Field(default_factory=list)  # 可选变体
    recommended_figures: list[str] = Field(default_factory=list)
    writing_baseline: dict[str, str] = Field(default_factory=dict)
    common_pitfalls: list[str] = Field(default_factory=list)
    exemplar_ids: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)  # 反馈回写权重
    version: str = "1.0"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GlobalStyleProfile(BaseModel):
    """L3 全局风格偏好：用户个人审美，独立于优秀论文。"""

    color_palette: list[str] = Field(default_factory=list)
    figure_preferences: list[str] = Field(default_factory=list)
    writing_preferences: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class ExemplarContext(BaseModel):
    """运行时注入包：检索结果 + 注入开关。"""

    active: bool = False
    guide: TypeStyleGuide | None = None
    cards: list[ExemplarPaper] = Field(default_factory=list)
    profile: GlobalStyleProfile | None = None  # L3 全局偏好
    craft: CraftGuide | None = None  # 行文技艺指南（题型级，深加工层）
    injection: dict[str, bool] = Field(
        default_factory=lambda: {"structure": True, "chart": True, "writing": True}
    )


class GraphState(TypedDict, total=False):
    static_ltm: Annotated[StaticLTM, overwrite_reducer]
    dynamic_ltm: Annotated[DynamicLTM, overwrite_reducer]
    ltm_archive: Annotated[list[LtmSnapshot], append_reducer]
    empirical: Annotated[EmpiricalLayer, merge_empirical_reducer]
    control: Annotated[ControlState, overwrite_reducer]
    artifacts: Annotated[ArtifactBundle, merge_artifacts_reducer]
    prompt_audit: Annotated[dict[str, str], merge_dict_reducer]
    exemplars: Annotated[ExemplarContext, overwrite_reducer]
    # V17 运行过程记录：逐节点留痕（先落盘 JSONL，state 内保留完整列表）
    process_log: Annotated[list[ProcessLogEntry], append_reducer]
