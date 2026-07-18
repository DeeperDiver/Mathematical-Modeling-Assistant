from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


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


class DataProfile(BaseModel):
    """真实数据附件的机器生成画像。"""

    file_paths: list[str] = Field(default_factory=list)
    total_rows: int = 0
    total_cols: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    sample_head: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


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
    for path in incoming.figure_paths:
        if path not in base.figure_paths:
            base.figure_paths.append(path)
    for path in incoming.result_paths:
        if path not in base.result_paths:
            base.result_paths.append(path)
    if incoming.latex_path:
        base.latex_path = incoming.latex_path
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
    hitl_stage: Literal["architecture", "final", "arbitration", "none"] = "none"
    rollback_to_version: str | None = None
    rollback_source: Literal["architecture_hitl", "final_hitl", "arbitration", "none"] = "none"
    # ── 实证反思与假设修正 ──
    coder_run_count: int = 0  # Coder 累计执行次数，用于生成 run_id
    trigger_clarifier_revision: bool = False  # Reflection 设置此标志触发回 Clarifier
    empirical_revision_count: int = 0  # 假设被推翻后触发 Clarifier 的已用次数
    empirical_revision_budget: int = 2  # 预算（防止无限循环）


class GraphState(TypedDict, total=False):
    static_ltm: Annotated[StaticLTM, overwrite_reducer]
    dynamic_ltm: Annotated[DynamicLTM, overwrite_reducer]
    ltm_archive: Annotated[list[LtmSnapshot], append_reducer]
    empirical: Annotated[EmpiricalLayer, merge_empirical_reducer]
    control: Annotated[ControlState, overwrite_reducer]
    artifacts: Annotated[ArtifactBundle, merge_artifacts_reducer]
    prompt_audit: Annotated[dict[str, str], merge_dict_reducer]
