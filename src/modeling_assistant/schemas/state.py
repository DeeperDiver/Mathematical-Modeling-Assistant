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


class StaticLTM(BaseModel):
    raw_problem: str = ""
    data_attachments: list[str] = Field(default_factory=list)
    data_schema: dict[str, str] = Field(default_factory=dict)
    problem_understanding: str = ""
    literature: list[LiteratureItem] = Field(default_factory=list)


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


class GraphState(TypedDict, total=False):
    static_ltm: Annotated[StaticLTM, overwrite_reducer]
    dynamic_ltm: Annotated[DynamicLTM, overwrite_reducer]
    ltm_archive: Annotated[list[LtmSnapshot], append_reducer]
    control: Annotated[ControlState, overwrite_reducer]
    artifacts: Annotated[ArtifactBundle, merge_artifacts_reducer]
    prompt_audit: Annotated[dict[str, str], merge_dict_reducer]
