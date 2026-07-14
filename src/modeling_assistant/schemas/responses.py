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


class ArchitectResponse(BaseModel):
    outline: dict[str, str] = Field(default_factory=dict)
    pseudocode: list[str] = Field(default_factory=list)


class CoderResponse(BaseModel):
    code: str = ""
    result_path: str = ""


class DrawerResponse(BaseModel):
    figure_code: str = ""
    figure_paths: list[str] = Field(default_factory=list)


class ArbiterResponse(BaseModel):
    action: str = "approve"
    rollback_version: str | None = None
    reason: str = ""
    requested_version: str | None = None


class WriterResponse(BaseModel):
    latex_content: str = ""