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


class ArchitectResponse(BaseModel):
    outline: dict[str, str] = Field(default_factory=dict)
    pseudocode: list[str] = Field(default_factory=list)


class CoderResponse(BaseModel):
    code: str = ""
    result_path: str = ""


class DrawerResponse(BaseModel):
    figure_code: str = ""
    figure_paths: list[str] = Field(default_factory=list)
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