from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from modeling_assistant.memory.archive import archive_summary


@dataclass(frozen=True, slots=True)
class PromptContext:
    static_ltm: BaseModel | None = None
    dynamic_ltm: BaseModel | None = None
    archive: list[BaseModel] | None = None
    control: BaseModel | None = None
    artifacts: BaseModel | None = None
    extra: dict[str, Any] | None = None

    def to_template_vars(self) -> dict[str, str]:
        extra = self.extra or {}
        # 从 control 中提取 top_k_plans 单独渲染，方便 Realist 模板引用
        top_k_plans_json = "[]"
        if self.control is not None:
            ctrl_data = self.control.model_dump(mode="json")
            top_k_plans_json = json.dumps(
                ctrl_data.get("top_k_plans", []),
                ensure_ascii=False,
                indent=2,
            )
        # 提取阈值用于 prompt 模板
        feasibility_threshold = "60"
        innovation_threshold = "60"
        branch_from_version = "无"
        rebrainstorm_feedback_json = "[]"
        coder_error_log_json = "[]"
        coder_error_count = "0"
        if self.control is not None:
            ctrl_data = self.control.model_dump()
            feasibility_threshold = str(ctrl_data.get("feasibility_threshold", 60))
            innovation_threshold = str(ctrl_data.get("innovation_threshold", 60))
            branch_from_version = str(ctrl_data.get("branch_from_version") or "无")
            rebrainstorm_feedback_json = json.dumps(
                ctrl_data.get("rebrainstorm_feedback", []),
                ensure_ascii=False,
                indent=2,
            )
            coder_error_log_json = json.dumps(
                ctrl_data.get("coder_error_log", []),
                ensure_ascii=False,
                indent=2,
            )
            coder_error_count = str(ctrl_data.get("coder_error_count", 0))
        # archive 摘要视图（轻量，仅含版本号与变更说明）
        archive_summary_json = json.dumps(
            archive_summary(list(self.archive or [])),
            ensure_ascii=False,
            indent=2,
        )
        return {
            "static_ltm_json": _json_model(self.static_ltm),
            "dynamic_ltm_json": _json_model(self.dynamic_ltm),
            "archive_json": _json_list(self.archive or []),
            "archive_summary_json": archive_summary_json,
            "control_json": _json_model(self.control),
            "artifacts_json": _json_model(self.artifacts),
            "top_k_plans_json": top_k_plans_json,
            "feasibility_threshold": feasibility_threshold,
            "innovation_threshold": innovation_threshold,
            "branch_from_version": branch_from_version,
            "rebrainstorm_feedback_json": rebrainstorm_feedback_json,
            "coder_error_log_json": coder_error_log_json,
            "coder_error_count": coder_error_count,
            **{key: str(value) for key, value in extra.items()},
        }


class PromptCatalog:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        self.prompt_dir = prompt_dir or Path(__file__).parent / "templates"

    def render(self, name: str, context: PromptContext) -> str:
        template_path = self.prompt_dir / f"{name}.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.format(**context.to_template_vars())


def _json_model(model: BaseModel | None) -> str:
    if model is None:
        return "{}"
    return model.model_dump_json(indent=2)


def _json_list(items: list[BaseModel]) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        indent=2,
    )
