from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class AppSettings(BaseModel):
    llm_model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_base_url: str = "https://api.deepseek.com"
    search_enabled: bool = True
    workspace_dir: Path = Field(default_factory=lambda: Path.cwd())
    output_dir: Path = Path("outputs")
    max_debate_rounds: int = 3
    innovation_threshold: int = 60
    feasibility_threshold: int = 60
    innovation_weight: float = 0.5
    feasibility_weight: float = 0.5

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


def load_settings(env_file: str | Path = ".env", **overrides: Any) -> AppSettings:
    env_path = Path(env_file)
    file_values = _read_env_file(env_path)
    raw_values = {
        "llm_model": os.getenv("MODELING_ASSISTANT_LLM_MODEL")
        or file_values.get("MODELING_ASSISTANT_LLM_MODEL"),
        "api_key_env": os.getenv("MODELING_ASSISTANT_API_KEY_ENV")
        or file_values.get("MODELING_ASSISTANT_API_KEY_ENV"),
        "api_base_url": os.getenv("MODELING_ASSISTANT_API_BASE_URL")
        or file_values.get("MODELING_ASSISTANT_API_BASE_URL"),
        "search_enabled": os.getenv("MODELING_ASSISTANT_SEARCH_ENABLED")
        or file_values.get("MODELING_ASSISTANT_SEARCH_ENABLED"),
        "workspace_dir": os.getenv("MODELING_ASSISTANT_WORKSPACE_DIR")
        or file_values.get("MODELING_ASSISTANT_WORKSPACE_DIR"),
        "output_dir": os.getenv("MODELING_ASSISTANT_OUTPUT_DIR")
        or file_values.get("MODELING_ASSISTANT_OUTPUT_DIR"),
        "max_debate_rounds": os.getenv("MODELING_ASSISTANT_MAX_DEBATE_ROUNDS")
        or file_values.get("MODELING_ASSISTANT_MAX_DEBATE_ROUNDS"),
        "innovation_threshold": os.getenv("MODELING_ASSISTANT_INNOVATION_THRESHOLD")
        or file_values.get("MODELING_ASSISTANT_INNOVATION_THRESHOLD"),
        "feasibility_threshold": os.getenv("MODELING_ASSISTANT_FEASIBILITY_THRESHOLD")
        or file_values.get("MODELING_ASSISTANT_FEASIBILITY_THRESHOLD"),
        "innovation_weight": os.getenv("MODELING_ASSISTANT_INNOVATION_WEIGHT")
        or file_values.get("MODELING_ASSISTANT_INNOVATION_WEIGHT"),
        "feasibility_weight": os.getenv("MODELING_ASSISTANT_FEASIBILITY_WEIGHT")
        or file_values.get("MODELING_ASSISTANT_FEASIBILITY_WEIGHT"),
    }

    values: dict[str, Any] = {key: value for key, value in raw_values.items() if value is not None}
    if "search_enabled" in values:
        values["search_enabled"] = _parse_bool(values["search_enabled"])
    for key in ("max_debate_rounds", "innovation_threshold", "feasibility_threshold"):
        if key in values:
            values[key] = int(values[key])
    for key in ("innovation_weight", "feasibility_weight"):
        if key in values:
            values[key] = float(values[key])

    values.update(overrides)
    return AppSettings(**values)
