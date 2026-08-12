from __future__ import annotations

import json
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
    # ── Exemplar Learning System 配置 ──
    exemplars_dir: Path = Path("exemplars")
    exemplar_min_relevance: float = 0.25  # TF-IDF/标签相关性阈值，低于则不注入
    exemplar_top_k: int = 2  # 每次注入的最多卡片数
    style_injection: dict[str, float] = Field(
        default_factory=lambda: {"structure": 1.0, "chart": 0.8, "writing": 0.5}
    )  # 注入强度分级
    style_dropout_rate: float = 0.3  # writing 卡片的随机丢弃率（防依赖）
    plagiarism_ngram: int = 8  # 查重护栏 n-gram 长度
    plagiarism_threshold: float = 0.15  # 重合率阈值，超过写入完整性警告
    feedback_alpha: float = 0.3  # 反馈回写滑动平均系数

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


def _resolve_env_path(env_file: str | Path) -> Path:
    """Resolve the .env file path robustly.

    If a relative path is given, first look in the current working directory.
    If it does not exist there, fall back to the project root directory
    (two levels above this settings module), so tests and scripts invoked
    from other directories still pick up the project's .env file.
    """
    env_path = Path(env_file)
    if env_path.is_absolute():
        return env_path

    cwd_path = Path.cwd() / env_path
    if cwd_path.exists():
        return cwd_path

    project_root = Path(__file__).resolve().parents[2]
    return project_root / env_path


def load_settings(env_file: str | Path = ".env", **overrides: Any) -> AppSettings:
    env_path = _resolve_env_path(env_file)
    file_values = _read_env_file(env_path)

    # 将 .env 中的 API key 注入当前进程的环境变量，让 AppSettings.api_key 能读取到。
    api_key_env = os.getenv("MODELING_ASSISTANT_API_KEY_ENV") or file_values.get("MODELING_ASSISTANT_API_KEY_ENV", "DEEPSEEK_API_KEY")
    if api_key_env in file_values and not os.getenv(api_key_env):
        os.environ[api_key_env] = file_values[api_key_env]

    raw_values = {
        "llm_model": os.getenv("MODELING_ASSISTANT_LLM_MODEL")
        or file_values.get("MODELING_ASSISTANT_LLM_MODEL")
        or file_values.get("DEEPSEEK_MODEL"),
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
        "exemplars_dir": os.getenv("MODELING_ASSISTANT_EXEMPLARS_DIR")
        or file_values.get("MODELING_ASSISTANT_EXEMPLARS_DIR"),
        "exemplar_min_relevance": os.getenv("MODELING_ASSISTANT_EXEMPLAR_MIN_RELEVANCE")
        or file_values.get("MODELING_ASSISTANT_EXEMPLAR_MIN_RELEVANCE"),
        "exemplar_top_k": os.getenv("MODELING_ASSISTANT_EXEMPLAR_TOP_K")
        or file_values.get("MODELING_ASSISTANT_EXEMPLAR_TOP_K"),
        "style_injection": os.getenv("MODELING_ASSISTANT_STYLE_INJECTION")
        or file_values.get("MODELING_ASSISTANT_STYLE_INJECTION"),
        "style_dropout_rate": os.getenv("MODELING_ASSISTANT_STYLE_DROPOUT_RATE")
        or file_values.get("MODELING_ASSISTANT_STYLE_DROPOUT_RATE"),
        "plagiarism_ngram": os.getenv("MODELING_ASSISTANT_PLAGIARISM_NGRAM")
        or file_values.get("MODELING_ASSISTANT_PLAGIARISM_NGRAM"),
        "plagiarism_threshold": os.getenv("MODELING_ASSISTANT_PLAGIARISM_THRESHOLD")
        or file_values.get("MODELING_ASSISTANT_PLAGIARISM_THRESHOLD"),
        "feedback_alpha": os.getenv("MODELING_ASSISTANT_FEEDBACK_ALPHA")
        or file_values.get("MODELING_ASSISTANT_FEEDBACK_ALPHA"),
    }

    values: dict[str, Any] = {key: value for key, value in raw_values.items() if value is not None}
    if "search_enabled" in values:
        values["search_enabled"] = _parse_bool(values["search_enabled"])
    for key in (
        "max_debate_rounds",
        "innovation_threshold",
        "feasibility_threshold",
        "exemplar_top_k",
        "plagiarism_ngram",
    ):
        if key in values:
            values[key] = int(values[key])
    for key in (
        "innovation_weight",
        "feasibility_weight",
        "exemplar_min_relevance",
        "style_dropout_rate",
        "plagiarism_threshold",
        "feedback_alpha",
    ):
        if key in values:
            values[key] = float(values[key])
    if "style_injection" in values:
        try:
            values["style_injection"] = json.loads(values["style_injection"])
        except json.JSONDecodeError:
            # 配置损坏时回退默认值
            values["style_injection"] = {"structure": 1.0, "chart": 0.8, "writing": 0.5}

    values.update(overrides)
    return AppSettings(**values)
