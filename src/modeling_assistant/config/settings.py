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
    # V13 新增：编程手模式
    # - builtin：主流程内置 Coder 生成代码（默认）
    # - codex：把"方案与实现架构说明书"打包后，调用本机 Codex CLI
    #   让另一个 AI 实例实现代码，主流程继续执行与验证
    coder_external_mode: str = "builtin"
    coder_external_timeout: int = 600  # 外部编程手单次实现超时（秒）
    # V15 新增：方法知识库（method knowledge）
    # 从 references/math_modeling_norms.md 按节点/题型切片注入
    # Mathematician / Realist / Coder / Clarifier / Drawer 的 prompt，
    # 只影响领域判断，不改变图结构。关闭时渲染行为与旧版本完全一致。
    method_knowledge_enabled: bool = True
    # V15 新增：论文 LaTeX 模板目录（当前内置国赛 CUMCM 模板）
    # writer 节点把模板复制到 output_dir/paper/ 并按实际子问题数量调整章节，
    # 论文以「模板格式骨架 + LLM 生成的 sections」方式成稿。
    # 目录不存在或 main.tex 缺失时回退到旧的「LLM 输出完整 main.tex」行为。
    paper_template_dir: Path = Path("templates/cumcm-latex")
    # V15.1 修复：推理模型（如 deepseek-v4-flash）会先消耗大量 reasoning tokens，
    # 若不显式设置 max_tokens，长 prompt 下可能返回空 content。
    # 显式给出输出上限，确保 JSON 结构化输出有足够预算。
    # 实测 clarifier（完整 LTM：假设/符号表/公式/目标/思路）输出可达 8K+ tokens，
    # 8192 会被 finish=length 截断，故默认 32768。
    llm_max_tokens: int = 32768
    # V17 分节点输出预算：coder/writer 需要长输出保留大上限；
    # 小节点压低，避免 reasoner 推理空转与异常发散。
    # 原则：cap 必须高于该节点正常峰值输出（+推理余量），否则会截断导致重试。
    llm_max_tokens_overrides: dict[str, int] = Field(
        default_factory=lambda: {
            "coder": 32768,
            "writer": 32768,
            "clarifier": 24576,
            "architect": 12288,
            "drawer": 12288,
            "analyst": 8192,
            "data_analyst": 8192,
            "mathematician": 8192,
            "realist": 8192,
            "reflection": 8192,
            "final_reviewer": 8192,
            "arbiter": 4096,
            "milestone_reviewer_1": 4096,
            "meta_router": 4096,
            "searcher": 2048,
            "exemplar_type_judge": 2048,
            "load_bearing_analyzer": 8192,
        }
    )

    def max_tokens_for(self, prompt_name: str) -> int:
        """按节点返回输出上限：有覆盖用覆盖，否则用全局默认。"""
        return self.llm_max_tokens_overrides.get(prompt_name, self.llm_max_tokens)

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
        "coder_external_mode": os.getenv("MODELING_ASSISTANT_CODER_EXTERNAL_MODE")
        or file_values.get("MODELING_ASSISTANT_CODER_EXTERNAL_MODE"),
        "coder_external_timeout": os.getenv("MODELING_ASSISTANT_CODER_EXTERNAL_TIMEOUT")
        or file_values.get("MODELING_ASSISTANT_CODER_EXTERNAL_TIMEOUT"),
        "method_knowledge_enabled": os.getenv("MODELING_ASSISTANT_METHOD_KNOWLEDGE_ENABLED")
        or file_values.get("MODELING_ASSISTANT_METHOD_KNOWLEDGE_ENABLED"),
        "paper_template_dir": os.getenv("MODELING_ASSISTANT_PAPER_TEMPLATE_DIR")
        or file_values.get("MODELING_ASSISTANT_PAPER_TEMPLATE_DIR"),
        "llm_max_tokens": os.getenv("MODELING_ASSISTANT_LLM_MAX_TOKENS")
        or file_values.get("MODELING_ASSISTANT_LLM_MAX_TOKENS"),
        "llm_max_tokens_overrides": os.getenv("MODELING_ASSISTANT_LLM_MAX_TOKENS_OVERRIDES")
        or file_values.get("MODELING_ASSISTANT_LLM_MAX_TOKENS_OVERRIDES"),
    }

    values: dict[str, Any] = {key: value for key, value in raw_values.items() if value is not None}
    if "search_enabled" in values:
        values["search_enabled"] = _parse_bool(values["search_enabled"])
    if "method_knowledge_enabled" in values:
        values["method_knowledge_enabled"] = _parse_bool(values["method_knowledge_enabled"])
    for key in (
        "max_debate_rounds",
        "innovation_threshold",
        "feasibility_threshold",
        "exemplar_top_k",
        "plagiarism_ngram",
        "coder_external_timeout",
        "llm_max_tokens",
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
    if "llm_max_tokens_overrides" in values:
        try:
            values["llm_max_tokens_overrides"] = json.loads(
                values["llm_max_tokens_overrides"]
            )
        except (json.JSONDecodeError, TypeError):
            # 配置损坏时回退默认覆盖
            values.pop("llm_max_tokens_overrides", None)

    values.update(overrides)
    return AppSettings(**values)
