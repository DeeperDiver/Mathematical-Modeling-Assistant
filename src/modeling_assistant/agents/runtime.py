from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from modeling_assistant.agents.searcher import ArxivSearcher, Searcher, StubSearcher
from modeling_assistant.config.settings import AppSettings, load_settings
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    GraphState,
    StaticLTM,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentRuntime:
    """统一的 LLM / 检索 / 绘图 / 执行能力接入层。

    所有节点通过此 Runtime 调用 LLM，不各自创建客户端。
    """

    settings: AppSettings
    prompts: PromptCatalog
    searcher: Searcher = field(default_factory=StubSearcher)
    client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        api_key = self.settings.api_key
        if api_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.settings.api_base_url + "/v1",
            )
        else:
            logger.warning(
                "未找到 API key (环境变量 %s)。Runtime 将以降级模式运行，"
                "所有 LLM 调用将触发 fallback 逻辑。",
                self.settings.api_key_env,
            )
            self.client = None

        if self.settings.search_enabled:
            try:
                self.searcher = ArxivSearcher()
            except Exception as exc:
                logger.warning(
                    "启用 ArXiv 检索失败，已降级为 StubSearcher: %s",
                    exc,
                )
                self.searcher = StubSearcher()

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> "AgentRuntime":
        resolved_settings = settings or load_settings()
        return cls(settings=resolved_settings, prompts=PromptCatalog())

    # ── prompt 渲染 ──────────────────────────────────────────────

    def render_prompt(self, name: str, state: GraphState) -> str:
        context = PromptContext(
            static_ltm=state.get("static_ltm", StaticLTM()),
            dynamic_ltm=state.get("dynamic_ltm", DynamicLTM()),
            archive=state.get("ltm_archive", []),
            control=state.get("control", ControlState()),
            artifacts=state.get("artifacts", ArtifactBundle()),
            extra={
                "llm_model": self.settings.llm_model,
                "output_dir": str(self.settings.output_dir),
            },
        )
        return self.prompts.render(name, context)

    # ── LLM 调用 ─────────────────────────────────────────────────

    def invoke(self, prompt_name: str, state: GraphState, system_prompt: str | None = None) -> str:
        """调用 LLM，返回原始文本响应。"""
        if self.client is None:
            raise RuntimeError(
                f"LLM client 未初始化。请设置环境变量 {self.settings.api_key_env}。"
            )
        if system_prompt is None:
            system_prompt = self.render_prompt(prompt_name, state)
        logger.info("Invoking LLM [%s] model=%s", prompt_name, self.settings.llm_model)

        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请执行你的任务，严格按要求的格式输出。"},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        logger.info("LLM [%s] 返回 %d 字符", prompt_name, len(content))
        return content

    def invoke_structured(
        self,
        prompt_name: str,
        state: GraphState,
        response_model: type[BaseModel],
        max_retries: int = 2,
        system_prompt: str | None = None,
    ) -> BaseModel:
        """调用 LLM 并解析为 Pydantic 模型，解析失败时自动重试。"""
        if system_prompt is None:
            system_prompt = self.render_prompt(prompt_name, state)
        for attempt in range(max_retries + 1):
            try:
                raw = self.invoke(prompt_name, state, system_prompt=system_prompt)
                json_str = _extract_json(raw)
                return response_model.model_validate_json(json_str)
            except Exception as exc:
                logger.warning(
                    "LLM [%s] 调用/解析失败 (attempt %d/%d): %s",
                    prompt_name,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"LLM [{prompt_name}] 在 {max_retries + 1} 次尝试后仍失败。"
                    ) from exc
        # unreachable
        raise RuntimeError(f"LLM [{prompt_name}] 结构化输出失败。")

    # ── 产物路径 ─────────────────────────────────────────────────

    def ensure_output_dirs(self) -> None:
        """确保 outputs/figures, outputs/results, outputs/paper 目录存在。"""
        for sub in ("figures", "results", "paper"):
            self.settings.output_dir.joinpath(sub).mkdir(parents=True, exist_ok=True)

    def output_path(self, *parts: str) -> str:
        return str(self.settings.output_dir.joinpath(*parts))

    def write_file(self, *parts: str, content: str) -> str:
        """写入产物文件，自动创建父目录，返回绝对路径。"""
        path = self.settings.output_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("写入文件: %s (%d 字符)", path, len(content))
        return str(path)

    def run_code(self, code: str, timeout: int = 120) -> tuple[bool, str, str]:
        """执行 Python 代码，返回 (success, stdout, stderr)。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.settings.output_dir),
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"执行超时（超过 {timeout} 秒）"
        finally:
            Path(script_path).unlink(missing_ok=True)


def _extract_json(text: str) -> str:
    """从 LLM 响应中提取 JSON 字符串。

    容忍以下情况：
    - 整段被 ```json ... ``` 包裹
    - LLM 在 JSON 前后加了解释文字
    - 多个代码块时取第一个
    """
    text = text.strip()
    # 1. 优先匹配 ```json ... ``` 或 ``` ... ``` 代码块
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # 2. 尝试找第一个平衡的 {...} 块
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    # 3. 尝试找第一个平衡的 [...] 块
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(0)
    # 4. 原样返回，让上层抛错
    return text


def get_default_runtime() -> AgentRuntime:
    """每次调用都构建一个新的 runtime 实例，避免测试间状态污染。"""
    return AgentRuntime.from_settings()