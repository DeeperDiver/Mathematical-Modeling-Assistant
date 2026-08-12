from __future__ import annotations

import ast
import logging
import os
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
    EmpiricalLayer,
    ExemplarContext,
    GraphState,
    StaticLTM,
)

logger = logging.getLogger(__name__)


# 禁止导入的库（未保证安装），与 coder.md/architect.md 约束一致
FORBIDDEN_IMPORTS = {
    "xgboost", "lightgbm", "imblearn", "shap", "lifelines",
    "pymer4", "seaborn", "plotly", "bokeh", "arviz",
    "torch", "tensorflow", "keras", "catboost", "statsmodels",
}
# statsmodels 实际已安装，但常被误用做高级模型；允许使用，从禁止列表移除
FORBIDDEN_IMPORTS.discard("statsmodels")


def precheck_code(code: str) -> str:
    """代码预检：语法检查 + 禁止库扫描。

    在执行代码前做轻量检查，避免明显错误浪费执行时间。
    返回空字符串表示通过，返回非空字符串表示错误信息（模拟 stderr 格式）。
    """
    # 1. 语法检查（ast.parse）
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (
            f"SyntaxError（预检拦截）: {e.msg} (line {e.lineno} col {e.offset})\n"
            f"请检查字符串字面量是否跨行、括号是否匹配、是否有非法字符。\n"
            f"完整 traceback:\n{e}"
        )

    # 2. 扫描 import 语句，发现禁止库
    forbidden_found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in FORBIDDEN_IMPORTS:
                    forbidden_found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module in FORBIDDEN_IMPORTS:
                    forbidden_found.append(node.module)

    if forbidden_found:
        libs = ", ".join(sorted(set(forbidden_found)))
        return (
            f"ModuleNotFoundError（预检拦截）: 以下库未安装: {libs}\n"
            f"允许的库：numpy, pandas, scipy, sklearn, statsmodels, matplotlib, networkx, pulp\n"
            f"替代方案：\n"
            f"- xgboost/lightgbm → sklearn.ensemble.GradientBoostingClassifier/Regressor\n"
            f"- imblearn → sklearn.utils.resample 或 class_weight 参数\n"
            f"- shap → sklearn 内置 feature_importances_ 属性\n"
            f"- seaborn/plotly → matplotlib\n"
            f"请移除禁止库的 import 并改用替代方案后重试。"
        )

    return ""


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

    def render_prompt(self, name: str, state: GraphState, extra: dict[str, Any] | None = None) -> str:
        merged_extra: dict[str, Any] = {
            "llm_model": self.settings.llm_model,
            "output_dir": str(self.settings.output_dir),
        }
        if extra:
            merged_extra.update(extra)
        context = PromptContext(
            static_ltm=state.get("static_ltm", StaticLTM()),
            dynamic_ltm=state.get("dynamic_ltm", DynamicLTM()),
            archive=state.get("ltm_archive", []),
            empirical=state.get("empirical", EmpiricalLayer()),
            control=state.get("control", ControlState()),
            artifacts=state.get("artifacts", ArtifactBundle()),
            exemplars=state.get("exemplars", ExemplarContext()),
            extra=merged_extra,
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
        fallback_parser: callable | None = None,
    ) -> BaseModel:
        """调用 LLM 并解析为 Pydantic 模型，解析失败时自动重试。

        V11.4：新增 fallback_parser 参数，用于在 JSON 解析失败时兜底。
        典型场景：Coder 偶发返回纯 Python 代码块（不带 JSON 包装），
        fallback_parser 从代码块中提取 code 字段构造 CoderResponse。
        fallback_parser 只在最后一次重试失败前调用，避免影响正常重试。
        """
        if system_prompt is None:
            system_prompt = self.render_prompt(prompt_name, state)
        for attempt in range(max_retries + 1):
            try:
                raw = self.invoke(prompt_name, state, system_prompt=system_prompt)
                json_str = _extract_json(raw)
                return response_model.model_validate_json(json_str)
            except Exception as exc:
                # V11.4：最后一次重试失败前，尝试 fallback_parser 兜底
                if fallback_parser is not None and attempt == max_retries:
                    try:
                        fallback_result = fallback_parser(raw)
                        if fallback_result is not None:
                            logger.info(
                                "LLM [%s] JSON 解析失败，fallback 兜底成功",
                                prompt_name,
                            )
                            return fallback_result
                    except Exception as fallback_exc:
                        logger.warning(
                            "LLM [%s] fallback 也失败: %s",
                            prompt_name,
                            fallback_exc,
                        )
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

    def run_code(
        self,
        code: str,
        timeout: int = 120,
        data_paths: list[str] | None = None,
    ) -> tuple[bool, str, str]:
        """执行 Python 代码，返回 (success, stdout, stderr)。

        如果提供了 data_paths，会把第一个路径通过环境变量 MODELING_DATA_PATH
        传入子进程，并把完整列表通过 MODELING_DATA_PATHS（JSON 数组）传入。
        """
        # 预检：语法检查 + 禁止库扫描（不消耗 budget，失败直接要求重写）
        precheck_error = precheck_code(code)
        if precheck_error:
            return False, "", precheck_error

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        env = os.environ.copy()
        # V11.2 修复（Bug 4）：禁用 .pyc 写入，避免 TRAE Sandbox 拦截标准库 __pycache__ 写入
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # 必须传入绝对路径：子进程 cwd 会被切到 output_dir，若传入相对路径，
        # Coder 代码中 os.path.join(OUTPUT_DIR, "results", "output.csv") 会在
        # cwd 下再次嵌套解析，导致结果写到 outputs2/outputs2/results/output.csv
        # 详见 real_test2 测试报告 Bug B
        env["MODELING_OUTPUT_DIR"] = str(self.settings.output_dir.resolve())
        if data_paths:
            # cwd 会被切换到 output_dir，子进程中的相对路径将基于此目录解析，
            # 因此必须把数据路径转为绝对路径，避免 'outputs/test_data.csv'
            # 被解析为 'outputs/outputs/test_data.csv'。
            abs_data_paths = [str(Path(p).resolve()) for p in data_paths]
            env["MODELING_DATA_PATH"] = abs_data_paths[0]
            env["MODELING_DATA_PATHS"] = __import__("json").dumps(abs_data_paths, ensure_ascii=False)

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.settings.output_dir),
                env=env,
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


def _coder_fallback_parser(raw: str):
    """V11.4：Coder 偶发返回纯 Python 代码块（不带 JSON 包装）时的兜底解析。

    场景：LLM 直接返回 ```python\nimport os\n...\n``` 而非 JSON 包装。
    fallback 提取代码块作为 code 字段构造 CoderResponse。

    加锚定：只匹配"整个返回就是纯代码块"的情况，
    避免误匹配 JSON 内部的代码块。
    """
    if not raw:
        return None
    from modeling_assistant.schemas.responses import CoderResponse

    # 加 ^\s* 和 \s*$ 锚定：整个返回必须是纯代码块（前后只允许空白）
    code_match = re.match(
        r"^\s*```(?:python)?\n(.*?)```\s*$",
        raw,
        re.DOTALL,
    )
    if code_match:
        return CoderResponse(
            code=code_match.group(1),
            result_path="results/output.csv",
        )
    return None


def get_default_runtime() -> AgentRuntime:
    """每次调用都构建一个新的 runtime 实例，避免测试间状态污染。"""
    return AgentRuntime.from_settings()
