"""Codex CLI 编程手适配器。

把"方案与实现架构说明书"交给本机另一个 Codex 实例实现：
它在任务目录内阅读 coder_task.md，编写 solution.py，
主流程负责后续的代码执行与 ResultReviewer 验证。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from modeling_assistant.config.settings import AppSettings

logger = logging.getLogger(__name__)


class CodexCoderAdapter:
    """调用本机 Codex CLI 作为外部编程手。"""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def generate(
        self,
        task_dir: Path,
        timeout: int | None = None,
    ) -> tuple[bool, str, str]:
        """运行 Codex 实现 solution.py，返回 (是否成功, 代码, 运行日志)。"""
        exe = shutil.which("codex") or shutil.which("codex.cmd")
        if not exe:
            return False, "", "未找到 Codex CLI（codex），请安装或改用 builtin 模式。"

        task_dir = task_dir.resolve()
        prompt = (
            "你是编程手（Coder）。请完整阅读本目录下的 coder_task.md 与 coder_task.json，"
            "严格按其中的建模设定、算法、结果契约和约束，在本目录编写 solution.py。"
            "要求：不改动建模设定；不生成其他文件；代码必须是完整可执行的 Python；"
            "如需读取真实数据，使用任务包中给出的数据文件路径（MODELING_DATA_PATHS / MODELING_DATA_PATH）。"
            "如果 coder_task.md 第 5 节声明了预期图表，另写 figures.py 并把图片保存到 figures/ 子目录。"
            "完成后确认 solution.py 已存在（figures.py 可选）。"
        )
        cmd = [
            exe,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "-C",
            str(task_dir),
            prompt,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.settings.coder_external_timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, "", f"外部编程手超时（>{timeout or self.settings.coder_external_timeout}s）"
        except Exception as exc:
            return False, "", f"调用外部编程手失败: {exc}"

        solution = task_dir / "solution.py"
        if not solution.exists():
            return (
                False,
                "",
                f"外部编程手未产出 solution.py。\nstdout:\n{result.stdout[:2000]}\nstderr:\n{result.stderr[:2000]}",
            )
        code = solution.read_text(encoding="utf-8")
        return True, code, result.stdout[:3000]
