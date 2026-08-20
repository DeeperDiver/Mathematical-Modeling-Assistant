"""环境健康检查（V15）：检查运行 Modeling_Assistant 所需的依赖与工具。

用法：`python -m modeling_assistant.cli --doctor`

检查项：
- 必选 Python 包：numpy/pandas/scipy/sklearn/statsmodels/matplotlib/networkx/pulp/
  openpyxl/pdfplumber/langgraph/openai/pydantic/arxiv/pyyaml
- 论文编译：xelatex（或 pdflatex）、typst（可选）
- PDF 视觉检查：pdftoppm / mutool / magick（三选一，可选）
- 非数据图：drawio（可选）
- 配置：LLM API key 是否就绪
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import shutil
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# (import 名, 展示名/发行包名)
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"),
    ("matplotlib", "matplotlib"),
    ("networkx", "networkx"),
    ("pulp", "pulp"),
    ("openpyxl", "openpyxl"),
    ("pdfplumber", "pdfplumber"),
    ("langgraph", "langgraph"),
    ("openai", "openai"),
    ("pydantic", "pydantic"),
    ("arxiv", "arxiv"),
    ("yaml", "PyYAML"),
]

OPTIONAL_PACKAGES: list[tuple[str, str]] = [
    ("plotly", "plotly"),
    ("langchain_chroma", "langchain-chroma"),
]

COMPILER_CANDIDATES = ("xelatex", "pdflatex", "typst")
PDF_RASTERIZERS = ("pdftoppm", "mutool", "magick")


@dataclass(slots=True)
class CheckItem:
    """单条检查结果。"""

    name: str
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class DoctorReport:
    """环境检查汇总报告。"""

    required: list[CheckItem] = field(default_factory=list)
    optional: list[CheckItem] = field(default_factory=list)
    config: list[CheckItem] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """全部必选项通过才算就绪。"""
        return all(item.ok for item in self.required) and all(item.ok for item in self.config)


def _package_version(import_name: str) -> str:
    try:
        return importlib.metadata.version(import_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def check_package(import_name: str, display_name: str) -> CheckItem:
    """检查单个 Python 包是否可导入。"""
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "") or _package_version(import_name)
        return CheckItem(
            name=display_name,
            ok=True,
            detail=version or "已安装",
        )
    except Exception as exc:
        return CheckItem(name=display_name, ok=False, detail=f"导入失败: {exc}")


def check_command(cmd: str) -> CheckItem:
    """检查可执行文件是否在 PATH 中。"""
    path = shutil.which(cmd)
    return CheckItem(name=cmd, ok=path is not None, detail=path or "未找到")


def run_doctor() -> DoctorReport:
    """执行全部检查，返回报告（不抛出异常）。"""
    report = DoctorReport()
    report.required = [check_package(name, disp) for name, disp in REQUIRED_PACKAGES]

    compilers = [check_command(c) for c in COMPILER_CANDIDATES]
    has_compiler = any(c.ok for c in compilers)
    report.required.append(
        CheckItem(
            name="论文编译器",
            ok=has_compiler,
            detail="、".join(c.name for c in compilers if c.ok) or "缺少 xelatex/pdflatex/typst",
        )
    )

    report.optional = [check_package(name, disp) for name, disp in OPTIONAL_PACKAGES]
    report.optional.extend(check_command(cmd) for cmd in PDF_RASTERIZERS)
    report.optional.append(check_command("drawio"))

    # 配置检查：LLM API key（读取 .env / 环境变量，不打印 key 本身）
    try:
        from modeling_assistant.config.settings import load_settings

        settings = load_settings()
        api_key_ok = bool(settings.api_key)
        report.config.append(
            CheckItem(
                name=f"LLM API Key（{settings.api_key_env}）",
                ok=api_key_ok,
                detail="已配置" if api_key_ok else "未配置（LLM 调用将走 fallback 降级）",
            )
        )
        report.config.append(
            CheckItem(
                name="输出目录",
                ok=True,
                detail=str(settings.output_dir),
            )
        )
    except Exception as exc:
        report.config.append(CheckItem(name="配置读取", ok=False, detail=str(exc)))

    return report


def print_report(report: DoctorReport) -> None:
    """以表格形式打印检查报告。"""
    print("\n" + "=" * 64)
    print("  Modeling Assistant 环境检查")
    print("=" * 64)
    print("  [必选项]")
    for item in report.required:
        mark = "[OK]" if item.ok else "[FAIL]"
        print(f"    {mark} {item.name:<20} {item.detail}")
    print("  [可选项]")
    for item in report.optional:
        mark = "[OK]" if item.ok else "[--]"
        print(f"    {mark} {item.name:<20} {item.detail}")
    print("  [配置]")
    for item in report.config:
        mark = "[OK]" if item.ok else "[FAIL]"
        print(f"    {mark} {item.name:<20} {item.detail}")
    print("-" * 64)
    print(f"  结论：{'环境就绪' if report.ready else '必选项缺失，请按提示安装'}")
    print("=" * 64)
