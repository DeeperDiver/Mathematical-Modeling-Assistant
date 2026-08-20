"""V15 环境健康检查测试：依赖检查、命令检查、报告汇总。"""

from __future__ import annotations

from modeling_assistant.doctor import (
    CheckItem,
    DoctorReport,
    check_command,
    check_package,
    run_doctor,
)


def test_check_package_detects_installed():
    """已安装的包应标记为 ok 且带版本信息。"""
    item = check_package("numpy", "numpy")
    assert item.ok is True
    assert item.detail


def test_check_package_detects_missing(monkeypatch):
    """不可导入的包应标记为失败且不崩溃。"""

    def fake_import(name):
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", fake_import)
    item = check_package("definitely_not_installed_pkg", "假包")
    assert item.ok is False
    assert "导入失败" in item.detail


def test_check_command_found_and_missing(monkeypatch):
    """命令存在/缺失应正确判定。"""
    monkeypatch.setattr("modeling_assistant.doctor.shutil.which", lambda c: "/usr/bin/" + c)
    assert check_command("xelatex").ok is True

    monkeypatch.setattr("modeling_assistant.doctor.shutil.which", lambda c: None)
    item = check_command("drawio")
    assert item.ok is False
    assert item.detail == "未找到"


def test_doctor_report_ready_requires_all_required_and_config():
    """ready 要求全部必选项与配置检查通过。"""
    report = DoctorReport()
    report.required = [CheckItem(name="numpy", ok=True), CheckItem(name="编译器", ok=True)]
    report.config = [CheckItem(name="API Key", ok=True)]
    assert report.ready is True

    report.config = [CheckItem(name="API Key", ok=False)]
    assert report.ready is False


def test_run_doctor_does_not_crash():
    """run_doctor 应返回报告且必选项列表完整。"""
    report = run_doctor()
    assert len(report.required) >= len(
        [
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "statsmodels",
            "matplotlib",
            "networkx",
            "pulp",
            "openpyxl",
            "pdfplumber",
            "langgraph",
            "openai",
            "pydantic",
            "arxiv",
            "yaml",
            "论文编译器",
        ]
    )
    assert report.optional  # 可选项检查至少有一条


def test_cli_doctor_flag_exits(monkeypatch):
    """--doctor 参数应被 CLI 解析且不进入建模主流程。"""
    import sys

    from modeling_assistant.cli import main

    captured = []

    def fake_parse(self):
        class Args:
            doctor = True

        return Args()

    monkeypatch.setattr(
        "modeling_assistant.cli.argparse.ArgumentParser.parse_args", fake_parse
    )
    monkeypatch.setattr("modeling_assistant.doctor.print_report", lambda r: captured.append(1))

    def fake_exit(code):
        captured.append(("exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)
    try:
        main()
    except SystemExit as exc:
        assert exc.code in (0, 1)
    assert captured
    assert any(item[0] == "exit" for item in captured if isinstance(item, tuple))
