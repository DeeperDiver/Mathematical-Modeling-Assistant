"""V15 论文确定性验收测试：占位符、泄露、章节、图片引用、编译降级。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.validation.paper_check import (
    check_paper,
    _check_malformed_image_commands,
    _check_front_matter_placeholders,
    _check_unresolved_refs,
    _find_placeholders,
    _find_internal_leaks,
)


def _make_paper(tmp_path: Path) -> Path:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{ctexart}\n"
        "\\begin{document}\n"
        "\\input{sections/1_restatement}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\n内容。\n", encoding="utf-8"
    )
    return paper


def test_find_placeholders_detects_common_markers():
    """应检测 TODO / 待补充 / 示例数据 等占位符。"""
    text = "本节内容 TODO 补充，示例数据待替换。"
    found = _find_placeholders(text)
    assert len(found) >= 2


def test_find_placeholders_detects_unresolved_table_ref():
    """V17：字面 `表??`/`图??` 也应视为占位符。"""
    text = "最优参数如表??所示。"
    found = _find_placeholders(text)
    assert found


def test_check_unresolved_refs_detects_missing_label(tmp_path):
    """V17：`\ref{tab:x}` 无对应 `\label{tab:x}` → 断链硬错误。"""
    tex = tmp_path / "bad.tex"
    tex.write_text(
        "\\section{问题}\n最优参数见表\\ref{tab:p1_params}。\n",
        encoding="utf-8",
    )
    issues = _check_unresolved_refs([tex])
    assert issues
    assert "tab:p1_params" in issues[0]


def test_check_unresolved_refs_ok_when_label_exists(tmp_path):
    """定义过 label 的引用不应被误报。"""
    tex = tmp_path / "good.tex"
    tex.write_text(
        "\\threelinetable[tab:p1]{参数}{cc}{参数 & 数值}{R & 1.0}\n"
        "见表\\ref{tab:p1}。\n",
        encoding="utf-8",
    )
    assert _check_unresolved_refs([tex]) == []


def test_check_paper_fails_on_unresolved_ref(tmp_path):
    """check_paper 应把未定义 label 的引用记为硬错误。"""
    paper = _make_paper(tmp_path)
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\n结果见表\\ref{tab:nope}。\n", encoding="utf-8"
    )
    report = check_paper(paper, compile_pdf=False)
    assert not report["passed"]
    assert "引用完整性" in report["checks"]
    assert any("tab:nope" in i for i in report["issues"])


def test_find_internal_leaks_detects_workflow_markers():
    """应检测 reports/ 等内部工作流标记。"""
    text = "模型结果见 reports/RESULTS_REPORT.md 与 coder_task.json。"
    leaks = _find_internal_leaks(text)
    assert "reports/" in leaks
    assert "coder_task" in leaks


def test_check_paper_passes_clean_paper(tmp_path):
    """干净的论文应通过确定性验收（无编译器时编译检查降级为警告）。"""
    paper = _make_paper(tmp_path)
    report = check_paper(paper)
    assert report["passed"], report["issues"]
    assert report["checks"]["入口"] == "存在"
    assert report["checks"]["占位符"] == "无"


def test_check_paper_fails_on_placeholder(tmp_path):
    """占位符是硬错误。"""
    paper = _make_paper(tmp_path)
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\nTODO 补充内容\n", encoding="utf-8"
    )
    report = check_paper(paper)
    assert not report["passed"]
    assert any("占位符" in i for i in report["issues"])


def test_check_paper_fails_on_missing_input(tmp_path):
    """main.tex 引用不存在的章节文件是硬错误。"""
    paper = _make_paper(tmp_path)
    (paper / "main.tex").write_text(
        "\\input{sections/9_missing}\n", encoding="utf-8"
    )
    report = check_paper(paper)
    assert not report["passed"]
    assert any("9_missing" in i for i in report["issues"])


def test_check_paper_fails_on_missing_image(tmp_path):
    """\\includegraphics 引用不存在的图片是硬错误。"""
    paper = _make_paper(tmp_path)
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\n"
        "\\includegraphics[width=0.8\\textwidth]{../figures/nonexistent.pdf}\n",
        encoding="utf-8",
    )
    report = check_paper(paper)
    assert not report["passed"]
    assert any("nonexistent.pdf" in i for i in report["issues"])


def test_check_malformed_image_commands_detects_bare_option(tmp_path):
    """`[width=0.85]../figures/xxx.png` 缺少 `\includegraphics` → 硬错误。"""
    tex = tmp_path / "bad.tex"
    tex.write_text(
        "三问共享同一反射渲染核心 $R_\\Theta$。\n"
        "[width=0.85]../figures/figroadmap.png\n"
        "图 1: 总体技术路线图\n",
        encoding="utf-8",
    )
    issues = _check_malformed_image_commands([tex])
    assert issues
    assert any("includegraphics" in issue and "figroadmap" in issue for issue in issues)


def test_check_malformed_image_commands_ignores_valid(tmp_path):
    """完整 `\includegraphics[width=...]{...}` 不应被误报。"""
    tex = tmp_path / "good.tex"
    tex.write_text(
        "\\includegraphics[width=0.85\\textwidth]{../figures/fig_roadmap.png}\n",
        encoding="utf-8",
    )
    assert _check_malformed_image_commands([tex]) == []


def test_check_paper_fails_on_bare_image_option(tmp_path):
    """论文正文出现裸 [width=...]figures 路径 → check_paper 打回。"""
    paper = _make_paper(tmp_path)
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\n[width=0.85]../figures/figroadmap.png\n",
        encoding="utf-8",
    )
    report = check_paper(paper, compile_pdf=False)
    assert not report["passed"]
    assert any("图片命令残缺" in issue for issue in report["issues"])


def test_check_front_matter_placeholders_detected(tmp_path):
    """main.tex 残留标题/摘要/关键词占位符 → 硬错误。"""
    paper = _make_paper(tmp_path)
    main = paper / "main.tex"
    text = main.read_text(encoding="utf-8")
    text += "\n\\papertitle{论文标题}\n\\abstractcn{%\n  中文摘要内容：占位%\n}{%\n  关键词1\\quad 关键词2%\n}\n"
    main.write_text(text, encoding="utf-8")
    issues = _check_front_matter_placeholders(main)
    assert len(issues) == 3
    report = check_paper(paper, compile_pdf=False)
    assert not report["passed"]
    assert any("占位符" in issue for issue in report["issues"])


def test_check_paper_handles_missing_paper_dir(tmp_path):
    """论文目录缺失时报告硬错误而非崩溃。"""
    report = check_paper(tmp_path / "no_such_paper")
    assert not report["passed"]
    assert report["checks"]["入口"] == "缺失"
