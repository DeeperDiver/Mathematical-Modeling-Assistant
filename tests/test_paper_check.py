"""V15 论文确定性验收测试：占位符、泄露、章节、图片引用、编译降级。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.validation.paper_check import (
    check_paper,
    _check_assumptions_section,
    _check_malformed_image_commands,
    _check_front_matter_placeholders,
    _check_problem_summaries,
    _check_unresolved_cites,
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


def test_check_unresolved_cites_detects_missing_bibitem(tmp_path):
    """V17：`\cite{x}` 无对应 `\bibitem{x}` → 断链硬错误。"""
    tex = tmp_path / "bad.tex"
    tex.write_text(
        "\\section{问题}\n该方法见\\cite{ref9}。\n", encoding="utf-8"
    )
    issues = _check_unresolved_cites([tex])
    assert issues
    assert "ref9" in issues[0]


def test_check_unresolved_cites_ok_when_bibitem_exists(tmp_path):
    """存在对应 bibitem 的引用不应被误报。"""
    tex = tmp_path / "good.tex"
    tex.write_text(
        "\\begin{thebibliography}{9}\n\\bibitem{ref1} 作者. 题名[J]. 刊名, 2024.\n"
        "\\end{thebibliography}\n见\\cite{ref1}。\n",
        encoding="utf-8",
    )
    assert _check_unresolved_cites([tex]) == []


def test_check_paper_fails_on_unresolved_cite(tmp_path):
    """check_paper 应把无 bibitem 的 cite 记为硬错误。"""
    paper = _make_paper(tmp_path)
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\n方法见\\cite{ref9}。\n", encoding="utf-8"
    )
    report = check_paper(paper, compile_pdf=False)
    assert not report["passed"]
    assert any("ref9" in i for i in report["issues"])


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


def test_check_problem_summaries_detects_missing(tmp_path):
    """V18：问题章节缺少「问题小结」收尾 → 硬错误。"""
    (tmp_path / "sections").mkdir(parents=True)
    (tmp_path / "sections" / "5_problem1.tex").write_text(
        "\\section{问题一的模型建立与求解}\n求解结果：R²=0.896。\n",
        encoding="utf-8",
    )
    issues = _check_problem_summaries(tmp_path / "sections")
    assert len(issues) == 1
    assert "5_problem1.tex" in issues[0]
    assert "问题小结" in issues[0]


def test_check_problem_summaries_passes_when_present(tmp_path):
    """V18：问题章节含「问题小结」时通过。"""
    (tmp_path / "sections").mkdir(parents=True)
    (tmp_path / "sections" / "5_problem1.tex").write_text(
        "\\section{问题一的模型建立与求解}\n求解结果：R²=0.896。\n"
        "\\subsection{问题小结}\n本题建立对比模型并得到最优结果，"
        "为下一题提供特征输入。\n",
        encoding="utf-8",
    )
    assert _check_problem_summaries(tmp_path / "sections") == []


def test_check_paper_fails_when_problem_summary_missing(tmp_path):
    """V18：check_paper 应将缺「问题小结」的问题章节记为硬错误。"""
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{ctexart}\n\\begin{document}\n"
        "\\input{sections/5_problem1}\n\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / "5_problem1.tex").write_text(
        "\\section{问题一的模型建立与求解}\n求解结果。\n", encoding="utf-8"
    )
    report = check_paper(paper, compile_pdf=False)
    assert not report["passed"]
    assert report["checks"]["问题小结"] != "通过"
    assert any("问题小结" in i for i in report["issues"])


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


def _write_assumptions(paper: Path, items: list[str]) -> None:
    """写入 3_assumptions.tex（enumerate + \\item 列表）。"""
    (paper / "sections" / "3_assumptions.tex").write_text(
        "\\section{模型假设}\n为简化问题，本文做出以下基本假设：\n"
        "\\begin{enumerate}\n"
        + "".join(f"  \\item {item}\n" for item in items)
        + "\\end{enumerate}\n",
        encoding="utf-8",
    )


def test_check_assumptions_section_fails_when_too_many(tmp_path):
    """V20：3_assumptions.tex 超过 6 条假设 → 硬错误。"""
    paper = _make_paper(tmp_path)
    _write_assumptions(paper, [f"【全文】假设{i}" for i in range(7)])
    issues = _check_assumptions_section(paper / "sections")
    assert len(issues) == 1
    assert "3_assumptions.tex" in issues[0]
    assert "超过上限" in issues[0]


def test_check_assumptions_section_fails_on_question_tag(tmp_path):
    """V20：【问题N】假设混入全文假设章 → 硬错误。"""
    paper = _make_paper(tmp_path)
    _write_assumptions(
        paper,
        ["【全文】题目所给数据真实可靠", "【问题1】浓度取值于(0,1)，在 logit 尺度建模"],
    )
    issues = _check_assumptions_section(paper / "sections")
    assert len(issues) == 1
    assert "问题假设被错误写入全文假设章" in issues[0]


def test_check_assumptions_section_passes_clean(tmp_path):
    """V20：≤6 条且无【问题N】标签的假设章通过。"""
    paper = _make_paper(tmp_path)
    _write_assumptions(
        paper,
        ["【全文】题目所给数据真实可靠", "【全文】系统在短期内处于稳定状态"],
    )
    assert _check_assumptions_section(paper / "sections") == []


def test_check_assumptions_section_missing_file_no_issue(tmp_path):
    """V20：假设章文件缺失时不误报（由章节存在性检查负责）。"""
    paper = _make_paper(tmp_path)
    assert _check_assumptions_section(paper / "sections") == []


def test_check_paper_fails_on_assumptions_overflow(tmp_path):
    """V20：check_paper 应将假设章违规记为硬错误并写入报告。"""
    paper = _make_paper(tmp_path)
    _write_assumptions(paper, [f"【全文】假设{i}" for i in range(7)])
    report = check_paper(paper, compile_pdf=False)
    assert not report["passed"]
    assert report["checks"]["假设章"] != "通过"
    assert any("3_assumptions.tex" in i for i in report["issues"])


def test_check_paper_passes_clean_assumptions(tmp_path):
    """V20：干净假设章不阻塞论文验收。"""
    paper = _make_paper(tmp_path)
    _write_assumptions(
        paper,
        ["【全文】题目所给数据真实可靠", "【全文】系统在短期内处于稳定状态"],
    )
    report = check_paper(paper, compile_pdf=False)
    assert report["passed"]
    assert report["checks"]["假设章"] == "通过"
