"""V15 国赛 LaTeX 模板测试：复制、子问题数量调整、结构清单。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.data.paper_template import (
    build_template_structure,
    copy_template,
    load_template_structure,
    _adjust_problem_inputs,
)


def _make_fake_template(tmp_path: Path) -> Path:
    """构造一个带 3 个问题章节的假模板。"""
    tpl = tmp_path / "cumcm-latex"
    (tpl / "sections").mkdir(parents=True)
    (tpl / "main.tex").write_text(
        "\\documentclass{ctexart}\n"
        "\\begin{document}\n"
        "\\input{sections/1_restatement}\n"
        "\\input{sections/2_analysis}\n"
        "\\input{sections/5_problem1}\n"
        "\\input{sections/6_problem2}\n"
        "\\input{sections/7_problem3}\n"
        "\\input{sections/8_sensitivity}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    for name in (
        "1_restatement",
        "2_analysis",
        "5_problem1",
        "6_problem2",
        "7_problem3",
        "8_sensitivity",
    ):
        (tpl / "sections" / f"{name}.tex").write_text(
            f"\\section{{{name}}}\n", encoding="utf-8"
        )
    return tpl


def test_adjust_problem_inputs_replaces_three_with_two():
    """3 个问题章节应被替换为 2 个。"""
    main_tex = (
        "\\input{sections/5_problem1}\n"
        "\\input{sections/6_problem2}\n"
        "\\input{sections/7_problem3}\n"
    )
    adjusted = _adjust_problem_inputs(main_tex, 2)
    assert "5_problem1" in adjusted
    assert "6_problem2" in adjusted
    assert "7_problem3" not in adjusted


def test_adjust_problem_inputs_extends_to_five():
    """子问题多于模板默认时，input 行应扩展到对应数量。"""
    main_tex = (
        "\\input{sections/5_problem1}\n"
        "\\input{sections/6_problem2}\n"
        "\\input{sections/7_problem3}\n"
    )
    adjusted = _adjust_problem_inputs(main_tex, 5)
    assert "\\input{sections/9_problem5}" in adjusted
    assert adjusted.count("_problem") == 5


def test_copy_template_adjusts_for_two_sub_questions(tmp_path):
    """复制模板后 main.tex 应为 2 个问题章节，且多余文件被清理。"""
    tpl = _make_fake_template(tmp_path)
    paper = tmp_path / "paper"

    structure = copy_template(tpl, paper, 2)

    assert structure is not None
    main_tex = (paper / "main.tex").read_text(encoding="utf-8")
    assert "6_problem2" in main_tex
    assert "7_problem3" not in main_tex
    assert not (paper / "sections" / "7_problem3.tex").exists()
    assert (paper / "sections" / "5_problem1.tex").exists()


def test_copy_template_returns_none_when_missing(tmp_path):
    """模板缺失时应返回 None（writer 回退旧行为）。"""
    assert copy_template(tmp_path / "missing", tmp_path / "paper", 3) is None


def test_build_template_structure_dynamic_problem_count():
    """结构清单应包含固定章节 + 动态问题章节 + 尾随章节。"""
    structure = build_template_structure(4)
    files = [s["file"] for s in structure]
    assert "1_restatement.tex" in files
    assert "4_symbols.tex" in files
    assert "5_problem1.tex" in files
    assert "8_problem4.tex" in files
    assert "8_sensitivity.tex" in files
    assert "9_evaluation.tex" in files


def test_load_template_structure_none_when_no_main(tmp_path):
    """无 main.tex 的目录不应被识别为模板。"""
    assert load_template_structure(tmp_path) is None
