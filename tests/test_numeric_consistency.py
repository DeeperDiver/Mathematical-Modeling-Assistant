"""V17 论文↔结果机器数字比对测试（含本次「Q1 章节抄 Q2 参数」事故回归）。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.validation.numeric_consistency import (
    check_numeric_consistency,
    extract_paper_value_refs,
)


def _make_paper(tmp_path: Path, section_name: str, table_tex: str, text: str = "") -> Path:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{ctexart}\n\\begin{document}\n\\input{sections/5_problem1}\n\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / f"{section_name}.tex").write_text(
        f"\\section{{问题}}\n{table_tex}\n{text}\n", encoding="utf-8"
    )
    return paper


def _write_q1_csv(tmp_path: Path) -> str:
    """与真实 q1.csv 同构：fig3/fig4 两行，R 为 0.0111/0.0698。"""
    p = tmp_path / "q1.csv"
    p.write_text(
        "image_id,R,H,x_c,z_c,phi0,y0,delta_phi,delta_y,CD,SSIM\n"
        "fig3,0.0110826202151898,0.1563507051774279,0.1499603084095052,"
        "0.0951080023735885,3.049627314935204,0.0316315659407253,"
        "0.1722580013431425,0.0302095842606953,0.0216525815046763,0.0278244760706581\n"
        "fig4,0.0698460260774311,0.1390528815284972,0.0773414193277007,"
        "0.0731307185072227,1.5807933159409873,0.0625688147386299,"
        "0.2520987071923236,0.0678066814348284,0.0336257377428021,0.4236519776887816\n",
        encoding="utf-8",
    )
    return str(p)


def _write_q2_csv(tmp_path: Path) -> str:
    """与真实 q2.csv 同构：best_params_text 含 R=0.0322 的共享参数串。"""
    p = tmp_path / "q2.csv"
    p.write_text(
        "sub_problem,target_image,threshold_tau,best_mirror_meaning,best_params_text\n"
        '1,none,0.7,0.470238,"R=0.0322,H=0.2163,x_c=0.0989,z_c=0.1449,'
        'phi0=1.4526,y0=0.0809,dphi=1.7711,dy=0.1159"\n',
        encoding="utf-8",
    )
    return str(p)


def test_extract_paper_value_refs_from_table_and_text(tmp_path):
    """三线表行与正文的「参数=数值」都应被提取（含 LaTeX 数学写法）。"""
    table = (
        "\\threelinetable[tab:p1]{参数}{cc}{参数 & 数值}"
        "{半径 $R$ (m) & 0.0322 \\\\ 中心方位角 $\\phi_0$ (rad) & 1.4526}"
    )
    text = "最优半径$R=0.0322$m；角向跨度$\\Delta\\phi=1.7711$rad。"
    refs = extract_paper_value_refs(table + "\n" + text)
    pairs = dict.fromkeys(refs)
    assert ("R", 0.0322) in pairs
    assert ("phi0", 1.4526) in pairs
    assert ("delta_phi", 1.7711) in pairs


def test_accident_regression_q1_section_uses_q2_params(tmp_path):
    """事故回归：5_problem1 的 R=0.0322 应报「未见于绑定结果文件」。"""
    q1 = _write_q1_csv(tmp_path)
    q2 = _write_q2_csv(tmp_path)
    table = (
        "\\threelinetable{图3与图4的最优参数}{cc}{参数 & 数值}"
        "{半径 $R$ (m) & 0.0322 \\\\ 高度 $H$ (m) & 0.2163}"
    )
    paper = _make_paper(tmp_path, "5_problem1", table)
    manifest = [
        {"index": 0, "result_paths": [q1]},
        {"index": 1, "result_paths": [q2]},
    ]
    issues, _warnings = check_numeric_consistency(paper, manifest, results_root=tmp_path)
    assert any("R=0.0322" in i and "q1.csv" in i for i in issues)
    assert any("H=0.2163" in i for i in issues)


def test_consistent_paper_passes(tmp_path):
    """数值与绑定文件一致（含保留位数差异）时不应误报。"""
    q1 = _write_q1_csv(tmp_path)
    table = (
        "\\threelinetable{最优参数}{cc}{参数 & 数值}"
        "{半径 $R$ (m) & 0.0111 \\\\ 高度 $H$ (m) & 0.1564}"
    )
    paper = _make_paper(tmp_path, "5_problem1", table)
    issues, _warnings = check_numeric_consistency(
        paper, [{"index": 0, "result_paths": [q1]}], results_root=tmp_path
    )
    assert issues == []


def test_tau_and_threshold_tau_match(tmp_path):
    """阈值 τ 应与 CSV 的 threshold_tau 列匹配，不误报。"""
    q2 = _write_q2_csv(tmp_path)
    text = "在阈值 $\\tau=0.7$ 下求解"
    paper = _make_paper(tmp_path, "6_problem2", "", text=text)
    issues, _warnings = check_numeric_consistency(
        paper, [{"index": 1, "result_paths": [q2]}], results_root=tmp_path
    )
    assert issues == []


def test_assumption_constant_not_flagged_when_absent_from_bound_file(tmp_path):
    """h_e 等固定假设常量在绑定文件缺失时不应被误报。"""
    q1 = _write_q1_csv(tmp_path)
    table = (
        "\\threelinetable{参数}{cc}{参数 & 数值}"
        "{眼睛高度 $h_e$ (m) & 0.3000}"
    )
    paper = _make_paper(tmp_path, "5_problem1", table)
    issues, _warnings = check_numeric_consistency(
        paper, [{"index": 0, "result_paths": [q1]}], results_root=tmp_path
    )
    assert issues == []


def test_skip_markers_exempt_lines(tmp_path):
    """含「待验证/理论推导」的行不参与比对。"""
    q1 = _write_q1_csv(tmp_path)
    text = "该参数为理论推导值：$R=9.9999$，待验证。"
    paper = _make_paper(tmp_path, "5_problem1", "", text=text)
    issues, _warnings = check_numeric_consistency(
        paper, [{"index": 0, "result_paths": [q1]}], results_root=tmp_path
    )
    assert issues == []
