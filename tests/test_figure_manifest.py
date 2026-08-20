"""V17 图表注册表与图表完整性闭环测试。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.schemas.responses import FigurePlan
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    merge_artifacts_reducer,
)
from modeling_assistant.validation.paper_check import check_paper


def _fig(id_: str, caption: str = "图注", required: bool = True) -> FigurePlan:
    return FigurePlan(
        id=id_,
        figure_type="scatter",
        kind="data",
        caption=caption,
        section="5_problem1.tex",
        required=required,
    )


def _make_paper(tmp_path: Path, section_tex: str) -> Path:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{ctexart}\n\\begin{document}\n"
        "\\input{sections/5_problem1}\n\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / "5_problem1.tex").write_text(
        f"\\section{{问题}}\n{section_tex}\n", encoding="utf-8"
    )
    return paper


def test_reducer_merges_figures_plan_by_id():
    """多轮 Architect 的图表计划应按 plan_id 合并而非整表覆盖。"""
    base = ArtifactBundle(figures_plan=[_fig("fig_a"), _fig("fig_b")])
    incoming = ArtifactBundle(
        figures_plan=[
            _fig("fig_b", caption="B 图（更新）"),
            _fig("fig_c"),
        ]
    )
    merged = merge_artifacts_reducer(base, incoming)
    ids = [f.id for f in merged.figures_plan]
    assert ids == ["fig_a", "fig_b", "fig_c"]
    by_id = {f.id: f for f in merged.figures_plan}
    assert by_id["fig_b"].caption == "B 图（更新）"


def test_reducer_merges_figure_manifest_by_id():
    """图表注册表按 plan_id 覆盖合并，多轮生成不互相覆盖。"""
    base = ArtifactBundle(
        figure_manifest={
            "fig_a": {"path": "figures/fig_a_v1.png", "run_id": "drawer_0", "status": "generated"}
        }
    )
    incoming = ArtifactBundle(
        figure_manifest={
            "fig_a": {"path": "figures/fig_a_v2.png", "run_id": "drawer_1", "status": "generated"},
            "fig_b": {"path": "figures/fig_b.png", "run_id": "drawer_1", "status": "generated"},
        }
    )
    merged = merge_artifacts_reducer(base, incoming)
    assert merged.figure_manifest["fig_a"]["path"] == "figures/fig_a_v2.png"
    assert merged.figure_manifest["fig_b"]["path"] == "figures/fig_b.png"


def test_reducer_clear_figure_manifest():
    """drawer 失败显式清空注册表（与 clear_result_paths 对称）。"""
    base = ArtifactBundle(figure_manifest={"fig_a": {"path": "x.png", "status": "generated"}})
    incoming = ArtifactBundle(clear_figure_manifest=True)
    merged = merge_artifacts_reducer(base, incoming)
    assert merged.figure_manifest == {}


def test_register_figure_manifest_matches_plan_ids():
    """人工/外部编程手交付的图应按文件名与 plan_id 匹配登记。"""
    from modeling_assistant.agents.nodes import _register_figure_manifest

    state = {"artifacts": ArtifactBundle(figures_plan=[_fig("fig_a"), _fig("fig_b")])}
    artifacts = ArtifactBundle()
    _register_figure_manifest(
        state,
        artifacts,
        real_figures=[
            "figures/fig_a.png",
            "figures/extra.png",  # 未规划 → 不登记
            "figures/fig_b.pdf",
        ],
        run_tag="human_3",
    )
    assert artifacts.figure_manifest["fig_a"]["path"] == "figures/fig_a.png"
    assert artifacts.figure_manifest["fig_b"]["path"] == "figures/fig_b.pdf"
    assert "extra" not in artifacts.figure_manifest
    assert artifacts.figure_manifest["fig_a"]["status"] == "generated"
    assert artifacts.figure_manifest["fig_a"]["run_id"] == "human_3"


def test_figure_completeness_passes_when_all_referenced(tmp_path):
    """规划→生成→引用→图注全部对账通过。"""
    figures = tmp_path / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "fig_a.png").write_bytes(b"x")
    paper = _make_paper(
        tmp_path,
        "\\begin{figure}\n"
        "\\includegraphics[width=0.8\\textwidth]{../figures/fig_a.png}\n"
        "\\caption{关键变量相关关系散点图}\n\\label{fig_a}\n\\end{figure}\n",
    )
    plan = [_fig("fig_a", caption="关键变量相关关系散点图").model_dump(mode="json")]
    manifest = {"fig_a": {"path": str(figures / "fig_a.png"), "status": "generated"}}
    report = check_paper(paper, compile_pdf=False, figures_plan=plan, figure_manifest=manifest)
    assert report["passed"], report["issues"]
    assert report["checks"]["图表完整性"] == "通过"


def test_figure_completeness_fails_when_required_missing(tmp_path):
    """C1：required 图未登记/文件不存在 → 硬错误。"""
    paper = _make_paper(tmp_path, "内容。")
    plan = [_fig("fig_a").model_dump(mode="json")]
    report = check_paper(paper, compile_pdf=False, figures_plan=plan, figure_manifest={})
    assert not report["passed"]
    assert any("未生成" in i and "fig_a" in i for i in report["issues"])

    # 登记了但文件不存在同样拦截
    manifest = {"fig_a": {"path": str(tmp_path / "figures" / "fig_a.png"), "status": "generated"}}
    report2 = check_paper(paper, compile_pdf=False, figures_plan=plan, figure_manifest=manifest)
    assert not report2["passed"]
    assert any("不存在" in i for i in report2["issues"])


def test_figure_completeness_fails_when_generated_but_not_referenced(tmp_path):
    """C2：required 图已生成但未被论文引用 → 硬错误。"""
    figures = tmp_path / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "fig_a.png").write_bytes(b"x")
    paper = _make_paper(tmp_path, "正文没有引用图。")
    plan = [_fig("fig_a").model_dump(mode="json")]
    manifest = {"fig_a": {"path": str(figures / "fig_a.png"), "status": "generated"}}
    report = check_paper(paper, compile_pdf=False, figures_plan=plan, figure_manifest=manifest)
    assert not report["passed"]
    assert any("未被论文引用" in i and "fig_a" in i for i in report["issues"])


def test_figure_completeness_warns_on_caption_mismatch_and_extra(tmp_path):
    """C4：图注与规划不一致 → 警告；C5：引用注册表外图片 → 警告。"""
    figures = tmp_path / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "fig_a.png").write_bytes(b"x")
    (figures / "fig_extra.png").write_bytes(b"x")
    paper = _make_paper(
        tmp_path,
        "\\begin{figure}\n"
        "\\includegraphics[width=0.8\\textwidth]{../figures/fig_a.png}\n"
        "\\includegraphics[width=0.8\\textwidth]{../figures/fig_extra.png}\n"
        "\\caption{完全不同的图注文案}\n\\end{figure}\n",
    )
    plan = [_fig("fig_a", caption="关键变量相关关系散点图").model_dump(mode="json")]
    manifest = {"fig_a": {"path": str(figures / "fig_a.png"), "status": "generated"}}
    report = check_paper(paper, compile_pdf=False, figures_plan=plan, figure_manifest=manifest)
    # 图注不一致 → 警告
    assert any("图注与规划不一致" in w for w in report["warnings"])
    # 引用了注册表之外的 fig_extra.png → 警告
    assert any("未登记图表注册表" in w for w in report["warnings"])


def test_figure_completeness_skipped_without_plan():
    """未提供 figures_plan 时跳过（旧流程向后兼容）。"""
    paper = _make_paper(tmp_path=Path(__import__("tempfile").mkdtemp()), section_tex="内容。")
    report = check_paper(paper, compile_pdf=False)
    assert report["checks"]["图表完整性"] == "跳过（未提供 figures_plan）"
