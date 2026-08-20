"""V15/V17 图表规划测试：FigurePlan 完整字段与 architect/drawer/writer 模板渲染。"""

from __future__ import annotations

from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.responses import FigurePlan
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    DynamicLTM,
    StaticLTM,
)


def test_figure_plan_kind_default_is_data():
    """FigurePlan.kind 默认应为 data（向后兼容）。"""
    plan = FigurePlan(id="f1", figure_type="scatter", purpose="相关分析")
    assert plan.kind == "data"
    assert plan.model_dump()["kind"] == "data"


def test_figure_plan_flowchart_kind_roundtrip():
    """flowchart/diagram 类别应能序列化。"""
    plan = FigurePlan(
        id="fig_roadmap",
        figure_type="roadmap",
        kind="flowchart",
        purpose="技术路线图",
        data_source="",
    )
    data = plan.model_dump()
    assert data["kind"] == "flowchart"
    restored = FigurePlan.model_validate(data)
    assert restored.kind == "flowchart"


def test_figure_plan_v17_fields_roundtrip():
    """V17：图注/章节/内容规格/必需标志应能序列化。"""
    plan = FigurePlan(
        id="fig_q1_corr",
        figure_type="scatter",
        kind="data",
        caption="关键变量相关关系散点图",
        section="5_problem1.tex",
        content_spec="用 x/y 两列绘制散点并标注 Pearson r",
        required=True,
        purpose="展示变量相关关系",
        data_source="results/q1.csv",
    )
    data = plan.model_dump()
    assert data["caption"] == "关键变量相关关系散点图"
    assert data["section"] == "5_problem1.tex"
    assert data["content_spec"].startswith("用 x/y 两列")
    restored = FigurePlan.model_validate(data)
    assert restored.section == "5_problem1.tex"


def test_architect_prompt_mentions_figure_kinds():
    """Architect 模板应包含图表规划与 kind 说明。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="测试题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
        ),
    )
    assert "图表规划" in prompt
    assert "flowchart" in prompt
    assert "技术路线图" in prompt


def test_architect_prompt_injects_craft_figure_placement_and_v17_fields():
    """V17：Architect 应拿到图片位置规划参考并规划图注/章节/内容规格。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="测试题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
        ),
    )
    assert "图片位置规划参考" in prompt
    assert "craft_figure_placement" not in prompt  # 变量已渲染
    assert "caption" in prompt
    assert "content_spec" in prompt
    assert "section" in prompt


def test_drawer_prompt_mentions_non_data_figures():
    """Drawer 模板应包含非数据图（技术路线图）绘制规范。"""
    prompt = PromptCatalog().render(
        "drawer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="测试题"),
            dynamic_ltm=DynamicLTM(),
        ),
    )
    assert "非数据图绘制规范" in prompt
    assert "FancyBboxPatch" in prompt
    assert "fig_roadmap" in prompt


def test_drawer_prompt_requires_figure_ids_and_plan_naming():
    """V17：Drawer 应按 plan.id 命名并输出 figure_ids。"""
    prompt = PromptCatalog().render(
        "drawer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="测试题"),
            dynamic_ltm=DynamicLTM(),
        ),
    )
    assert "图表计划执行" in prompt
    assert "figure_ids" in prompt
    assert "plan_id" in prompt


def test_writer_prompt_injects_figure_manifest():
    """V17：Writer 应拿到图表注册表并受图注绑定约束。"""
    artifacts = ArtifactBundle(
        figure_manifest={
            "fig_a": {"path": "figures/fig_a.png", "run_id": "drawer_0", "status": "generated"}
        },
        figures_plan=[
            FigurePlan(id="fig_a", caption="A 图", section="5_problem1.tex")
        ],
    )
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="测试题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            artifacts=artifacts,
            extra={
                "integrity_warnings": "无",
                "paper_template_structure": "[]",
                "paper_template_active": "false",
            },
        ),
    )
    assert "图表引用与图注绑定" in prompt
    assert "fig_a.png" in prompt
    assert "status=generated" in prompt
