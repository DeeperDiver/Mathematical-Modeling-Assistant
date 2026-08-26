"""V15 方法知识库测试：规范解析、题型切片、prompt 渲染与降级开关。"""

from __future__ import annotations

from modeling_assistant.data.method_knowledge import (
    build_knowledge_payload,
    get_node_knowledge,
    get_type_knowledge,
    load_norm_sections,
    parse_sections,
)
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import DynamicLTM, StaticLTM


def test_parse_sections_splits_by_second_level_heading():
    """`## ` 二级标题应正确切分，且保留节内三级标题内容。"""
    text = (
        "# 知识库\n"
        "## 题型防错速查\n"
        "### 优化、调度\n"
        "- 不要漏非负约束\n"
        "## 假设与模型建立\n"
        "- 假设必须必要、可解释、可参数化\n"
    )
    sections = parse_sections(text)
    assert "题型防错速查" in sections
    assert "优化、调度" in sections["题型防错速查"]
    assert "假设与模型建立" in sections
    assert "可参数化" in sections["假设与模型建立"]


def test_load_norm_sections_finds_key_sections():
    """打包的 math_modeling_norms.md 应能解析出关键小节。"""
    sections = load_norm_sections()
    assert sections
    for key in (
        "模型大分类与选型速查",
        "题型防错速查",
        "假设与模型建立",
        "编码阶段常见错误",
        "图表与可视化",
        "论文写作",
        "论文写作规范补充",
        "论文验收与一致性",
        "决策/分组/分段类问题",
        "优化类模型详细指南",
        "评价类模型详细指南",
    ):
        assert key in sections, f"缺少规范小节: {key}"


def test_get_type_knowledge_maps_problem_types():
    """各题型应映射到对应的专属指南小节。"""
    optimization = get_type_knowledge("optimization")
    assert "优化" in optimization
    assert "scipy" in optimization  # 优化类指南包含求解器防错

    evaluation = get_type_knowledge("evaluation")
    assert "TOPSIS" in evaluation or "AHP" in evaluation

    # 未知题型回退到通用题型防错速查
    fallback = get_type_knowledge("unknown_type")
    assert fallback
    assert "约束" in fallback or "防错" in fallback


def test_get_node_knowledge_returns_node_specific_sections():
    """按节点切片应只返回该节点相关的小节。"""
    coding = get_node_knowledge("coding")
    assert "数据泄露" in coding or "scipy" in coding or "求解器" in coding

    chart = get_node_knowledge("chart")
    assert "图表" in chart

    writing = get_node_knowledge("writing")
    assert "摘要" in writing
    assert "灵敏度" in writing
    assert "硬错误" in writing

    model_selection = get_node_knowledge("model_selection")
    assert "决策变量" in model_selection
    assert "目标函数" in model_selection

    assumptions = get_node_knowledge("assumptions")
    assert "分组/分段质量度量" in assumptions
    assert "约束" in assumptions


def test_build_knowledge_payload_contains_all_injection_keys():
    """知识包应包含全部注入键，且 active 标志正确。"""
    payload = build_knowledge_payload("optimization")
    assert payload["method_knowledge_active"] == "true"
    assert payload["problem_type"] == "optimization"
    for key in (
        "model_selection_knowledge",
        "type_knowledge",
        "assumption_knowledge",
        "coding_knowledge",
        "chart_knowledge",
        "writing_knowledge",
    ):
        assert payload[key], f"知识包字段为空: {key}"


def test_prompt_catalog_renders_method_knowledge_when_enabled():
    """开启方法知识库时，Mathematician / Coder / Drawer 模板应包含知识内容。"""
    catalog = PromptCatalog()
    state = PromptContext(
        static_ltm=StaticLTM(
            raw_problem="给定城市交通流量数据，预测拥堵并优化信号灯配时。"
        ),
        dynamic_ltm=DynamicLTM(),
    )

    mathematician = catalog.render("mathematician", state)
    assert "模型选型与方法知识" in mathematician
    assert "method_knowledge_active=true" in mathematician
    assert "选型" in mathematician

    coder = catalog.render("coder", state)
    assert "编码阶段常见错误" in coder
    assert "当前题型专属指南" in coder

    drawer = catalog.render("drawer", state)
    assert "图表与可视化规范" in drawer

    writer = catalog.render("writer", state)
    assert "论文写作规范" in writer
    assert "摘要写作要点" in writer

    final_reviewer = catalog.render("final_reviewer", state)
    assert "论文写作与验收规范" in final_reviewer
    assert "验收应先识别项目实际布局" in final_reviewer
    assert "硬错误包括" in final_reviewer


def test_prompt_catalog_renders_without_knowledge_when_disabled():
    """关闭方法知识库时，模板应正常渲染且知识变量为空（与旧行为一致）。"""
    catalog = PromptCatalog()
    state = PromptContext(
        static_ltm=StaticLTM(raw_problem="测试问题"),
        dynamic_ltm=DynamicLTM(),
        extra={"method_knowledge_enabled": False},
    )
    prompt = catalog.render("mathematician", state)
    assert "method_knowledge_active=false" in prompt
    assert "model_selection_knowledge}" not in prompt  # 占位符必须已被替换


def test_prompt_catalog_renders_unknown_type_without_crash():
    """无题面文本时题型判定应回退 unknown，渲染不崩溃。"""
    catalog = PromptCatalog()
    state = PromptContext(
        static_ltm=StaticLTM(raw_problem=""),
        dynamic_ltm=DynamicLTM(),
    )
    prompt = catalog.render("realist", state)
    assert "题型防错速查" in prompt
    assert "unknown" in prompt or "method_knowledge_active" in prompt


def test_build_knowledge_payload_unknown_type_has_empty_type_guidance():
    """unknown 题型不应注入具体题型专属指南（避免误导）。"""
    payload = build_knowledge_payload("unknown")
    assert payload["problem_type"] == "unknown"
    assert payload["type_knowledge"] == ""
    # 通用知识仍应注入
    assert payload["model_selection_knowledge"]
    assert payload["method_knowledge_active"] == "true"


def test_math_modeling_norms_packaged(monkeypatch):
    """规范文件缺失时应优雅降级为空知识包。"""
    import modeling_assistant.data.method_knowledge as mk

    monkeypatch.setattr(mk, "_NORMS_PATH", mk.Path("not_exist/norms.md"))
    monkeypatch.setattr(mk, "_sections_cache", None)
    payload = build_knowledge_payload("optimization")
    assert payload["method_knowledge_active"] == "false"
    assert payload["model_selection_knowledge"] == ""
