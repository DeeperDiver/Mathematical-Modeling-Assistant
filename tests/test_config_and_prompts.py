from pathlib import Path

from modeling_assistant.config import load_settings
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import ControlState, DynamicLTM, StaticLTM


def test_load_settings_from_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODELING_ASSISTANT_LLM_MODEL=deepseek-test",
                "MODELING_ASSISTANT_SEARCH_ENABLED=true",
                "MODELING_ASSISTANT_MAX_DEBATE_ROUNDS=4",
                "MODELING_ASSISTANT_INNOVATION_WEIGHT=0.6",
                "MODELING_ASSISTANT_FEASIBILITY_WEIGHT=0.4",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.llm_model == "deepseek-test"
    assert settings.search_enabled is True
    assert settings.max_debate_rounds == 4
    assert settings.innovation_weight == 0.6
    assert settings.feasibility_weight == 0.4


def test_default_search_enabled():
    """Searcher 默认应启用真实检索。"""
    settings = load_settings()
    assert settings.search_enabled is True


def test_prompt_catalog_renders_ltm_without_history():
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(
                assumptions=["只使用当前 LTM"],
                equations=["y = ax + b"],
                objective="线性拟合",
            ),
        ),
    )

    assert "不接收完整对话历史" in prompt
    assert "线性拟合" in prompt
    assert "y = ax + b" in prompt


def test_prompt_catalog_renders_branch_from_version():
    """PromptContext 应正确渲染 branch_from_version。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(branch_from_version="v1.0"),
        ),
    )

    assert "v1.0" in prompt
    assert "分支重建" in prompt


def test_prompt_catalog_renders_realist_thresholds():
    """Realist 模板应包含阈值与候选方案。"""
    from modeling_assistant.schemas.state import PlanCandidate

    prompt = PromptCatalog().render(
        "realist",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            control=ControlState(
                innovation_threshold=70,
                feasibility_threshold=65,
                top_k_plans=[
                    PlanCandidate(
                        id="p1",
                        title="方案",
                        description="描述",
                        innovation_score=80,
                        feasibility_score=70,
                    )
                ],
            ),
        ),
    )

    assert "70" in prompt  # innovation_threshold
    assert "65" in prompt  # feasibility_threshold
    assert "方案" in prompt  # plan title


def test_prompt_catalog_renders_coder_error_log():
    """Coder Prompt 应注入历史错误日志。"""
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(coder_error_log=["SyntaxError: invalid syntax"]),
        ),
    )
    assert "SyntaxError" in prompt
    assert "历史错误日志" in prompt


def test_prompt_catalog_renders_rebrainstorm_feedback():
    """Mathematician Prompt 应注入 Milestone Reviewer 1 的反馈。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(rebrainstorm_feedback=["假设列表为空。"]),
        ),
    )
    assert "假设列表为空" in prompt
