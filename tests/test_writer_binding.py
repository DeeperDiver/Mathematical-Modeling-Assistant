"""V17 Writer 章节-结果绑定测试：manifest 与绑定规则注入 prompt。"""

from __future__ import annotations

from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import (
    AuthoritativeResult,
    ControlState,
    DynamicLTM,
    StaticLTM,
)


def _render_writer(control: ControlState) -> str:
    return PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="测试题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=control,
            extra={
                "integrity_warnings": "无",
                "paper_template_structure": "[]",
                "paper_template_active": "false",
            },
        ),
    )


def test_writer_prompt_injects_manifest_and_binding():
    """manifest 与章节绑定都应出现在 Writer prompt 中。"""
    control = ControlState(
        sub_questions=["问题1", "问题2"],
        current_sub_question_index=2,
        results_manifest=[
            AuthoritativeResult(index=0, title="问题1", result_paths=["out/q1.csv"]),
            AuthoritativeResult(index=1, title="问题2", result_paths=["out/q2.csv"]),
        ],
    )
    prompt = _render_writer(control)

    assert "结果文件预览（按章节绑定" in prompt
    assert "章节-结果绑定规则" in prompt
    assert '"5_problem1.tex": 0' in prompt
    assert '"6_problem2.tex": 1' in prompt
    assert "q1.csv" in prompt
    assert "q2.csv" in prompt
    assert "threelinetable[label]" in prompt
    assert "best_params_text" in prompt


def test_writer_prompt_binding_falls_back_without_sub_questions():
    """无小题清单时绑定退化为 5_problem1.tex → 0。"""
    control = ControlState(
        results_manifest=[AuthoritativeResult(index=0, result_paths=["out/output.csv"])]
    )
    prompt = _render_writer(control)
    assert '"5_problem1.tex": 0' in prompt
    assert "output.csv" in prompt


def test_writer_prompt_empty_manifest_renders():
    """manifest 为空时 prompt 仍应正常渲染（旧流程兼容）。"""
    prompt = _render_writer(ControlState())
    assert "结果文件预览（按章节绑定" in prompt
    assert "结果文件预览（旧平铺" in prompt


def test_writer_prompt_table_label_requirements():
    """Writer 指令应统一为带 label 的三线表写法，并给出示例。"""
    prompt = _render_writer(ControlState())
    # 新写法 + 示例
    assert "\\threelinetable[label]" in prompt
    assert "`label` 必填" in prompt
    assert "tab:p1_params" in prompt
    # 不应再出现旧的无 label 4 参写法指令
    assert "\\threelinetable{{表题}}{{列格式}}{{表头}}{{内容}}" not in prompt
