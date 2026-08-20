"""V13/V17 编程手任务包测试：交付要求含 plan_id 图命名。"""

from __future__ import annotations

import json
from pathlib import Path

from modeling_assistant.handoff.spec import write_coder_task_package
from modeling_assistant.schemas.responses import FigurePlan
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    StaticLTM,
)


def _state(tmp_path: Path) -> dict:
    return {
        "static_ltm": StaticLTM(raw_problem="测试题"),
        "dynamic_ltm": DynamicLTM(objective="目标", assumptions=["假设1"]),
        "artifacts": ArtifactBundle(
            figures_plan=[
                FigurePlan(
                    id="fig_q1_corr",
                    figure_type="scatter",
                    kind="data",
                    caption="相关散点图",
                    section="5_problem1.tex",
                )
            ]
        ),
        "control": ControlState(
            sub_questions=["问题1", "问题2"], current_sub_question_index=0
        ),
    }


def test_coder_task_md_requires_plan_id_naming(tmp_path):
    """coder_task.md 应要求 figures.py 按 plan_id 命名图片。"""
    md_path, _json_path = write_coder_task_package(_state(tmp_path), tmp_path)
    md = md_path.read_text(encoding="utf-8")
    assert "figures_plan" in md
    assert "plan_id" in md
    assert "fig_q1_corr" in md
    assert "未按 plan_id 命名的图片不会被论文引用" in md


def test_coder_task_json_carries_figures_plan(tmp_path):
    """coder_task.json 应携带结构化 figures_plan 与小题信息。"""
    _md_path, json_path = write_coder_task_package(_state(tmp_path), tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["task"] == "coder_implementation"
    assert payload["sub_question"]["index"] == 0
    assert payload["artifacts"]["figures_plan"][0]["id"] == "fig_q1_corr"
    assert payload["artifacts"]["figures_plan"][0]["section"] == "5_problem1.tex"
