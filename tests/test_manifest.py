"""V17 Result Manifest 测试：验收锁定、替换、单题兜底、回滚截断。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.agents.nodes import (
    _finalize_authoritative_result,
    _truncate_manifest,
)
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    AuthoritativeResult,
    ControlState,
    GraphState,
    StaticLTM,
)


def _state(tmp_path: Path) -> GraphState:
    return {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "control": ControlState(),
        "ltm_archive": [],
        "artifacts": ArtifactBundle(),
    }


def _write_csv(tmp_path: Path, name: str, content: str = "x,y\n1,2\n") -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_finalize_manifest_sub_question_mode_filters_current_file(tmp_path):
    """小题模式下只锁定当前小题的 q{i}.csv，不含累积的旧路径。"""
    q1 = _write_csv(tmp_path, "q1.csv")
    q2 = _write_csv(tmp_path, "q2.csv")
    control = ControlState(
        sub_questions=["问题1", "问题2"],
        current_sub_question_index=0,
        coder_run_count=3,
    )
    artifacts = ArtifactBundle(
        result_paths=[q1, q2],
        figure_paths=["figs/roadmap.png"],
    )
    control = _finalize_authoritative_result(
        _state(tmp_path), control, artifacts, status="passed"
    )

    assert len(control.results_manifest) == 1
    entry = control.results_manifest[0]
    assert entry.index == 0
    assert entry.result_paths == [q1]
    assert entry.run_id == "run_2"
    assert entry.status == "passed"
    assert entry.metrics["q1.csv"]["rows"] == 1


def test_finalize_manifest_replaces_existing_index(tmp_path):
    """同一小题重新验收时应替换旧条目而非追加。"""
    p = _write_csv(tmp_path, "q1.csv")
    control = ControlState(sub_questions=["问题1"], current_sub_question_index=0)
    artifacts = ArtifactBundle(result_paths=[p])
    control = _finalize_authoritative_result(_state(tmp_path), control, artifacts)
    control = _finalize_authoritative_result(
        _state(tmp_path), control, artifacts, status="degraded"
    )
    assert len(control.results_manifest) == 1
    assert control.results_manifest[0].status == "degraded"


def test_finalize_manifest_single_question_mode(tmp_path):
    """单题模式（无小题清单）以 index=0 锁定全部结果路径。"""
    out = _write_csv(tmp_path, "output.csv")
    control = ControlState(coder_run_count=5)
    artifacts = ArtifactBundle(result_paths=[out])
    control = _finalize_authoritative_result(
        _state(tmp_path), control, artifacts, status="degraded"
    )
    assert len(control.results_manifest) == 1
    entry = control.results_manifest[0]
    assert entry.index == 0
    assert entry.result_paths == [out]
    assert entry.run_id == "run_4"


def test_manifest_serializes_through_control_state():
    """manifest 应能随 ControlState 正常序列化（LangGraph checkpoint 兼容）。"""
    control = ControlState(
        sub_questions=["q1"],
        results_manifest=[
            AuthoritativeResult(index=0, title="q1", result_paths=["out/q1.csv"])
        ],
    )
    dumped = control.model_dump(mode="json")
    restored = ControlState.model_validate(dumped)
    assert restored.results_manifest[0].result_paths == ["out/q1.csv"]
    assert restored.results_manifest[0].index == 0


def test_truncate_manifest_keeps_before_rollback_point():
    """跨小题回滚截断：保留 index < rollback_idx，被重做条目一并丢弃。"""
    control = ControlState(
        results_manifest=[
            AuthoritativeResult(index=0, title="q1"),
            AuthoritativeResult(index=1, title="q2"),
            AuthoritativeResult(index=2, title="q3"),
        ]
    )
    control = _truncate_manifest(control, keep_up_to=1)
    assert [e.index for e in control.results_manifest] == [0]
