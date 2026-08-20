"""V17 运行过程记录测试：落盘、报告生成、节点埋点。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config.settings import AppSettings
from modeling_assistant.recording.process_log import (
    ProcessLogEntry,
    archive_prompt,
    build_process_report,
    load_process_log,
    make_entry,
    summarize_usage,
    write_log_line,
    write_process_report,
)
from modeling_assistant.schemas.state import (
    append_reducer,
    ControlState,
)


def test_make_entry_and_jsonl_roundtrip(tmp_path):
    """记录应能落盘 JSONL 并原样读回（崩溃后可恢复）。"""
    control = ControlState(phase="model_brainstorming")
    entry = make_entry(
        control,
        "mathematician",
        "plans_generated",
        "第 1 轮发散：生成 3 个候选方案",
        {"debate_round": 1, "plans": [{"id": "plan_1", "title": "方案A"}]},
        seq=1,
    )
    path = write_log_line(tmp_path, entry)
    assert path is not None and path.exists()
    restored = load_process_log(tmp_path)
    assert len(restored) == 1
    assert restored[0].stage == "mathematician"
    assert restored[0].details["plans"][0]["id"] == "plan_1"


def test_append_reducer_accumulates_process_log():
    """process_log 在 GraphState 中应追加累积。"""
    e1 = make_entry(ControlState(), "problem", "run_started", "启动")
    e2 = make_entry(ControlState(), "mathematician", "plans_generated", "发散")
    merged = append_reducer([], [e1])
    merged = append_reducer(merged, [e2])
    assert [e.stage for e in merged] == ["problem", "mathematician"]


def test_archive_prompt_writes_file(tmp_path):
    """建模阶段 prompt 应存档到 logs/prompts/。"""
    path = archive_prompt(tmp_path, "mathematician", "round1", "你是 Mathematician…")
    assert path is not None and path.exists()
    assert "mathematician_round1" in path.name
    assert path.read_text(encoding="utf-8").startswith("你是 Mathematician")


def _modeling_entries() -> list[ProcessLogEntry]:
    entries = [
        make_entry(
            ControlState(phase="model_brainstorming"),
            "mathematician",
            "plans_generated",
            "第 1 轮发散：生成 2 个候选方案",
            {
                "debate_round": 1,
                "plans": [
                    {"id": "plan_1", "title": "几何光路逆映射", "innovation_score": 78, "feasibility_score": 85},
                    {"id": "plan_2", "title": "纯数值拟合", "innovation_score": 55, "feasibility_score": 70},
                ],
            },
            seq=1,
        ),
        make_entry(
            ControlState(phase="plan_scored"),
            "realist",
            "plans_scored",
            "剪枝评分：选中 plan_1",
            {
                "evaluations": [
                    {"plan_id": "plan_1", "innovation_score": 78, "feasibility_score": 85, "verdict": "keep"},
                    {"plan_id": "plan_2", "innovation_score": 55, "feasibility_score": 70, "verdict": "reject"},
                ],
                "selected_plan_id": "plan_1",
                "innovation_threshold": 60,
                "feasibility_threshold": 60,
            },
            seq=2,
        ),
        make_entry(
            ControlState(phase="dynamic_ltm_committed"),
            "clarifier",
            "ltm_committed",
            "提交动态 LTM v1.0",
            {
                "version": "v1.0",
                "objective": "最小化 Chamfer 距离",
                "assumptions": ["A4 纸面为理想平面"],
                "equations": ["L = CD + λ(1-SSIM)"],
            },
            seq=3,
        ),
    ]
    return entries


def test_build_process_report_contains_modeling_details():
    """报告应包含时间线与建模阶段详细（方案/评分/LTM），供重新评估。"""
    report = build_process_report(
        _modeling_entries(), meta={"LLM 模型": "deepseek-v4-flash"}
    )
    assert "## 一、时间线" in report
    assert "## 二、建模阶段详细（可据此重新评估方案）" in report
    assert "几何光路逆映射" in report
    assert "78" in report
    assert "reject" in report
    assert "最小化 Chamfer 距离" in report
    assert "A4 纸面为理想平面" in report
    assert "deepseek-v4-flash" in report


def test_write_process_report_creates_file(tmp_path):
    """报告应写入 outputs/logs/process_report.md。"""
    path = write_process_report(tmp_path, _modeling_entries())
    assert path.exists()
    assert path.name == "process_report.md"


def _runtime(tmp_path: Path) -> AgentRuntime:
    return AgentRuntime.from_settings(
        AppSettings(output_dir=tmp_path, api_key_env="MISSING_KEY_FOR_TEST")
    )


def test_final_reviewer_logs_stage_final_reviewer(tmp_path):
    """回归：final_reviewer 埋点必须标 stage=final_reviewer（防错位到 arbiter）。"""
    from modeling_assistant.agents.nodes import final_reviewer_node
    from modeling_assistant.schemas.state import (
        ArtifactBundle,
        DynamicLTM,
        GraphState,
        StaticLTM,
    )

    runtime = _runtime(tmp_path)
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(objective="目标"),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    result = final_reviewer_node(state, runtime=runtime)
    entries = result.get("process_log") or []
    assert entries, "final_reviewer 应产出 process_log 条目"
    assert entries[-1].stage == "final_reviewer"
    assert entries[-1].event == "paper_reviewed"


def test_searcher_node_records_literature_titles_and_authors(tmp_path):
    """检索结果（标题/作者）应写入 process_log，并同步 static_ltm.literature。"""
    from modeling_assistant.agents.nodes import searcher_node
    from modeling_assistant.schemas.state import (
        ArtifactBundle,
        DynamicLTM,
        GraphState,
        StaticLTM,
    )

    runtime = AgentRuntime.from_settings(
        AppSettings(output_dir=tmp_path, api_key_env="NONE", search_enabled=False)
    )
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="测试题"),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    result = searcher_node(state, runtime=runtime)

    entries = result.get("process_log") or []
    assert entries and entries[-1].stage == "searcher"
    assert entries[-1].event == "literature_retrieved"
    detail = entries[-1].details
    assert detail["count"] == len(result["static_ltm"].literature) >= 1
    assert all("title" in item and "authors" in item for item in detail["literature"])
    # 落盘 JSONL 可读回
    restored = load_process_log(tmp_path)
    assert any(e.stage == "searcher" for e in restored)


def test_prompts_note_literature_as_reference_only():
    """writer/mathematician 指令应写明「文献可作为启发和参考」。"""
    from modeling_assistant.prompts import PromptCatalog, PromptContext
    from modeling_assistant.schemas.state import DynamicLTM, StaticLTM

    for name in ("writer", "mathematician"):
        prompt = PromptCatalog().render(
            name,
            PromptContext(
                static_ltm=StaticLTM(raw_problem="测试题"),
                dynamic_ltm=DynamicLTM(objective="目标"),
                extra={"integrity_warnings": "无"},
            ),
        )
        assert "文献可作为启发和参考" in prompt


def test_summarize_usage_aggregates():
    """usage 汇总：调用次数、输入/输出/缓存、输出 top 节点。"""
    entries = [
        {"prompt_name": "writer", "prompt_tokens": 1000, "completion_tokens": 5000},
        {"prompt_name": "writer", "prompt_tokens": 1000, "completion_tokens": 3000},
        {"prompt_name": "coder", "prompt_tokens": 800, "completion_tokens": 7000},
    ]
    summary = summarize_usage(entries)
    assert summary["calls"] == 3
    assert summary["prompt_tokens"] == 2800
    assert summary["completion_tokens"] == 15000
    assert summary["total_tokens"] == 17800
    assert summary["cache_hit_tokens"] == 0
    assert summary["cache_hit_rate"] == 0.0
    top = summary["top_nodes_by_completion"]
    assert top[0]["node"] == "writer"
    assert top[0]["completion_tokens"] == 8000
    assert top[1]["node"] == "coder"
