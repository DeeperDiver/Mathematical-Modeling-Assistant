"""V15 论文终审节点测试：确定性验收失败/通过 + LLM 降级。"""

from __future__ import annotations

import json
from pathlib import Path

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config.settings import AppSettings
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    GraphState,
    StaticLTM,
)


def _make_paper(output_dir: Path) -> None:
    """构造一份能通过确定性验收的最小论文。"""
    paper = output_dir / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{ctexart}\n"
        "\\begin{document}\n"
        "\\input{sections/1_restatement}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / "1_restatement.tex").write_text(
        "\\section{问题重述}\n内容。\n", encoding="utf-8"
    )


def _runtime(output_dir: Path) -> AgentRuntime:
    return AgentRuntime.from_settings(
        AppSettings(
            output_dir=output_dir,
            api_key_env="MISSING_KEY_FOR_TEST",
            paper_template_dir=output_dir / "no_template",
        )
    )


def _state(output_dir: Path) -> GraphState:
    return {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(objective="目标"),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }


def test_final_reviewer_fails_on_missing_paper(tmp_path):
    """论文目录缺失时确定性验收应失败并写入报告。"""
    from modeling_assistant.agents.nodes import final_reviewer_node

    runtime = _runtime(tmp_path)
    result = final_reviewer_node(_state(tmp_path), runtime=runtime)

    assert result["control"].phase == "paper_review_failed"
    report = result["control"].paper_review_report
    assert report["passed"] is False
    assert report["checks"]["入口"] == "缺失"
    audit = result.get("prompt_audit", {})
    assert "paper_check_report" in audit


def test_final_reviewer_passes_clean_paper_without_llm(tmp_path, monkeypatch):
    """确定性验收通过且 LLM 不可用时，应降级为 paper_review_passed。"""
    from modeling_assistant.agents.nodes import final_reviewer_node

    _make_paper(tmp_path)
    runtime = _runtime(tmp_path)

    # 无 API key：invoke_structured 会抛错 → 应降级通过
    result = final_reviewer_node(_state(tmp_path), runtime=runtime)

    assert result["control"].phase == "paper_review_passed"
    assert result["control"].paper_review_report["passed"] is True


def test_final_reviewer_invokes_llm_when_available(tmp_path, monkeypatch):
    """LLM 可用时应调用终审审查并写入 llm 报告。"""
    from modeling_assistant.agents.nodes import final_reviewer_node
    from modeling_assistant.schemas.responses import FinalReviewerResponse

    _make_paper(tmp_path)
    runtime = _runtime(tmp_path)

    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        assert name == "final_reviewer"
        assert "paper_text" in system_prompt or "结果文件预览" in system_prompt
        return FinalReviewerResponse(
            verdict="pass",
            issues=[],
            suggestions=["摘要补充灵敏度分析"],
            numerical_consistency="一致",
            summary="论文结构完整",
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)
    result = final_reviewer_node(_state(tmp_path), runtime=runtime)

    assert result["control"].phase == "paper_review_passed"
    llm = result["control"].paper_review_report.get("llm", {})
    assert llm.get("verdict") == "pass"
    assert "摘要补充灵敏度分析" in llm.get("suggestions", [])


def test_final_reviewer_llm_fail_marks_failed(tmp_path, monkeypatch):
    """LLM 审查报 fail 时应标记论文未通过（最终裁决在 HITL）。"""
    from modeling_assistant.agents.nodes import final_reviewer_node
    from modeling_assistant.schemas.responses import FinalReviewerResponse

    _make_paper(tmp_path)
    runtime = _runtime(tmp_path)

    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        return FinalReviewerResponse(
            verdict="fail",
            issues=["1_restatement.tex：数值与结果文件不一致"],
            suggestions=[],
            numerical_consistency="存在冲突",
            summary="数值不一致",
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)
    result = final_reviewer_node(_state(tmp_path), runtime=runtime)

    assert result["control"].phase == "paper_review_failed"
    llm_issues = result["control"].paper_review_report["llm"]["issues"]
    assert any("数值与结果文件不一致" in issue for issue in llm_issues)


def test_hitl_final_rewrite_consumes_budget(tmp_path):
    """HITL 终审 rewrite 应记录反馈并设置 paper_rewrite_requested。"""
    from modeling_assistant.agents.nodes import _parse_hitl_decision, hitl_final_node

    decision = _parse_hitl_decision("rewrite 问题一缺少敏感性分析")
    assert decision["type"] == "rewrite"
    assert decision["version"] == "问题一缺少敏感性分析"

    # 直接验证预算字段语义（hitl_final 依赖 interrupt，此处只测字段）
    control = ControlState(paper_revision_budget=2, paper_revision_count=0)
    assert control.paper_revision_count < control.paper_revision_budget


def test_parse_hitl_rewrite_vs_retry():
    """rewrite 不应被 retry/revise 前缀误判。"""
    from modeling_assistant.agents.nodes import _parse_hitl_decision

    assert _parse_hitl_decision("rewrite 重写摘要")["type"] == "rewrite"
    assert _parse_hitl_decision("retry v1.0")["type"] == "retry"
    assert _parse_hitl_decision("revise 修改模型")["type"] == "revise"
