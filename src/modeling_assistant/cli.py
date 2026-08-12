from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from langgraph.types import Command

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config import load_settings
from modeling_assistant.graph.builder import build_graph
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    ExemplarContext,
    StaticLTM,
)

logger = logging.getLogger(__name__)


def _print_interrupt_info(interrupt_data: dict) -> None:
    """打印中断信息给用户。"""
    print("\n" + "=" * 60)
    print(f"  [HITL] {interrupt_data.get('stage', 'unknown').upper()} 阶段 — 需要人类决策")
    print("=" * 60)
    print(f"  {interrupt_data.get('message', '')}")
    print(f"  {interrupt_data.get('hint', '')}")
    print("-" * 60)

    if "dynamic_ltm" in interrupt_data:
        ltm = interrupt_data["dynamic_ltm"]
        print(f"  目标: {ltm.get('objective', 'N/A')}")
        print(f"  假设: {ltm.get('assumptions', [])}")
        print(f"  公式: {ltm.get('equations', [])}")

    if "artifacts_summary" in interrupt_data:
        arts = interrupt_data["artifacts_summary"]
        if "figure_paths" in arts:
            print(f"  图表: {arts.get('figure_paths', [])}")
        print(f"  结果: {arts.get('result_paths', [])}")
        if "latex_path" in arts:
            print(f"  LaTeX: {arts.get('latex_path', 'N/A')}")
        if "has_backup_results" in arts:
            print(f"  有备份结果: {arts.get('has_backup_results', 'N/A')}")

    if "control_summary" in interrupt_data:
        ctrl = interrupt_data["control_summary"]
        print(f"  方案: {ctrl.get('selected_plan_id', 'N/A')}")
        if "innovation_score" in ctrl:
            print(f"  创新分: {ctrl.get('innovation_score', 'N/A')}, 可行性分: {ctrl.get('feasibility_score', 'N/A')}")
        if "budget_used" in ctrl:
            print(f"  预算: {ctrl.get('budget_used', 'N/A')}/{ctrl.get('budget_limit', 'N/A')}")
            print(f"  Meta决策: {ctrl.get('meta_decision', 'N/A')}")
            print(f"  方向提示: {ctrl.get('meta_direction_hint', 'N/A')}")

    print("-" * 60)


def _get_user_decision() -> str:
    """获取用户输入。"""
    try:
        return input("  >>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  已取消。")
        sys.exit(0)


def _collect_attachment_paths(paths: list[str]) -> list[str]:
    """收集附件路径：文件直接保留，目录递归收集其中文件。"""
    collected: list[str] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    collected.append(str(child.resolve()))
        elif path.is_file():
            collected.append(str(path.resolve()))
        else:
            logger.warning("附件路径不存在，已忽略: %s", path_str)
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Modeling Assistant graph skeleton.")
    parser.add_argument("--problem", required=True, help="Raw mathematical modeling problem text.")
    parser.add_argument(
        "--data-attachment",
        action="append",
        default=[],
        dest="data_attachments",
        help="数据附件路径（文件或目录），可多次指定。",
    )
    parser.add_argument("--env-file", default=".env", help="Path to optional .env configuration file.")
    parser.add_argument("--llm-model", default=None, help="Override MODELING_ASSISTANT_LLM_MODEL.")
    parser.add_argument("--api-base-url", default=None, help="Override MODELING_ASSISTANT_API_BASE_URL.")
    parser.add_argument("--output-dir", default=None, help="Override MODELING_ASSISTANT_OUTPUT_DIR.")
    parser.add_argument(
        "--exemplars-dir",
        default=None,
        help="Override MODELING_ASSISTANT_EXEMPLARS_DIR (优秀论文知识库目录).",
    )
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all HITL checkpoints (no pause).")
    args = parser.parse_args()

    overrides = {}
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    if args.api_base_url:
        overrides["api_base_url"] = args.api_base_url
    if args.output_dir:
        overrides["output_dir"] = Path(args.output_dir)
    if args.exemplars_dir:
        overrides["exemplars_dir"] = Path(args.exemplars_dir)
    settings = load_settings(args.env_file, **overrides)
    runtime = AgentRuntime.from_settings(settings)
    app = build_graph(runtime=runtime)

    attachment_paths = _collect_attachment_paths(args.data_attachments)
    initial_state = {
        "static_ltm": StaticLTM(raw_problem=args.problem, data_attachments=attachment_paths),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "exemplars": ExemplarContext(),
        "prompt_audit": {},
    }

    config = {"configurable": {"thread_id": "cli-session"}}

    # ── 主循环：运行图，处理 HITL 中断 ──
    current_input: dict | Command = initial_state
    final_state: dict | None = None

    while True:
        result = app.invoke(current_input, config)
        control = result.get("control", ControlState())

        if not control.hitl_required:
            final_state = result
            break

        if args.auto_approve:
            logger.info("auto-approve: 跳过 HITL 中断，自动放行。")
            current_input = Command(resume="approve")
            continue

        # 显示中断信息
        if control.hitl_stage == "architecture":
            _print_interrupt_info({
                "stage": "architecture",
                "message": "请审核当前建模方案。",
                "hint": "输入 'approve' 放行进入架构设计，或 'rollback <version>' 回滚到指定版本。",
                "dynamic_ltm": result.get("dynamic_ltm", DynamicLTM()).model_dump(),
                "control_summary": {
                    "phase": control.phase,
                    "selected_plan_id": control.selected_plan_id,
                    "innovation_score": control.innovation_score,
                    "feasibility_score": control.feasibility_score,
                },
            })
        elif control.hitl_stage == "final":
            _print_interrupt_info({
                "stage": "final",
                "message": "请审核最终论文。",
                "hint": "输入 'approve' 完成流程，或 'retry' 回到建模阶段重新打磨。",
                "artifacts_summary": result.get("artifacts", ArtifactBundle()).model_dump(),
            })
        elif control.hitl_stage == "arbitration":
            _print_interrupt_info({
                "stage": "arbitration",
                "message": f"Arbiter 建议回滚到版本 {control.rollback_to_version}。",
                "hint": "输入 'approve' 接受回滚，或 'reject' 拒绝回滚继续进入 Clarifier。",
                "control_summary": {
                    "phase": control.phase,
                    "debate_round": control.debate_round,
                    "selected_plan_id": control.selected_plan_id,
                    "innovation_score": control.innovation_score,
                    "feasibility_score": control.feasibility_score,
                },
            })
        elif control.hitl_stage == "modeling":
            artifacts = result.get("artifacts", ArtifactBundle())
            _print_interrupt_info({
                "stage": "modeling",
                "message": (
                    f"建模预算已耗尽（{control.modeling_revision_count}/{control.modeling_revision_budget}）。"
                    "系统多次尝试未能产出通过验证的结果，请人类决断下一步。"
                ),
                "hint": (
                    "输入 'accept' 接受失败并产出'待验证'论文；"
                    "输入 'retry' 重置预算并回到 Architect 重试当前方案；"
                    "输入 'redirect <方向提示>' 重置预算并回到 Mathematician 重新发散。"
                ),
                "control_summary": {
                    "phase": control.phase,
                    "budget_used": control.modeling_revision_count,
                    "budget_limit": control.modeling_revision_budget,
                    "selected_plan_id": control.selected_plan_id,
                    "meta_decision": control.meta_decision,
                    "meta_direction_hint": control.meta_direction_hint,
                },
                "artifacts_summary": {
                    "result_paths": getattr(artifacts, "result_paths", []),
                    "has_backup_results": bool(getattr(artifacts, "result_paths", None)),
                },
            })
        else:
            logger.info("未知 HITL 阶段: %s，自动放行。", control.hitl_stage)
            current_input = Command(resume="approve")
            continue

        decision = _get_user_decision()
        current_input = Command(resume=decision)

    # ── 输出最终摘要 ──
    summary = {
        "phase": final_state["control"].phase,
        "archive_versions": [snapshot.version for snapshot in final_state["ltm_archive"]],
        "objective": final_state["dynamic_ltm"].objective,
        "artifacts": final_state["artifacts"].model_dump(),
        "prompt_audit_keys": sorted(final_state.get("prompt_audit", {}).keys()),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
