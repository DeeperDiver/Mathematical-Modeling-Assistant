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

    review = interrupt_data.get("assumption_review")
    if review:
        print("  假设分类：")
        print(f"    【全文】: {review.get('full', [])}")
        print(f"    【问题N】: {review.get('question', [])}")
        print(f"    【关键】: {review.get('critical', [])}")
        if review.get("unlabeled"):
            print(f"    未分类: {review.get('unlabeled', [])}")

    plan_pool = interrupt_data.get("plan_pool")
    if plan_pool:
        print("  方案池（实现路径，测试后定夺）：")
        for p in plan_pool:
            print(
                f"    - {p.get('id')} [{p.get('verdict', '')}] "
                f"{p.get('title', '')}: {(p.get('description') or '')[:120]}"
            )

    if "dynamic_ltm" in interrupt_data:
        ltm = interrupt_data["dynamic_ltm"]
        print(f"  目标: {ltm.get('objective', 'N/A')}")
        if not review:
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

    if "paper_review_report" in interrupt_data:
        report = interrupt_data["paper_review_report"] or {}
        print(f"  确定性验收: {'通过' if report.get('passed') else '未通过'}")
        for issue in (report.get("issues") or [])[:5]:
            print(f"    [硬错误] {issue[:120]}")
        llm = report.get("llm") or {}
        if llm:
            print(f"  LLM 审查: {llm.get('verdict', 'N/A')} — {llm.get('summary', '')[:120]}")
            for issue in (llm.get("issues") or [])[:5]:
                print(f"    [审查] {issue[:120]}")

    assumptions_section = interrupt_data.get("assumptions_section")
    if assumptions_section:
        print("  ---- 3_assumptions.tex ----")
        print("  " + assumptions_section[:2000].replace("\n", "\n  "))

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
    # V16 修复：配置根日志，让长流程的 LLM 调用/阶段切换可观测
    # （runtime 已用 logger.info 记录每次调用与返回字符数，缺 basicConfig 时不输出）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the Modeling Assistant graph skeleton.")
    parser.add_argument("--problem", required=False, help="Raw mathematical modeling problem text.")
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
    parser.add_argument(
        "--coder-external-mode",
        choices=["builtin", "codex", "human"],
        default=None,
        help=(
            "编程手模式（默认 human）：human=人工实现（人编写 solution.py 与可选 "
            "figures.py，Coder/Drawer 任务均由人执行）；builtin=内置 Coder；"
            "codex=调用本机 Codex CLI。"
        ),
    )
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all HITL checkpoints (no pause).")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="运行环境健康检查后退出（依赖、编译器、API Key）。",
    )
    args = parser.parse_args()

    if args.doctor:
        from modeling_assistant.doctor import print_report, run_doctor

        report = run_doctor()
        print_report(report)
        sys.exit(0 if report.ready else 1)

    if not args.problem:
        parser.error("--problem 是必填参数（--doctor 模式除外）。")

    overrides = {}
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    if args.api_base_url:
        overrides["api_base_url"] = args.api_base_url
    if args.output_dir:
        overrides["output_dir"] = Path(args.output_dir)
    if args.exemplars_dir:
        overrides["exemplars_dir"] = Path(args.exemplars_dir)
    if args.coder_external_mode:
        overrides["coder_external_mode"] = args.coder_external_mode
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

        # LangGraph 在节点中断时，invoke 返回的是"中断前"的 state，
        # 因此不能依赖节点内设置的 control.hitl_required 判断暂停；
        # 必须检查 __interrupt__ 载荷（stage/message/hint 都在里面）。
        interrupts = result.get("__interrupt__")
        if not interrupts:
            final_state = result
            break

        interrupt_value = (
            interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        )
        stage = (
            interrupt_value.get("stage")
            if isinstance(interrupt_value, dict)
            else str(interrupt_value)
        )

        if args.auto_approve:
            logger.info("auto-approve: 跳过 HITL 中断（%s），自动放行。", stage)
            if stage == "implementation_human":
                resume = "auto"
            elif stage == "sub_question_acceptance":
                resume = "pass"
            elif stage == "cross_sub_question":
                resume = "accept"
            else:
                resume = "approve"
            current_input = Command(resume=resume)
            continue

        # 显示中断信息：stage/message/hint 都来自节点 interrupt 载荷
        _print_interrupt_info(interrupt_value)
        if stage == "implementation_architecture":
            spec_md = interrupt_value.get("architecture_spec_md", "") or ""
            if spec_md:
                print("\n" + spec_md[:4000])
                print("-" * 60)
        elif stage == "implementation_human":
            print(f"  任务目录: {interrupt_value.get('task_dir', '')}")
            print("-" * 60)
        elif stage == "sub_question_split":
            print("  小题清单：")
            for i, q in enumerate(interrupt_value.get("sub_questions", []), start=1):
                print(f"    {i}. {q[:200]}")
            print("-" * 60)
        elif stage == "sub_question_acceptance":
            print(f"  当前小题: {interrupt_value.get('sub_question_text', '')[:300]}")
            print(f"  结果文件: {interrupt_value.get('result_paths', [])}")
            print(f"  图表: {interrupt_value.get('figure_paths', [])}")
            preview = interrupt_value.get("result_preview", "")
            if preview:
                print("\n  结果预览：\n" + preview[:2000])
            warnings = interrupt_value.get("review_warnings", "")
            if warnings:
                print("\n  机械校验提示：\n" + warnings[:1000])
            print("-" * 60)
        elif stage == "cross_sub_question":
            print(
                f"  目标小题: {interrupt_value.get('target_index', -1) + 1}"
                if interrupt_value.get("target_index", -1) >= 0
                else "  目标小题: 未知"
            )
            print(f"  可用快照: {interrupt_value.get('archive_versions', [])}")
            print("-" * 60)

        decision = _get_user_decision()
        current_input = Command(resume=decision)

    # ── 输出最终摘要 ──
    summary = {
        "phase": final_state["control"].phase,
        "archive_versions": [snapshot.version for snapshot in final_state["ltm_archive"]],
        "objective": final_state["dynamic_ltm"].objective,
        "artifacts": final_state["artifacts"].model_dump(),
        "prompt_audit_keys": sorted(final_state.get("prompt_audit", {}).keys()),
        "process_log_count": len(final_state.get("process_log", []) or []),
    }
    # V17：token usage 汇总（runtime 每次调用已落盘 usage.jsonl）
    usage_summary: dict | None = None
    try:
        from modeling_assistant.recording.process_log import summarize_usage

        usage_summary = summarize_usage(runtime.usage_log)
        summary["usage"] = usage_summary
    except Exception as exc:
        logger.warning("usage 汇总失败: %s", exc)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    # V17：生成运行过程报告（含建模阶段详细，供重新评估方案）
    try:
        from modeling_assistant.recording.process_log import write_process_report

        meta = {
            "问题": (args.problem or "")[:150],
            "LLM 模型": settings.llm_model,
            "输出目录": str(settings.output_dir),
            "结束阶段": final_state["control"].phase,
            "LLM 调用次数": str(usage_summary["calls"]) if usage_summary else "N/A",
            "输入 tokens": str(usage_summary["prompt_tokens"]) if usage_summary else "N/A",
            "输出 tokens": str(usage_summary["completion_tokens"]) if usage_summary else "N/A",
            "缓存命中": (
                f"{usage_summary['cache_hit_tokens']} / {usage_summary['prompt_tokens']}"
                f"（命中率 {usage_summary['cache_hit_rate']:.1%}）"
                if usage_summary
                else "N/A"
            ),
            "输出 top 节点": (
                "、".join(
                    f"{t['node']} {t['completion_tokens']}"
                    for t in usage_summary["top_nodes_by_completion"]
                )
                if usage_summary
                else "N/A"
            ),
        }

        report_path = write_process_report(
            settings.output_dir,
            final_state.get("process_log") or [],
            meta=meta,
        )
        print(f"\n运行过程报告: {report_path}")
    except Exception as exc:
        logger.warning("运行过程报告生成失败: %s", exc)


if __name__ == "__main__":
    main()
