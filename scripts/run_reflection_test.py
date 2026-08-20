"""运行 test/ 下 2026 华中杯 B 题「反射的艺术」端到端测试。

HITL 中断由外部输入执行（PTY 交互）：每次中断打印阶段信息后等待 stdin 决策。
模型由环境变量 MODELING_ASSISTANT_LLM_MODEL 控制（本测试用 deepseek-v4-flash）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def main() -> None:
    problem_text = (ROOT / "test" / "problem.txt").read_text(encoding="utf-8")
    attach_dir = ROOT / "test" / "附件"
    attachments = [str(p) for p in sorted(attach_dir.iterdir()) if p.is_file()]

    settings = load_settings()
    print(f"模型: {settings.llm_model}", flush=True)
    print(f"题面字符: {len(problem_text)}", flush=True)
    print(f"附件: {attachments}", flush=True)

    settings.output_dir = ROOT / "outputs" / "test_reflection"
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = AgentRuntime.from_settings(settings)
    app = build_graph(runtime=runtime)

    state = {
        "static_ltm": StaticLTM(raw_problem=problem_text, data_attachments=attachments),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "exemplars": ExemplarContext(),
        "prompt_audit": {},
    }
    config = {"configurable": {"thread_id": "reflection-art"}}

    current = state
    final = None
    step = 0
    while True:
        step += 1
        print(f"\n===== Graph invoke step {step} =====", flush=True)
        result = app.invoke(current, config)
        control = result.get("control", ControlState())
        interrupts = result.get("__interrupt__")
        print(
            f"phase={control.phase} hitl={control.hitl_required} "
            f"stage={control.hitl_stage} budget={control.modeling_revision_count}/"
            f"{control.modeling_revision_budget} coder_error={control.coder_error_count}",
            flush=True,
        )
        if not control.hitl_required and not interrupts:
            final = result
            break

        if interrupts:
            # LangGraph interrupt（子问题拆分确认 / HITL 节点）
            print("=" * 60, flush=True)
            for it in interrupts:
                value = getattr(it, "value", it)
                if isinstance(value, dict):
                    stage = value.get("stage", "interrupt")
                    print(f"[INTERRUPT {stage}] {value.get('message', '')}", flush=True)
                    print(f"  提示: {value.get('hint', '')}", flush=True)
                    for key in ("sub_questions", "dynamic_ltm", "control_summary", "artifacts_summary"):
                        if key in value:
                            print(f"  {key}: {str(value[key])[:600]}", flush=True)
                else:
                    print(f"[INTERRUPT] {value}", flush=True)
            decision = input(">>> 人类决策: ").strip()
            print(f"<<< 收到决策: {decision}", flush=True)
            current = Command(resume=decision)
            continue

        stage = control.hitl_stage
        print("=" * 60, flush=True)
        print(f"[HITL {stage}] 需要人类决策", flush=True)
        if stage == "architecture":
            dltm = result.get("dynamic_ltm")
            print("目标:", (getattr(dltm, "objective", "") or "")[:150], flush=True)
            print(
                "假设数:", len(getattr(dltm, "assumptions", [])),
                "公式数:", len(getattr(dltm, "equations", [])),
                flush=True,
            )
            print(
                "评分 创新:", control.innovation_score,
                "可行:", control.feasibility_score,
                flush=True,
            )
            print(
                "提示: approve 放行 / rollback v1.0 回滚 / approve score 80 反馈",
                flush=True,
            )
        elif stage == "arbitration":
            print("提示: approve 接受回滚 / reject 拒绝继续", flush=True)
        elif stage == "modeling":
            print(
                "预算耗尽 评分 创新:", control.innovation_score,
                "可行:", control.feasibility_score,
                flush=True,
            )
            print(
                "提示: accept 接受失败 / retry 回 Architect 重试 / redirect <方向> 重新发散",
                flush=True,
            )
        elif stage == "final":
            arts = result.get("artifacts")
            print("图:", getattr(arts, "figure_paths", []), flush=True)
            print("结果:", getattr(arts, "result_paths", []), flush=True)
            print("LaTeX:", getattr(arts, "latex_path", None), flush=True)
            print("提示: approve 完成 / retry 重打磨 / approve score 80 完成并反馈", flush=True)

        decision = input(">>> 人类决策: ").strip()
        print(f"<<< 收到决策: {decision}", flush=True)
        current = Command(resume=decision)

    print("\n===== 流程完成 =====", flush=True)
    print("phase:", final["control"].phase, flush=True)
    print("archive:", [s.version for s in final["ltm_archive"]], flush=True)
    print("objective:", (final["dynamic_ltm"].objective or "")[:120], flush=True)
    arts = final["artifacts"]
    print("figures:", arts.figure_paths, flush=True)
    print("results:", arts.result_paths, flush=True)
    print("latex:", arts.latex_path, flush=True)
    print("prompt_audit keys:", sorted(final.get("prompt_audit", {}).keys()), flush=True)


if __name__ == "__main__":
    main()
