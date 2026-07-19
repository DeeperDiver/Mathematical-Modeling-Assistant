"""端到端运行 real_test4（2024 华中杯 A 题：城市绿色物流配送调度），启用详细日志并捕获所有产物。

用法：
    python run_real_test4.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# V11.2 修复（Bug 4）：主进程也禁用 .pyc 写入，避免 TRAE Sandbox 拦截
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _json_default(obj):
    """处理 datetime 等不可直接序列化的对象。"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# 预先创建输出目录，避免 FileHandler 初始化失败
OUTPUT_DIR = "outputs4_v11_4"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from langgraph.types import Command

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config import load_settings
from modeling_assistant.graph.builder import build_graph
from modeling_assistant.schemas.state import ArtifactBundle, ControlState, DynamicLTM, StaticLTM

logger = logging.getLogger("e2e_test4")


def main() -> None:
    problem_text = Path("real_test4/problem.txt").read_text(encoding="utf-8")
    logger.info("读取问题文本: %d 字符", len(problem_text))

    settings = load_settings(".env")
    settings.output_dir = Path(OUTPUT_DIR)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Settings: model=%s, search_enabled=%s, output_dir=%s, max_debate=%d",
        settings.llm_model, settings.search_enabled, settings.output_dir, settings.max_debate_rounds,
    )
    runtime = AgentRuntime.from_settings(settings)
    app = build_graph(runtime=runtime)

    # real_test4 有 4 个附件（订单信息/距离矩阵/客户坐标信息/时间窗）
    attachment_paths = [
        str(Path("real_test4/附件/订单信息.xlsx").resolve()),
        str(Path("real_test4/附件/距离矩阵.xlsx").resolve()),
        str(Path("real_test4/附件/客户坐标信息.xlsx").resolve()),
        str(Path("real_test4/附件/时间窗.xlsx").resolve()),
    ]
    logger.info("数据附件: %s", attachment_paths)

    initial_state = {
        "static_ltm": StaticLTM(raw_problem=problem_text, data_attachments=attachment_paths),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    config = {"configurable": {"thread_id": "e2e-real-test4"}}

    current_input = initial_state
    final_state = None
    step = 0
    auto_approve = True

    while True:
        step += 1
        logger.info("=" * 60)
        logger.info("Graph invoke step %d", step)
        logger.info("=" * 60)
        try:
            result = app.invoke(current_input, config)
        except Exception as exc:
            logger.exception("Graph invoke 失败: %s", exc)
            return

        control = result.get("control", ControlState())
        logger.info(
            "Step %d 完成: phase=%s, hitl_required=%s, hitl_stage=%s, "
            "modeling_budget=%d/%d, coder_error=%d, trigger_clarifier=%s",
            step, control.phase, control.hitl_required, control.hitl_stage,
            control.modeling_revision_count, control.modeling_revision_budget,
            control.coder_error_count, control.trigger_clarifier_revision,
        )

        if not control.hitl_required:
            final_state = result
            break

        if auto_approve:
            logger.info("auto-approve: 跳过 HITL (%s)", control.hitl_stage)
            current_input = Command(resume="approve")
            continue

        break

    if final_state is None:
        logger.error("未获得 final_state")
        return

    empirical = final_state.get("empirical")
    summary = {
        "phase": final_state["control"].phase,
        "archive_versions": [s.version for s in final_state["ltm_archive"]],
        "archive_summaries": [
            {
                "version": s.version,
                "commit_summary": s.commit_summary,
                "reason": s.reason,
                "objective": s.dynamic_ltm.objective,
            }
            for s in final_state["ltm_archive"]
        ],
        "final_objective": final_state["dynamic_ltm"].objective,
        "final_assumptions": final_state["dynamic_ltm"].assumptions,
        "final_equations": final_state["dynamic_ltm"].equations,
        "final_nomenclature": final_state["dynamic_ltm"].nomenclature,
        "artifacts": final_state["artifacts"].model_dump(),
        "empirical_findings": [f.model_dump() for f in empirical.findings] if empirical else [],
        "empirical_refuted": empirical.refuted_assumptions if empirical else [],
        "empirical_open_questions": empirical.open_questions if empirical else [],
        "prompt_audit_keys": sorted(final_state.get("prompt_audit", {}).keys()),
        "control_final": final_state["control"].model_dump(),
    }
    summary_path = settings.output_dir / "final_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    logger.info("最终摘要已写入 %s", summary_path)
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
