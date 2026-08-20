"""从 process_log.jsonl 重建运行过程报告。

用途：主流程异常退出（如 PTY 崩溃）导致 CLI 未生成报告时，
用本脚本从已落盘的 JSONL 记录重建 Markdown 报告。

用法：
    python scripts/build_run_report.py --output-dir outputs \
        --problem "2026 华中杯 B 题" --llm-model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modeling_assistant.recording.process_log import (
    load_process_log,
    write_process_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="重建运行过程报告")
    parser.add_argument(
        "--output-dir", default="outputs", help="输出目录（含 logs/process_log.jsonl）"
    )
    parser.add_argument("--problem", default="", help="题面摘要（可选，进报告头）")
    parser.add_argument("--llm-model", default="", help="LLM 模型（可选，进报告头）")
    args = parser.parse_args()

    out = Path(args.output_dir)
    entries = load_process_log(out)
    meta: dict[str, str] = {}
    if args.problem:
        meta["问题"] = args.problem[:150]
    if args.llm_model:
        meta["LLM 模型"] = args.llm_model
    path = write_process_report(out, entries, meta=meta)
    print(f"已从 {len(entries)} 条记录重建报告：{path}")


if __name__ == "__main__":
    main()
