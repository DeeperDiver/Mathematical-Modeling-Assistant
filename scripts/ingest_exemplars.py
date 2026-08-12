"""批量摄入优秀论文，生成 L1 卡片并聚合 L2 题型指南。

用法：
    python scripts/ingest_exemplars.py --input exemplars/raw --output exemplars
    python scripts/ingest_exemplars.py --input papers/2024_国赛_优秀论文.pdf --contest 国赛

约定：
- --input 可以是单个文件或目录（递归收集）。
- 若论文所在目录存在 problem.txt / 题目.txt，自动作为该论文的题面。
- 无 LLM API key 时自动降级为确定性卡片（章节启发式 + 关键词题型判定）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config import load_settings
from modeling_assistant.data.exemplar_ingest import (
    SUPPORTED_EXTENSIONS,
    aggregate_guides,
    find_problem_file,
    ingest_paper,
)
from modeling_assistant.data.exemplars import load_cards, save_card, save_guide


def collect_inputs(path: str | Path) -> list[Path]:
    """收集单个文件或目录下的所有支持文件。"""
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS or root.suffix.lower() == ".json" else []
    if root.is_dir():
        files: list[Path] = []
        for candidate in sorted(root.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in (
                SUPPORTED_EXTENSIONS | {".json"}
            ):
                if candidate.name in ("problem.txt", "题目.txt"):
                    continue
                files.append(candidate)
        return files
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="摄取优秀论文并生成表达知识库。")
    parser.add_argument("--input", required=True, help="论文文件或目录。")
    parser.add_argument("--output", default="exemplars", help="知识库根目录（默认 exemplars）。")
    parser.add_argument("--contest", default="", help="赛事语境（国赛/美赛/华中杯/...）。")
    parser.add_argument("--problem-type", default="", help="强制题型（覆盖自动判定）。")
    parser.add_argument("--problem-text", default="", help="题面文本（当目录内无 problem.txt 时使用）。")
    parser.add_argument("--min-occurrences", type=int, default=3, help="聚合共性所需最少卡片数（默认 3）。")
    parser.add_argument("--verbose", action="store_true", help="输出 INFO 日志。")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("ingest_exemplars")

    output_root = Path(args.output)
    cards_dir = output_root / "cards"
    guides_dir = output_root / "guides"

    settings = load_settings()
    runtime = AgentRuntime.from_settings(settings)
    files = collect_inputs(args.input)
    if not files:
        logger.error("未找到任何支持的文件（pdf/tex/md/txt/json）：%s", args.input)
        sys.exit(1)

    ingested: list[str] = []
    for path in files:
        problem_file = find_problem_file(path.parent)
        problem_text = (
            problem_file.read_text(encoding="utf-8", errors="replace")
            if problem_file
            else args.problem_text
        )
        card = ingest_paper(
            path,
            problem_text or "",
            runtime=runtime,
            contest=args.contest,
            problem_type=args.problem_type,
        )
        if card is not None:
            save_card(card, cards_dir)
            ingested.append(str(path))
            print(f"[card] {card.id} ({card.problem_type}) <- {path}")

    all_cards = load_cards(cards_dir)
    guides = aggregate_guides(
        all_cards, min_occurrences=args.min_occurrences, runtime=runtime
    )
    for guide in guides:
        save_guide(guide, guides_dir)
        print(
            f"[guide] {guide.problem_type}"
            + (f"/{guide.contest}" if guide.contest else "")
            + f" 共性章节={len(guide.common_structure)} 推荐图={len(guide.recommended_figures)}"
        )

    print(f"\n完成：新摄入 {len(ingested)} 篇，知识库现有 {len(all_cards)} 张卡片、{len(guides)} 份指南。")


if __name__ == "__main__":
    main()
