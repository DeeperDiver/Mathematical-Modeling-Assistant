"""行文技艺深加工：为优秀论文卡片提炼六大行文技艺（Writing Craft）。

用法：
    python scripts/ingest_craft.py --cards 2023_B_B477,2023_B_B226,2020_C_C109
    python scripts/ingest_craft.py --cards-dir exemplars/cards --output exemplars

质量优先：默认使用 deepseek-v4-pro（可用 --model 或环境变量
MODELING_ASSISTANT_LLM_MODEL 覆盖）；单次调用 180s 超时、重试 2 次，
仍失败则跳过该篇（不降级为低质占位）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modeling_assistant.agents.runtime import AgentRuntime, _extract_json
from modeling_assistant.config import load_settings
from modeling_assistant.schemas.craft import WritingCraft

logger = logging.getLogger("ingest_craft")


def _read_paper_text(card: dict) -> str:
    """读取论文全文：优先 OCR/文本缓存，其次 tex/md/txt 源码。"""
    source = card.get("source_path", "")
    if not source:
        return ""
    src = Path(source)
    cache = src.with_suffix(src.suffix + ".ocr.txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    if src.suffix.lower() in (".tex", ".md", ".txt"):
        try:
            return src.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    return ""


def extract_craft(card: dict, runtime: AgentRuntime) -> WritingCraft | None:
    card_id = card["id"]
    paper_text = _read_paper_text(card)
    if len(paper_text.strip()) < 1000:
        logger.warning("跳过 %s：论文文本不足（%d 字符）", card_id, len(paper_text.strip()))
        return None

    template_path = ROOT / "src" / "modeling_assistant" / "prompts" / "templates" / "exemplar_craft_ingest.md"
    template = template_path.read_text(encoding="utf-8")
    system_prompt = template.format(
        paper_title=card.get("title", card_id),
        problem_type=card.get("problem_type", ""),
        paper_text=paper_text[:30000],
        card_id=card_id,
    )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            # 直接调用：由 runtime 内部 httpx 300s 超时 + openai 重试兜底，
            # 避免线程池包装造成的超时计数与实际完成不同步。
            raw = runtime.invoke("exemplar_craft_ingest", {}, system_prompt)
            data = json.loads(_extract_json(raw))
            data["card_id"] = card_id
            return WritingCraft.model_validate(data)
        except Exception as exc:
            last_exc = exc
            logger.warning("提炼失败 %s（attempt %d/3）: %s", card_id, attempt + 1, exc)
    logger.error("提炼最终失败 %s: %s", card_id, last_exc)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="行文技艺深加工。")
    parser.add_argument("--cards", default="", help="逗号分隔的卡片 id 列表。")
    parser.add_argument(
        "--cards-file",
        default="",
        help="卡片 id 列表文件（每行一个，UTF-8；支持中文 id）。",
    )
    parser.add_argument("--cards-dir", default="exemplars/cards", help="卡片目录。")
    parser.add_argument("--output", default="exemplars", help="知识库根目录。")
    parser.add_argument("--model", default="", help="覆盖 LLM 模型（默认 deepseek-v4-pro）。")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="聚合生成题型级行文技艺指南（exemplars/craft_guides/）。",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.model:
        import os

        os.environ["MODELING_ASSISTANT_LLM_MODEL"] = args.model
    runtime = AgentRuntime.from_settings(load_settings())
    logger.info("深加工模型: %s", runtime.settings.llm_model)

    cards_dir = Path(args.cards_dir)
    if args.cards:
        ids = [x.strip() for x in args.cards.split(",") if x.strip()]
        paths = [cards_dir / f"{cid}.json" for cid in ids]
    elif args.cards_file:
        ids = [
            line.strip()
            for line in Path(args.cards_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        paths = [cards_dir / f"{cid}.json" for cid in ids]
    else:
        paths = sorted(cards_dir.glob("*.json"))

    craft_dir = Path(args.output) / "craft"
    craft_dir.mkdir(parents=True, exist_ok=True)

    done, skipped, failed = 0, 0, 0
    for path in paths:
        if not path.exists():
            logger.warning("卡片不存在: %s", path)
            failed += 1
            continue
        card = json.loads(path.read_text(encoding="utf-8"))
        card_id = card["id"]
        out_path = craft_dir / f"{card_id}.json"
        if out_path.exists():
            skipped += 1
            print(f"[skip] {card_id}")
            continue
        t0 = time.time()
        craft = extract_craft(card, runtime)
        if craft is None:
            failed += 1
            print(f"[fail] {card_id} ({time.time()-t0:.0f}s)")
            continue
        out_path.write_text(craft.model_dump_json(indent=2), encoding="utf-8")
        done += 1
        n_der = len(craft.derivation)
        n_alg = len(craft.algorithm)
        n_int = len(craft.interpretation)
        n_wri = len(craft.writing)
        n_fig = len(craft.figure_placements)
        n_sec = len(craft.section_focuses)
        print(
            f"[ok] {card_id} deriv={n_der} alg={n_alg} interp={n_int} "
            f"writing={n_wri} figpos={n_fig} section={n_sec} ({time.time()-t0:.0f}s)"
        )

    print(f"\n完成：深加工 {done} 篇，跳过 {skipped} 篇，失败 {failed} 篇。")
    print(f"产物目录：{craft_dir}")

    if args.aggregate:
        from modeling_assistant.data.craft_aggregate import (
            aggregate_craft_guides,
            load_crafts,
            save_craft_guide,
        )

        all_crafts = load_crafts(craft_dir)
        card_types: dict[str, str] = {}
        for p in Path(args.cards_dir).glob("*.json"):
            card = json.loads(p.read_text(encoding="utf-8"))
            card_types[card["id"]] = card.get("problem_type", "")
        guides = aggregate_craft_guides(all_crafts, card_types=card_types)
        guides_dir = Path(args.output) / "craft_guides"
        for guide in guides:
            save_craft_guide(guide, guides_dir)
            print(
                f"[guide] {guide.problem_type}: deriv={len(guide.derivation_common)} "
                f"alg={len(guide.algorithm_common)} interp={len(guide.interpretation_common)} "
                f"writing={len(guide.writing_common)} figpos={len(guide.figure_placement_common)} "
                f"section={len(guide.section_focus_common)}"
            )
        print(f"题型指南目录：{guides_dir}")


if __name__ == "__main__":
    main()
