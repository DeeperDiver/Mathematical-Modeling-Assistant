"""留一验证：量化每张示例卡片对题型表达指南的独有贡献。

做法：剔除第 i 张卡片 → 用同题型其余卡片重新聚合 → 检查被剔除卡片的
highlights 是否仍被其余卡片的文本覆盖。独有特征丢失越多，贡献分越高。

用法：
    python scripts/leave_one_out_eval.py --exemplars exemplars
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling_assistant.data.exemplars import load_cards
from modeling_assistant.validation.originality import overlap_ratio


def _coverage(highlight: str, texts: list[str]) -> float:
    """高亮特征被其余卡片文本覆盖的程度（字符 3-gram 重合率）。"""
    best = 0.0
    for text in texts:
        best = max(best, overlap_ratio(highlight, [text], n=3))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="留一验证示例卡片贡献。")
    parser.add_argument("--exemplars", default="exemplars", help="知识库根目录。")
    parser.add_argument("--output", default="exemplars/leave_one_out_report.json")
    args = parser.parse_args()

    cards = load_cards(Path(args.exemplars) / "cards")
    if not cards:
        print("知识库为空，无卡片可验证。")
        sys.exit(0)

    report: list[dict] = []
    for card in cards:
        others = [c for c in cards if c.id != card.id and c.problem_type == card.problem_type]
        if not others:
            report.append(
                {
                    "card_id": card.id,
                    "problem_type": card.problem_type,
                    "exclusive_highlights": card.highlights,
                    "coverage": 0.0,
                    "contribution_score": 1.0,
                    "note": "同题型无其他卡片，无法做留一比较",
                }
            )
            continue
        other_texts = [
            " ".join(
                [
                    c.title,
                    " ".join(c.structure.keys()),
                    " ".join(c.highlights),
                    " ".join(c.tags),
                ]
            )
            for c in others
        ]
        coverages = [_coverage(h, other_texts) for h in card.highlights] or [0.0]
        coverage = sum(coverages) / len(coverages)
        report.append(
            {
                "card_id": card.id,
                "problem_type": card.problem_type,
                "highlight_count": len(card.highlights),
                "coverage": round(coverage, 3),
                "contribution_score": round(1.0 - coverage, 3),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {out_path}")


if __name__ == "__main__":
    main()
