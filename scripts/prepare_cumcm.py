"""预处理 CUMCM 数模资料：为优秀论文配对历年题面，生成摄入暂存目录。

输出结构（exemplars/raw/cumcm/）：
    <年份>_<题号>/problem.txt       题面纯文本（无法解析则为空）
    <年份>_<题号>/<年份>_<题号>_<原文件名>.pdf   论文副本（文件名唯一 → 卡片 id 唯一）
    _review/<年份>_<原文件名>.pdf   解析/评述类文章（不配题面）

用法：
    python scripts/prepare_cumcm.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "CUMCM数模资料"
PAPERS_ROOT = DATA_ROOT / "优秀论文"
PROBLEMS_ROOT = DATA_ROOT / "历年试题"
STAGING = ROOT / "exemplars" / "raw" / "cumcm"

REVIEW_MARKERS = ("解析", "评述", "评论", "研究_", "_研究")


def extract_pdf_text(path: Path) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_docx_text(path: Path) -> str:
    """不依赖 python-docx，直接从 docx 的 document.xml 提取段落文本。"""
    with zipfile.ZipFile(path) as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paras: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("}p"):
            texts = [
                (child.text or "")
                for child in elem.iter()
                if child.tag.endswith("}t") and child.text
            ]
            para = "".join(texts).strip()
            if para:
                paras.append(para)
    return "\n".join(paras)


def extract_any_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            return extract_pdf_text(path)
        except Exception as exc:
            print(f"  [warn] PDF 解析失败 {path.name}: {exc}")
            return ""
    if suffix == ".docx":
        try:
            return extract_docx_text(path)
        except Exception as exc:
            print(f"  [warn] DOCX 解析失败 {path.name}: {exc}")
            return ""
    return ""


def find_problem_file(year: str, topic: str) -> Path | None:
    """在历年试题目录中查找 (年份, 题号) 对应的题目文件。"""
    year_dir = PROBLEMS_ROOT / f"{year}年赛题"
    if not year_dir.exists():
        return None
    topic_dir = year_dir / f"{topic}题"
    candidates: list[Path] = []
    if topic_dir.exists():
        for p in topic_dir.iterdir():
            if p.suffix.lower() in (".pdf", ".docx", ".doc") and "附录" not in p.name and "附件" not in p.name:
                candidates.append(p)
    else:
        for p in year_dir.iterdir():
            if p.suffix.lower() == ".pdf" and p.stem.startswith(topic):
                candidates.append(p)
    if not candidates:
        return None
    # 优先 docx/pdf 中文件名含题号或 Problem 的；否则取第一个
    def score(p: Path) -> int:
        name = p.name
        s = 0
        if topic in name:
            s += 4
        if "Problem" in name or "problem" in name:
            s += 2
        if p.suffix.lower() == ".doc":
            s -= 3
        return s

    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    if best.suffix.lower() == ".doc":
        return None  # 旧版 .doc 无法直接提取
    return best


def parse_year_topic(path: Path) -> tuple[str, str] | None:
    """从论文路径解析 (年份, 题号)；无法解析时返回 None（评述/解析类）。"""
    year = re.search(r"(20\d{2})", str(path.parent.name))
    year = year.group(1) if year else ""
    name = path.stem
    # 题号字母出现在文件名开头的年份之后（如 2020A：…）或直接开头（如 A001.pdf）。
    # 不能加 (?<!\d)，否则会漏掉与年份紧邻的题号；标题中的 A-E 字母极少且不影响
    # 取第一个匹配（年份后的题号总是最先出现）。
    m = re.search(r"([A-EＡ-Ｅ])", name)
    if not m:
        return None
    topic = m.group(1).upper()
    if topic in "ＡＢＣＤＥ":
        topic = chr(ord("A") + "ＡＢＣＤＥ".index(topic))
    return year, topic


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)
    review_dir = STAGING / "_review"
    review_dir.mkdir(exist_ok=True)

    stats: dict[str, int] = {"papers": 0, "with_problem": 0, "review": 0, "skip_dup": 0}
    report: list[dict] = []
    seen_ids: set[str] = set()

    for year_dir in sorted(PAPERS_ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        for pdf in sorted(year_dir.glob("*.pdf")):
            parsed = parse_year_topic(pdf)
            if parsed is None:
                # 评述/解析类文章：不配题面
                dest = review_dir / f"{year_dir.name[:4]}_{pdf.stem[:80]}.pdf"
                shutil.copy2(pdf, dest)
                stats["review"] += 1
                report.append(
                    {"id": dest.stem, "year": year_dir.name[:4], "topic": "review", "paper": pdf.name}
                )
                continue
            year, topic = parsed
            card_id = f"{year}_{topic}_{pdf.stem[:60]}"
            if card_id in seen_ids:
                stats["skip_dup"] += 1
                continue
            seen_ids.add(card_id)
            target_dir = STAGING / f"{year}_{topic}"
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_pdf = target_dir / f"{card_id}.pdf"
            shutil.copy2(pdf, dest_pdf)

            problem_file = find_problem_file(year, topic)
            problem_text = ""
            if problem_file is not None:
                problem_text = extract_any_text(problem_file)
                if problem_text.strip():
                    stats["with_problem"] += 1
            (target_dir / "problem.txt").write_text(problem_text.strip(), encoding="utf-8")
            stats["papers"] += 1
            report.append(
                {
                    "id": card_id,
                    "year": year,
                    "topic": topic,
                    "paper": pdf.name,
                    "problem_file": problem_file.name if problem_file else None,
                    "problem_chars": len(problem_text.strip()),
                }
            )

    (DATA_ROOT / "_prep_report.json").write_text(
        json.dumps({"stats": stats, "items": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"暂存目录：{STAGING}")
    print(f"报告：{DATA_ROOT / '_prep_report.json'}")


if __name__ == "__main__":
    main()
