"""对扫描版获奖论文 PDF 执行 OCR，生成文本缓存（<pdf>.ocr.txt）。

用法：
    python scripts/ocr_cumcm.py [--workers 4] [--dpi 150] [--limit 10]

只处理文本层为空（扫描件）的 PDF；已有 .ocr.txt 缓存的自动跳过。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 限制每个 ONNX 会话的线程数，避免多个 OCR worker 之间 CPU 超订互相拖慢
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("ORT_NUM_THREADS", "2")


def _engine():
    """线程本地 RapidOCR 实例，避免跨线程复用。"""
    local = threading.local()
    if not hasattr(local, "engine"):
        from rapidocr_onnxruntime import RapidOCR

        local.engine = RapidOCR()
    return local.engine


def ocr_one(pdf_path: Path, dpi: int, max_pages: int = 0) -> str:
    import numpy as np

    import fitz

    engine = _engine()
    doc = fitz.open(pdf_path)
    parts: list[str] = []
    for idx, page in enumerate(doc):
        if max_pages > 0 and idx >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # RGBA → RGB
            img = img[:, :, :3]
        result, _elapse = engine(img)
        if result:
            page_text = "\n".join(str(item[1]) for item in result)
            parts.append(page_text)
    doc.close()
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描版获奖论文 OCR。")
    parser.add_argument("--report", default=str(ROOT / "CUMCM数模资料" / "_text_layer_report.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=0, help="每篇只 OCR 前 N 页（0=全部）。")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（0=全部）。")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    targets = [Path(item["path"]) for item in report.get("empty", [])]
    if args.limit > 0:
        targets = targets[: args.limit]

    todo: list[Path] = []
    for p in targets:
        ocr_cache = p.with_suffix(p.suffix + ".ocr.txt")
        if ocr_cache.exists() and ocr_cache.stat().st_size > 200:
            continue
        todo.append(p)

    if not todo:
        print("没有需要 OCR 的文件。")
        return
    print(f"待 OCR：{len(todo)} 篇（workers={args.workers}, dpi={args.dpi}, max_pages={args.max_pages}）")

    def _run(p: Path) -> tuple[Path, int]:
        text = ocr_one(p, args.dpi, args.max_pages)
        cache = p.with_suffix(p.suffix + ".ocr.txt")
        cache.write_text(text, encoding="utf-8")
        return p, len(text)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, p): p for p in todo}
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            try:
                _path, chars = future.result()
                done += 1
                print(f"[{done}/{len(todo)}] OK {p.name} chars={chars}", flush=True)
            except Exception as exc:
                print(f"[ERR] {p.name}: {exc}", flush=True)

    print("OCR 完成。")


if __name__ == "__main__":
    main()
