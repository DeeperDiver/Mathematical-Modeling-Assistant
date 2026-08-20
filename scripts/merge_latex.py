"""把 LaTeX 论文的 \input 引用递归展开，合并为单个文件。

用法：
    python scripts/merge_latex.py --paper-dir outputs/test_reflection/paper
    python scripts/merge_latex.py --paper-dir outputs/test_reflection/paper --output merged.tex

说明：
- 以 --paper-dir/main.tex 为入口，递归展开 \input{...}（相对 paper 目录解析）。
- 输出默认 paper/main_merged.tex；图片相对路径（../figures/ 等）保持不变，
  因为合并文件仍在同一目录。
- 循环引用自动防护；缺失文件保留原 \input 行并给出警告。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_INPUT_RE = re.compile(r"^[ \t]*\\input\{([^}]+)\}[ \t]*%?[ \t]*$", re.MULTILINE)


def merge_latex(paper_dir: str | Path, output: str | Path | None = None) -> Path:
    root = Path(paper_dir)
    main_tex = root / "main.tex"
    if not main_tex.exists():
        raise FileNotFoundError(f"未找到 {main_tex}")

    expanded: dict[Path, str] = {}

    def expand(path: Path, visited: set[Path]) -> str:
        path = path.resolve()
        if path in visited:
            raise RuntimeError(f"循环引用：{path.name}")
        if path in expanded:
            return expanded[path]
        text = path.read_text(encoding="utf-8", errors="replace")
        visited = visited | {path}

        def repl(match: re.Match) -> str:
            ref = match.group(1).strip()
            if not ref.endswith(".tex"):
                ref += ".tex"
            target = (path.parent / ref).resolve()
            if not target.exists():
                print(f"[warn] 引用的文件不存在，保留原行：{ref}", flush=True)
                return match.group(0)
            return expand(target, visited)

        result = _INPUT_RE.sub(repl, text)
        expanded[path] = result
        return result

    merged = expand(main_tex, set())
    out_path = root / (output or "main_merged.tex")
    out_path.write_text(merged, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LaTeX 论文为单文件。")
    parser.add_argument("--paper-dir", default="outputs/test_reflection/paper")
    parser.add_argument("--output", default=None, help="输出文件名（相对 paper 目录，默认 main_merged.tex）。")
    args = parser.parse_args()

    out = merge_latex(args.paper_dir, args.output)
    text = out.read_text(encoding="utf-8")
    remaining = _INPUT_RE.findall(text)
    print(f"合并完成：{out}")
    print(f"总行数：{len(text.splitlines())}，总字符：{len(text)}")
    print(f"残留 \\input：{len(remaining)}（{remaining[:5]}）")


if __name__ == "__main__":
    main()
