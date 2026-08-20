"""V17 论文↔结果机器数字比对。

把论文章节中的「参数名 = 数值」（三线表行 + 正文）与 Result Manifest
绑定的结果文件逐项比对：

- R1（硬错误）：论文数值必须在绑定文件中存在（容差内）；
- R2（硬错误）：跨文件污染——论文数值只存在于非绑定文件时，报
  「疑似引用错误文件」（用于抓「问题 1 章节抄问题 2 参数」这类事故）；
- R3（警告）：绑定文件的关键参数未被章节提及（防漏报）。

只对白名单参数比对，避免公式编号/年份/系数误报；
含「待验证/理论推导」标记的段落跳过。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 参数名归一化白名单：各种写法 → 规范名
PARAM_ALIASES = {
    "r": "R",
    "h": "H",
    "xc": "x_c",
    "x_c": "x_c",
    "zc": "z_c",
    "z_c": "z_c",
    "phi0": "phi0",
    "phi_0": "phi0",
    "y0": "y0",
    "y_0": "y0",
    "deltaphi": "delta_phi",
    "dphi": "delta_phi",
    "deltay": "delta_y",
    "dy": "delta_y",
    "h_e": "h_e",
    "he": "h_e",
    "d_view": "d_view",
    "dview": "d_view",
    "tau": "tau",
    "τ": "tau",
    "thresholdtau": "tau",  # CSV 列 threshold_tau
    "cd": "CD",
    "ssim": "SSIM",
}

# 只有这些「结果关键参数」参与跨文件污染（R2）判定；
# h_e / d_view / tau 等固定假设或扫描参数不在其列，避免误报。
RESULT_KEY_PARAMS = {
    "R", "H", "x_c", "z_c", "phi0", "y0", "delta_phi", "delta_y", "CD", "SSIM",
}

# 正文文本中 "参数 = 数值"（容忍 $R$ = 0.0322 的闭合 $）
_PARAM_EQ_RE = re.compile(
    r"([A-Za-z_]{1,12}|τ)\s*(?:\$\s*)?=\s*(-?\d+(?:\.\d+)?)"
)

_SKIP_MARKERS = ("待验证", "理论推导", "未验证", "待数值验证")

# 三线表单元格中的参数名识别（单元格含中文/单位/LaTeX，如 "半径 $R$ (m)"）
_PARAM_PATTERNS: dict[str, re.Pattern] = {
    "R": re.compile(r"\bR\b"),
    "H": re.compile(r"\bH\b"),
    "x_c": re.compile(r"x\s*[_}]?\s*\{?\s*c\b|\bxc\b", re.IGNORECASE),
    "z_c": re.compile(r"z\s*[_}]?\s*\{?\s*c\b|\bzc\b", re.IGNORECASE),
    "phi0": re.compile(
        r"(?:\\?phi|\\?varphi)\s*[_}]?\s*\{?\s*0\b|φ\s*[_}]?\s*0\b|\bphi0\b",
        re.IGNORECASE,
    ),
    "y0": re.compile(r"\by\s*[_}]?\s*\{?\s*0\b|\by0\b", re.IGNORECASE),
    "delta_phi": re.compile(
        r"\\?Delta\s*\\?phi\b|Δ\s*φ\b|\bdphi\b|\bdelta_phi\b", re.IGNORECASE
    ),
    "delta_y": re.compile(
        r"\\?Delta\s*y\b|Δ\s*y\b|\bdy\b|\bdelta_y\b", re.IGNORECASE
    ),
    "h_e": re.compile(r"\bh\s*[_}]?\s*\{?\s*e\b|\bhe\b", re.IGNORECASE),
    "d_view": re.compile(
        r"\bd\s*[_}]?\s*\{?\s*view\b|\bdview\b|\bd_view\b", re.IGNORECASE
    ),
    "tau": re.compile(r"\\?tau\b|τ\b|\btau\b", re.IGNORECASE),
    "CD": re.compile(r"\bCD\b"),
    "SSIM": re.compile(r"\bSSIM\b"),
}


def _clean_param(raw: str) -> str | None:
    """把 LaTeX 参数名归一化为白名单规范名；不在白名单返回 None。"""
    name = str(raw).strip()
    name = name.replace("{", "").replace("}", "")
    # \Delta\phi → \phi → phi；\delta\phi 同理
    name = re.sub(r"\\(?:Delta|delta)\s*", "", name)
    name = re.sub(r"\\phi\b", "phi", name)
    name = re.sub(r"\\tau\b", "tau", name)
    name = name.lower().replace("_", "").replace(" ", "")
    return PARAM_ALIASES.get(name)


def _normalize_text(line: str) -> str:
    """把 LaTeX 数学写法归一化为简单参数名，便于 "参数=数值" 正则匹配。"""
    line = re.sub(r"\\Delta\s*\\?phi\b", "delta_phi", line, flags=re.IGNORECASE)
    line = re.sub(r"\\Delta\s*y\b", "delta_y", line, flags=re.IGNORECASE)
    line = re.sub(r"\\?varphi\b", "phi", line, flags=re.IGNORECASE)
    line = re.sub(r"\\?phi\s*[_}]?\s*\{?\s*0\b", "phi0", line, flags=re.IGNORECASE)
    line = re.sub(r"\\?tau\b", "tau", line, flags=re.IGNORECASE)
    return line


def _find_param_in_cell(cell: str) -> str | None:
    """在表格单元格（含中文/单位/LaTeX 数学）中识别参数名。"""
    direct = _clean_param(cell)
    if direct is not None:
        return direct
    for param, pattern in _PARAM_PATTERNS.items():
        if pattern.search(cell):
            return param
    return None


def _parse_threelinetable_args(text: str) -> list[tuple[list[str], int]]:
    """定位所有 \\threelinetable 并解析参数（brace-aware，容忍嵌套花括号）。

    返回 [(参数列表, 结束位置)]；参数列表为 [可选label, 标题, 列格式, 表头, 内容]。
    """
    out: list[tuple[list[str], int]] = []
    start = 0
    while True:
        pos = text.find("\\threelinetable", start)
        if pos == -1:
            break
        args: list[str] = []
        i = pos + len("\\threelinetable")
        if i < len(text) and text[i] == "[":
            end = text.find("]", i)
            if end == -1:
                break
            args.append(text[i + 1 : end])
            i = end + 1
        while i < len(text) and text[i] in " \n\r\t":
            i += 1
        while i < len(text) and text[i] == "{":
            depth = 1
            j = i + 1
            while j < len(text) and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            args.append(text[i + 1 : j - 1])
            i = j
            while i < len(text) and text[i] in " \n\r\t":
                i += 1
        if args:
            out.append((args, pos + len("\\threelinetable")))
        start = pos + len("\\threelinetable")
    return out


def _extract_table_value_refs(section_text: str) -> list[tuple[str, float]]:
    """从三线表内容行提取 (参数, 数值)。"""
    refs: list[tuple[str, float]] = []
    for args, _pos in _parse_threelinetable_args(section_text):
        if not args:
            continue
        content = args[-1]
        for row in content.split(r"\\"):
            cells = [c.strip() for c in row.split("&")]
            if len(cells) != 2:
                continue
            param = _find_param_in_cell(cells[0])
            if param is None:
                continue
            m = re.search(r"-?\d+(?:\.\d+)?", cells[1])
            if m:
                refs.append((param, float(m.group())))
    return refs


def _extract_text_value_refs(section_text: str) -> list[tuple[str, float]]:
    """从正文提取 (参数, 数值)；含待验证标记的行跳过。"""
    refs: list[tuple[str, float]] = []
    for line in section_text.splitlines():
        if any(mk in line for mk in _SKIP_MARKERS):
            continue
        norm = _normalize_text(line)
        for m in _PARAM_EQ_RE.finditer(norm):
            param = _clean_param(m.group(1))
            if param is None:
                continue
            try:
                refs.append((param, float(m.group(2))))
            except ValueError:
                continue
    return refs


def extract_paper_value_refs(section_text: str) -> list[tuple[str, float]]:
    """合并三线表与正文的 (参数, 数值) 引用（供测试直接使用）。"""
    return _extract_table_value_refs(section_text) + _extract_text_value_refs(section_text)


def _read_result_values(path: Path) -> list[dict[str, float]]:
    """读取结果文件，返回行字典列表（列名归一化 → 数值）。

    支持两种形态：
    - 表格形态：列名=参数名（如 q1.csv 的 R/H/x_c...）；
    - 文本串形态：某列内容为 "R=0.0322,H=0.2163,..."（如 best_params_text）。
    """
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    suffix = path.suffix.lower()
    try:
        import pandas as pd

        df = pd.read_csv(path) if suffix == ".csv" else pd.read_excel(path)
    except Exception as exc:
        logger.warning("数值比对：读取结果文件失败 %s: %s", path, exc)
        return rows

    # 1) 文本串形态：整列内容为 "K=V,K=V,..."
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        s = str(sample.iloc[0])
        if "," not in s or not re.search(r"[A-Za-z_]{1,12}=(-?\d+(?:\.\d+)?)", s):
            continue
        for val in sample.tolist():
            row: dict[str, float] = {}
            for m in re.finditer(r"([A-Za-z_]{1,12})=(-?\d+(?:\.\d+)?)", str(val)):
                param = _clean_param(m.group(1))
                if param is not None:
                    row[param] = float(m.group(2))
            if row:
                rows.append(row)

    # 2) 表格形态：列名 = 参数名
    for _, row in df.iterrows():
        r: dict[str, float] = {}
        for col in df.columns:
            param = _clean_param(str(col))
            if param is None:
                continue
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            r[param] = v
        if r:
            rows.append(r)
    return rows


def _close(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 1e-3) -> bool:
    return abs(a - b) <= max(rel_tol * abs(b), abs_tol)


def check_numeric_consistency(
    paper_dir: str | Path,
    manifest: list[dict],
    *,
    results_root: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """比对论文章节数字与 Result Manifest 绑定文件。返回 (issues, warnings)。"""
    root = Path(paper_dir)
    rroot = Path(results_root) if results_root else root.parent
    issues: list[str] = []
    warnings: list[str] = []

    if not manifest:
        return issues, warnings

    # 全部 manifest 条目的值集合（供跨文件污染检查）
    all_file_values: dict[str, list[dict[str, float]]] = {}
    entries: list[tuple[dict, list[dict[str, float]]]] = []
    for entry in manifest:
        e = dict(entry)
        paths = e.get("result_paths") or []
        rows_all: list[dict[str, float]] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = rroot / pp
            rows = _read_result_values(pp)
            rows_all.extend(rows)
            all_file_values.setdefault(pp.name, []).extend(rows)
        entries.append((e, rows_all))

    sections_dir = root / "sections"
    if not sections_dir.exists():
        issues.append("数值一致性：章节目录缺失，无法比对")
        return issues, warnings

    for entry, bound_rows in entries:
        idx = int(entry.get("index", 0))
        section_file = sections_dir / f"{4 + idx + 1}_problem{idx + 1}.tex"
        if not section_file.exists():
            warnings.append(
                f"数值一致性：未找到绑定章节 {section_file.name}（小题 {idx + 1}）"
            )
            continue
        try:
            text = section_file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            warnings.append(f"数值一致性：读取章节 {section_file.name} 失败: {exc}")
            continue

        paper_refs = extract_paper_value_refs(text)
        bound_names = [Path(p).name for p in entry.get("result_paths") or []]

        for param, value in dict.fromkeys(paper_refs):
            bound_hit = any(
                _close(value, r.get(param)) for r in bound_rows if param in r
            )
            if bound_hit:
                continue
            bound_present = any(param in r for r in bound_rows)
            if bound_present:
                issues.append(
                    f"{section_file.name}: 参数 {param}={value} 未见于绑定结果文件"
                    f"（{bound_names}）"
                )
                continue
            # 绑定文件无此参数：仅当属于结果关键参数且存在于其他文件时判污染
            if param in RESULT_KEY_PARAMS:
                elsewhere = [
                    fname
                    for fname, rows in all_file_values.items()
                    if fname not in bound_names
                    and any(_close(value, r.get(param)) for r in rows if param in r)
                ]
                if elsewhere:
                    issues.append(
                        f"{section_file.name}: 参数 {param}={value} 未见于绑定结果文件，"
                        f"仅见于其他小题文件 {sorted(elsewhere)}（疑似引用错误文件）"
                    )

        # R3：绑定文件的 CD/SSIM 质量指标未被章节提及 → 警告（防漏报）
        if bound_rows:
            file_params = {p for row in bound_rows for p in row}
            key_params = [p for p in ("CD", "SSIM") if p in file_params]
            mentioned = {p for p, _ in paper_refs}
            missing = [p for p in key_params if p not in mentioned]
            if missing and paper_refs:
                warnings.append(
                    f"{section_file.name}: 绑定结果文件包含关键参数 {missing} "
                    "但章节未提及（防漏报）"
                )
    return issues, warnings
