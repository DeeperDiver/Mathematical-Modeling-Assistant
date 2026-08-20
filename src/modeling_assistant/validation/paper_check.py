"""论文级确定性验收（V15）：结构、占位符、泄露、图表引用、编译检查。

移植自 MathModelAgent `6verity` 的验收理念，但保持本项目
「零 LLM 机器校验」哲学：本模块只做确定性文本/文件系统检查，
语义层面的灵活审查（数值一致性、表达质量）由 final_reviewer 节点的
LLM 审查完成。

硬错误（FAIL）会导致 final_reviewer 标记论文未通过；
警告（WARN）只记录，最终裁决权交给 HITL 终审。
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 占位符（硬错误）
PLACEHOLDER_PATTERNS = [
    re.compile(r"TODO", re.IGNORECASE),
    re.compile(r"PLACEHOLDER", re.IGNORECASE),
    re.compile(r"待补充|待续写|待写|示例数据|占位|此处填写"),
    re.compile(r"Lorem\s+ipsum", re.IGNORECASE),
    re.compile(r"内容待定|XXX|yyy"),
    # V17 修复：表格/图片引用断链（Writer 输出 \ref 但无对应 \label）
    re.compile(r"表\?\?|图\?\?"),
]

# 内部工作流路径/文件泄露（硬错误）
INTERNAL_PATH_MARKERS = [
    "reports/",
    "figures/",  # 论文正文不应出现工作流目录名（但 \includegraphics{../figures/...} 合法）
    "coder_task",
    "tasks/",
    "CLAUDE.md",
    "prompt_audit",
    "outputs/logs/",
    "run_",
]

# 允许的图片引用前缀（相对 paper/ 目录）
_ALLOWED_IMAGE_PREFIXES = ("../figures/", "figures/", "./figures/")

# V18 承重契约对账：实验类型 → 论文中必须出现的标记
_EXPERIMENT_KEYWORDS = {
    "calibration": ("校准", "标定", "基准", "真值"),
    "perturbation": ("扰动", "敏感性", "灵敏度", "扫描"),
    "cross_check": ("交叉", "复核", "独立验证", "对照"),
    "contrast": ("对照", "有无", "对比"),
    "case_study": ("案例", "典型", "边界", "极端"),
    "artifact": ("示意图", "可视化", "分区", "实物", "照片", "重建"),
}
_ANCHOR_KEYWORDS = ("锚点", "分区", "示意", "可视化", "绑定")
_FALLBACK_KEYWORDS = ("兜底", "边界", "对照", "案例", "极端", "反例", "翻转")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("读取文件失败 %s: %s", path, exc)
        return ""


def _find_placeholders(text: str) -> list[str]:
    found: list[str] = []
    for pattern in PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 15) : match.end() + 15]
            snippet = " ".join(snippet.split())
            if len(snippet) > 50:
                snippet = snippet[:50] + "..."
            found.append(snippet)
    return found[:20]


def _find_internal_leaks(text: str) -> list[str]:
    found: list[str] = []
    for marker in INTERNAL_PATH_MARKERS:
        if marker in text:
            # 排除图片引用行（../figures/ 是合法路径）
            if marker == "figures/" and any(
                prefix in text for prefix in _ALLOWED_IMAGE_PREFIXES
            ):
                continue
            found.append(marker)
    return found


def _check_section_headers(sections_dir: Path) -> list[str]:
    """每个章节文件必须有一级标题（\\section{...}）。"""
    issues: list[str] = []
    if not sections_dir.exists():
        return [f"章节目录缺失：{sections_dir}"]
    for path in sorted(sections_dir.glob("*.tex")):
        # 附录（A_code.tex）的标题由 main.tex 的 \appendixcn 提供，不需要自带一级标题
        if path.name == "A_code.tex":
            continue
        text = _read_text(path)
        if not text.strip():
            issues.append(f"章节文件为空：{path.name}")
            continue
        if not re.search(r"\\section\s*\{", text):
            issues.append(f"章节文件缺少一级标题 \\section{{...}}：{path.name}")
    return issues


def _check_problem_summaries(sections_dir: Path) -> list[str]:
    """每个问题章节必须以「问题小结」收尾（承上启下，说明本题贡献）。

    Writer 输出约定：`5_problemN.tex` 等建模章节的最后一节为
    `\subsection{问题小结}`（做了什么 → 得到什么 → 对下一题的支撑）。
    这里只做确定性存在性检查；内容质量由 final_reviewer 的 LLM 审查把关。
    """
    issues: list[str] = []
    if not sections_dir.exists():
        return [f"章节目录缺失：{sections_dir}"]
    for path in sorted(sections_dir.glob("*_problem*.tex")):
        text = _read_text(path)
        if "问题小结" not in text:
            issues.append(
                f"{path.name} 缺少「问题小结」收尾：每个问题章节末尾必须包含 "
                "\\subsection{问题小结}，写清「本题做了什么 → 得到什么 → 对下一题的支撑」。"
            )
    return issues


def _check_image_references(main_tex: Path, sections_dir: Path, paper_dir: Path) -> list[str]:
    """\\includegraphics 引用的图片必须真实存在（相对 paper/ 目录解析）。"""
    issues: list[str] = []
    tex_files = [main_tex] if main_tex.exists() else []
    if sections_dir.exists():
        tex_files.extend(sorted(sections_dir.glob("*.tex")))

    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    for tex_file in tex_files:
        text = _read_text(tex_file)
        for match in pattern.finditer(text):
            img_rel = match.group(1).strip()
            if img_rel.startswith("http"):
                continue
            # 相对 paper/ 目录：模板图片路径形如 ../figures/xxx.pdf
            candidate = (paper_dir / img_rel).resolve()
            if not candidate.exists():
                issues.append(
                    f"{tex_file.name} 引用的图片不存在：{img_rel}"
                )
    return issues


def _check_malformed_image_commands(tex_files: list[Path]) -> list[str]:
    """检测"残缺图片命令"：LLM 输出 LaTeX 时可能丢失 \\includegraphics，
    只留下裸参数（如 `[width=0.85]../figures/figroadmap.png` 或 `{../figures/x.png}`），
    编译后图片不会显示而是作为普通文本输出。硬错误，必须打回重写。
    """
    issues: list[str] = []
    # 完整命令（\includegraphics[...]{...}）先整体剔除，避免大括号模式误报
    complete_cmd = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}")
    # 模式1：裸 [width=...]/[height=...] 后跟 figures/ 路径
    bare_option = re.compile(
        r"\[(?:width|height)=[^\]]*\]\s*(?:\.\./)?figures/[^\s}\]]+"
    )
    # 模式2：裸 {figures/...} 路径
    bare_brace = re.compile(r"\{(?:\.\./)?figures/[^}]*\}")
    for tex_file in tex_files:
        text = _read_text(tex_file)
        stripped = complete_cmd.sub("", text)
        for pattern, label in (
            (bare_option, "图片命令残缺（缺少 \\includegraphics，只剩 [width=...] 参数）"),
            (bare_brace, "图片路径大括号裸露（缺少 \\includegraphics 命令）"),
        ):
            for match in pattern.finditer(stripped):
                snippet = " ".join(match.group(0).split())[:80]
                issues.append(f"{tex_file.name}: {label}: {snippet}")
                if len(issues) >= 20:
                    return issues
    return issues


def _check_front_matter_placeholders(main_tex: Path) -> list[str]:
    """检查 main.tex 的标题/摘要/关键词占位符是否被替换（writer 未注入时残留）。"""
    issues: list[str] = []
    if not main_tex.exists():
        return issues
    text = _read_text(main_tex)
    markers = [
        ("\\papertitle{论文标题}", "标题占位符未替换"),
        ("中文摘要内容", "摘要占位符未替换"),
        ("关键词1", "关键词占位符未替换"),
    ]
    for marker, label in markers:
        if marker in text:
            issues.append(f"main.tex {label}：{marker}")
    return issues


def _check_input_files(main_tex: Path) -> list[str]:
    """main.tex 引用的 \\input 文件必须存在。"""
    issues: list[str] = []
    if not main_tex.exists():
        return [f"论文入口缺失：{main_tex}"]
    text = _read_text(main_tex)
    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        ref = match.group(1).strip()
        if not ref.endswith(".tex"):
            ref += ".tex"
        candidate = (main_tex.parent / ref).resolve()
        if not candidate.exists():
            issues.append(f"main.tex 引用的章节文件不存在：{ref}")
    return issues


def _check_unresolved_refs(tex_files: list[Path]) -> list[str]:
    r"""检测 \ref{...}/\eqref{...} 引用了不存在的 \label{...}。

    V17 修复：Writer 可能输出 `表\ref{tab:x}` 但三线表宏没有对应的
    `\label{tab:x}`，编译后渲染成 `表??`。在 tex 源码层直接拦截，
    比等编译后匹配字面 `??` 更可靠。
    """
    labels: set[str] = set()
    refs: list[tuple[str, str]] = []
    for tex_file in tex_files:
        text = _read_text(tex_file)
        labels.update(re.findall(r"\\label\{([^}]+)\}", text))
        # V17：\threelinetable[label]{...} 的 label 由宏内部 \label{#1} 生成，
        # 源码中不出现 \label 命令，需把可选 label 视为已定义。
        labels.update(re.findall(r"\\threelinetable\[([^\]]+)\]", text))
        for m in re.finditer(
            r"\\eqref\{([^}]+)\}|\\(?:page)?ref\{([^}]+)\}", text
        ):
            name = m.group(1) or m.group(2)
            if name:
                refs.append((tex_file.name, name))
    issues: list[str] = []
    for fname, name in refs:
        if name not in labels:
            issues.append(f"{fname}: 引用未定义的 \\label {{{name}}}（将渲染为 ??）")
    return issues[:20]


def _check_unresolved_cites(tex_files: list[Path]) -> list[str]:
    r"""检测 \cite{...} 引用了不存在的 \bibitem（可能渲染为 [?]）。

    与 \ref 断链同源：参考文献是占位/缺失时，引用也会变成问号占位。
    """
    bibitems: set[str] = set()
    refs: list[tuple[str, str]] = []
    for tex_file in tex_files:
        text = _read_text(tex_file)
        bibitems.update(re.findall(r"\\bibitem\{([^}]+)\}", text))
        for m in re.finditer(r"\\cite\{([^}]+)\}", text):
            for name in m.group(1).split(","):
                name = name.strip()
                if name:
                    refs.append((tex_file.name, name))
    issues = [
        f"{fname}: \\cite{{{name}}} 无对应 \\bibitem（将渲染为 [?]）"
        for fname, name in refs
        if name not in bibitems
    ]
    return issues[:20]


def _check_load_bearing_contract(
    root: Path,
    figures_plan: list[dict] | None,
    figure_manifest: dict | None,
    load_bearing_map,
) -> tuple[list[str], list[str]]:
    """V18 承重契约对账（确定性）。

    - 契约 required item：验收锚点章节存在，且出现对应实验类型的标记；
    - anchor_gaps 构造：论文有可视化或显式锚点论证，二者必居其一；
    - fallback_required 结论：论文出现兜底/边界/对照/案例类表述。
    降级分析（analysis_incomplete）时只记警告，不阻塞。
    """
    issues: list[str] = []
    warnings: list[str] = []
    if load_bearing_map is None:
        return issues, warnings
    if isinstance(load_bearing_map, dict):
        try:
            from modeling_assistant.schemas.state import LoadBearingMap

            load_bearing_map = LoadBearingMap.model_validate(load_bearing_map)
        except Exception:
            return issues, warnings

    if load_bearing_map.analysis_incomplete:
        warnings.append(
            "承重图为降级分析（analysis_incomplete），承重契约检查降级为警告"
        )

    tex_files: list[Path] = []
    main_tex = root / "main.tex"
    if main_tex.exists():
        tex_files.append(main_tex)
    sections_dir = root / "sections"
    if sections_dir.exists():
        tex_files.extend(sorted(sections_dir.glob("*.tex")))
    full_text = "\n".join(_read_text(f) for f in tex_files)

    anchors = load_bearing_map.contract.acceptance_anchors or {}
    for item in load_bearing_map.contract.required_items:
        section = anchors.get(item.id, "8_sensitivity.tex")
        path = root / "sections" / section
        text = _read_text(path) if path.exists() else ""
        if not path.exists():
            issues.append(
                f"承重契约：构造 {item.construct} 的验证锚点章节缺失 {section}"
            )
            continue
        # 剔除章节标题行，避免标题自带关键词（如"敏感性分析"）造成空内容误通过
        body = re.sub(r"\\section\s*\{[^}]*\}", "", text)
        keywords = _EXPERIMENT_KEYWORDS.get(item.required_experiment, ())
        if keywords and not any(kw in body for kw in keywords):
            issues.append(
                f"承重契约：{item.construct} 在 {section} 中未出现"
                f"「{item.required_experiment}」类验证表述（{keywords[0]} 等）"
            )

    if not load_bearing_map.analysis_incomplete:
        for item in load_bearing_map.constructs:
            if item.physical_anchor:
                continue
            has_any_figure = any(
                Path(entry.get("path", "")).name in full_text
                for entry in (figure_manifest or {}).values()
                if entry.get("status") == "generated"
            )
            if not has_any_figure and not any(kw in full_text for kw in _ANCHOR_KEYWORDS):
                issues.append(
                    f"承重契约：无物理锚点的构造 {item.construct} "
                    "既无可视化，也未给出显式锚点论证"
                )

        for conclusion in load_bearing_map.conclusions:
            if conclusion.fallback_required and not any(
                kw in full_text for kw in _FALLBACK_KEYWORDS
            ):
                issues.append(
                    f"承重契约：结论 {conclusion.id} 要求兜底/边界对照"
                    "（fallback_required），但论文未出现相关表述"
                )
    return issues, warnings


def _check_figure_completeness(
    root: Path,
    figures_plan: list[dict] | None,
    figure_manifest: dict | None,
) -> tuple[list[str], list[str]]:
    r"""V17 图表完整性闭环：规划 → 生成 → 引用 → 图注 对账。

    - C1（硬错误）：required 图已生成（注册表有路径且文件存在）；
    - C2（硬错误）：required 图被论文引用；
    - C3（硬错误）：论文引用的图片真实存在（由现有图片引用检查兜底）；
    - C4（警告）：\caption 图注与 plan.caption 一致（前缀匹配）；
    - C5（警告）：论文引用了未登记图表注册表的图片。
    """
    issues: list[str] = []
    warnings: list[str] = []
    if not figures_plan:
        return issues, warnings

    tex_files: list[Path] = []
    main_tex = root / "main.tex"
    if main_tex.exists():
        tex_files.append(main_tex)
    sections_dir = root / "sections"
    if sections_dir.exists():
        tex_files.extend(sorted(sections_dir.glob("*.tex")))
    full_text = "\n".join(_read_text(f) for f in tex_files)

    referenced = {
        Path(m.group(1)).name
        for m in re.finditer(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", full_text
        )
        if not m.group(1).startswith("http")
    }
    captions = re.findall(r"\\caption\{([^}]*)\}", full_text)
    manifest = figure_manifest or {}

    for plan in figures_plan:
        if not plan.get("required", True):
            continue
        pid = str(plan.get("id", ""))
        if not pid:
            continue
        entry = manifest.get(pid)
        # C1：必需图已生成
        if not entry:
            issues.append(f"图表规划必需图未生成：{pid}（未登记到图表注册表）")
            continue
        path = Path(entry.get("path", ""))
        if not path.exists():
            issues.append(f"图表规划必需图未生成：{pid}（{path} 不存在）")
            continue
        # C2：必需图被论文引用
        if path.name not in referenced:
            issues.append(f"图表规划必需图未被论文引用：{pid}（{path.name}）")
        # C4：图注与规划一致（前缀匹配，警告级）
        cap = str(plan.get("caption") or "").strip()
        if cap and len(cap) >= 8 and captions:
            if not any(cap[:8] in c or c[:8] in cap for c in captions):
                warnings.append(
                    f"图注与规划不一致：{pid}（规划图注「{cap[:20]}…」"
                    "未出现在论文 \\caption 中）"
                )

    # C5：论文引用了注册表之外的图（警告级）
    if manifest:
        generated_names = {
            Path(e.get("path", "")).name
            for e in manifest.values()
            if e.get("status") == "generated"
        }
        extra = sorted(referenced - generated_names)
        if extra:
            warnings.append(
                f"论文引用了未登记图表注册表的图片：{extra[:8]}"
            )
    return issues, warnings


def _compile_pdf(main_tex: Path) -> tuple[bool, str]:
    """用 xelatex/pdflatex 编译两遍（解决交叉引用），返回 (是否成功, 说明)。"""
    compiler = shutil.which("xelatex") or shutil.which("pdflatex")
    if not compiler:
        return False, "未找到 xelatex/pdflatex，跳过编译检查（非硬错误）"
    try:
        for _ in range(2):
            subprocess.run(
                [compiler, "-interaction=nonstopmode", main_tex.name],
                cwd=main_tex.parent,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        pdf = main_tex.with_suffix(".pdf")
        if pdf.exists() and pdf.stat().st_size > 0:
            return True, f"编译成功：{pdf.name}（{pdf.stat().st_size} 字节）"
        return False, "编译结束但未生成有效 PDF"
    except subprocess.TimeoutExpired:
        return False, "编译超时（>180s）"
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "")[-800:]
        return False, f"编译失败：{tail}"
    except Exception as exc:
        return False, f"编译异常：{exc}"


def check_paper(
    paper_dir: str | Path,
    *,
    compile_pdf: bool = True,
    manifest: list[dict] | None = None,
    results_root: str | Path | None = None,
    figures_plan: list[dict] | None = None,
    figure_manifest: dict | None = None,
    load_bearing_map=None,
) -> dict:
    """对论文目录做确定性验收，返回报告 dict。

    Args:
        paper_dir: 论文目录（含 main.tex / sections/）。
        compile_pdf: 是否尝试编译 PDF（无编译器时降级为警告）。
        manifest: V17 Result Manifest 条目列表（AuthoritativeResult 的 dict），
            提供时执行「论文数字 ↔ 结果文件」机器比对；缺省跳过（向后兼容）。
        results_root: 结果文件相对路径的解析根目录（默认 paper_dir 的父目录）。
        figures_plan: V17 图表规划（FigurePlan 的 dict 列表）；提供时执行
            图表完整性检查（规划→生成→引用），缺省跳过。
        figure_manifest: V17 图表注册表（plan_id -> {path, status}）。
        load_bearing_map: V18 承重图（LoadBearingMap 或其 dict），提供时执行
            承重契约对账（根构造验证锚点、锚点缺口可视化/论证、结论形态兜底）。

    Returns:
        {
          "passed": bool,
          "issues": list[str],      # 硬错误
          "warnings": list[str],    # 警告
          "checks": dict[str, str], # 各检查项结论
        }
    """
    root = Path(paper_dir)
    main_tex = root / "main.tex"
    sections_dir = root / "sections"
    issues: list[str] = []
    warnings: list[str] = []
    checks: dict[str, str] = {}

    # 1. 入口与章节文件
    checks["入口"] = "存在" if main_tex.exists() else "缺失"
    issues.extend(_check_input_files(main_tex))
    issues.extend(_check_section_headers(sections_dir))
    summary_issues = _check_problem_summaries(sections_dir)
    if summary_issues:
        checks["问题小结"] = f"{len(summary_issues)} 个问题章节缺失"
        issues.extend(summary_issues)
    else:
        checks["问题小结"] = "通过"

    # 2. 占位符与泄露（正文 = main + sections + references）
    tex_files = []
    if main_tex.exists():
        tex_files.append(main_tex)
    if sections_dir.exists():
        tex_files.extend(sorted(sections_dir.glob("*.tex")))
    references = root / "references.tex"
    if references.exists():
        tex_files.append(references)
    full_text = "\n".join(_read_text(f) for f in tex_files)

    placeholders = _find_placeholders(full_text)
    if placeholders:
        checks["占位符"] = f"发现 {len(placeholders)} 处"
        issues.append(f"正文存在占位符：{placeholders[:5]}")
    else:
        checks["占位符"] = "无"

    leaks = _find_internal_leaks(full_text)
    if leaks:
        checks["内部泄露"] = f"发现 {len(leaks)} 处"
        issues.append(f"正文泄露内部工作流标记：{leaks}")
    else:
        checks["内部泄露"] = "无"

    # 2.1 引用完整性：\ref/\eqref 必须有对应 \label，\cite 必须有 \bibitem
    ref_issues = _check_unresolved_refs(tex_files) + _check_unresolved_cites(tex_files)
    if ref_issues:
        checks["引用完整性"] = f"{len(ref_issues)} 处断链"
        issues.extend(ref_issues)
    else:
        checks["引用完整性"] = "通过"

    # 3. 图表引用
    img_issues = _check_image_references(main_tex, sections_dir, root)
    if img_issues:
        issues.extend(img_issues)
        checks["图表引用"] = f"{len(img_issues)} 处图片引用无效"
    else:
        checks["图表引用"] = "通过"

    # 3.1 图片命令完整性（残缺 includegraphics）
    malformed = _check_malformed_image_commands(tex_files)
    if malformed:
        checks["图片命令完整性"] = f"{len(malformed)} 处残缺"
        issues.extend(malformed)
    else:
        checks["图片命令完整性"] = "通过"

    # 3.2 前置占位符（标题/摘要/关键词）
    front_issues = _check_front_matter_placeholders(main_tex)
    if front_issues:
        checks["前置占位符"] = f"{len(front_issues)} 处未替换"
        issues.extend(front_issues)
    else:
        checks["前置占位符"] = "通过"

    # 3.3 数值一致性（V17）：论文数字 ↔ Result Manifest 绑定结果文件
    if manifest:
        from modeling_assistant.validation.numeric_consistency import (
            check_numeric_consistency,
        )

        num_root = Path(results_root) if results_root else root.parent
        num_issues, num_warnings = check_numeric_consistency(
            root, manifest, results_root=num_root
        )
        if num_issues:
            checks["数值一致性"] = f"{len(num_issues)} 处不一致"
            issues.extend(num_issues)
        else:
            checks["数值一致性"] = "通过"
        warnings.extend(num_warnings)
    else:
        checks["数值一致性"] = "跳过（未提供 Result Manifest）"

    # 3.4 图表完整性（V17）：规划 → 生成 → 引用 对账
    if figures_plan:
        fig_issues, fig_warnings = _check_figure_completeness(
            root, figures_plan, figure_manifest
        )
        if fig_issues:
            checks["图表完整性"] = f"{len(fig_issues)} 处缺失/未引用"
            issues.extend(fig_issues)
        else:
            checks["图表完整性"] = "通过"
        warnings.extend(fig_warnings)
    else:
        checks["图表完整性"] = "跳过（未提供 figures_plan）"

    # 3.5 承重契约对账（V18）
    if load_bearing_map is not None:
        lb_issues, lb_warnings = _check_load_bearing_contract(
            root, figures_plan, figure_manifest, load_bearing_map
        )
        if lb_issues:
            checks["承重契约"] = f"{len(lb_issues)} 处缺项"
            issues.extend(lb_issues)
        else:
            checks["承重契约"] = "通过"
        warnings.extend(lb_warnings)
    else:
        checks["承重契约"] = "跳过（未提供承重图）"

    # 4. 编译
    if compile_pdf and main_tex.exists():
        ok, message = _compile_pdf(main_tex)
        checks["编译"] = message
        if not ok and "未找到" not in message:
            issues.append(message)
        elif "未找到" in message:
            warnings.append(message)

    checks["章节文件数"] = str(len(list(sections_dir.glob("*.tex")))) if sections_dir.exists() else "0"

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
    }
