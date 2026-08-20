"""国赛（CUMCM）LaTeX 论文模板管理。

V15：writer 节点把内置模板目录复制到 output_dir/paper/，
按实际子问题数量调整 main.tex 的问题章节 input 行，
并把「章节文件 → 标题/用途」结构注入 writer prompt，让 LLM 按模板骨架写各 section。

模板目录缺失或 main.tex 缺失时返回 None，writer 回退到旧的
「LLM 输出完整 main.tex」行为，保证无模板环境完全可用。
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# main.tex 中连续的问题章节 input 行（如 \input{sections/5_problem1} 三行）
_PROBLEM_INPUT_RE = re.compile(r"(?:\\input\{sections/\d+_problem\d+\}\s*\n)+")

# 固定章节元信息（问题章节按子问题数量动态生成）
_FIXED_SECTIONS: list[dict[str, str]] = [
    {
        "file": "1_restatement.tex",
        "title": "问题重述",
        "purpose": "问题背景（可引文献）与问题内容重述，不照抄题面",
    },
    {
        "file": "2_analysis.tex",
        "title": "问题分析",
        "purpose": "每个子问题的类型判断、分析思路与选型理由",
    },
    {
        "file": "3_assumptions.tex",
        "title": "模型假设",
        "purpose": "假设 + 合理性说明，每条假设在正文建模处被引用",
    },
    {
        "file": "4_symbols.tex",
        "title": "符号说明",
        "purpose": "符号表（符号/含义/单位），用 LaTeX 数学公式渲染",
    },
]

_TRAILING_SECTIONS: list[dict[str, str]] = [
    {
        "file": "8_sensitivity.tex",
        "title": "灵敏度分析",
        "purpose": "关键参数 ±20% 扰动对结果的影响、鲁棒性验证",
    },
    {
        "file": "9_evaluation.tex",
        "title": "模型的评价、改进与推广",
        "purpose": "优缺点（优点多于缺点）、改进方向与推广",
    },
]

# 行文技艺章节名 → 模板章节文件的映射关键词（顺序优先，命中即返回）
_CRAFT_SECTION_TO_TEMPLATE: list[tuple[tuple[str, ...], str]] = [
    (("摘要",), "main.tex"),
    (("问题重述", "问题背景", "背景"), "1_restatement.tex"),
    (("灵敏度", "敏感性", "鲁棒性"), "8_sensitivity.tex"),
    (("符号说明", "符号"), "4_symbols.tex"),
    (("模型假设", "假设"), "3_assumptions.tex"),
    (("问题分析", "分析"), "2_analysis.tex"),
    (("模型评价", "评价", "结论", "推广", "改进", "模型检验"), "9_evaluation.tex"),
    (("模型建立", "模型求解", "模型分析", "模型", "问题一", "问题二", "问题三", "算法"), "5_problem1.tex"),
    (("附录", "代码"), "A_code.tex"),
]


def map_craft_section_to_template(section: str) -> str | None:
    """把行文技艺的章节名（如"模型建立"）映射到模板章节文件名。

    用于让 Writer 明确"这个写作重点应该落到哪个模板章节文件"；
    无法映射时返回 None（由 Writer 自行判断）。
    """
    if not section:
        return None
    from modeling_assistant.data.exemplar_ingest import _normalize_section_name

    norm = _normalize_section_name(section)
    for keywords, filename in _CRAFT_SECTION_TO_TEMPLATE:
        if any(kw in norm for kw in keywords):
            return filename
    return None


def _problem_sections(n: int) -> list[dict[str, str]]:
    """生成 n 个子问题的章节元信息。"""
    return [
        {
            "file": f"{4 + i}_problem{i}.tex",
            "title": f"问题{i}的模型建立与求解",
            "purpose": (
                f"问题{i}：数据预处理 → 模型建立（公式推导）→ 求解方法 → "
                "结果展示（图表 + 3 行以上解读）→ 结果分析 → 以「问题小结」收尾"
                f"（小结必须覆盖三段式：本题做了什么 → 得到什么 → 对问题{i + 1}"
                "（最后一题为：对全文结论/评价）的支撑，形成承上启下的论证链条）"
            ),
        }
        for i in range(1, n + 1)
    ]


def build_template_structure(n_sub_questions: int) -> list[dict[str, str]]:
    """构建模板章节结构清单（固定章节 + 动态问题章节 + 尾随章节）。"""
    return (
        list(_FIXED_SECTIONS)
        + _problem_sections(max(1, n_sub_questions))
        + list(_TRAILING_SECTIONS)
    )


def build_section_result_binding(n_sub_questions: int) -> dict[str, int]:
    """构建「模板章节文件 → 小题索引」绑定（V17 P2）。

    如 {5_problem1.tex: 0, 6_problem2.tex: 1, ...}，
    供 Writer 约束「问题 N 章节只能引用第 N 小题的权威结果文件」。
    """
    n = max(1, n_sub_questions)
    return {f"{4 + i}_problem{i}.tex": i - 1 for i in range(1, n + 1)}


def load_template_structure(template_dir: str | Path) -> list[dict[str, str]] | None:
    """加载模板章节结构；模板缺失时返回 None（writer 回退旧行为）。"""
    root = Path(template_dir)
    if not (root / "main.tex").exists():
        return None
    # 模板自带 3 个问题章节；结构仅用于注入 prompt，问题数量在复制时按实际调整
    return build_template_structure(3)


def _adjust_problem_inputs(main_tex: str, n: int) -> str:
    """把 main.tex 中连续的问题章节 input 行替换为 n 行。"""
    replacement = "".join(
        f"\\input{{sections/{4 + i}_problem{i}}}\n" for i in range(1, n + 1)
    )
    # lambda 返回字面文本，避免 re.sub 把 replacement 中的 `\i` 解析为转义
    return _PROBLEM_INPUT_RE.sub(lambda _match: replacement, main_tex)


def copy_template(
    template_dir: str | Path,
    paper_dir: str | Path,
    n_sub_questions: int,
) -> list[dict[str, str]] | None:
    """把模板整目录复制到 paper_dir，并按子问题数量调整 main.tex。

    Args:
        template_dir: 模板源目录（含 main.tex）。
        paper_dir: 论文产物目录（output_dir/paper）。
        n_sub_questions: 实际子问题数量（≥1）。

    Returns:
        调整后的章节结构清单；模板缺失时返回 None。
    """
    src = Path(template_dir)
    dst = Path(paper_dir)
    main_src = src / "main.tex"
    if not main_src.exists():
        logger.warning("论文模板缺失（%s），回退旧 Writer 行为", main_src)
        return None

    try:
        dst.mkdir(parents=True, exist_ok=True)
        # 整目录复制（覆盖旧产物，保留 main.tex/references.tex/sections/*）
        shutil.copytree(src, dst, dirs_exist_ok=True)
        main_tex = (dst / "main.tex").read_text(encoding="utf-8")
        adjusted = _adjust_problem_inputs(main_tex, max(1, n_sub_questions))
        if adjusted != main_tex:
            (dst / "main.tex").write_text(adjusted, encoding="utf-8")
        # 清理多余的问题章节文件，确保与 main.tex 的 input 一一对应
        problem_files = sorted(dst.glob("sections/*_problem*.tex"))
        keep = {f"{4 + i}_problem{i}.tex" for i in range(1, n_sub_questions + 1)}
        for f in problem_files:
            if f.name not in keep:
                f.unlink(missing_ok=True)
        # 补齐缺失的问题章节文件（空占位，由 writer 填充）
        for i in range(1, n_sub_questions + 1):
            f = dst / "sections" / f"{4 + i}_problem{i}.tex"
            if not f.exists():
                f.write_text(f"\\section{{问题{i}的模型建立与求解}}\n", encoding="utf-8")
        logger.info("论文模板已复制并调整（%d 个子问题）→ %s", n_sub_questions, dst)
    except Exception as exc:
        logger.error("论文模板复制失败，回退旧 Writer 行为: %s", exc)
        return None

    return build_template_structure(n_sub_questions)


def read_paper_text(paper_dir: str | Path, max_chars: int = 12000) -> str:
    """拼接论文全部 .tex 文本（供 final_reviewer LLM 审查），按字节截断。"""
    root = Path(paper_dir)
    if not root.exists():
        return ""
    parts: list[str] = []
    total = 0
    for path in sorted(root.rglob("*.tex")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("读取论文文件失败 %s: %s", path, exc)
            continue
        parts.append(f"===== {path.relative_to(root)} =====\n{text}")
        total += len(parts[-1])
        if total >= max_chars:
            break
    return "\n\n".join(parts)[:max_chars]
