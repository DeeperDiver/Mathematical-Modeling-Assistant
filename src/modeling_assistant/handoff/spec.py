"""生成"方案与实现架构说明书"与编程手任务包。

V13 设计：主流程把建模方案、算法、预期图表/表格、结果契约打包成
一个自包含的任务包（Markdown + JSON），经人类审核后交给编程手
（另一个 AI / Codex 实例 / 人工会话）实现。主流程只负责执行与验证。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modeling_assistant.prompts.catalog import _compact_profile_dict
from modeling_assistant.schemas.responses import ResultContract
from modeling_assistant.schemas.state import GraphState


def _contract_dict(contract: ResultContract | None) -> dict[str, Any]:
    if contract is None:
        return {}
    if isinstance(contract, dict):
        return contract
    return contract.model_dump(mode="json")


def build_architecture_spec_md(state: GraphState) -> str:
    """把当前方案与实现架构渲染成人类可读的说明书。"""
    static = state.get("static_ltm")
    dynamic = state.get("dynamic_ltm")
    artifacts = state.get("artifacts")
    control = state.get("control")

    lines: list[str] = []
    lines.append("# 建模方案与实现架构说明书")
    lines.append("")

    # 0. 当前小题
    sub_questions = getattr(control, "sub_questions", None) or []
    idx = getattr(control, "current_sub_question_index", 0) or 0
    lines.append("## 0. 当前小题")
    if sub_questions:
        lines.append(f"**当前小题（{idx + 1}/{len(sub_questions)}）**：{sub_questions[idx]}")
    lines.append("")

    # 1. 题目与目标
    lines.append("## 1. 题目与目标")
    raw = getattr(static, "raw_problem", "") or ""
    lines.append(f"**题目**：{raw[:500]}")
    objective = getattr(dynamic, "objective", "") or ""
    lines.append(f"**建模目标**：{objective}")
    lines.append("")

    # 2. 建模设定
    lines.append("## 2. 建模设定（动态 LTM，编程手不得修改）")
    assumptions = getattr(dynamic, "assumptions", []) or []
    nomenclature = getattr(dynamic, "nomenclature", {}) or {}
    equations = getattr(dynamic, "equations", []) or []
    outline = getattr(dynamic, "solution_outline", "") or ""
    lines.append("**假设**：")
    lines.extend(f"- {a}" for a in assumptions)
    lines.append("**符号表**：")
    lines.extend(f"- {k}: {v}" for k, v in nomenclature.items())
    lines.append("**公式/方程**：")
    lines.extend(f"- {e}" for e in equations)
    lines.append(f"**解题思路**：{outline}")
    lines.append("")

    # 3. 算法与求解
    lines.append("## 3. 算法与求解")
    algorithms = getattr(artifacts, "algorithms_summary", "") or ""
    lines.append(f"**算法摘要**：{algorithms}")
    pseudocode = getattr(artifacts, "pseudocode", []) or []
    lines.append("**伪代码/实现步骤**：")
    lines.extend(f"{i}. {p}" for i, p in enumerate(pseudocode, start=1))
    lines.append("")

    # 4. 数据使用
    lines.append("## 4. 数据使用")
    profile = getattr(static, "data_profile", None)
    if profile is not None:
        lines.append("**数据概要**（原始数据只由代码运行时读取）：")
        lines.append("```json")
        lines.append(json.dumps(_compact_profile_dict(profile), ensure_ascii=False, indent=2))
        lines.append("```")
    intelligence = getattr(static, "data_intelligence", []) or []
    if intelligence:
        lines.append("**数据智能摘要**：")
        lines.extend(f"- {i}" for i in intelligence)
    prev_results = getattr(control, "sub_results", None) or []
    if prev_results:
        lines.append("**已完成小题的结果（供本题复用）**：")
        for r in prev_results:
            paths = ", ".join(getattr(r, "result_paths", []) or [])
            lines.append(f"- 小题 {getattr(r, 'index', 0) + 1}（{getattr(r, 'status', '')}）：{paths}")
    lines.append("")

    # 5. 预期图表
    lines.append("## 5. 预期图表")
    figures = getattr(artifacts, "figures_plan", []) or []
    if figures:
        for fig in figures:
            lines.append(
                f"- {getattr(fig, 'id', '')} [{getattr(fig, 'figure_type', '')}] "
                f"{getattr(fig, 'purpose', '')}（数据来源：{getattr(fig, 'data_source', '')}）"
            )
    else:
        lines.append("（未声明；Drawer 可自行补充）")
    lines.append("")

    # 6. 预期表格
    lines.append("## 6. 预期表格")
    tables = getattr(artifacts, "tables_plan", []) or []
    if tables:
        for t in tables:
            cols = ", ".join(getattr(t, "columns", []) or [])
            lines.append(
                f"- {getattr(t, 'id', '')}：{getattr(t, 'title', '')}"
                f"（列：{cols or '待定'}；{getattr(t, 'purpose', '')}）"
            )
    else:
        lines.append("（未声明；Coder 按结果契约产出）")
    lines.append("")

    # 7. 结果契约
    lines.append("## 7. 结果契约")
    contract = getattr(artifacts, "result_contract", None)
    lines.append("```json")
    lines.append(json.dumps(_contract_dict(contract), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    # 8. 实现约束
    lines.append("## 8. 实现约束")
    lines.append("- 只允许使用：numpy、pandas、scipy、sklearn、statsmodels、matplotlib、networkx、pulp")
    lines.append("- 代码总执行时间不超过 90 秒")
    lines.append("- 结果必须写入 MODELING_OUTPUT_DIR/results/output.csv")
    lines.append("- 数值常量必须与题目 problem_facts 一致，不得自创物理参数")
    lines.append("")
    return "\n".join(lines)


def write_coder_task_package(
    state: GraphState,
    output_dir: Path,
    recent_stderr: str = "",
) -> tuple[Path, Path]:
    """把编程手任务包写到 output_dir/tasks/ 下。

    返回 (markdown 路径, json 路径)。编程手只负责实现，不参与建模决策。
    """
    static = state.get("static_ltm")
    dynamic = state.get("dynamic_ltm")
    artifacts = state.get("artifacts")
    control = state.get("control")

    idx = getattr(control, "current_sub_question_index", 0) or 0
    task_dir = Path(output_dir) / "tasks" / f"q{idx + 1}"
    task_dir.mkdir(parents=True, exist_ok=True)

    md_path = task_dir / "coder_task.md"
    json_path = task_dir / "coder_task.json"

    spec_md = build_architecture_spec_md(state)
    if recent_stderr:
        spec_md += (
            "\n## 9. 最近一次失败（必须针对性修复，不得生成相同代码）\n"
            f"```\n{recent_stderr[:4000]}\n```\n"
        )
    spec_md += (
        "\n## 10. 编程手交付要求\n"
        "- 只在本任务目录编写 `solution.py`，不要修改任何建模设定文件。\n"
        "- `solution.py` 必须是完整可执行的 Python 代码，遵守第 8 节约束。\n"
        "- 如需图表，另写 `figures.py`，把图片保存到 `figures/` 子目录（如 `figures/figure1.png`）。\n"
        "- V17：`figures.py` 的图片必须按 figures_plan 的 `id` 命名"
        "（如 `figures/fig_q1_corr.png`，文件名 = plan.id）；\n"
        "  系统按文件名把图登记到图表注册表，未按 plan_id 命名的图片不会被论文引用。\n"
        "- 数据文件路径通过环境变量 `MODELING_DATA_PATHS`（JSON 数组）与 `MODELING_DATA_PATH`（第一个文件）传入；\n"
        "  多附件时按第 4 节数据概要中的文件边界分别读取，不要假设已合并成一张表。\n"
        "- 不要读取/写入任务包之外的原始数据之外的内容；数据路径见第 4 节。\n"
    )
    md_path.write_text(spec_md, encoding="utf-8")

    payload = {
        "task": "coder_implementation",
        "sub_question": {
            "index": idx,
            "text": (getattr(control, "sub_questions", None) or [""])[idx]
            if idx < len(getattr(control, "sub_questions", None) or [])
            else "",
            "total": len(getattr(control, "sub_questions", None) or []),
        },
        "previous_sub_results": [
            {
                "index": r.index,
                "status": r.status,
                "ltm_version": r.ltm_version,
                "result_paths": list(r.result_paths or []),
                "figure_paths": list(r.figure_paths or []),
            }
            for r in (getattr(control, "sub_results", None) or [])
        ],
        "previous_sub_ltms": [
            {
                "objective": ltm.objective,
                "assumptions": list(ltm.assumptions or []),
                "equations": list(ltm.equations or []),
            }
            for ltm in (getattr(control, "sub_ltms", None) or [])
        ],
        "problem_facts": [
            f.model_dump(mode="json") for f in (getattr(static, "problem_facts", []) or [])
        ],
        "dynamic_ltm": dynamic.model_dump(mode="json") if dynamic is not None else {},
        "artifacts": {
            "outline": getattr(artifacts, "outline", {}),
            "pseudocode": getattr(artifacts, "pseudocode", []),
            "algorithms_summary": getattr(artifacts, "algorithms_summary", ""),
            "figures_plan": [
                f.model_dump(mode="json") for f in (getattr(artifacts, "figures_plan", []) or [])
            ],
            "tables_plan": [
                t.model_dump(mode="json") for t in (getattr(artifacts, "tables_plan", []) or [])
            ],
            "result_contract": _contract_dict(getattr(artifacts, "result_contract", None)),
        },
        "data": {
            "file_paths": list(getattr(static, "data_attachments", []) or []),
            "data_intelligence": list(getattr(static, "data_intelligence", []) or []),
            "data_findings": list(getattr(static, "data_findings", []) or []),
            "profile_summary": (
                _compact_profile_dict(static.data_profile)
                if getattr(static, "data_profile", None) is not None
                else {}
            ),
        },
        "recent_stderr": recent_stderr[:4000],
        "constraints": {
            "allowed_libraries": [
                "numpy", "pandas", "scipy", "sklearn", "statsmodels",
                "matplotlib", "networkx", "pulp",
            ],
            "execution_timeout_seconds": 90,
            "output_path": f"MODELING_OUTPUT_DIR/results/q{idx + 1}.csv",
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md_path, json_path
