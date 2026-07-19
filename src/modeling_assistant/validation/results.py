"""结果验证：检查 Coder 产出的结果是否合理。

ResultReviewer 不调用 LLM，完全基于确定性规则对代码输出做合理性检查。

扩展（假设检验）：基于动态 LTM 中的假设文本做机械检验（正态/线性/独立等），
产出结构化 EmpiricalFinding，与 Reflection 节点形成互补：
- ResultReviewer：纯规则、零成本、覆盖常见分布检验
- Reflection：LLM 提炼、能处理非结构化洞察
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    EmpiricalFinding,
    EmpiricalLayer,
    GraphState,
    REFUTED_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _read_result(path: Path) -> pd.DataFrame | None:
    """尝试读取结果文件。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
        # 尝试按 csv 读取
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("读取结果文件失败 %s: %s", path, exc)
        return None


def _check_nan_inf(df: pd.DataFrame) -> list[str]:
    """检查结果中是否有 NaN 或 Inf。"""
    issues: list[str] = []
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return issues

    nan_cols = numeric_df.columns[numeric_df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"以下数值列包含 NaN：{nan_cols}")

    inf_cols = numeric_df.columns[numeric_df.isin([float("inf"), float("-inf")]).any()].tolist()
    if inf_cols:
        issues.append(f"以下数值列包含 Inf：{inf_cols}")

    return issues


def _check_reasonable_ranges(df: pd.DataFrame) -> list[str]:
    """检查常见指标是否在合理范围内。"""
    issues: list[str] = []
    numeric_df = df.select_dtypes(include=["number"])

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue

        col_lower = str(col).lower()
        min_val = float(series.min())
        max_val = float(series.max())

        # 准确率、概率类指标应在 [0, 1]
        if any(k in col_lower for k in ("accuracy", "precision", "recall", "f1", "auc", "prob", "probability", "score")):
            if min_val < 0 or max_val > 1:
                issues.append(f"指标 '{col}' 疑似概率/准确率，但范围不在 [0, 1] 内：[{min_val}, {max_val}]")

        # 误差类指标应非负
        if any(k in col_lower for k in ("error", "rmse", "mae", "mse", "loss")):
            if min_val < 0:
                issues.append(f"误差指标 '{col}' 出现负值：{min_val}")

        # V11.2 修复（Bug 5）：R² 类指标应在 [0, 1] 区间
        # 负 R² 表示拟合劣于常数模型，是算法未收敛的强信号
        if any(k in col_lower for k in ("r2", "r_squared", "r²", "r_2", "r2_score", "r_square")):
            if min_val < -0.001:  # 容忍浮点误差
                issues.append(
                    f"R² 指标 '{col}' 出现负值 {min_val:.4f}，拟合劣于常数模型，"
                    f"可能未收敛或模型设定错误"
                )

    return issues


def _check_empty_or_trivial(df: pd.DataFrame) -> list[str]:
    """检查结果是否为空或过于简单。"""
    issues: list[str] = []
    if df.empty:
        issues.append("结果文件为空表。")
        return issues

    if len(df) == 1:
        issues.append("结果文件只有一行，可能缺少详细输出。")

    numeric_df = df.select_dtypes(include=["number"])
    for col in numeric_df.columns:
        if numeric_df[col].nunique(dropna=True) <= 1:
            issues.append(f"数值列 '{col}' 为常量，无区分信息。")

    return issues


def validate_result(result_path: str | Path) -> dict[str, Any]:
    """验证单个结果文件，返回验证报告。"""
    path = Path(result_path)
    report: dict[str, Any] = {
        "passed": False,
        "path": str(path),
        "issues": [],
        "metrics": {},
        "sanity_checks": [],
    }

    if not path.exists():
        report["issues"].append(f"结果文件不存在：{path}")
        return report

    df = _read_result(path)
    if df is None:
        report["issues"].append("无法读取结果文件。")
        return report

    issues: list[str] = []
    issues.extend(_check_empty_or_trivial(df))
    issues.extend(_check_nan_inf(df))
    issues.extend(_check_reasonable_ranges(df))

    # 基础统计指标
    numeric_df = df.select_dtypes(include=["number"])
    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
    }
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if not series.empty:
            metrics[str(col)] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()) if not pd.isna(series.std()) else None,
            }
    report["metrics"] = metrics

    sanity_checks = [
        f"结果文件存在，大小为 {path.stat().st_size} 字节。",
        f"结果包含 {len(df)} 行、{len(df.columns)} 列。",
    ]
    if numeric_df.empty:
        sanity_checks.append("结果中无数值列，无法做数值合理性检查。")
    else:
        sanity_checks.append(f"数值列：{list(numeric_df.columns)}")

    report["issues"] = issues
    report["sanity_checks"] = sanity_checks
    report["passed"] = len(issues) == 0

    return report


def validate_results(result_paths: list[str]) -> dict[str, Any]:
    """验证多个结果文件，返回汇总报告。"""
    reports = [validate_result(p) for p in result_paths]
    all_passed = all(r["passed"] for r in reports)
    all_issues = []
    for r in reports:
        all_issues.extend([f"[{r['path']}] {issue}" for issue in r["issues"]])

    return {
        "passed": all_passed,
        "reports": reports,
        "issues": all_issues,
    }


def _check_assumption_violations(
    df: pd.DataFrame, dynamic_ltm: DynamicLTM
) -> list[EmpiricalFinding]:
    """V11.2 修复（Bug 6）：回归纯机械检查，不做分布/关系分析。

    项目约定：ResultReviewer only performs mechanical checks
    (NaN/Inf, reasonable ranges, empty/trivial results) without
    analyzing data distributions or variable relationships.

    分布形态（Shapiro-Wilk）和变量关系（Pearson/R²）属于 LLM 提炼范畴，
    应交给 Reflection 节点基于完整 stdout 分析，而非在 ResultReviewer 用
    统计检验自动产出 refuted finding 误触发 Clarifier 修正。

    保留：
    - 样本独立性检验（ID 类列重复值）：属于边界条件检查，非分布/关系分析
    """
    findings: list[EmpiricalFinding] = []
    if dynamic_ltm is None:
        return findings

    # 样本独立性检验（ID 类列重复值）— 边界条件检查，保留
    for col in df.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ("id", "编号", "样本号")) and not df[col].isna().all():
            dup_count = int(df[col].duplicated().sum())
            if dup_count > 0:
                findings.append(EmpiricalFinding(
                    id=f"auto_independence_{col}",
                    run_id="auto",
                    source_node="result_reviewer",
                    assumption_tested=f"{col} 样本独立性",
                    evidence=f"存在 {dup_count} 个重复 ID（总样本 {len(df)}）",
                    verdict="refuted",
                    confidence=0.85,
                    suggested_fix="使用混合效应模型 / 聚类稳健标准误 / 删除重复样本",
                ))

    return findings


def result_reviewer_node(
    state: GraphState,
    runtime: Any | None = None,
    config: dict | None = None,
) -> GraphState:
    """结果审查节点：验证 Coder 产出的结果是否合理。

    该节点不调用 LLM。若验证失败，会按错误类型决定回滚目标。

    扩展（假设检验）：验证通过后，对结果做机械假设检验，产出 EmpiricalFinding
    写入 empirical 层。若有高置信度 refuted 发现，设置 trigger_clarifier_revision
    让 Reflection 路径直接触发 Clarifier 修正（避免无谓的 LLM Reflection 调用）。
    """
    artifacts = state.get("artifacts", ArtifactBundle()).model_copy(deep=True)
    control = state.get("control", ControlState()).model_copy(deep=True)
    empirical = state.get("empirical", EmpiricalLayer()).model_copy(deep=True)

    if not artifacts.result_paths:
        control.phase = "result_review_failed"
        control.coder_error_count += 1
        control.coder_error_log.append("ResultReviewer: 没有结果文件路径。")
        control.coder_rollback_target = "architect"
        # V9 修复：标记清空 result_paths（虽然此处已为空，但保持一致性）
        artifacts.clear_result_paths = True
        return {"control": control, "artifacts": artifacts, "empirical": empirical}

    report = validate_results(artifacts.result_paths)
    if not report["passed"]:
        control.phase = "result_review_failed"
        control.coder_error_count += 1
        control.coder_error_log.extend(report["issues"])

        # V10 修复：保存 ResultReviewer 拒绝原因，供 Architect 针对性调整模型设计
        # 让 Architect 区分 "Coder 执行失败"（语法/API）和 "Coder 成功但结果质量不通过"
        control.last_result_review_issues = list(report["issues"])

        # V10 修复：ResultReviewer 拒绝时强制走 architect（而非根据 issue 文本判断）
        # 原逻辑：根据 issue 文本决定回退到 architect 或 clarifier，但 ResultReviewer
        # 拒绝的本质是"模型设计导致结果质量不通过"（如常量列、边界值），这是 Architect
        # 的职责而非 Clarifier 的职责。Clarifier 修正的是假设，Architect 修正的是模型约束。
        control.coder_rollback_target = "architect"

        # V9 修复：清空旧 result_paths，避免 writer 误用旧结果文件
        # 同时让 route_after_reflection 看到 result_paths 为空，正确走回退路径
        artifacts.result_paths = []
        artifacts.clear_result_paths = True
        logger.warning("ResultReviewer 失败：%s", report["issues"])
        return {"control": control, "artifacts": artifacts, "empirical": empirical}

    # ── 验证通过后做假设检验（机械、零 LLM 成本）──
    control.phase = "result_review_passed"
    logger.info("ResultReviewer 通过：%s", artifacts.result_paths)

    dynamic_ltm = state.get("dynamic_ltm", DynamicLTM())
    auto_findings: list[EmpiricalFinding] = []
    for path_str in artifacts.result_paths:
        path = Path(path_str)
        if not path.exists():
            continue
        df = _read_result(path)
        if df is None or df.empty:
            continue
        auto_findings.extend(_check_assumption_violations(df, dynamic_ltm))

    if auto_findings:
        # 与已有 findings 合并（同 assumption_tested 取置信度高者，由 reducer 处理）
        empirical.findings.extend(auto_findings)
        # 重新派生 refuted/open_questions（与 reducer 逻辑保持一致）
        from modeling_assistant.schemas.state import _rebuild_empirical_derived_fields
        _rebuild_empirical_derived_fields(empirical)

        # 判定是否直接触发 Clarifier 修正（跳过 Reflection LLM 调用）
        has_refuted = any(
            f.verdict == "refuted" and f.confidence >= REFUTED_CONFIDENCE_THRESHOLD
            for f in auto_findings
        )
        if has_refuted and control.empirical_revision_count < control.empirical_revision_budget:
            control.trigger_clarifier_revision = True
            control.empirical_revision_count += 1
            logger.info(
                "ResultReviewer 机械检验触发 Clarifier 修正（已用 %d/%d 预算）",
                control.empirical_revision_count,
                control.empirical_revision_budget,
            )
        logger.info("ResultReviewer 假设检验产出 %d 条发现", len(auto_findings))

    return {"control": control, "artifacts": artifacts, "empirical": empirical}
