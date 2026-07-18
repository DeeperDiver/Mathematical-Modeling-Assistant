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
    """基于动态 LTM 中的假设文本做机械检验，不调 LLM。

    覆盖三类常见假设：
    1. 正态性：对结果文件的数值列无条件做 Shapiro-Wilk 检验
       （分布形态是数据的客观属性，不应依赖假设文本关键词；
       若假设未提正态但数据实际非正态，仍应产出 refuted 让 Clarifier 修正）
    2. 线性关系：假设文本提到「线性」时，检查最大 Pearson 相关系数
       （「线性」是建模假设而非数据属性，保留关键词触发）
    3. 样本独立性：检查 ID 类列是否有重复值

    检验失败（p<0.05 或规则违反）才产出 refuted 发现；
    检验通过产出 confirmed 发现（让下游知道假设已被验证）。
    其余情况不产出，避免污染 empirical 层。
    """
    findings: list[EmpiricalFinding] = []
    if dynamic_ltm is None:
        return findings

    # 假设文本拼合（小写）
    assumptions_text = " ".join(getattr(dynamic_ltm, "assumptions", []) or []).lower()

    numeric_df = df.select_dtypes(include=["number"])

    # 1. 正态性检验（无条件，对所有数值列做）
    # 分布形态是数据客观属性，不应依赖假设文本是否提到「正态」
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        # Shapiro-Wilk 对样本量有限制（建议 8 ≤ n ≤ 5000）
        if 8 <= len(series) <= 5000:
            try:
                from scipy import stats

                stat, p = stats.shapiro(series)
                if p < 0.05:
                    findings.append(EmpiricalFinding(
                        id=f"auto_normality_{col}",
                        run_id="auto",
                        source_node="result_reviewer",
                        assumption_tested=f"{col} 列正态性",
                        evidence=f"Shapiro-Wilk 统计量={stat:.3f}, p={p:.4f} < 0.05",
                        verdict="refuted",
                        confidence=0.9,
                        suggested_fix="对数变换 / Box-Cox 变换 / 改用非参数检验",
                    ))
                else:
                    findings.append(EmpiricalFinding(
                        id=f"auto_normality_{col}_ok",
                        run_id="auto",
                        source_node="result_reviewer",
                        assumption_tested=f"{col} 列正态性",
                        evidence=f"Shapiro-Wilk p={p:.4f} ≥ 0.05",
                        verdict="confirmed",
                        confidence=0.75,
                    ))
            except Exception as exc:
                logger.debug("Shapiro 检验失败 %s: %s", col, exc)

    # 2. 线性关系检验（保留关键词触发：「线性」是建模假设而非数据属性）
    if any(k in assumptions_text for k in ("线性", "linear")) and len(numeric_df.columns) >= 2:
        try:
            corr_matrix = numeric_df.corr(method="pearson").abs()
            # 屏蔽对角线，取最大相关系数
            # 注意：pandas DataFrame.values 可能是只读视图，必须 .copy() 才能修改
            corr_vals = corr_matrix.values.copy()
            np.fill_diagonal(corr_vals, 0)
            max_corr = float(pd.DataFrame(corr_vals, index=corr_matrix.index, columns=corr_matrix.columns).max().max())
            if not np.isnan(max_corr):
                if max_corr < 0.3:
                    findings.append(EmpiricalFinding(
                        id="auto_linear_check",
                        run_id="auto",
                        source_node="result_reviewer",
                        assumption_tested="变量间线性关系",
                        evidence=f"最大 Pearson 相关系数 {max_corr:.3f}，线性关系微弱",
                        verdict="refuted",
                        confidence=0.7,
                        suggested_fix="考虑非线性模型、交互项或核方法",
                    ))
                elif max_corr > 0.7:
                    findings.append(EmpiricalFinding(
                        id="auto_linear_check_ok",
                        run_id="auto",
                        source_node="result_reviewer",
                        assumption_tested="变量间线性关系",
                        evidence=f"最大 Pearson 相关系数 {max_corr:.3f}",
                        verdict="confirmed",
                        confidence=0.75,
                    ))
            # 2b. 残差非线性检验：对最强相关变量对做线性 vs 二次拟合 R² 对比
            # 无条件执行（不依赖 max_corr 阈值）：即使线性相关弱，也可能存在强非线性关系（如 U 型）
            try:
                # 找到最强相关变量对（按绝对值）
                corr_matrix_raw = numeric_df.corr(method="pearson")
                abs_corr = corr_matrix_raw.abs()
                abs_vals = abs_corr.values.copy()
                np.fill_diagonal(abs_vals, 0)
                # 找最大值位置
                max_idx = np.unravel_index(np.nanargmax(abs_vals), abs_vals.shape)
                col1, col2 = numeric_df.columns[max_idx[0]], numeric_df.columns[max_idx[1]]
                x = numeric_df[col1].dropna()
                y = numeric_df[col2].dropna()
                # 对齐索引
                common = x.index.intersection(y.index)
                x = x.loc[common].values
                y = y.loc[common].values
                if len(x) >= 10 and np.std(x) > 0 and np.std(y) > 0:
                    # 线性拟合 R²
                    coeffs_lin = np.polyfit(x, y, 1)
                    y_pred_lin = np.polyval(coeffs_lin, x)
                    ss_res_lin = np.sum((y - y_pred_lin) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0
                    # 二次拟合 R²
                    coeffs_quad = np.polyfit(x, y, 2)
                    y_pred_quad = np.polyval(coeffs_quad, x)
                    ss_res_quad = np.sum((y - y_pred_quad) ** 2)
                    r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
                    # 若二次拟合 R² 显著高于线性（提升 > 0.1），说明存在非线性
                    if r2_quad - r2_lin > 0.1:
                        findings.append(EmpiricalFinding(
                            id=f"auto_nonlinear_{col1}_{col2}",
                            run_id="auto",
                            source_node="result_reviewer",
                            assumption_tested=f"{col1} 与 {col2} 间线性关系",
                            evidence=f"线性 R²={r2_lin:.3f}, 二次 R²={r2_quad:.3f}（提升 {r2_quad - r2_lin:.3f}，存在非线性）",
                            verdict="refuted",
                            confidence=0.8,
                            suggested_fix="改用二次多项式、样条回归或树模型",
                        ))
            except Exception as exc:
                logger.debug("残差非线性检验失败: %s", exc)
        except Exception as exc:
            logger.debug("线性关系检验失败: %s", exc)

    # 3. 样本独立性检验（ID 类列重复值）
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
        return {"control": control, "artifacts": artifacts, "empirical": empirical}

    report = validate_results(artifacts.result_paths)
    if not report["passed"]:
        control.phase = "result_review_failed"
        control.coder_error_count += 1
        control.coder_error_log.extend(report["issues"])

        # 根据问题类型选择回滚目标
        issues_text = "\n".join(report["issues"]).lower()
        if any(k in issues_text for k in ("不存在", "无法读取", "为空表", "没有结果")):
            control.coder_rollback_target = "architect"
        else:
            control.coder_rollback_target = "clarifier"

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
