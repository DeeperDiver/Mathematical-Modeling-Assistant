"""数据加载与画像生成。

读取用户提供的真实数据附件，生成机器可验证的 DataProfile，
供后续 Agent 基于真实数据结构做建模决策。

扩展（数据认知）：在画像阶段对数值列做分布检验（正态性、偏度峰度、
时序自相关），把发现写入 empirical 层。这让「看到数据分布后才发现
正态假设不成立」能在数据加载阶段就发生，而非等到 Coder 执行后被动触发。
此阶段产出 inconclusive 发现（数据特性描述），不直接判定 refuted
（因为 dynamic_ltm 还没生成，无假设可反驳）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling_assistant.data.facts import annotate_parse_hints
from modeling_assistant.schemas.state import (
    ColumnProfile,
    ControlState,
    DataProfile,
    EmpiricalFinding,
    EmpiricalLayer,
    GraphState,
    StaticLTM,
)

logger = logging.getLogger(__name__)


def _infer_dtype(series: pd.Series) -> str:
    """推断列类型，返回简化标签。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_categorical_dtype(series) or series.nunique() / max(len(series), 1) < 0.1:
        return "category"
    return "text"


def _compute_column_profile(name: str, series: pd.Series) -> ColumnProfile:
    """为单列生成画像。"""
    dtype = _infer_dtype(series)
    missing_rate = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))

    sample_values: list[Any] = []
    try:
        sample_values = (
            series.dropna().head(5).astype(str).tolist()
            if len(series) > 0
            else []
        )
    except Exception:
        sample_values = []

    profile = ColumnProfile(
        name=name,
        dtype=dtype,
        missing_rate=missing_rate,
        unique_count=unique_count,
        sample_values=sample_values,
    )

    if dtype in ("int", "float"):
        try:
            profile.min = float(series.min()) if not pd.isna(series.min()) else None
            profile.max = float(series.max()) if not pd.isna(series.max()) else None
            profile.mean = float(series.mean()) if not pd.isna(series.mean()) else None
            profile.std = float(series.std()) if not pd.isna(series.std()) else None
        except Exception:
            pass

    return profile


def _detect_issues(df: pd.DataFrame) -> list[str]:
    """检测数据质量问题。"""
    issues: list[str] = []

    if df.empty:
        issues.append("数据表为空。")
        return issues

    # 完全空列
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        issues.append(f"以下列完全缺失：{empty_cols}")

    # 缺失率过高的列
    high_missing = [col for col in df.columns if df[col].isna().mean() > 0.5]
    if high_missing:
        issues.append(f"以下列缺失率超过 50%：{high_missing}")

    # 重复行
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"存在 {dup_count} 行重复数据。")

    # 常量列（无信息）
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    if constant_cols:
        issues.append(f"以下列为常量（无区分信息）：{constant_cols}")

    return issues


def _check_distribution(df: pd.DataFrame) -> list[EmpiricalFinding]:
    """对数值列做分布检验，产出数据特性发现。

    此阶段是「数据认知」而非「假设检验」：dynamic_ltm 还没生成，无法判定
    假设是否被反驳。因此所有发现都标记为 inconclusive，让下游 Mathematician
    在建模时参考（如：发现某列非正态，建模时可主动避免正态假设）。

    检验项：
    1. 正态性：Shapiro-Wilk（8 ≤ n ≤ 5000）
    2. 偏度峰度：|skewness| > 1 或 |kurtosis| > 3 视为显著偏离
    3. 时序自相关：对疑似时序列做 lag-1 自相关检验
    4. 强非线性关系：两两数值列 Spearman 相关显著但 Pearson 差异大
    """
    findings: list[EmpiricalFinding] = []
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return findings

    finding_counter = 0

    # 1 & 2. 正态性 + 偏度峰度
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 8:
            continue

        # 偏度峰度（无样本量限制）
        try:
            skewness = float(series.skew())
            kurtosis = float(series.kurtosis())  # 超额峰度
            if abs(skewness) > 1 or abs(kurtosis) > 3:
                finding_counter += 1
                findings.append(EmpiricalFinding(
                    id=f"data_skew_{finding_counter}",
                    run_id="data_profile",
                    source_node="data_profile",
                    assumption_tested=f"{col} 列分布形态",
                    evidence=f"偏度={skewness:.3f}, 超额峰度={kurtosis:.3f}（显著偏离正态）",
                    verdict="inconclusive",
                    confidence=0.6,
                    suggested_fix="建模时避免对该列直接做正态假设，考虑对数/Box-Cox 变换",
                ))
        except Exception as exc:
            logger.debug("偏度峰度计算失败 %s: %s", col, exc)

        # Shapiro-Wilk（样本量限制）
        if 8 <= len(series) <= 5000:
            try:
                from scipy import stats

                stat, p = stats.shapiro(series)
                if p < 0.05:
                    finding_counter += 1
                    findings.append(EmpiricalFinding(
                        id=f"data_shapiro_{finding_counter}",
                        run_id="data_profile",
                        source_node="data_profile",
                        assumption_tested=f"{col} 列正态性",
                        evidence=f"Shapiro-Wilk 统计量={stat:.3f}, p={p:.4f} < 0.05（非正态）",
                        verdict="inconclusive",
                        confidence=0.65,
                        suggested_fix="若建模需要正态假设，需做变换或改用非参数方法",
                    ))
            except Exception as exc:
                logger.debug("Shapiro 检验失败 %s: %s", col, exc)

    # 3. 时序自相关（对疑似时序列）
    # 启发式：列名含 time/date/年/月/日 或列为单调递增的整数
    time_keywords = ("time", "date", "year", "month", "day", "时", "年", "月", "日")
    for col in numeric_df.columns:
        col_lower = str(col).lower()
        series = numeric_df[col].dropna()
        if len(series) < 10:
            continue
        is_time_col = any(k in col_lower for k in time_keywords)
        is_monotonic = series.is_monotonic_increasing
        if not (is_time_col or is_monotonic):
            continue
        try:
            # lag-1 自相关
            n = len(series)
            mean = series.mean()
            var = series.var()
            if var > 0:
                autocorr = float(((series.iloc[:-1] - mean) * (series.iloc[1:] - mean)).sum() / (n * var))
                if abs(autocorr) > 0.5:
                    finding_counter += 1
                    findings.append(EmpiricalFinding(
                        id=f"data_autocorr_{finding_counter}",
                        run_id="data_profile",
                        source_node="data_profile",
                        assumption_tested=f"{col} 列样本独立性（时序自相关）",
                        evidence=f"lag-1 自相关系数={autocorr:.3f}（>0.5，存在强时序自相关）",
                        verdict="inconclusive",
                        confidence=0.7,
                        suggested_fix="该列疑似时序列，建模时需考虑时序结构（如 ARIMA、混合效应模型），避免假设样本独立",
                    ))
        except Exception as exc:
            logger.debug("自相关检验失败 %s: %s", col, exc)

    # 4. 非线性关系检测已移除
    # 设计反思：原方案用 Pearson r² 与 Spearman ρ² 的比值检测非线性，
    # 但实测发现对单调非线性函数（sqrt/log/x³），Pearson 也很高，差异不大；
    # 而非单调非线性（U 型）时 Spearman 也接近 0。此方法不可靠。
    # 真正的非线性检测应在 Coder 执行后由 ResultReviewer 做残差分析
    # （比较线性拟合 vs 多项式拟合的 R²），那时才有建模残差可用。
    # data_profile 阶段无法可靠检测非线性关系，这是设计边界。

    return findings


def _read_file(path: Path) -> pd.DataFrame | None:
    """根据扩展名读取单个文件。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
        logger.warning("不支持的文件格式，已跳过：%s", path)
        return None
    except Exception as exc:
        logger.warning("读取文件失败 %s: %s", path, exc)
        return None


def load_data_profile(file_paths: list[str]) -> DataProfile:
    """读取多个数据文件，合并生成统一的数据画像。"""
    profile = DataProfile(file_paths=file_paths)

    if not file_paths:
        return profile

    frames: list[pd.DataFrame] = []
    readable_paths: list[str] = []

    for p in file_paths:
        path = Path(p)
        if not path.exists():
            logger.warning("数据文件不存在：%s", p)
            continue
        df = _read_file(path)
        if df is not None:
            frames.append(df)
            readable_paths.append(str(path.resolve()))

    if not frames:
        profile.issues.append("未成功读取任何数据文件。")
        return profile

    profile.file_paths = readable_paths

    # 简单策略：如果只有一个文件，直接用它；多个文件时按列拼接
    if len(frames) == 1:
        df = frames[0]
    else:
        try:
            df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        except Exception as exc:
            logger.warning("合并多个数据文件失败：%s", exc)
            df = frames[0]

    profile.total_rows = int(len(df))
    profile.total_cols = int(len(df.columns))

    # 列画像
    profile.columns = [
        _compute_column_profile(str(col), df[col]) for col in df.columns
    ]

    # V11 修复：为 text 列自动生成解析建议（纯机器推断，不调用 LLM）
    annotate_parse_hints(profile.columns)

    # 问题检测
    profile.issues.extend(_detect_issues(df))

    # 样本（最多 5 行，字符串化）
    try:
        # V11.4 修复：距离矩阵等文件的列名可能是 int（如 0,1,2,...），
        # to_dict 后 dict key 是 int，与 schema list[dict[str, Any]] 不符，
        # 导致 Pydantic 序列化失败 → checkpoint 反序列化 data_profile 为 dict。
        # 这里强制把 key 转成 str，保证 schema 一致性。
        raw_records = df.head(5).astype(str).to_dict(orient="records")
        profile.sample_head = [
            {str(k): v for k, v in record.items()}
            for record in raw_records
        ]
    except Exception:
        profile.sample_head = []

    # 数值列相关性矩阵
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty and len(numeric_df.columns) >= 2:
        try:
            corr = numeric_df.corr()
            profile.correlation_matrix = {
                str(row): {str(col): float(val) for col, val in corr.loc[row].items()}
                for row in corr.index
            }
        except Exception:
            pass

    return profile


def data_profile_node(
    state: GraphState,
    runtime: Any | None = None,
    config: dict | None = None,
) -> GraphState:
    """数据画像节点：读取附件并生成 DataProfile。

    该节点不调用 LLM，完全基于确定性代码生成真实数据画像。

    扩展（数据认知）：画像后对数值列做分布检验，产出 EmpiricalFinding 写入
    empirical 层。这让「看到数据分布后才发现正态假设不成立」能在数据加载
    阶段就发生，而非等到 Coder 执行后被动触发。同时把发现追加到
    static_ltm.data_findings，让数据认知更新可被下游 LTM 感知。
    """
    static_ltm = state.get("static_ltm", StaticLTM()).model_copy(deep=True)
    control = state.get("control", ControlState()).model_copy(deep=True)
    empirical = state.get("empirical", EmpiricalLayer()).model_copy(deep=True)

    if static_ltm.data_attachments:
        static_ltm.data_profile = load_data_profile(static_ltm.data_attachments)
        if static_ltm.data_profile.issues:
            logger.warning("数据画像发现问题：%s", static_ltm.data_profile.issues)

        # 数据认知：对原始数据做分布检验，产出 inconclusive 发现
        # 重新读取合并后的 DataFrame 用于检验
        frames: list[pd.DataFrame] = []
        for p in static_ltm.data_profile.file_paths:
            df = _read_file(Path(p))
            if df is not None:
                frames.append(df)
        if frames:
            try:
                merged_df = frames[0] if len(frames) == 1 else pd.concat(frames, axis=0, ignore_index=True, sort=False)
                dist_findings = _check_distribution(merged_df)
                if dist_findings:
                    empirical.findings.extend(dist_findings)
                    # 重新派生 open_questions
                    from modeling_assistant.schemas.state import _rebuild_empirical_derived_fields
                    _rebuild_empirical_derived_fields(empirical)
                    # 同步追加到 static_ltm.data_findings（数据认知更新）
                    for f in dist_findings:
                        static_ltm.data_findings.append(
                            f"{f.assumption_tested}: {f.evidence}"
                        )
                    logger.info("数据画像产出 %d 条分布发现", len(dist_findings))
            except Exception as exc:
                logger.warning("分布检验失败: %s", exc)
    else:
        static_ltm.data_profile = DataProfile()

    # V11.4 修复：data_profile 准备好后，重新对 problem_facts 做 category 分类
    # 原因：fact_extractor_node 在 data_profile 之前运行，此时 columns=None，
    # 所有 fact 都被标为 physical_param/count，data_range 类（如"GC 含量正常范围 40%-60%"）
    # 无法被识别。这里在 data_profile 完成后用真实列名重新分类，让校验器能正确跳过
    # data_range 类 fact。
    if static_ltm.problem_facts and static_ltm.data_profile and static_ltm.data_profile.columns:
        from modeling_assistant.data.facts import classify_fact
        reclassified = 0
        for fact in static_ltm.problem_facts:
            new_category = classify_fact(fact, static_ltm.data_profile.columns)
            if new_category != fact.category:
                fact.category = new_category
                reclassified += 1
        if reclassified:
            logger.info(
                "data_profile_node: V11.4 重新分类 %d 个 fact（基于 %d 个数据列）",
                reclassified, len(static_ltm.data_profile.columns),
            )

    control.phase = "data_profile_loaded"
    return {"static_ltm": static_ltm, "empirical": empirical, "control": control}
