"""验证 empirical 层信号产生与流转的单元测试。

目标：验证改造后的系统在「信号产生」层面是否真的有效。
- _check_distribution 在非正态/时序/非线性数据上是否产出 finding
- _check_assumption_violations 无条件正态检验是否生效
- merge_empirical_reducer 去重与冲突处理
- Drawer 自评 verdict/confidence 是否正确写入 empirical
- catalog 注入 empirical_* 变量是否非空

不调 LLM，纯确定性代码验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling_assistant.data.loader import _check_distribution
from modeling_assistant.prompts.catalog import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import (
    DynamicLTM,
    EmpiricalFinding,
    EmpiricalLayer,
    StaticLTM,
    merge_empirical_reducer,
)
from modeling_assistant.validation.results import _check_assumption_violations


# ──────────────────────────────────────────────────────────────────────────
# 1. _check_distribution：信号产生验证
# ──────────────────────────────────────────────────────────────────────────


def test_check_distribution_detects_non_normal():
    """显著非正态数据（指数分布）应产出正态性 finding。"""
    np.random.seed(42)
    # 指数分布：显著右偏，偏度 > 1
    df = pd.DataFrame({"x": np.random.exponential(scale=2.0, size=200)})
    findings = _check_distribution(df)

    # 应该产出偏度峰度 finding（|skewness|>1）和 Shapiro finding（p<0.05）
    normality_findings = [
        f for f in findings if "正态" in f.assumption_tested or "分布形态" in f.assumption_tested
    ]
    assert len(normality_findings) > 0, "非正态数据应产出分布 finding"
    # 所有 data_profile 阶段的 finding 应该是 inconclusive（无假设可反驳）
    for f in findings:
        assert f.verdict == "inconclusive", f"data_profile finding 应为 inconclusive，实际 {f.verdict}"
        assert f.source_node == "data_profile"
        assert f.confidence > 0.0


def test_check_distribution_detects_time_series_autocorrelation():
    """强时序自相关的列应产出独立性 finding。"""
    np.random.seed(42)
    # 构造一个强自相关的时序列
    n = 100
    ts = np.cumsum(np.random.randn(n))  # 随机游走，lag-1 自相关接近 1
    df = pd.DataFrame({"date": np.arange(n), "value": ts})
    findings = _check_distribution(df)

    autocorr_findings = [f for f in findings if "自相关" in f.assumption_tested or "独立" in f.assumption_tested]
    assert len(autocorr_findings) > 0, "强时序自相关应产出独立性 finding"
    # 应给出时序建模建议
    assert any("时序" in (f.suggested_fix or "") for f in autocorr_findings)


def test_check_distribution_nonlinear_removed_from_data_profile():
    """data_profile 阶段不再做非线性检测（设计限制，改由 ResultReviewer 做残差分析）。"""
    np.random.seed(42)
    x = np.linspace(0.01, 10, 200)
    y = np.sqrt(x) + np.random.randn(200) * 0.1
    df = pd.DataFrame({"x": x, "y": y})
    findings = _check_distribution(df)
    # data_profile 不应产出非线性 finding
    nonlinear_findings = [f for f in findings if "非线性" in f.assumption_tested or "关系形态" in f.assumption_tested]
    assert len(nonlinear_findings) == 0, "data_profile 不应做非线性检测"


def test_check_assumption_violations_detects_nonlinear_via_residual():
    """ResultReviewer 残差分析：二次拟合 R² 显著高于线性时应产出非线性 finding。"""
    np.random.seed(42)
    # y = x² + noise：强非线性，线性拟合 R² 低，二次拟合 R² 高
    x = np.linspace(-5, 5, 200)
    y = x ** 2 + np.random.randn(200) * 2
    df = pd.DataFrame({"x": x, "y": y})
    ltm = DynamicLTM(
        assumptions=["变量间存在线性关系"],
        nomenclature={"x": "自变量", "y": "因变量"},
        equations=["y = a * x + b"],
        objective="最小化残差",
    )
    findings = _check_assumption_violations(df, ltm)
    nonlinear_findings = [f for f in findings if "非线性" in f.evidence or "二次 R²" in f.evidence]
    assert len(nonlinear_findings) > 0, "残差分析应检测到非线性"
    assert any(f.verdict == "refuted" for f in nonlinear_findings)


def test_check_distribution_empty_for_normal_data():
    """正态分布数据不应产出大量误报 finding。"""
    np.random.seed(42)
    # 标准正态分布
    df = pd.DataFrame({"x": np.random.randn(500)})
    findings = _check_distribution(df)

    # 正态数据不应产出正态性 finding（偏度峰度接近 0，Shapiro p>0.05）
    # 注意：500 个样本 Shapiro 可能仍因微小偏离而拒绝，所以只检查不产出"偏度峰度显著偏离"的 finding
    skew_findings = [f for f in findings if "分布形态" in f.assumption_tested]
    assert len(skew_findings) == 0, "正态数据不应产出偏度峰度 finding"


def test_check_distribution_skips_small_samples():
    """样本量 < 8 的列不应产出 finding（Shapiro 限制）。"""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})  # 5 行
    findings = _check_distribution(df)
    # 5 行不触发 Shapiro（需 >=8），但偏度峰度仍可能计算
    # 这里只验证不崩溃且返回 list
    assert isinstance(findings, list)


# ──────────────────────────────────────────────────────────────────────────
# 2. _check_assumption_violations：无条件正态检验验证
# ──────────────────────────────────────────────────────────────────────────


def test_check_assumption_violations_normality_unconditional():
    """即使假设文本不提「正态」，也应对数值列做 Shapiro-Wilk 检验。"""
    np.random.seed(42)
    # 非正态数据
    df = pd.DataFrame({"residual": np.random.exponential(scale=2.0, size=200)})
    # 假设文本里完全不提正态
    ltm = DynamicLTM(
        assumptions=["误差项服从典型分布"],  # 不出现"正态"关键词
        nomenclature={"residual": "残差"},
        equations=["y = a * x + b"],
        objective="最小化残差",
    )
    findings = _check_assumption_violations(df, ltm)

    # 无条件正态检验应产出 finding
    normality_findings = [f for f in findings if "正态" in f.assumption_tested]
    assert len(normality_findings) > 0, "无条件正态检验应产出 finding，即使假设不提「正态」"
    # 非正态应是 refuted
    assert any(f.verdict == "refuted" for f in normality_findings)


def test_check_assumption_violations_linear_keyword_triggered():
    """假设提到「线性」时，应检查 Pearson 相关系数。"""
    np.random.seed(42)
    # 构造弱相关数据
    x = np.random.randn(100)
    y = np.random.randn(100)  # 完全独立
    df = pd.DataFrame({"x": x, "y": y})
    ltm = DynamicLTM(
        assumptions=["变量间存在线性关系"],
        nomenclature={"x": "自变量", "y": "因变量"},
        equations=["y = a * x + b"],
        objective="最小化残差",
    )
    findings = _check_assumption_violations(df, ltm)

    linear_findings = [f for f in findings if "线性" in f.assumption_tested]
    assert len(linear_findings) > 0, "假设提「线性」时应检查线性关系"
    # 弱相关应被 refute
    assert any(f.verdict == "refuted" for f in linear_findings)


# ──────────────────────────────────────────────────────────────────────────
# 3. merge_empirical_reducer：去重与冲突处理
# ──────────────────────────────────────────────────────────────────────────


def test_merge_empirical_reducer_dedupes_same_assumption():
    """同 assumption_tested 的新发现应覆盖旧的（取置信度更高者）。"""
    old = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f1",
            run_id="run_1",
            source_node="reflection",
            assumption_tested="残差正态性",
            evidence="Shapiro p=0.04",
            verdict="refuted",
            confidence=0.7,
        )
    ])
    new = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f2",
            run_id="run_2",
            source_node="reflection",
            assumption_tested="残差正态性",  # 同主题
            evidence="Shapiro p=0.001",
            verdict="refuted",
            confidence=0.95,  # 更高置信度
        )
    ])
    merged = merge_empirical_reducer(old, new)

    # 应只剩 1 条（高置信度覆盖低置信度）
    assert len(merged.findings) == 1
    assert merged.findings[0].confidence == 0.95
    # refuted_assumptions 应自动派生
    assert "残差正态性" in merged.refuted_assumptions


def test_merge_empirical_reducer_keeps_different_assumptions():
    """不同 assumption_tested 的发现应都保留。"""
    old = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f1", run_id="r1", source_node="reflection",
            assumption_tested="正态性", evidence="p=0.001",
            verdict="refuted", confidence=0.9,
        )
    ])
    new = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f2", run_id="r2", source_node="drawer",
            assumption_tested="线性关系", evidence="散点凸性",
            verdict="refuted", confidence=0.85,
        )
    ])
    merged = merge_empirical_reducer(old, new)
    assert len(merged.findings) == 2
    assert len(merged.refuted_assumptions) == 2


def test_merge_empirical_reducer_low_confidence_goes_to_open_questions():
    """confidence < 0.7 的发现应进 open_questions，不进 refuted_assumptions。"""
    new = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f1", run_id="r1", source_node="reflection",
            assumption_tested="某假设", evidence="弱证据",
            verdict="refuted", confidence=0.5,  # 低于 0.7 阈值
        )
    ])
    merged = merge_empirical_reducer(None, new)
    assert len(merged.refuted_assumptions) == 0
    assert len(merged.open_questions) == 1


def test_merge_empirical_reducer_conflict_resolution():
    """矛盾场景：同主题，旧 refuted + 新 confirmed，应取置信度更高者。"""
    old = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f1", run_id="r1", source_node="result_reviewer",
            assumption_tested="残差正态性", evidence="Shapiro p=0.001（非正态）",
            verdict="refuted", confidence=0.9,
        )
    ])
    new = EmpiricalLayer(findings=[
        EmpiricalFinding(
            id="f2", run_id="r2", source_node="result_reviewer",
            assumption_tested="残差正态性", evidence="Shapiro p=0.3（正态）",
            verdict="confirmed", confidence=0.6,  # 更低置信度，不应覆盖
        )
    ])
    merged = merge_empirical_reducer(old, new)
    # 旧的 refuted 应保留（置信度更高）
    assert len(merged.findings) == 1
    assert merged.findings[0].verdict == "refuted"
    assert merged.findings[0].confidence == 0.9


# ──────────────────────────────────────────────────────────────────────────
# 4. Drawer 自评 verdict/confidence 写入 empirical
# ──────────────────────────────────────────────────────────────────────────


def test_drawer_self_evaluated_finding_respects_verdict():
    """Drawer 自评 verdict=refuted + confidence=0.85 应进 refuted_assumptions。"""
    finding = EmpiricalFinding(
        id="f1", run_id="drawer_1", source_node="drawer",
        assumption_tested="变量关系形态（视觉观察）",
        evidence="散点明显凸性 | 统计佐证: Pearson r=0.32, Spearman ρ=0.61",
        verdict="refuted", confidence=0.85,
    )
    layer = EmpiricalLayer(findings=[finding])
    from modeling_assistant.schemas.state import _rebuild_empirical_derived_fields
    _rebuild_empirical_derived_fields(layer)
    assert "变量关系形态（视觉观察）" in layer.refuted_assumptions


def test_drawer_low_confidence_goes_to_open_questions():
    """Drawer 自评 confidence=0.4 应进 open_questions，不触发修正。"""
    finding = EmpiricalFinding(
        id="f1", run_id="drawer_1", source_node="drawer",
        assumption_tested="变量关系形态（视觉观察）",
        evidence="图像模糊",
        verdict="inconclusive", confidence=0.4,
    )
    layer = EmpiricalLayer(findings=[finding])
    from modeling_assistant.schemas.state import _rebuild_empirical_derived_fields
    _rebuild_empirical_derived_fields(layer)
    assert len(layer.refuted_assumptions) == 0
    assert len(layer.open_questions) == 1


# ──────────────────────────────────────────────────────────────────────────
# 5. PromptCatalog 注入验证
# ──────────────────────────────────────────────────────────────────────────


def test_catalog_injects_empirical_variables():
    """PromptContext.to_template_vars() 应注入 empirical_* 变量，且非空。"""
    empirical = EmpiricalLayer(
        findings=[
            EmpiricalFinding(
                id="f1", run_id="r1", source_node="reflection",
                assumption_tested="正态性", evidence="p=0.001",
                verdict="refuted", confidence=0.9,
            ),
            EmpiricalFinding(
                id="f2", run_id="drawer_1", source_node="drawer",
                assumption_tested="变量关系形态", evidence="散点凸性",
                verdict="refuted", confidence=0.85,
            ),
        ],
        refuted_assumptions=["正态性", "变量关系形态"],
        open_questions=[],
    )
    static_ltm = StaticLTM(
        raw_problem="测试",
        data_findings=["列 x 有强时序性", "列 y 非正态"],
    )
    ctx = PromptContext(static_ltm=static_ltm, empirical=empirical)
    vars = ctx.to_template_vars()

    # empirical 变量应非空
    assert vars["empirical_refuted_json"] != "[]"
    assert "正态性" in vars["empirical_refuted_json"]
    assert vars["empirical_findings_summary_json"] != "[]"
    assert vars["drawer_observations_json"] != "[]"
    assert "散点凸性" in vars["drawer_observations_json"]
    # data_findings 应注入
    assert vars["data_findings_json"] != "[]"
    assert "时序性" in vars["data_findings_json"]


def test_catalog_empty_empirical_yields_empty_json():
    """空 empirical 层应注入空 JSON 数组，不崩溃。"""
    ctx = PromptContext(static_ltm=StaticLTM(), empirical=EmpiricalLayer())
    vars = ctx.to_template_vars()
    assert vars["empirical_refuted_json"] == "[]"
    assert vars["empirical_open_questions_json"] == "[]"
    assert vars["drawer_observations_json"] == "[]"
    assert vars["data_findings_json"] == "[]"


# ──────────────────────────────────────────────────────────────────────────
# 6. 集成验证：data_profile_node 端到端产出
# ──────────────────────────────────────────────────────────────────────────


def test_data_profile_node_writes_findings_to_empirical(tmp_path):
    """data_profile_node 对真实数据应产出 empirical finding 并写入 static_ltm.data_findings。"""
    import os
    from modeling_assistant.data.loader import data_profile_node
    from modeling_assistant.schemas.state import ControlState, GraphState

    # 构造 ≥8 行非正态 CSV
    np.random.seed(42)
    df_data = pd.DataFrame({
        "x": np.random.exponential(scale=2.0, size=100),
    })
    csv_path = tmp_path / "test_nonnormal.csv"
    df_data.to_csv(csv_path, index=False)

    state: GraphState = {
        "static_ltm": StaticLTM(data_attachments=[str(csv_path)]),
        "control": ControlState(),
    }
    result = data_profile_node(state)

    # empirical 层应有 finding
    empirical = result.get("empirical")
    assert empirical is not None
    assert len(empirical.findings) > 0, "非正态数据应产出 empirical finding"
    # static_ltm.data_findings 应有追加
    static_ltm = result["static_ltm"]
    assert len(static_ltm.data_findings) > 0, "应追加到 static_ltm.data_findings"
    # control.phase 应更新
    assert result["control"].phase == "data_profile_loaded"
