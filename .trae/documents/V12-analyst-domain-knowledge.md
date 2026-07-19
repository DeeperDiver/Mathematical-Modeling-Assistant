# V12 优化方案：强化 Analyst 为「领域知识包」生成器

## 一、Summary（摘要）

当前 Analyst 节点只产出 `problem_understanding` + `data_schema` 两段自由文本，LLM 对题目的"理解"没有结构化传递给下游。这是所有"换题就要修复"问题的根因。

本次优化将 Analyst 从"破题者"升级为"领域知识包生成器"——让 LLM 看完题目后，产出完整的结构化领域知识（`DomainKnowledge`），覆盖：题目类型、数据关系、常量语义、求解策略、校验规则、执行提示、预期输出结构。下游节点（Coder/Architect/ResultReviewer/Clarifier）通过 PromptCatalog 注入读取，按题目专属知识工作，而非用通用规则硬猜。

**完全替换 V11.4 的 `classify_fact` 机器分类机制**，改由 Analyst LLM 标注每个常量的语义角色。 Analyst 失败时回退为保守默认（全部 physical_param），保留机器提取的数值事实但不依赖机器分类。

## 二、Current State Analysis（现状分析）

### 2.1 Analyst 当前产出（过薄）

[`schemas/responses.py:10-12`](src/modeling_assistant/schemas/responses.py)
```python
class AnalystResponse(BaseModel):
    problem_understanding: str = ""
    data_schema: dict[str, str] = Field(default_factory=dict)
```
只有两段文本，LLM 的理解装进自由文本后，下游无法机械读取。

### 2.2 下游节点用通用规则硬猜（失败根源）

| 节点 | 通用规则 | 失败案例 |
|---|---|---|
| `fact_extractor` + `classify_fact` | 双重判据（强信号词+列名匹配） | real_test2 的 "范 围" 空格导致误判 |
| `ResultReviewer` | "常量列=无信息" | real_test4 的 StartCost=400 被误杀 |
| `Coder` | 模板 A/B 按是否有数据文件选择 | real_test4 多附件关联错误 |
| `Architect` | 通用依赖约束+复杂度约束 | 不知道题目推荐什么算法 |

### 2.3 V11.4 机制的问题

[`data/facts.py:42-84`](src/modeling_assistant/data/facts.py) 的 `classify_fact`：
- 用正则 + 关键词列表硬猜语义类型
- 需要 PDF 空格修复、空格容忍等补丁
- 无法识别"业务常量"（如 StartCost=400 是合理的）
- 每遇到新题目就要加新关键词

**结论**：机器分类永远覆盖不了所有情况，必须让 LLM 标注。

### 2.4 流程顺序问题

当前流程：`problem → fact_extractor → analyst → data_profile → ...`

- `fact_extractor` 在 `analyst` 之前，无法用 Analyst 的标注
- V11.4 的临时方案：在 `data_profile_node` 末尾重新分类（[`data/loader.py:367-384`](src/modeling_assistant/data/loader.py)）
- V12 方案：`fact_extractor` 只提取数值（category 全默认 physical_param），`analyst_node` 末尾用 `constant_semantics` 覆盖 category

## 三、Proposed Changes（具体改动）

### 3.1 新增 `DomainKnowledge` schema

**文件**：[`src/modeling_assistant/schemas/state.py`](src/modeling_assistant/schemas/state.py)

在 `ProblemFact` 之后、`StaticLTM` 之前新增：

```python
class DataRelation(BaseModel):
    """多附件之间的关联关系。"""
    source_file: str  # 文件名或路径
    source_key: list[str]  # 主键列名（可复合）
    target_file: str
    target_key: list[str]
    relation_type: Literal["one_to_one", "one_to_many", "many_to_many", "matrix_index"] = "one_to_one"
    # matrix_index: 距离矩阵类，行列索引是客户/节点编号
    description: str = ""


class ConstantSemantic(BaseModel):
    """单个题目常量的语义标注。"""
    value: float
    unit: str
    semantic_type: Literal[
        "physical_param",      # 物理参数，代码必须字面量出现（如 v=300 m/s）
        "data_range",          # 数据列范围描述，代码不需要字面量（如 GC 40%-60%）
        "business_constant",   # 业务常量，结果列允许为该常量（如 StartCost=400）
        "count",               # 纯计数单位，不参与校验（如 3 枚）
        "threshold",           # 阈值参数，代码需字面量但结果可基于它计算（如 4% 达标线）
        "boundary",            # 边界值，允许结果落在边界（如孕周 10-25）
    ]
    role: str  # 该常量的语义角色描述（如 "导弹速度"、"启动成本"）
    code_usage: Literal[
        "literal_required",    # 必须以字面量出现
        "literal_optional",    # 可字面量也可计算得出
        "data_driven",         # 由数据决定，代码不写字面量
        "not_used",            # 代码中不使用
    ] = "literal_required"


class SolverStrategy(BaseModel):
    """求解策略建议。"""
    problem_type: Literal[
        "statistical",     # 统计建模（回归/分类/假设检验）
        "optimization",    # 优化问题（LP/MIP/VRP/调度）
        "geometric",       # 几何建模（相交/距离/轨迹）
        "time_series",     # 时序建模
        "simulation",      # 仿真/蒙特卡洛
        "mixed",           # 混合类型
        "other",
    ]
    recommended_algorithms: list[str]  # 如 ["ALNS", "MILP"]，按优先级排序
    complexity_limit: str = ""  # 如 "Bootstrap≤200, 网格步长≥0.5"
    fallback_strategy: str = ""  # 失败时的降级策略


class ValidationRule(BaseModel):
    """题目专属的校验规则。"""
    allow_constant_columns: bool = False  # VRP 题允许 StartCost 常量列
    allow_boundary_values: bool = False   # 允许结果落在搜索边界
    allow_data_range_constants: bool = True  # 允许代码不写 data_range 字面量
    required_columns: list[str] = []  # 结果必须包含的列名（如 ["vehicle_id", "route", "cost"]）
    custom_checks: list[str] = []  # 自定义校验描述（如 "路径必须连续"、"容量约束必须满足"）


class ExecutionHint(BaseModel):
    """给 Coder/Drawer 的执行提示。"""
    data_reading_pattern: str = ""  # 如 "多表 join" / "矩阵索引" / "单表"
    dependencies_allowed: list[str] = []  # 允许的额外库（如 ["shapely"]）
    dependencies_forbidden: list[str] = []  # 明确禁止的库
    time_limit_hint: str = ""  # 执行时间提示
    output_file_format: str = "csv"  # 结果文件格式


class OutputSchema(BaseModel):
    """预期结果的结构描述。"""
    columns: list[dict[str, str]]  # 如 [{"name": "vehicle_id", "type": "int", "meaning": "车辆编号"}]
    expected_rows: str = ""  # 如 "每辆车一行" / "每个客户一行"
    key_metrics: list[str] = []  # 关键指标列名（如 ["total_cost", "service_rate"]）


class DomainKnowledge(BaseModel):
    """Analyst 产出的完整领域知识包。

    这是 LLM 对题目的结构化理解，替代自由文本的 problem_understanding，
    让下游节点能机械读取领域知识，而非用通用规则硬猜。
    """
    problem_type: str = "other"  # 简化标签，供路由参考
    problem_understanding: str = ""  # 保留原字段，自由文本概述
    data_relations: list[DataRelation] = []
    constant_semantics: list[ConstantSemantic] = []
    solver_strategy: SolverStrategy | None = None
    validation_rules: ValidationRule | None = None
    execution_hints: ExecutionHint | None = None
    output_schema: OutputSchema | None = None
```

### 3.2 扩展 `StaticLTM`

**文件**：[`src/modeling_assistant/schemas/state.py`](src/modeling_assistant/schemas/state.py)（`StaticLTM` 类内）

在 `fact_role_mapping` 之后新增字段：

```python
class StaticLTM(BaseModel):
    # ... 现有字段 ...
    fact_role_mapping: dict[str, str] = Field(default_factory=dict)
    # V12：Analyst 产出的完整领域知识包
    domain_knowledge: DomainKnowledge | None = None
```

### 3.3 扩展 `AnalystResponse`

**文件**：[`src/modeling_assistant/schemas/responses.py`](src/modeling_assistant/schemas/responses.py)

```python
class AnalystResponse(BaseModel):
    problem_understanding: str = ""
    data_schema: dict[str, str] = Field(default_factory=dict)
    # V12：领域知识包
    domain_knowledge: DomainKnowledge | None = None
```

### 3.4 重写 `analyst.md` 模板

**文件**：[`src/modeling_assistant/prompts/templates/analyst.md`](src/modeling_assistant/prompts/templates/analyst.md)

要求 LLM 产出完整的 `DomainKnowledge` JSON。核心指令：

```markdown
你是数学建模框架中的 Analyst（破题者 + 领域知识工程师）。

任务：
- 只基于原始问题建立静态 LTM
- 提取核心矛盾、数据字段、约束、评价指标
- **产出完整的领域知识包（domain_knowledge）**，让下游节点能机械读取，而非用通用规则硬猜

## 领域知识包要求

### 1. problem_type
标注题目类型：statistical / optimization / geometric / time_series / simulation / mixed / other

### 2. data_relations（多附件关联）
如果有多附件，必须说明它们如何关联：
- 主键/外键关系
- 距离矩阵的行列索引含义
- 合并方式（join/concat）

### 3. constant_semantics（常量语义表）
对 problem_facts 中的每个常量，标注：
- semantic_type: physical_param / data_range / business_constant / count / threshold / boundary
- role: 该常量的语义角色（如 "导弹速度"、"启动成本"）
- code_usage: literal_required / literal_optional / data_driven / not_used

判断规则：
- 物理参数（如 300 m/s 导弹速度）→ physical_param, literal_required
- 数据列范围描述（如 GC 含量 40%-60%）→ data_range, data_driven
- 业务常量（如 启动成本 400 元）→ business_constant, literal_required
- 纯计数（如 3 枚）→ count, not_used
- 达标阈值（如 4% 达标线）→ threshold, literal_required
- 搜索边界（如 孕周 10-25）→ boundary, literal_optional

### 4. solver_strategy
- 推荐算法（按优先级排序）
- 复杂度上限
- 降级策略

### 5. validation_rules
- 是否允许常量列（VRP 题的 StartCost 是合理的）
- 是否允许边界值
- 结果必须包含的列名
- 自定义校验（如 "路径必须连续"）

### 6. execution_hints
- 数据读取模式（单表/多表join/矩阵索引）
- 允许/禁止的额外库
- 执行时间提示

### 7. output_schema
- 预期结果列名和类型
- 预期行数（每辆车一行？每个客户一行？）
- 关键指标列名

## 真实数据画像
{data_profile_json}

## 机器提取的题目常量
{problem_facts_json}

## 必须严格按以下 JSON 格式输出
```json
{{
  "problem_understanding": "...",
  "data_schema": {{}},
  "domain_knowledge": {{
    "problem_type": "optimization",
    "problem_understanding": "...",
    "data_relations": [],
    "constant_semantics": [],
    "solver_strategy": {{}},
    "validation_rules": {{}},
    "execution_hints": {{}},
    "output_schema": {{}}
  }}
}}
```
```

**注意**：fact_extractor 在 analyst 之前运行，所以 problem_facts_json 已经可用。但 data_profile 在 analyst 之后运行，所以 data_profile_json 为空。**需要调整流程顺序**（见 3.5）。

### 3.5 调整流程顺序

**文件**：[`src/modeling_assistant/graph/builder.py`](src/modeling_assistant/graph/builder.py)

当前：`problem → fact_extractor → analyst → data_profile → searcher`

改为：`problem → fact_extractor → data_profile → analyst → searcher`

**原因**：Analyst 需要 data_profile 才能标注 data_relations 和 output_schema。fact_extractor 仍在前（提供 problem_facts），data_profile 前移到 analyst 之前。

**改动**：调整 `graph.add_edge` 顺序：
```python
graph.add_edge("problem", "fact_extractor")
graph.add_edge("fact_extractor", "data_profile")  # 原来是 analyst
graph.add_edge("data_profile", "analyst")          # 原来是 data_profile
graph.add_edge("analyst", "searcher")
```

### 3.6 修改 `analyst_node`

**文件**：[`src/modeling_assistant/agents/nodes.py`](src/modeling_assistant/agents/nodes.py)

```python
def analyst_node(state, runtime=None, config=None):
    resolved_runtime = _runtime(runtime)
    static_ltm = _static_ltm(state)

    system_prompt, audit = _prompt_audit("analyst", state, runtime)
    try:
        response = resolved_runtime.invoke_structured(
            "analyst", state, AnalystResponse, system_prompt=system_prompt
        )
        static_ltm.problem_understanding = response.problem_understanding
        static_ltm.data_schema = response.data_schema
        # V12：写入领域知识包
        if response.domain_knowledge:
            static_ltm.domain_knowledge = response.domain_knowledge
            # 用 constant_semantics 覆盖 problem_facts 的 category
            _apply_constant_semantics(static_ltm)
    except Exception as exc:
        logger.error("Analyst LLM 调用失败: %s", exc)
        # fallback：保留 problem_facts，category 全部默认 physical_param
        if static_ltm.raw_problem and not static_ltm.problem_understanding:
            static_ltm.problem_understanding = "围绕赛题目标、数据可得性、约束条件和评价指标建立结构化理解。"

    control = _control(state)
    control.phase = "static_ltm_initialized"
    return {"static_ltm": static_ltm, "control": control, "prompt_audit": audit}


def _apply_constant_semantics(static_ltm: StaticLTM) -> None:
    """V12：用 Analyst 的 constant_semantics 覆盖 problem_facts 的 category。

    替代 V11.4 的 classify_fact 机器分类。
    """
    if not static_ltm.domain_knowledge or not static_ltm.problem_facts:
        return
    semantics_by_value_unit = {
        (s.value, s.unit): s for s in static_ltm.domain_knowledge.constant_semantics
    }
    for fact in static_ltm.problem_facts:
        semantic = semantics_by_value_unit.get((fact.value, fact.unit))
        if semantic:
            # 映射 semantic_type 到 ProblemFact.category
            type_mapping = {
                "physical_param": "physical_param",
                "data_range": "data_range",
                "business_constant": "physical_param",  # 业务常量保留校验
                "count": "count",
                "threshold": "physical_param",         # 阈值保留校验
                "boundary": "data_range",              # 边界值豁免字面量检查
            }
            fact.category = type_mapping.get(semantic.semantic_type, "physical_param")
            if semantic.role and not fact.role_hint:
                fact.role_hint = semantic.role
```

### 3.7 删除 V11.4 的机器分类机制

**文件**：[`src/modeling_assistant/data/facts.py`](src/modeling_assistant/data/facts.py)

删除：
- `classify_fact` 函数
- `_DATA_RANGE_STRONG_KEYWORDS` 常量
- `_COUNT_UNITS` 常量
- `extract_facts_from_problem` 的 `columns` 参数

简化为纯提取：
```python
def extract_facts_from_problem(raw_problem: str) -> list[ProblemFact]:
    """从题目原文用正则提取所有数值常量。

    纯机器操作，不调用 LLM。提取 (value, unit, context) 三元组。
    category 字段全部默认 physical_param，由 Analyst 的 constant_semantics 覆盖。
    """
    # ... 提取逻辑不变 ...
    fact = ProblemFact(
        value=value, unit=unit, context=context, role_hint=role_hint,
        # category 默认 physical_param，由 analyst_node 覆盖
    )
    facts.append(fact)
```

### 3.8 删除 `data_profile_node` 的 V11.4 重新分类逻辑

**文件**：[`src/modeling_assistant/data/loader.py`](src/modeling_assistant/data/loader.py)

删除 [`loader.py:367-384`](src/modeling_assistant/data/loader.py) 的 V11.4 重新分类块：
```python
# 删除整段：
# if static_ltm.problem_facts and static_ltm.data_profile and static_ltm.data_profile.columns:
#     from modeling_assistant.data.facts import classify_fact
#     ...
```

**原因**：V12 流程中 data_profile 在 analyst 之前，analyst_node 会用 constant_semantics 覆盖 category，不需要 data_profile_node 再做。

### 3.9 修改 `fact_extractor_node`

**文件**：[`src/modeling_assistant/agents/nodes.py`](src/modeling_assistant/agents/nodes.py)

```python
def fact_extractor_node(state, runtime=None, config=None):
    static_ltm = _static_ltm(state)
    control = _control(state)

    if not static_ltm.raw_problem:
        logger.warning("fact_extractor_node: raw_problem 为空，跳过提取")
        control.phase = "facts_extracted"
        return {"static_ltm": static_ltm, "control": control}

    # V12：不再传 columns 参数，纯提取数值
    facts = extract_facts_from_problem(static_ltm.raw_problem)
    static_ltm.problem_facts = facts
    # category 全部默认 physical_param，由 analyst_node 覆盖

    if facts:
        logger.info(
            "fact_extractor_node: 提取到 %d 个数值常量（category 待 Analyst 标注）",
            len(facts),
        )

    control.phase = "facts_extracted"
    return {"static_ltm": static_ltm, "control": control}
```

### 3.10 扩展 `PromptCatalog` 注入

**文件**：[`src/modeling_assistant/prompts/catalog.py`](src/modeling_assistant/prompts/catalog.py)

在 `to_template_vars` 中新增：
```python
# V12：领域知识包注入
domain_knowledge_json = "{}"
if self.static_ltm is not None:
    dk = getattr(self.static_ltm, "domain_knowledge", None)
    if dk is not None:
        domain_knowledge_json = dk.model_dump_json(indent=2)

# 加入返回字典
return {
    # ... 现有字段 ...
    "domain_knowledge_json": domain_knowledge_json,
}
```

### 3.11 修改 `coder.md` 模板

**文件**：[`src/modeling_assistant/prompts/templates/coder.md`](src/modeling_assistant/prompts/templates/coder.md)

注入 `{domain_knowledge_json}`，并调整常量校验规则：
```markdown
## 领域知识包（Analyst 标注，必须遵守）
{domain_knowledge_json}

## 常量校验规则（V12 更新）
- code_usage=literal_required 的常量必须以字面量出现
- code_usage=data_driven 的常量不需要字面量（由数据决定）
- code_usage=not_used 的常量不参与校验
```

### 3.12 修改 `architect.md` 模板

**文件**：[`src/modeling_assistant/prompts/templates/architect.md`](src/modeling_assistant/prompts/templates/architect.md)

注入 `{domain_knowledge_json}`，让 Architect 按 solver_strategy 设计伪代码。

### 3.13 修改 `ResultReviewer`

**文件**：[`src/modeling_assistant/validation/results.py`](src/modeling_assistant/validation/results.py)

读取 `validation_rules`，按规则定制校验：

```python
def _check_empty_or_trivial(df, validation_rules=None):
    issues = []
    if df.empty:
        issues.append("结果文件为空表。")
        return issues
    if len(df) == 1:
        issues.append("结果文件只有一行，可能缺少详细输出。")

    numeric_df = df.select_dtypes(include=["number"])
    allow_constant = (
        validation_rules.allow_constant_columns if validation_rules else False
    )
    for col in numeric_df.columns:
        if numeric_df[col].nunique(dropna=True) <= 1:
            if not allow_constant:
                issues.append(f"数值列 '{col}' 为常量，无区分信息。")
            # V12：允许常量列时不告警（如 VRP 的 StartCost）
    return issues
```

`result_reviewer_node` 读取 `static_ltm.domain_knowledge.validation_rules` 传入校验函数。

### 3.14 修改 `check_code_against_facts`

**文件**：[`src/modeling_assistant/validation/constants.py`](src/modeling_assistant/validation/constants.py)

V12 调整：不再读 `fact.category`（已由 Analyst 标注），改为读 `domain_knowledge.constant_semantics` 的 `code_usage`：

```python
def check_code_against_facts(code, static_ltm, artifacts):
    # V12：从 domain_knowledge.constant_semantics 读取 code_usage
    semantics = []
    if static_ltm.domain_knowledge:
        semantics = static_ltm.domain_knowledge.constant_semantics

    # 只校验 code_usage=literal_required 的常量
    for semantic in semantics:
        if semantic.code_usage != "literal_required":
            continue
        # ... 原有字面量检查逻辑 ...
```

### 3.15 修改 `check_ltm_against_facts`

**文件**：[`src/modeling_assistant/validation/constants.py`](src/modeling_assistant/validation/constants.py)

同样改为读 `code_usage`，跳过 `data_driven` 和 `not_used`。

## 四、Assumptions & Decisions（假设与决策）

### 4.1 假设
- LLM（DeepSeek V4-Pro/Flash）能稳定产出符合 DomainKnowledge schema 的 JSON
- 流程顺序调整（data_profile 前移）不影响 checkpoint 序列化
- 删除 classify_fact 不会破坏现有 pytest（需更新测试）

### 4.2 决策
1. **完全替换 V11.4 机器分类**：删除 classify_fact，依赖 Analyst 标注。失败时回退为全 physical_param。
2. **流程顺序调整**：data_profile 前移到 analyst 之前，让 Analyst 能看到真实数据画像。
3. **category 映射策略**：Analyst 的 semantic_type 有 6 种，ProblemFact.category 只有 3 种。映射规则：
   - physical_param/threshold/business_constant → physical_param（保留校验）
   - data_range/boundary → data_range（豁免字面量）
   - count → count（完全豁免）
4. **业务常量保留字面量校验**：StartCost=400 必须在代码里写 400，但 ResultReviewer 允许结果列为常量。这是两个独立校验。
5. **向后兼容**：domain_knowledge 为 None 时（Analyst 失败），所有 fact 默认 physical_param，校验器按最严格规则工作（与 V11.4 前一致）。

## 五、Verification Steps（验证步骤）

### 5.1 单元测试

更新 [`tests/test_config_and_prompts.py`](tests/test_config_and_prompts.py)：
- 删除 V11.4 的 classify_fact 相关测试（test_classify_fact_*）
- 新增 V12 测试：
  - `test_analyst_node_writes_domain_knowledge`：验证 analyst_node 写入 domain_knowledge
  - `test_apply_constant_semantics_overrides_category`：验证 category 覆盖
  - `test_check_code_against_facts_uses_code_usage`：验证校验器读 code_usage
  - `test_result_reviewer_allows_constant_columns`：验证 validation_rules 生效
  - `test_fact_extractor_no_longer_classifies`：验证 fact_extractor 不再分类
  - `test_domain_knowledge_injected_to_prompt`：验证 PromptCatalog 注入

### 5.2 端到端测试

重跑三个真实测试，对比效果：

| 测试 | V11.4 结果 | V12 预期 |
|---|---|---|
| real_test2 (NIPT) | Coder run_4 成功 | Coder 应更早成功（常量语义正确标注） |
| real_test4 (VRP) | budget 4/4 耗尽，降级 | StartCost 不再被误杀，budget 应未耗尽 |
| real_test3 (烟幕弹) | 未测 | 应能处理（geometric 类型标注） |

### 5.3 回归验证

- pytest 全量通过（除 graph 偶发 LLM 错误）
- 现有 V11.4 修复点（PDF 空格、双重判据）被删除后，功能由 Analyst 接管

## 六、实施顺序（推荐）

1. **Schema 层**：新增 DomainKnowledge + 扩展 StaticLTM + AnalystResponse
2. **Prompt 层**：重写 analyst.md + 扩展 coder.md/architect.md + catalog.py 注入
3. **节点层**：修改 analyst_node + fact_extractor_node + 删除 data_profile_node 的 V11.4 块
4. **校验层**：修改 constants.py（check_ltm_against_facts + check_code_against_facts）+ results.py
5. **流程层**：调整 builder.py 的边顺序
6. **清理层**：删除 facts.py 的 classify_fact + 相关常量
7. **测试层**：更新 pytest + 重跑 real_test2/3/4

## 七、风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 产出格式错误的 DomainKnowledge | invoke_structured 已有重试 + fallback_parser 机制；失败时 domain_knowledge=None，回退保守模式 |
| Analyst 失败导致 category 全错 | 保留 fact_extractor 的纯提取；category 默认 physical_param 是最严格校验，不会放过错误 |
| 流程顺序调整破坏 checkpoint | data_profile 是纯函数节点，无状态依赖，前移安全 |
| LLM 标注的 constant_semantics 不全 | 校验器对未标注的 fact 默认 physical_param（最严格），不会因遗漏而放过错误 |
| DomainKnowledge schema 过于复杂 | 分层设计，所有子 schema 都有默认值，LLM 可以只填关键字段 |
