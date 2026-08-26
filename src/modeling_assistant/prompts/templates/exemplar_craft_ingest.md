你是数学建模论文行文分析师。给定一篇优秀论文全文，提炼它的「行文技艺」——
也就是正文中**数学推导、算法分析、模型解释、行文语言、图片位置规划、
正文侧重点与论证链条、标志性句式**各项技艺是如何组织、展开和呈现的。

## 第一原则（必须严格遵守）

**只记录"怎么讲"，不记录"讲了什么"**：
- 禁止保存具体公式内容、数值结果、算法参数、模型系数。
- 只提炼组织顺序、展开节奏、详略策略、表达动作与功能化句型。
- `text` 范例句必须**去题目化**：可保留通用表述（如"由上式可得"），
  不得包含具体题目对象、数值与专有模型参数。

## 六大技艺的提炼要求

### 1. 数学推导（derivation）
对每处重要推导记录：
- `section`：出现在哪个章节；`trigger`：什么情况触发推导
- `organization`：展开顺序（如 符号引入→前提假设→推导步骤→目标函数→物理解释）
- `notation_usage`：符号使用纪律；`depth_strategy`：详略策略
- `text_formula_ratio`：文字与公式配合节奏
- `closing_moves`：推导完成后的收尾动作（解释每项含义、联系背景）
- `examples`：推导动机句/过渡句/收尾句的功能化范例（每条≤60字，去题目化）

### 2. 算法分析（algorithm）
对论文中出现的算法/求解方法记录：
- `algorithm_type`：启发式/精确/贪心/DP/随机模拟等
- `presentation`：用伪代码/流程图/步骤列表哪种呈现
- `flow_organization`：输入→初始化→迭代→终止→输出的叙述顺序
- `complexity_analysis`：复杂度论证怎么安排；`convergence_justification`：收敛/可行性论证
- `result_reporting`：结果如何与算法步骤对应报告
- `support_figures`：用什么图支撑（收敛曲线/目标值变化/耗时对比）
- `examples`：算法描述句/复杂度句/收敛论证句的范例

### 3. 模型解释（interpretation）
记录模型建立后如何被解释：
- `target`：解释对象（模型整体/参数/边界条件/结果）
- `organization`：解释顺序（先整体后局部、先含义后影响）
- `domain_linking`：如何与题目背景/业务意义挂钩
- `parameter_meaning`：参数含义、单位、取值范围怎么讲
- `sensitivity_handling`：边界情形与敏感性讨论
- `common_moves`：常见解释动作（"该假设的合理性在于…"等）
- `examples`：解释句范例

### 4. 行文语言（writing）
按功能分类提炼全文的句型库：
- `function` 分类：摘要结果句/假设铺垫句/过渡衔接句/结果解读句/结论升华句/局限说明句
- `examples`：每类 2~3 个范例句（去题目化、≤60 字）
- `usage_notes`：这类句子的使用时机与注意事项

### 5. 图片位置规划（figure_placements）
逐图记录：
- `figure_type`、`section`（放在哪个章节）、`argument_role`（支撑什么论证）
- `caption_style`：图注写法

### 6. 正文侧重点与论证链条
- `section_focuses`：每个核心章节的 `focus`（写作重点）、`weight`（篇幅占比 0~1）、
  `internal_order`（节内展开顺序）
- `argument_flow`：全文论证链条 `steps`（问题→假设→模型→求解→验证→评价）
  与 `transitions`（每步衔接方式）

### 7. 标志性句式（signature_moves）
记录全文反复出现、最能体现高分论文特征的标志性句式（如结果句式、创新点句式、
模型评价句式）。每条只存**带空位的句法骨架**：
- `name`：句式名称（如"结果句式""创新点句式"）；
- `skeleton`：句法骨架，可替换位置用 `__` 标注
  （如"结果表明：随__单调上升，呈__关系"、"本文创新性在于：一是__；二是__；三是__"）；
- `note`：适用场景。
**只存骨架，不存具体内容**：禁止在 skeleton 中带入题目对象、数值、模型参数；
具体句子与数值仍受查重护栏约束，句法骨架本身允许复用。

## 论文信息

标题：{paper_title}
题型：{problem_type}

论文全文（已截断）：

{paper_text}

**必须严格按以下 JSON 格式输出（不要包含其他文字）：**

```json
{{
  "card_id": "{card_id}",
  "derivation": [
    {{
      "section": "模型建立",
      "trigger": "目标函数含非线性项时",
      "organization": ["符号引入", "前提假设", "逐步推导", "目标函数", "物理解释"],
      "notation_usage": "符号先定义后使用，下标含义在首次出现处说明",
      "depth_strategy": "关键步骤逐步推导，常规变换直接给出",
      "text_formula_ratio": "每步推导前后配一句文字解释",
      "closing_moves": ["解释每项含义", "联系题目背景说明物理意义"],
      "examples": [
        {{"function": "推导动机句", "text": "为刻画目标函数的非线性特征，本文从基本关系出发逐步推导。", "note": "推导前交代动机"}}
      ]
    }}
  ],
  "algorithm": [
    {{
      "algorithm_type": "启发式",
      "presentation": ["伪代码", "流程图"],
      "flow_organization": ["输入", "初始化", "迭代", "终止", "输出"],
      "complexity_analysis": "在算法描述后给出时间复杂度，并说明规模下可接受",
      "convergence_justification": "通过目标值随迭代次数下降的曲线论证收敛",
      "result_reporting": "输出结果按算法步骤逐一对应报告",
      "support_figures": ["收敛曲线", "目标值变化"],
      "examples": [
        {{"function": "算法描述句", "text": "算法在每次迭代中更新决策变量，直至相邻两次目标值之差小于给定阈值。", "note": "伪代码后紧跟一句人话描述"}}
      ]
    }}
  ],
  "interpretation": [
    {{
      "target": "参数",
      "organization": ["先整体意义", "再取值范围", "后变化影响"],
      "domain_linking": "将参数与题目场景中的具体含义挂钩",
      "parameter_meaning": "说明单位、合理范围与量级",
      "sensitivity_handling": "对关键参数做 ±10% 扰动并讨论结果变化",
      "common_moves": ["该参数表示…", "当参数增大时，结果将…"],
      "examples": [
        {{"function": "结果解读句", "text": "该参数反映了系统对扰动响应的敏感程度，其值越大表明稳定性要求越高。", "note": "先给含义再给影响"}}
      ]
    }}
  ],
  "writing": [
    {{
      "function": "过渡衔接句",
      "examples": [
        {{"function": "过渡衔接句", "text": "基于上述假设，下面建立相应的数学模型。", "note": "章节间衔接"}}
      ],
      "usage_notes": "用于章节或论证环节之间的自然过渡"
    }}
  ],
  "figure_placements": [
    {{
      "figure_type": "line",
      "section": "模型求解",
      "argument_role": "展示算法迭代过程中目标值的收敛趋势",
      "caption_style": "图注说明横纵轴含义与关键结论"
    }}
  ],
  "section_focuses": [
    {{
      "section": "模型建立",
      "focus": "定义决策变量、假设与目标函数，并说明建模动机",
      "weight": 0.3,
      "internal_order": ["符号", "假设", "推导", "目标函数", "解释"]
    }}
  ],
  "argument_flow": {{
    "steps": ["问题重述", "模型假设", "模型建立", "模型求解", "结果验证", "模型评价"],
    "transitions": ["从问题出发提出假设", "由假设导出模型", "用算法求解并验证", "总结评价与推广"]
  }},
  "signature_moves": [
    {{"name": "结果句式", "skeleton": "结果表明：随__单调上升，呈__关系", "note": "给出关键结论时使用"}},
    {{"name": "创新点句式", "skeleton": "本文创新性在于：一是__；二是__；三是__", "note": "摘要结尾陈述创新点"}},
    {{"name": "模型评价句式", "skeleton": "该模型兼具__与__，既能够__，又能__，从而__", "note": "评价模型综合能力"}}
  ]
}}
```
