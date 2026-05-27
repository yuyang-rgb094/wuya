# 无涯 (WuYa) 系统架构

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户交互层                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  学者投稿    │  │  期刊审稿    │  │  编委反馈    │  │  系统管理    │      │
│  │  推荐请求    │  │  评审请求    │  │  输入        │  │  后台        │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │                 │
          └─────────────────┴─────────┬───────┴─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Router（路由智能体）                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  职责：                                                              │   │
│  │  1. 解析论文（PDF/文本提取、学科识别、关键词提取）                      │   │
│  │  2. 识别用户意图（投稿推荐 / 期刊审稿 / 交互反馈）                      │   │
│  │  3. 两阶段路由调度                                                    │   │
│  │  4. 结果聚合与输出格式化                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │     阶段 1：CUDOS      │           │   前沿发现 Sub-agent   │
        │     守门检查           │           │   （后台持续运行）      │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
        ┌───────────┴───────────┐                       │ 积累
        │ 通过                   │ 否决                  ▼
        ▼                       ▼           ┌───────────────────────┐
┌───────────────────┐   ┌───────────────┐   │   前沿面知识库        │
│   阶段 2：并行     │   │  返回 CUDOS   │   │   （DEA 参考集）      │
│   评价 Sub-agents  │   │  否决报告      │   └───────────┬───────────┘
└─────────┬─────────┘   └───────────────┘               │
          │                                             │
    ┌─────┼─────┬─────────┬─────────┐                   │
    ▼     ▼     ▼         ▼         ▼                   │
┌─────┐┌─────┐┌─────┐ ┌─────┐ ┌─────┐                  │
│创新 ││方法 ││证据 │ │应用 │ │规范 │                  │
│性   ││有效性││置信度│ │前景 │ │CUDOS│                  │
└──┬──┘└──┬──┘└──┬──┘ └──┬──┘ └──┬──┘                  │
   └──────┴──────┴───────┴───────┘                      │
                  │                                     │
                  ▼                                     │
    ┌─────────────────────────────┐                     │
    │     5 维评分向量             │                     │
    │  {创新性, 方法, 证据, 应用}  │                     │
    │  各维度子维度评分            │                     │
    └─────────────┬───────────────┘                     │
                  │                                     │
        ┌─────────┴─────────┐                           │
        ▼                   ▼                           │
┌───────────────────┐ ┌───────────────────┐             │
│    路径 A          │ │    路径 B          │             │
│  （快速定位）       │ │  （深度定位）       │◄────────────┘
│                   │ │                   │
│  LLM as Mapper    │ │  DEA 效率分析      │
│  + 学科先验分布    │ │  + Bootstrap      │
│                   │ │                   │
│  输入：            │ │  输入：            │
│  - 5 维评分        │ │  - 5 维评分        │
│  - 学科标识        │ │  - 目标期刊历史数据 │
│  - P0(分区|学科)   │ │  - 前沿面知识库    │
│                   │ │                   │
│  输出：            │ │  输出：            │
│  - 估计分区        │ │  - 效率得分        │
│  - 置信度          │ │  - 置信区间        │
│  - 推理链          │ │  - 前沿面距离      │
└─────────┬─────────┘ └─────────┬─────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
        ┌─────────────────────────────┐
        │      交叉验证与 RAG          │
        │                             │
        │  一致 → 高置信度输出          │
        │  矛盾 → 触发 RAG 解释原因     │
        └─────────────┬───────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────┐   ┌───────────────────┐
│   投稿推荐输出      │   │   审稿报告输出     │
│                   │   │                   │
│  - 推荐期刊列表     │   │  - 5 维评审详情    │
│  - 匹配度排序       │   │  - 估计 vs 实际分区 │
│  - 每本推荐理由     │   │  - 改进建议        │
│  - 理论引用支撑     │   │  - 理论引用支撑    │
└───────────────────┘   └───────────────────┘
```

## Sub-agents 详细设计

### 1. CUDOS Sub-agent

**职责**：学术共同体规范审查，执行默顿 CUDOS 四原则

**输入**：
- 论文全文（PDF 或文本）
- 元数据（作者信息、机构、资助声明等）

**输出**：
```json
{
  "gate_pass": true/false,
  "dimensions": {
    "universalism": {"score": 1-5, "issues": [...]},
    "communism": {"score": 1-5, "issues": [...]},
    "disinterestedness": {"score": 1-5, "issues": [...]},
    "organized_skepticism": {"score": 1-5, "issues": [...]}
  },
  "veto_reason": "如果 gate_pass=false，说明原因",
  "rag_citations": [...]
}
```

**RAG 触发条件**：
- 发现潜在伦理问题时，检索默顿 CUDOS 原文解释为什么这是问题

### 2. 选题创新性 Sub-agent

**理论基础**：熊彼特创新理论 → 库恩范式转换 → 拉卡托斯研究纲领 → Mokyr 知识双循环 → Aghion 创造性破坏

**子维度**：
| 子维度 | 说明 | 评分 |
|--------|------|------|
| novelty_level | 创新层级：填补空白/增量改进/方法迁移/理论突破/范式转换 | 1-5 |
| knowledge_bridge | Q-知识与 A-知识的连接强度 | 1-5 |
| future_potential | 为后续研究开辟新路径的潜力 | 1-5 |
| creative_destruction | 对现有理论的替代/颠覆程度 | 1-5 |

**RAG 触发条件**：
- 评价创新层级时，检索库恩、拉卡托斯原文解释范式转换 vs 常规科学的区别
- 给出改进建议时，检索 Mokyr 知识双循环理论，建议如何增强 Q-A 知识连接

### 3. 方法有效性 Sub-agent

**理论基础**：Fisher 实验设计 → Campbell & Stanley 有效性理论 → Pearl 因果推断

**子维度**：
| 子维度 | 说明 | 评分 |
|--------|------|------|
| causal_identification | 因果识别的清晰度 | 1-5 |
| internal_validity | 内部有效性控制 | 1-5 |
| external_validity | 外部有效性/可推广性 | 1-5 |
| statistical_rigor | 统计方法的严谨性 | 1-5 |

**RAG 触发条件**：
- 发现因果推断缺陷时，检索 Pearl 因果图理论
- 评价实验设计时，检索 Campbell & Stanley 对威胁内部有效性因素的分类

### 4. 证据置信度 Sub-agent

**理论基础**：波普尔可证伪性 → 拉卡托斯精致证伪主义 → 贝叶斯推断

**子维度**：
| 子维度 | 说明 | 评分 |
|--------|------|------|
| falsifiability | 假设的可证伪性 | 1-5 |
| evidence_strength | 证据的累积强度/贝叶斯更新程度 | 1-5 |
| replicability | 结果的可重复性 | 1-5 |
| research_program | 是否属于进步的研究纲领 | 1-5 |

**RAG 触发条件**：
- 评价可证伪性时，检索波普尔《科学发现的逻辑》
- 评价研究纲领时，检索拉卡托斯原文解释进步 vs 退化纲领

### 5. 产业应用前景 Sub-agent

**理论基础**：Bush 线性模型 → NASA TRL → Mokyr 双向知识互动

**子维度**：
| 子维度 | 说明 | 评分 |
|--------|------|------|
| trl_level | 技术就绪水平估计（TRL 1-9 映射到 1-5） | 1-5 |
| translation_path | 从基础到应用的转化路径清晰度 | 1-5 |
| bidirectional_feedback | 产业反馈对基础研究的反哺潜力 | 1-5 |
| market_readiness | 市场/社会需求的匹配度 | 1-5 |

**RAG 触发条件**：
- 评价 TRL 时，检索 NASA TRL 等级定义
- 建议增强应用前景时，检索 Bush《科学：无尽的前沿》

### 6. 前沿发现 Sub-agent（后台）

**职责**：从交互过程中动态发现和归纳领域前沿节点

**输入**：
- 论文 5 维评分向量
- 编委/作者的反馈文本
- 期刊的历史接受/拒稿决策

**输出**：
```json
{
  "journal_id": "期刊标识",
  "frontier_nodes": [
    {
      "dimension": "创新性",
      "preference": "偏好宏大叙事 vs 精细雕琢",
      "evidence": ["编委反馈1", "编委反馈2"],
      "confidence": 0.85
    }
  ],
  "frontier_surface": {
    "description": "该期刊当前的前沿面特征",
    "representative_papers": ["论文ID1", "论文ID2"]
  }
}
```

**运行机制**：
- 每次审稿完成后，接收 Router 传递的反馈数据
- 定期（每周/每月）聚类分析，更新前沿面知识库
- 为 DEA 路径 B 提供参考集构建依据

## Router 编排逻辑

### 主流程

```python
async def process_paper(paper: Paper, user_intent: Intent) -> Result:
    # 1. 解析论文
    parsed = await parse_paper(paper)
    discipline = identify_discipline(parsed)
    
    # 2. 阶段 1：CUDOS 守门
    cudos_result = await call_subagent("cudos", parsed)
    if not cudos_result.gate_pass:
        return generate_veto_report(cudos_result)
    
    # 3. 阶段 2：并行评价
    evaluation_tasks = [
        call_subagent("innovation", parsed),
        call_subagent("method", parsed),
        call_subagent("evidence", parsed),
        call_subagent("application", parsed)
    ]
    scores = await asyncio.gather(*evaluation_tasks)
    score_vector = aggregate_scores(scores)
    
    # 4. 分层定位
    # 路径 A：始终运行
    path_a_result = llm_mapper(score_vector, discipline, get_prior(discipline))
    
    # 路径 B：条件运行（如果目标期刊有足够历史数据）
    path_b_result = None
    if user_intent.target_journal and has_sufficient_data(user_intent.target_journal):
        path_b_result = dea_analysis(score_vector, user_intent.target_journal)
    
    # 5. 交叉验证
    if path_b_result:
        consistency = check_consistency(path_a_result, path_b_result)
        if not consistency.is_consistent:
            rag_explanation = await generate_rag_explanation(path_a_result, path_b_result)
    
    # 6. 生成输出
    if user_intent.type == "recommendation":
        return generate_recommendation(path_a_result, path_b_result, rag_explanation)
    else:
        return generate_review_report(score_vector, path_a_result, path_b_result)
```

### 路径 A：LLM as Mapper

```python
def llm_mapper(score_vector: ScoreVector, discipline: str, prior: Distribution) -> PathAResult:
    prompt = f"""
    你是一个科研质量评价专家。已知以下5维评分标准：
    - 创新性：1(填补空白) ~ 5(范式转换)
    - 方法：1(有缺陷) ~ 5(革命性新方法)
    - 证据：1(无证据) ~ 5(多重复、独立验证)
    - 应用：1(纯理论) ~ 5(立即产业化)
    
    学科：{discipline}
    评分：{score_vector}
    
    该学科的先验分区分布为：{prior}
    
    请按以下格式输出：
    推理过程：...
    估计分区：Q?
    置信度：高/中/低
    """
    
    response = llm.generate(prompt, temperature=0.1)
    return parse_response(response)
```

### 路径 B：DEA 效率分析

```python
def dea_analysis(score_vector: ScoreVector, journal_id: str) -> PathBResult:
    # 1. 获取目标期刊的历史论文数据
    reference_set = get_journal_papers(journal_id, min_count=50)
    
    # 2. 转换投入-产出
    inputs = {
        "method_flaws": 6 - score_vector.method,  # 方法漏洞 = 6 - 方法评分
        "evidence_gap": 6 - score_vector.evidence  # 证据不足 = 6 - 证据评分
    }
    outputs = {
        "innovation": score_vector.innovation,
        "application": score_vector.application
    }
    
    # 3. 构建 DEA 前沿面
    frontier = build_dea_frontier(reference_set)
    
    # 4. 计算超效率得分
    efficiency_score = calculate_super_efficiency(inputs, outputs, frontier)
    
    # 5. Bootstrap 置信区间
    confidence_interval = bootstrap_dea(inputs, outputs, reference_set, n_iterations=200)
    
    return PathBResult(
        efficiency_score=efficiency_score,
        confidence_interval=confidence_interval,
        frontier_distance=calculate_distance(inputs, outputs, frontier)
    )
```

## 数据流与存储

### 核心数据模型

```
Paper
├── id: UUID
├── content: Text/PDF
├── metadata: {title, authors, abstract, keywords, discipline}
├── parsed: {entities, citations, methods, claims}
└── evaluations: [EvaluationResult]

EvaluationResult
├── paper_id: UUID
├── subagent: Enum
├── scores: {dimension: Score}
├── narrative: Text
├── rag_citations: [Citation]
└── timestamp: DateTime

ScoreVector
├── innovation: {novelty, bridge, potential, destruction}
├── method: {causal, internal, external, statistical}
├── evidence: {falsifiability, strength, replicability, program}
├── application: {trl, translation, feedback, market}
└── overall: {discipline, estimated_tier, confidence}

JournalProfile
├── id: String
├── name: String
├── discipline: String
├── quartile: Q1-Q4
├── tier: A-D
├── frontier_surface: FrontierSurface
├── historical_papers: [Paper]
└── preferences: {dimension: Preference}

FrontierSurface
├── journal_id: String
├── nodes: [FrontierNode]
├── representative_papers: [Paper]
└── updated_at: DateTime
```

### 外部依赖

| 依赖 | 用途 | 接口 |
|------|------|------|
| 期刊分区表 API | 获取学科先验分布、期刊元数据 | REST API |
| 文献数据库 | DEA 参考集构建、前沿发现 | 待接入（Web of Science / Scopus / OpenAlex） |
| RAG 向量库 | 经典理论原文检索 | 本地部署（Milvus / Pinecone） |
| LLM 服务 | Sub-agent 推理、路径 A 映射 | OpenAI API / Claude API / 本地部署 |

## 实现优先级

### Phase 1：MVP（4-6 周）
- [ ] Router 基础框架（论文解析、两阶段路由）
- [ ] CUDOS Sub-agent（基础守门功能）
- [ ] 4 个评价 Sub-agent（基础评分，无 RAG）
- [ ] 路径 A：LLM as Mapper（基础版本，无先验校准）
- [ ] 投稿推荐输出（基础版本）

### Phase 2：增强（4-6 周）
- [ ] RAG 系统集成（经典理论原文检索）
- [ ] 路径 A：先验校准（接入期刊分区表 API）
- [ ] 路径 B：DEA 分析（基础版本）
- [ ] 审稿报告输出
- [ ] 交叉验证与 RAG 解释

### Phase 3：智能化（6-8 周）
- [ ] 前沿发现 Sub-agent
- [ ] 前沿面知识库
- [ ] DEA 参考集自动构建
- [ ] 系统反馈闭环（从编委反馈中学习）

### Phase 4：规模化（持续）
- [ ] 多语言支持
- [ ] 更多学科覆盖
- [ ] 性能优化（缓存、并行化）
- [ ] 可解释性增强（可视化、交互式探索）
