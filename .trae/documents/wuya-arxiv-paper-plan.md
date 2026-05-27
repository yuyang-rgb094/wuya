# 无涯 (WuYa) arXiv 论文撰写计划（更新版）

## 一、论文定位

### 类型
系统论文（System Paper），面向 AI 应用领域

### 目标读者
- AI/LLM 应用研究者（Agent 架构、RAG、多智能体系统、Self-Improving Agent）
- 科学计量学与学术评价研究者
- 数字图书馆与学术信息服务从业者
- 对自动化审稿感兴趣的期刊编委与出版商

### 标题候选（突出检索与自改进耦合）
1. **WuYa: Coupling Retrieval-Augmented Reasoning with Self-Improving Frontier Discovery for Academic Paper Evaluation**
2. **Retrieval-Driven Self-Improvement: A Multi-Agent Architecture for Theory-Grounded Academic Review**
3. **WuYa: An Open-Ended Academic Review Agent with Retrieval-Augmented Evaluation and Dynamic Frontier Learning**

---

## 二、核心创新点（五大贡献）

| # | 创新点 | 一句话描述 |
|---|--------|-----------|
| 1 | **检索-评价耦合架构** | RAG 不仅是知识补充，更是评价触发的核心机制——"需要解释时检索，检索后重新评价" |
| 2 | 两阶段路由架构 | CUDOS 守门 + 并行评价，实现"守门人"语义 |
| 3 | 混合知识策略 | Prompt 内化"道"（原则）+ RAG 调用"术"（原文金句） |
| 4 | LLM as Mapper + 先验校准 | 用学科先验分布校准 LLM 推理，防止幻觉 |
| 5 | **自改进前沿发现** | 借鉴 Voyager/Reflexion 思路，从编委反馈中动态归纳领域前沿偏好，持续优化 DEA 参考集 |

---

## 三、与 Self-Improving Agent 研究的关联

### 关键文献引用

#### 1. Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)
- **核心思想**：Agent 在执行动作后对结果做语言化自我反思，将反思写入 Episodic Memory，在下一轮试错中复用
- **在无涯中的应用**：前沿发现 Sub-agent 借鉴 Reflexion 的"语言化反思"机制，将编委的文本反馈转化为结构化的前沿节点描述

#### 2. Voyager: An Open-Ended Embodied Agent with Large Language Models (Wang et al., 2023)
- **核心思想**：通过技能库（Skill Library）实现终身学习，自动发现、编码和复用可执行技能
- **在无涯中的应用**：前沿发现 Sub-agent 维护"前沿面知识库"，类似 Voyager 的技能库，但存储的是领域偏好模式而非可执行代码

#### 3. Self-Improving Agent (OpenClaw, 2024)
- **核心思想**：将学习内容、错误和修正记录到持久化存储，实现跨会话的持续改进
- **在无涯中的应用**：前沿面知识库的持久化存储，支持跨期刊、跨学科的长期积累

### 无涯的独特贡献

与现有 Self-Improving Agent 研究相比，无涯的创新在于：

| 维度 | 现有研究（Reflexion/Voyager） | 无涯 |
|------|------------------------------|------|
| **改进来源** | 任务执行的成功/失败反馈 | 编委/作者的专业评价反馈 |
| **改进对象** | Agent 自身的执行策略 | 评价标准的前沿面定义 |
| **知识形式** | 可执行技能/反思文本 | 领域偏好模式（DEA 参考集） |
| **应用场景** | 编程、游戏等可验证任务 | 学术评价这一主观性领域 |

---

## 四、论文结构（约 20-25 页）

### Abstract（约 250 词）
- 问题背景：同行评议的挑战、现有 AI 审稿工具缺乏理论根基
- 研究空白：现有 Self-Improving Agent 多聚焦任务执行，缺乏对评价标准本身的动态学习
- 系统贡献：
  - **检索-评价耦合**：RAG 触发与评价流程深度整合
  - **理论驱动**：科学哲学经典理论转化为 Agent 架构
  - **自改进前沿发现**：从交互中持续学习领域偏好
- 实验结果：投稿推荐准确率、审稿一致性
- 意义：首次建立"检索增强 + 自改进"的学术评价 Agent 范式

### 1. Introduction（约 3 页）
- 1.1 研究动机：
  - 同行评议痛点
  - AI 辅助审稿的兴起与局限
  - **Self-Improving Agent 的兴起（引用 Reflexion, Voyager）**
- 1.2 理论空白：
  - "术"与"道"的分离
  - 跨学科一致性缺失
  - **现有 Self-Improving Agent 未涉及评价标准的学习**
- 1.3 本文贡献：五大创新点逐条阐述
- 1.4 论文组织

### 2. Related Work（约 3 页）
- 2.1 AI 辅助审稿
- 2.2 多智能体系统
- 2.3 RAG 与知识增强
- **2.4 Self-Improving Agent（新增）**
  - Reflexion: Verbal Reinforcement Learning (Shinn et al., 2023)
  - Voyager: Lifelong Learning with Skill Library (Wang et al., 2023)
  - 与无涯的对比：现有研究聚焦任务执行，无涯聚焦评价标准学习
- 2.5 DEA 在学术评价中的应用
- 2.6 科学哲学与科学计量学

### 3. Theoretical Foundations（约 3 页）
- 3.1 评价维度的理论溯源
- 3.2 从理论到 Agent 的映射方法论
- **3.3 检索-自改进耦合的理论基础（新增）**
  - 为什么检索增强需要与自改进结合：静态知识库无法适应领域演化

### 4. System Architecture（约 5 页）
- 4.1 整体架构概览（架构图）
- **4.2 检索-评价耦合机制（创新点 1，重点展开）**
  - RAG 触发条件与评价流程的整合
  - "解释为什么"→检索→"重新评价"的闭环
- 4.3 两阶段路由架构
- 4.4 Sub-agent 设计详解
- 4.5 混合知识策略

### 5. Paper Localization Methods（约 4 页）
- 5.1 问题定义
- 5.2 路径 A：LLM as Mapper + 先验校准
- 5.3 路径 B：DEA 效率分析
- 5.4 双路径交叉验证

### 6. Self-Improving Frontier Discovery（约 4 页，重点章节）
- **6.1 设计动机（创新点 5）**
  - 现有 Self-Improving Agent 的局限
  - 学术评价领域的特殊性：评价标准本身是动态演化的
- **6.2 与 Reflexion/Voyager 的对比**
  - Reflexion：从失败中学习执行策略
  - Voyager：从探索中发现可复用技能
  - 无涯：从反馈中学习评价偏好
- **6.3 前沿发现 Sub-agent 设计**
  - 输入：编委反馈文本 + 论文评分 + 期刊决策
  - 处理：
    - 语言化反思（借鉴 Reflexion）：将反馈转化为结构化描述
    - 模式归纳（借鉴 Voyager 的技能库）：聚类形成前沿节点
  - 输出：前沿节点 + 前沿面更新
- **6.4 知识库设计：Frontier Surface Library**
  - 类似 Voyager 的 Skill Library，但存储领域偏好模式
  - 支持跨期刊、跨学科的检索与复用
- **6.5 自改进闭环**
  - 反馈 → 反思 → 归纳 → 更新 → 应用
- 6.6 与传统知识库的对比

### 7. Implementation Details（约 2 页）
- 7.1 技术栈
- 7.2 核心数据模型
- 7.3 Router 编排逻辑
- **7.4 自改进模块的工程实现**

### 8. Experiments（约 4 页）
- 8.1 实验设置
- 8.2 主实验结果
- **8.3 自改进效果验证（新增）**
  - 前沿发现准确率随反馈数量的变化
  - 与无自改进基线的对比
- 8.4 消融实验
- 8.5 案例分析

### 9. Discussion（约 2 页）
- 9.1 理论贡献
- 9.2 工程贡献
- **9.3 对 Self-Improving Agent 研究的启示**
- 9.4 局限性
- 9.5 伦理考量

### 10. Conclusion（约 1 页）
- 总结五大创新点
- 展望未来工作

### References（约 2 页）

---

## 五、关键参考文献（更新版）

### A. 科学哲学经典
1. Popper, K. R. (1959). The Logic of Scientific Discovery.
2. Kuhn, T. S. (1970). The Structure of Scientific Revolutions.
3. Lakatos, I. (1970). Falsification and the methodology of scientific research programmes.
4. Merton, R. K. (1973). The normative structure of science.

### B. 创新与知识经济理论
5. Schumpeter, J. A. (1934). The Theory of Economic Development.
6. Mokyr, J. (2002). The Gifts of Athena.
7. Aghion, P., & Howitt, P. (1992). A model of growth through creative destruction.

### C. 统计与因果推断
8. Fisher, R. A. (1925). Statistical Methods for Research Workers.
9. Campbell, D. T., & Stanley, J. C. (1963). Experimental and Quasi-Experimental Designs.
10. Pearl, J. (2009). Causality.

### D. 科学政策与技术评估
11. Bush, V. (1945). Science: The Endless Frontier.
12. NASA. (1995). Technology Readiness Level.

### E. Agent 与 LLM 相关
13. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting. *NeurIPS*.
14. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
15. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv*.

### F. Self-Improving Agent（新增重点）
16. **Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv:2303.11366*.**
    - 核心：语言化反思 + Episodic Memory
    - 无涯借鉴：编委反馈的反思转化机制
17. **Wang, G., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv:2305.16291*.**
    - 核心：Skill Library + 终身学习
    - 无涯借鉴：Frontier Surface Library 设计
18. **Madaan, A., et al. (2024). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS*.**
    - 核心：自我反馈迭代优化
19. **Zhang, Y., et al. (2024). CLIN: A Continually Learning Language Agent for Rapid Task Adaptation. *arXiv*.**
    - 核心：持续学习 + 任务适应

### G. RAG 与知识增强
20. Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML*.
21. Izacard, G., et al. (2022). Few-Shot Learning with Retrieval Augmented Language Models. *arXiv*.
22. Asai, A., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv*.

### H. DEA 相关
23. Charnes, A., Cooper, W. W., & Rhodes, E. (1978). Measuring the efficiency of decision making units. *European Journal of Operational Research*.
24. Banker, R. D., et al. (1984). Some models for estimating technical and scale inefficiencies.

### I. AI 辅助审稿
25. Kang, D., et al. (2018). PeerRead: A dataset of peer reviews. *NAACL*.

---

## 六、图表清单（更新版）

| 图表 | 位置 | 内容 |
|------|------|------|
| Figure 1 | Section 4.1 | 整体架构图（突出检索-评价耦合 + 自改进循环） |
| Figure 2 | Section 3.3 | 检索-自改进耦合的理论框架 |
| Figure 3 | Section 4.2 | RAG 触发与评价流程的整合机制 |
| Figure 4 | Section 3.1 | 理论-架构映射图 |
| Figure 5 | Section 4.5 | 混合知识策略示意图 |
| Figure 6 | Section 5 | 双路径定位流程图 |
| Figure 7 | Section 6.3 | **自改进前沿发现机制（对比 Reflexion/Voyager）** |
| Figure 8 | Section 6.4 | **Frontier Surface Library 设计（对比 Voyager Skill Library）** |
| Figure 9 | Section 8.3 | 自改进效果验证曲线 |
| Figure 10 | Section 8.5 | 案例分析图 |
| Table 1 | Section 8.2 | 投稿推荐准确率对比 |
| Table 2 | Section 8.4 | 消融实验结果 |
| **Table 3** | **Section 6.2** | **无涯与现有 Self-Improving Agent 的对比** |

---

## 七、可复用资源

| 源文件 | 可复用内容 |
|--------|-----------|
| CONTEXT.md | 领域术语定义、关系描述 |
| ARCHITECTURE.md | 系统架构图、Sub-agent 设计、Router 编排逻辑 |
| docs/adr/0001-*.md | 两阶段路由的设计动机 |
| docs/adr/0002-*.md | 混合知识策略的设计动机 |
| docs/adr/0003-*.md | LLM as Mapper + 先验校准的设计动机 |
| docs/adr/0004-*.md | DEA-Agent 耦合、前沿发现 Sub-agent 的设计动机 |
| wuya.md | 产品理念、核心价值观 |
| paper.md | 理论综述（五大维度的哲学溯源） |

---

## 八、投稿建议

### 目标会议/期刊
- **首选**：ACL、EMNLP、AAAI（AI 应用 Track）
- **备选**：JCDL、iConference（数字图书馆领域）
- **arXiv**：cs.AI, cs.CL, cs.DL

### 突出亮点
1. **检索-评价耦合**：不仅是 RAG 应用，更是架构层面的创新
2. **Self-Improving 的学术评价应用**：填补现有研究空白
3. **理论驱动**：科学哲学与 Agent 架构的深度结合

### 补充材料
- 代码仓库链接
- 演示视频或在线 Demo
- Frontier Surface Library 示例数据
