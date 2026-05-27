# 无涯论文 LaTeX 整合实施计划

## 一、项目概述

将分散的 LaTeX 章节文件整合为单一完整文档，并创建 4 个缺失的图表。

## 二、任务分解

### 任务 1: 创建单一完整 LaTeX 文档

**输入文件**:
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/wuya_paper.tex` (主框架)
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/1-introduction.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/2-related-work.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/3-theoretical-foundations.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/4-system-architecture.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/5-paper-localization.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/6-frontier-discovery.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/7-implementation.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/8-experiments.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/9-discussion.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/sections/appendix.tex`

**输出文件**:
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/wuya_paper_complete.tex`

**实施步骤**:
1. 读取主文件 `wuya_paper.tex`
2. 递归替换所有 `\input{...}` 指令为对应文件内容
3. 确保参考文献引用正确（`\bibliography{references}` 保留）
4. 处理附录中的嵌套输入

### 任务 2: 使用 JSON Canvas 创建 4 个图表

**图表 1: 系统架构图** (`figures/architecture.pdf`)
- 内容: 两阶段路由架构
  - Phase 1: CUDOS Gatekeeping
  - Phase 2: 并行专家评估 (Innovation, Method, Evidence, Application)
  - Router 作为中央协调器
  - RAG 检索模块

**图表 2: 混合知识策略图** (`figures/hybrid-knowledge.pdf`)
- 内容: 双层知识架构
  - Layer 1: Prompt-Internalized Principles ("Dao")
  - Layer 2: RAG-Retrieved Evidence ("Shu")

**图表 3: 自改进循环图** (`figures/self-improvement-loop.pdf`)
- 内容: 反馈学习循环
  - Feedback Collection
  - Reflection Generation
  - Pattern Generalization
  - Library Update
  - Application

**图表 4: 学习曲线图** (`figures/self-improvement-curve.pdf`)
- 内容: 推荐准确率随 Frontier Library 增长
- X轴: Frontier Library Size (0-500 review cycles)
- Y轴: Top-5 Accuracy (%)
- 从 65.4% 提升到 70.3%

### 任务 3: LaTeX 语法检查和编译验证

**检查项目**:
1. 宏包依赖完整性
2. 交叉引用配对 (\label / \ref)
3. 引用有效性 (\citep / \citet)
4. 数学环境完整性
5. 图表引用有效性

**编译流程**:
```
pdflatex wuya_paper_complete.tex
bibtex wuya_paper_complete
pdflatex wuya_paper_complete.tex
pdflatex wuya_paper_complete.tex
```

## 三、关键文件位置

**源文件** (只读):
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/`

**输出文件**:
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/wuya_paper_complete.tex`
- `/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/figures/*.pdf`

## 四、潜在问题

| 问题 | 解决方案 |
|------|----------|
| 参考文献重复键 | 检查并去重 references.bib |
| 图表引用失效 | 确保 \label 在 \caption 之后 |
| 算法环境冲突 | 确保 algorithm 和 algorithmic 版本兼容 |
| 图片路径 | 使用相对路径 `figures/xxx.pdf` |

## 五、实施顺序

1. 创建图表 (JSON Canvas)
2. 整合 LaTeX 文档
3. 语法检查和编译验证
