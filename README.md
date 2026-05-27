# WuYa (无涯) - Theory-Grounded Academic Review System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**English** | [中文](#中文介绍)

WuYa (无涯, meaning "boundless") is a theory-driven multi-agent system that transforms classical philosophical principles into an operational architecture for academic paper evaluation and journal recommendation.

## 🌟 Key Features

- **Retrieval-Evaluation Coupling**: RAG serves as the core triggering mechanism for judgment, not just citation
- **Two-Phase Routing**: CUDOS gatekeeping + parallel expert evaluation (Innovation, Method, Evidence, Application)
- **Hybrid Knowledge Strategy**: Combines prompt-internalized "principles" (dao) with RAG-retrieved "evidence" (shu)
- **LLM-as-Mapper with Discipline Priors**: Cross-disciplinary paper localization calibrated by citation network analysis
- **Self-Improving Frontier Discovery**: Learns evaluation preferences from editor feedback
- **DEA Efficiency Analysis**: Data Envelopment Analysis for quantitative paper-to-journal matching

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Router                               │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐      ┌─────────────────┐
│  CUDOS   │      │  Evaluation     │
│  Gate    │      │  Sub-agents     │
│ (Phase 1)│      │  (Phase 2)      │
└──────────┘      └────────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌──────────┐     ┌──────────┐      ┌──────────┐
   │Innovation│     │  Method  │      │ Evidence │
   └──────────┘     └──────────┘      └──────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
              ┌─────────────────────┐
              │   Paper Localization │
              │  Path A: LLM-Mapper  │
              │  Path B: DEA Analysis│
              └─────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/wuya.git
cd wuya
pip install -r requirements.txt
```

### DEA Sub-agent Usage

```python
import asyncio
from wuya_agents import DEASubAgentService, ScoreVector

async def main():
    # Initialize service
    service = DEASubAgentService({"min_reference_papers": 50})
    await service.startup()
    
    # Prepare evaluation request
    request = {
        "paper_id": "paper_001",
        "score_vector": {
            "innovation": 4.5,
            "method": 4.2,
            "evidence": 3.5,
            "application": 4.0,
            "cudos": 4.8
        },
        "target_journal": "Nature Machine Intelligence",
        "reference_papers": [...],  # Historical papers from target journal
        "path_a_estimate": "Q1-A"   # Optional: LLM-as-Mapper estimate
    }
    
    # Execute DEA evaluation
    result = await service.evaluate(request)
    
    print(f"Efficiency Score: {result['dea_result']['efficiency_score']:.3f}")
    print(f"Explanation: {result['dea_result']['explanation']}")
    
    await service.shutdown()

asyncio.run(main())
```

## 📁 Project Structure

```
wuya/
├── wuya_agents/           # Agent implementations
│   ├── __init__.py
│   └── dea_subagent.py    # DEA Sub-agent
├── paper/                 # Research paper (arXiv)
│   ├── wuya_paper.tex
│   ├── references.bib
│   └── figures/
├── docs/                  # Documentation
│   └── adr/               # Architecture Decision Records
├── requirements.txt
└── README.md
```

## 📄 Research Paper

The theoretical foundations and system architecture are described in our research paper:

**"Retrieval-Driven Self-Improvement: A Multi-Agent Architecture for Theory-Grounded Academic Review"**

See [`paper/WuYa_Paper.pdf`](paper/WuYa_Paper.pdf)

## 🧠 Theoretical Foundations

| Dimension | Primary Theorists | Key Concepts |
|-----------|------------------|--------------|
| Innovation | Kuhn, Schumpeter, Mokyr | Paradigm shift, Creative destruction |
| Method | Pearl, Campbell, Fisher | Causal ladder, Internal/External validity |
| Evidence | Popper, Lakatos | Falsification, Research programmes |
| Application | Bush, Mokyr | Innovation pipeline, TRL |
| CUDOS | Merton | Scientific norms (Communalism, Universalism, etc.) |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Philosophical foundations drawn from classical works in philosophy of science
- DEA methodology based on Charnes, Cooper, and Rhodes (1978)
- Self-improving agent design inspired by Reflexion and Voyager

---

# 中文介绍

**WuYa (无涯)** 是一个基于理论驱动的多智能体系统，将经典哲学原理转化为学术论文评估和期刊推荐的运营架构。

## 核心特性

- **检索-评估耦合机制**: RAG 作为判断的核心触发机制，而非仅用于引用
- **两阶段路由**: CUDOS 把关 + 并行专家评估（创新性、方法、证据、应用）
- **混合知识策略**: 提示内化的"道"与 RAG 检索的"术"相结合
- **LLM-as-Mapper**: 基于引文网络分析的学科先验校准
- **自改进前沿发现**: 从编辑反馈中学习评估偏好
- **DEA 效率分析**: 数据包络分析用于定量的论文-期刊匹配

## 理论基础

| 评估维度 | 主要理论家 | 核心概念 |
|---------|-----------|---------|
| 创新性 | 库恩、熊彼特、莫基尔 | 范式转换、创造性破坏 |
| 方法 | 珀尔、坎贝尔、费舍尔 | 因果阶梯、内部/外部效度 |
| 证据 | 波普尔、拉卡托斯 | 证伪主义、研究纲领 |
| 应用 | 布什、莫基尔 | 创新管道、技术就绪水平 |
| CUDOS | 默顿 | 科学规范（共有主义、普遍主义等） |

## 快速开始

```bash
git clone https://github.com/yourusername/wuya.git
cd wuya
pip install -r requirements.txt
```

运行 DEA Sub-agent 示例：

```bash
python -m wuya_agents.dea_subagent
```
