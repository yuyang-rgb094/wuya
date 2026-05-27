# WuYa (无涯) - Theory-Grounded Academic Review System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-218%20passed-green.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-61%25-yellow.svg)](#testing)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](.github/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](#docker)
[![PyPI version](https://img.shields.io/badge/v0.1.0-orange.svg)](CHANGELOG.md)

**English** | [中文](#中文介绍)

WuYa (无涯, meaning "boundless") is a theory-driven multi-agent system that transforms classical philosophical principles into an operational architecture for academic paper evaluation and journal recommendation.

## 🌟 Key Features

- **Retrieval-Evaluation Coupling**: RAG serves as the core triggering mechanism for judgment, not just citation
- **Two-Phase Routing**: CUDOS gatekeeping + parallel expert evaluation (Innovation, Method, Evidence, Application)
- **Hybrid Knowledge Strategy**: Combines prompt-internalized "principles" (dao) with RAG-retrieved "evidence" (shu)
- **LLM-as-Mapper with Discipline Priors**: Cross-disciplinary paper localization calibrated by citation network analysis
- **Self-Improving Frontier Discovery**: Learns evaluation preferences from editor feedback
- **DEA Efficiency Analysis**: Data Envelopment Analysis for quantitative paper-to-journal matching
- **CLI & Docker**: One-command evaluation, containerized deployment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wuya-team/wuya.git
cd wuya

# Install (with all optional dependencies)
pip install -e ".[all]"

# Or install core only
pip install -e .
```

### Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=sk-...
# WUYA_LLM_PROVIDER=openai
```

### CLI Usage

```bash
# Evaluate a single paper
wuya evaluate paper.pdf --journal "Nature Machine Intelligence" -o report.md

# Batch evaluate papers in a directory
wuya batch ./papers/ --output-dir ./reports/

# Show current configuration
wuya config show

# Start API server
wuya serve --port 8000

# Show version
wuya --version
```

### Python API

```python
import asyncio
from wuya_agents.router import TwoPhaseRouter
from wuya_agents.subagents import (
    CUDOSSubAgent, InnovationSubAgent, MethodSubAgent,
    EvidenceSubAgent, ApplicationSubAgent
)
from wuya_agents.dea_subagent import DEASubAgent, DEAEngine
from wuya_agents.rag.client import RAGClientImpl
from wuya_agents.base import ParsedPaper
from tests.conftest import MockLLMClient, MockRAGClient  # For testing

async def main():
    # Initialize agents (use real clients in production)
    mock_llm = MockLLMClient()
    mock_rag = MockRAGClient()

    router = TwoPhaseRouter(
        cudos_agent=CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        innovation_agent=InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        method_agent=MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        evidence_agent=EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        application_agent=ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        rag_client=mock_rag,
    )

    # Prepare paper
    paper = ParsedPaper(
        paper_id="paper_001",
        title="Your Paper Title",
        abstract="Paper abstract...",
        content="Full paper content...",
        authors=["Author Name"],
        keywords=["keyword1", "keyword2"],
        discipline="computer science",
    )

    # Run evaluation
    report = await router.route(
        paper,
        target_journal="Nature",
        reference_papers=[...]  # Optional: for DEA analysis
    )

    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Tier Estimate: {report.tier_estimate}")
    print(f"Status: {report.status}")

asyncio.run(main())
```

## 🐳 Docker

### Quick Start with Docker Compose

```bash
# Start all services (WuYa + ChromaDB)
docker compose up -d

# Run evaluation
docker compose exec wuya wuya evaluate /app/data/paper.pdf

# View logs
docker compose logs -f wuya
```

### Build and Run Manually

```bash
# Build the image
docker build -t wuya-agents .

# Run CLI commands
docker run --rm -it wuya-agents wuya --help
docker run --rm -it -v $(pwd)/papers:/app/data wuya-agents wuya evaluate /app/data/paper.pdf

# Start API server
docker run --rm -it -p 8000:8000 wuya-agents wuya serve

# Run tests
docker run --rm -it wuya-agents test
```

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

## 📁 Project Structure

```
wuya/
├── wuya_agents/           # Agent implementations
│   ├── __init__.py        # Package exports and version
│   ├── config.py          # Global configuration (pydantic-settings)
│   ├── cli.py             # CLI entry point (typer)
│   ├── base.py            # Base classes and data models
│   ├── router.py          # TwoPhaseRouter orchestration
│   ├── aggregator.py      # Result aggregation
│   ├── parser.py          # Paper parsing (PDF/text)
│   ├── dea_subagent.py    # DEA Sub-agent
│   ├── subagents/         # Evaluation sub-agents
│   │   ├── cudos.py       # CUDOS gate
│   │   ├── innovation.py  # Innovation evaluation
│   │   ├── method.py      # Methodology evaluation
│   │   ├── evidence.py    # Evidence evaluation
│   │   ├── application.py # Application relevance
│   │   └── frontier.py    # Frontier discovery
│   ├── llm_client/        # LLM client abstraction
│   │   ├── config.py      # LLM configuration
│   │   ├── client.py      # Unified client
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   └── rag/               # RAG components
│       ├── client.py      # RAG client
│       ├── embedding.py   # Embedding providers
│       └── vector_store.py # Vector storage
├── tests/                 # Test suite (218 tests)
├── scripts/               # Utility scripts
├── docs/                  # Documentation & ADRs
├── .github/workflows/     # CI/CD (GitHub Actions)
├── Dockerfile             # Docker image
├── docker-compose.yml     # Docker Compose
├── pyproject.toml         # Package configuration
├── .env.example           # Environment template
├── CHANGELOG.md           # Version history
├── CONTRIBUTING.md        # Contribution guide
├── SECURITY.md            # Security policy
├── LICENSE                # MIT License
└── README.md
```

## 🔬 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=wuya_agents --cov-report=term-missing

# Run specific module
pytest tests/test_subagents/ -v

# Skip slow tests
pytest tests/ -v -m "not slow and not e2e"
```

### Test Results

```
============================= test session starts ==============================
218 passed in 4.11s
```

### Coverage Report

| Module | Coverage |
|--------|----------|
| Total | 61% |
| wuya_agents/subagents/* | 80-92% |
| wuya_agents/router.py | 88% |
| wuya_agents/rag/client.py | 90% |
| wuya_agents/base.py | 69% |
| wuya_agents/dea_subagent.py | 68% |

## ⚙️ Configuration

All configuration is managed through environment variables or `.env` files:

| Variable | Default | Description |
|----------|---------|-------------|
| `WUYA_ENVIRONMENT` | `development` | Environment: development/testing/production |
| `WUYA_LLM_PROVIDER` | `openai` | LLM provider: openai/anthropic |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `WUYA_LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `WUYA_RAG_ENABLED` | `true` | Enable RAG retrieval |
| `WUYA_RAG_VECTOR_STORE_TYPE` | `in_memory` | Vector store: in_memory/chromadb |
| `WUYA_EVAL_CUDOS_THRESHOLD` | `3.0` | CUDOS gate threshold |

See [.env.example](.env.example) for the complete list.

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

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🔒 Security

Please see [SECURITY.md](SECURITY.md) for our security policy and vulnerability reporting guidelines.

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
- **CLI 和 Docker**: 一键评估，容器化部署

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/wuya-team/wuya.git
cd wuya

# 安装
pip install -e ".[all]"

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API 密钥

# 评估单篇论文
wuya evaluate paper.pdf --journal "Nature Machine Intelligence" -o report.md

# 批量评估
wuya batch ./papers/ --output-dir ./reports/

# Docker 一键启动
docker compose up -d
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行带覆盖率报告
pytest tests/ --cov=wuya_agents --cov-report=term-missing
```

**测试结果**: 218 个测试全部通过

## 理论基础

| 评估维度 | 主要理论家 | 核心概念 |
|---------|-----------|---------|
| 创新性 | 库恩、熊彼特、莫基尔 | 范式转换、创造性破坏 |
| 方法 | 珀尔、坎贝尔、费舍尔 | 因果阶梯、内部/外部效度 |
| 证据 | 波普尔、拉卡托斯 | 证伪主义、研究纲领 |
| 应用 | 布什、莫基尔 | 创新管道、技术就绪水平 |
| CUDOS | 默顿 | 科学规范（共有主义、普遍主义等） |
