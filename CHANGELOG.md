# Changelog

All notable changes to the WuYa Agents project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-27

### Added

#### Core Architecture
- **Two-Phase Router**: CUDOS gatekeeping (Phase 1) + parallel expert evaluation (Phase 2)
- **5 Evaluation Sub-agents**:
  - CUDOS Sub-agent: Merton's scientific norms gate (Communalism, Universalism, Disinterestedness, Organized Skepticism)
  - Innovation Sub-agent: Novelty, Significance, Advancement evaluation
  - Method Sub-agent: Rigor, Validity, Reproducibility evaluation
  - Evidence Sub-agent: Strength, Consistency, Sufficiency evaluation
  - Application Sub-agent: Relevance, Feasibility, Impact evaluation
- **DEA Sub-agent**: Data Envelopment Analysis for quantitative paper-to-journal matching
- **Frontier Discovery Sub-agent**: Self-improving agent that learns from editorial feedback
- **Result Aggregator**: Multi-dimensional report generation with journal recommendations

#### LLM Client
- Unified LLM client supporting OpenAI and Anthropic providers
- Exponential backoff retry with jitter
- Dual-dimension rate limiting (RPM/TPM)
- Token counting and cost estimation
- Custom API base URL support for proxies

#### RAG (Retrieval-Augmented Generation)
- Hybrid retriever combining semantic and keyword search
- In-memory and ChromaDB vector store backends
- OpenAI and mock embedding providers
- Embedding cache for performance optimization

#### Paper Parser
- PDF extraction with PyPDF2
- Mock extractor for testing
- Parse cache to avoid redundant processing

#### Configuration & CLI
- Global configuration management with pydantic-settings
- Environment variable and .env file support
- Multi-environment support (development/testing/production)
- CLI tool with `wuya evaluate`, `wuya batch`, `wuya serve`, `wuya config` commands
- Rich-formatted console output with progress bars

#### Deployment
- Docker support with multi-stage build
- Docker Compose with ChromaDB service
- GitHub Actions CI/CD pipeline
- Matrix testing across Python 3.9/3.10/3.11
- Code quality checks (ruff, mypy)

#### Documentation
- Comprehensive README with quick start guide
- Architecture documentation (English and Chinese)
- API documentation
- Architecture Decision Records (ADRs)
- Contributing guide
- Security policy

### Testing
- 218 tests covering all components
- 61% overall code coverage
- Sub-agent modules: 80-92% coverage
- End-to-end workflow tests
- Mock LLM and RAG clients for isolated testing

### Theory
- Retrieval-Evaluation Coupling: RAG as core judgment trigger
- Hybrid Knowledge Strategy: "Dao" (principles) + "Shu" (evidence)
- LLM-as-Mapper with discipline priors from citation networks
- Philosophical foundations: Kuhn, Popper, Pearl, Merton, Campbell, Fisher

## [Unreleased]

_No unreleased changes._
