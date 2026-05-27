# Contributing to WuYa Agents

Thank you for your interest in contributing to WuYa (无涯)! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Be respectful, constructive, and inclusive. We welcome contributions from everyone regardless of background, experience level, or identity.

## Getting Started

### Prerequisites

- Python 3.9+
- pip or poetry
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/wuya-team/wuya.git
cd wuya

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Run tests
pytest tests/ -v
```

## Development Setup

### Install Optional Dependencies

```bash
# LLM integration (OpenAI + Anthropic)
pip install -e ".[llm]"

# API server (FastAPI + Uvicorn)
pip install -e ".[server]"

# RAG with ChromaDB
pip install -e ".[rag]"

# Everything
pip install -e ".[all]"
```

### Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `WUYA_LLM_PROVIDER`: Set to `openai` or `anthropic`
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: Your API keys
- `WUYA_ENVIRONMENT`: `development`, `testing`, or `production`

## Project Structure

```
wuya_agents/
├── __init__.py          # Package exports and version
├── config.py            # Global configuration (pydantic-settings)
├── cli.py               # CLI entry point (typer)
├── base.py              # Base classes and data models
├── router.py            # Two-phase routing orchestration
├── aggregator.py        # Result aggregation
├── parser.py            # Paper parsing (PDF/text)
├── dea_subagent.py      # DEA analysis sub-agent
├── subagents/           # Evaluation sub-agents
│   ├── cudos.py         # CUDOS gate
│   ├── innovation.py    # Innovation evaluation
│   ├── method.py        # Method evaluation
│   ├── evidence.py      # Evidence evaluation
│   ├── application.py   # Application evaluation
│   └── frontier.py      # Frontier discovery
├── llm_client/          # LLM client abstraction
│   ├── config.py        # LLM configuration
│   ├── client.py        # Unified client
│   ├── openai_provider.py
│   └── anthropic_provider.py
└── rag/                 # RAG components
    ├── client.py        # RAG client
    ├── embedding.py     # Embedding providers
    └── vector_store.py  # Vector stores
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** and write tests
4. **Run checks**: `pytest tests/ -v && ruff check wuya_agents/`
5. **Commit** with a descriptive message
6. **Push** and open a Pull Request

## Coding Standards

### Formatting

We use **Black** for code formatting and **isort** for import sorting:

```bash
# Format code
black wuya_agents/ tests/
isort wuya_agents/ tests/

# Check formatting
black --check wuya_agents/ tests/
isort --check wuya_agents/ tests/
```

### Linting

We use **ruff** as a fast linter:

```bash
ruff check wuya_agents/
ruff check --fix wuya_agents/  # Auto-fix
```

### Type Hints

Use type hints for all public functions and methods:

```python
def evaluate_paper(
    paper: ParsedPaper,
    target_journal: str | None = None,
) -> EvaluationReport:
    ...
```

## Testing Guidelines

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=wuya_agents --cov-report=term-missing

# Specific module
pytest tests/test_subagents/test_cudos.py -v

# Specific test
pytest tests/test_router.py::TestTwoPhaseRouter::test_full_workflow -v

# Only fast tests (skip slow/e2e)
pytest tests/ -v -m "not slow and not e2e"
```

### Writing Tests

- Place unit tests alongside the module they test
- Use `tests/conftest.py` fixtures (`MockLLMClient`, `MockRAGClient`)
- Test both happy paths and error cases
- Use descriptive test names: `test_cudos_gate_rejects_low_score`
- Aim for >80% coverage on new code

### Test Structure

```python
import pytest
from wuya_agents.subagents import CUDOSSubAgent
from tests.conftest import MockLLMClient, MockRAGClient

class TestCUDOSGate:
    """Tests for CUDOS gatekeeping logic."""

    def test_rejects_below_threshold(self):
        """CUDOS gate should reject papers below threshold."""
        agent = CUDOSSubAgent(
            llm_client=MockLLMClient(),
            rag_client=MockRAGClient(),
        )
        # ... test implementation
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat(router): add parallel evaluation support
fix(dea): correct efficiency score calculation
test(cudos): add gate rejection tests
docs(readme): update installation instructions
refactor(llm): unify provider interface
chore(deps): update numpy to 1.24.0
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

## Pull Request Process

1. **Title**: Use a conventional commit format
2. **Description**: Explain what, why, and how
3. **Tests**: Include tests for new functionality
4. **CI**: Ensure all CI checks pass
5. **Review**: Address reviewer feedback promptly
6. **Squash**: Keep commit history clean (maintainer will squash)

## Reporting Issues

When reporting bugs, please include:

- **Python version** and OS
- **Minimal reproducible example**
- **Expected vs. actual behavior**
- **Error messages and stack traces**
- **Configuration** (with API keys redacted)

For feature requests, describe the use case and expected behavior.

---

Thank you for contributing to WuYa! 🎉
