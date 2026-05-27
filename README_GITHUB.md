# WuYa (无涯) - Theory-Grounded Academic Review System

WuYa (无涯, meaning 'boundless') is a theory-driven multi-agent system that transforms classical philosophical principles into an operational architecture for academic paper evaluation and journal recommendation.

## Key Features

- **Retrieval-Evaluation Coupling**: RAG serves as the core triggering mechanism for judgment
- **Two-Phase Routing**: CUDOS gatekeeping + parallel expert evaluation
- **Hybrid Knowledge Strategy**: Combines prompt-internalized 'principles' (dao) with RAG-retrieved 'evidence' (shu)
- **LLM-as-Mapper with Discipline Priors**: Cross-disciplinary paper localization
- **Self-Improving Frontier Discovery**: Learns evaluation preferences from editor feedback
- **DEA Efficiency Analysis**: Data Envelopment Analysis for paper-to-journal matching

## Quick Start

```bash
git clone https://github.com/yangyu2010/wuya.git
cd wuya
pip install -r requirements.txt
```

## Research Paper

See `paper/WuYa_Paper.pdf` for the full research paper.

## License

MIT License
