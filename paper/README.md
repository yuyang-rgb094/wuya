# WuYa: Retrieval-Driven Self-Improvement Paper

**Title**: Retrieval-Driven Self-Improvement: A Multi-Agent Architecture for Theory-Grounded Academic Review

## Paper Structure

```
wuya_paper.tex          # Main LaTeX document
sections/
  1-introduction.tex    # Introduction (3 pages)
  2-related-work.tex    # Related Work (3 pages)
  3-theoretical-foundations.tex  # Theoretical Foundations (3 pages)
  4-system-architecture.tex      # System Architecture (5 pages)
  5-paper-localization.tex       # Paper Localization Methods (4 pages)
  6-frontier-discovery.tex      # Self-Improving Frontier Discovery (4 pages)
  7-implementation.tex           # Implementation Details (2 pages)
  8-experiments.tex             # Experiments (4 pages)
  9-discussion.tex              # Discussion and Conclusion (2 pages)
  appendix.tex                  # Appendix
references.bib                  # BibTeX references
```

## Figures (to be created)

1. `figures/architecture.pdf` - Overall system architecture
2. `figures/hybrid-knowledge.pdf` - Hybrid knowledge strategy diagram
3. `figures/self-improvement-loop.pdf` - Self-improvement feedback loop
4. `figures/self-improvement-curve.pdf` - Self-improvement learning curve

## Compilation

```bash
# Compile with pdflatex
pdflatex wuya_paper.tex
bibtex wuya_paper
pdflatex wuya_paper.tex
pdflatex wuya_paper.tex
```

## Key Contributions

1. **Retrieval-Evaluation Coupling**: RAG as core triggering mechanism
2. **Two-Phase Routing**: CUDOS gatekeeping + parallel evaluation
3. **Hybrid Knowledge Strategy**: Prompt-internalized principles + RAG-retrieved evidence
4. **LLM-as-Mapper with Discipline Priors**: Calibrated paper localization
5. **Self-Improving Frontier Discovery**: Learning from editor feedback

## arXiv Submission

Target: cs.AI, cs.CL, or cs.DL

## Status

- [x] Abstract
- [x] Introduction
- [x] Related Work
- [x] Theoretical Foundations
- [x] System Architecture
- [x] Paper Localization Methods
- [x] Self-Improving Frontier Discovery
- [x] Implementation Details
- [x] Experiments
- [x] Discussion and Conclusion
- [x] References
- [x] Appendix
- [ ] Figures (need to be created)
- [ ] Full compilation and PDF generation
