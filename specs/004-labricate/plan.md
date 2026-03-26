# Implementation Plan: Labricate - Hyperparameter Experimentation Framework

**Branch**: `004-labricate` | **Date**: March 22, 2026 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-labricate/spec.md`

## Summary

Framework for running clustering pipeline experiments with varying hyperparameters, evaluating results with Metricate, and comparing outcomes. **Uses BERTopic library** for the default UMAP → HDBSCAN/K-Means pipeline, leveraging its modularity while skipping topic representation overhead.

## Technical Context

**Language/Version**: Python 3.10+ (matching Metricate)  
**Primary Dependencies**: bertopic (minimal install, no embedding backends), scikit-learn, numpy, pandas, matplotlib, tqdm >=4.60, click  
**Storage**: File-based (JSON/CSV for results, checkpoint.json for resume)  
**Testing**: pytest (matching existing infrastructure)  
**Target Platform**: macOS/Linux (development), cross-platform Python  
**Project Type**: Single package (metricate submodule)  
**Performance Goals**: 10 runs on 10k points within 10 minutes (excluding O(n²) metrics)  
**Constraints**: Handle up to 100k embeddings without memory errors (sequential mode)  
**Scale/Scope**: Research tool for hyperparameter exploration

## Constitution Check

*GATE: All gates pass ✅*

| Gate | Status | Notes |
|------|--------|-------|
| MAX_PACKAGES=1 | ✅ PASS | Labricate is submodule of `metricate` package |
| NO_ORM | ✅ PASS | File-based storage with JSON/CSV |
| NO_ASYNC | ✅ PASS | Synchronous ProcessPoolExecutor for parallelism |
| PREFER_COMPOSITION | ✅ PASS | Functions + dataclasses, minimal class hierarchy |

## Project Structure

### Documentation (this feature)

```text
specs/004-labricate/
├── plan.md              # This file
├── research.md          # Phase 0: Technical decisions
├── data-model.md        # Phase 1: Entity definitions
├── quickstart.md        # Phase 1: Usage examples
├── contracts/           # Phase 1: API contracts
│   ├── python-api.md
│   └── cli.md
└── tasks.md             # Phase 2: Implementation tasks
```

### Source Code (repository root)

```text
metricate/
├── __init__.py          # Updated: re-export labricate
├── labricate/           # NEW: Experimentation submodule
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── experiment.py    # Experiment class, results
│   │   ├── config.py        # Config validation, dot-notation
│   │   └── loader.py        # Embeddings loading (CSV/NPY/NPZ)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── base.py          # Pipeline protocol
│   │   └── bertopic.py      # BERTopic wrapper (default pipeline)
│   ├── output/
│   │   ├── __init__.py
│   │   ├── storage.py       # JSON/CSV output
│   │   └── visualization.py # Line charts, heatmaps
│   └── utils/
│       ├── __init__.py
│       ├── logging.py       # Progress bars, timing
│       └── parallel.py      # ProcessPoolExecutor wrapper
└── cli/
    ├── main.py              # Updated: add labricate group
    └── labricate.py         # NEW: labricate CLI commands

tests/
└── unit/
    └── labricate/           # NEW: Unit tests
        ├── test_config.py
        ├── test_experiment.py
        ├── test_loader.py
        └── test_pipelines.py
```

**Structure Decision**: Single package submodule structure per constitution. Labricate lives at `metricate/labricate/` to maintain single-package requirement while providing clear separation.

## Key Changes: BERTopic Integration (2026-03-22)

| Decision | Implementation |
|----------|----------------|
| Use BERTopic library | Replace direct umap-learn/hdbscan with `bertopic` package |
| Extract outputs | `topic_model.umap_model.embedding_` for reduced embeddings, `topic_model.topics_` for labels |
| Skip topic representation | Default: `representation_model=None`, `calculate_probabilities=False` (configurable) |
| Handle doc requirement | Pass placeholder empty strings: `fit_transform([""] * n, embeddings=embeddings)` |
| Minimal install | `pip install bertopic` without embedding backends (users provide pre-computed embeddings) |

## Complexity Tracking

> No constitution violations - all gates pass.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
