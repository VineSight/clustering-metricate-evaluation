# Implementation Plan: Labricate Weighted Evaluation & Computation Modes

**Branch**: `005-labricate-weighted-eval` | **Date**: 2026-03-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-labricate-weighted-eval/spec.md`

## Summary

Add weighted evaluation support to Labricate experiments, enabling compound scores from trained weights JSON files, light/heavy computation modes to control O(n²) metric inclusion, and `best_run` identification with tie handling. Reuses existing `metricate.training.weights.MetricWeights` and `METRIC_REFERENCE.skip_large` for seamless integration.

## Technical Context

**Language/Version**: Python 3.10+ (type hints required per constitution)  
**Primary Dependencies**: numpy, pandas, click (CLI), metricate.training.weights (existing)  
**Storage**: JSON files for weights, CSV/JSON for experiment results  
**Testing**: pytest with existing `tests/` structure  
**Target Platform**: macOS/Linux CLI + Python API  
**Project Type**: Single package (`metricate`)  
**Performance Goals**: Light mode 30%+ faster than heavy on 5k+ samples (SC-002)  
**Constraints**: No async (constitution), composition over classes  
**Scale/Scope**: Extends existing Labricate experiment framework (720 lines experiment.py)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| **MAX_PACKAGES: 1** | ✅ PASS | All code in `metricate` package |
| **NO_ORM** | ✅ PASS | Direct pandas/numpy, JSON file storage |
| **NO_ASYNC** | ✅ PASS | Synchronous evaluation only |
| **PREFER_COMPOSITION** | ✅ PASS | New functions in existing modules, minimal new classes |
| **Type hints required** | ✅ PASS | All new code will use 3.10+ type hints |
| **CLI + Library API** | ✅ PASS | Both `--weights`/`--mode` CLI and `weights`/`mode` params |

## Project Structure

### Documentation (this feature)

```text
specs/005-labricate-weighted-eval/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
metricate/
├── labricate/
│   ├── core/
│   │   ├── experiment.py       # MODIFY: Add weights, mode params; best_run logic
│   │   ├── modes.py            # NEW: Light/heavy mode definitions
│   │   └── scoring.py          # NEW: Compound score computation for experiments
│   └── output/
│       └── storage.py          # MODIFY: Include best_run in JSON output
├── training/
│   └── weights.py              # EXISTING: MetricWeights, compute_compound_score (reuse)
├── core/
│   └── reference.py            # EXISTING: METRIC_REFERENCE with skip_large flags
└── cli/
    └── labricate.py            # MODIFY: Add --weights, --mode options

tests/
├── unit/
│   ├── test_labricate_weights.py      # NEW: Weights integration tests
│   └── test_labricate_modes.py        # NEW: Mode filtering tests
└── test_evaluator.py                   # EXISTING
```

**Structure Decision**: Single package structure maintained. Two new modules (`modes.py`, `scoring.py`) for separation of concerns. Heavy reuse of existing `MetricWeights` and `METRIC_REFERENCE`.

## Complexity Tracking

> No constitution violations. All changes fit within existing architecture.

| Aspect | Approach | Constitution Compliance |
|--------|----------|-------------------------|
| Weights loading | Reuse `load_weights()` | ✅ No new dependencies |
| Mode filtering | New `modes.py` module | ✅ Composition over classes |
| CLI options | Extend existing Click commands | ✅ CLI + Library API parity |

## Key Integration Points

### 1. Existing Components to Reuse

| Component | Location | Usage |
|-----------|----------|-------|
| `MetricWeights` | `metricate/training/weights.py` | Load/validate weights JSON |
| `load_weights()` | `metricate/training/weights.py` | Load weights from file path |
| `compute_compound_score()` | `metricate/training/weights.py` | Calculate weighted score |
| `validate_weights_schema()` | `metricate/training/weights.py` | Schema validation |
| `METRIC_REFERENCE` | `metricate/core/reference.py` | Get `skip_large: True` metrics |

### 2. Expensive Metrics (skip_large: True)

From `METRIC_REFERENCE`, the following 6 metrics have `skip_large: True`:
- **Gamma** - O(n²) concordant-discordant pairs
- **G-plus** - O(n²) normalized discordant pairs  
- **Tau** - O(n²) normalized concordance
- **Point-Biserial** - O(n²) distance-cluster correlation
- **McClain-Rao** - O(n²) mean within/between distance
- **NIVA** - O(n²) nearest intra/inter distance ratio

### 3. Experiment Flow Changes

```
Experiment.__init__()
├── NEW: weights parameter (str | dict | None)
├── NEW: Validate weights if provided
└── Store self._weights: MetricWeights | None

Experiment.run() / run_grid()
├── NEW: mode parameter ("light" | "heavy", default="heavy")
├── NEW: Apply mode filtering to exclude_metrics
├── For each run:
│   ├── Compute metrics (filtered by mode)
│   ├── NEW: Compute compound_score if weights
│   └── Store RunResult
├── NEW: Determine best_run (compound_score or metric)
├── NEW: Detect ties, store all tied run IDs
└── Return ExperimentResult with best_run

ExperimentResult
├── NEW: best_run: BestRunInfo | None
├── to_dataframe(): NEW compound_score, is_best_run columns
└── to_json(): NEW best_run top-level field
```

## Phase 0 & 1 Outputs

See generated artifacts:
- [research.md](research.md) - Technical research findings
- [data-model.md](data-model.md) - Entity definitions
- [contracts/](contracts/) - API contracts
- [quickstart.md](quickstart.md) - Implementation guide
