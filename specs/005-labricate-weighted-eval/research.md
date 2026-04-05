# Research: Labricate Weighted Evaluation & Computation Modes

**Phase**: 0 (Research)  
**Date**: 2026-03-26

## Research Tasks

### R1: Weights Integration Architecture

**Question**: How should weights be integrated with the existing Experiment class?

**Research Findings**:

The existing `metricate.training.weights` module provides everything needed:
- `MetricWeights` dataclass with `coefficients`, `bias`, and metadata
- `load_weights(path)` function handles file loading + validation
- `compute_compound_score(metrics, weights)` computes weighted score with renormalization
- `validate_weights_schema(dict)` returns `(is_valid, errors)` for upfront validation

**Decision**: Reuse existing module entirely. Add `weights` parameter to `Experiment.__init__()` that accepts:
- `str | Path` → call `load_weights(path)`
- `dict` → construct `MetricWeights` from dict
- `None` → no weighted scoring (current behavior)

**Alternatives Considered**:
1. Create separate LabrweightsExperiment subclass → Rejected: violates PREFER_COMPOSITION
2. Pass weights only to run() → Rejected: weights validation should happen at construction time

---

### R2: Mode-Based Metric Filtering

**Question**: How to implement light/heavy mode metric filtering?

**Research Findings**:

Current exclude pattern in `experiment.py` line 351:
```python
eval_result = evaluate(
    tmp_path,
    label_col="cluster_id",
    exclude=exclude_metrics,
)
```

`METRIC_REFERENCE` in `metricate/core/reference.py` has `skip_large: True` for 6 metrics:
- Gamma, G-plus, Tau, Point-Biserial, McClain-Rao, NIVA

**Decision**: Create `metricate/labricate/core/modes.py` with:
```python
def get_expensive_metrics() -> list[str]:
    """Return metrics with skip_large: True from METRIC_REFERENCE."""
    return [name for name, info in METRIC_REFERENCE.items() if info.get("skip_large")]

def apply_mode_exclusions(
    mode: Literal["light", "heavy"],
    exclude_metrics: list[str] | None,
) -> list[str]:
    """Merge mode-based exclusions with user-provided exclusions."""
```

Light mode adds expensive metrics to exclude list; user's explicit `include_metrics` takes precedence (handled by metricate's evaluate).

**Alternatives Considered**:
1. Hardcode the 6 metric names → Rejected: would break if METRIC_REFERENCE changes
2. Add mode logic directly in experiment.py → Rejected: better separation of concerns

---

### R3: Best Run Determination with Ties

**Question**: How to implement best_run identification with tie handling (FR-015, FR-016)?

**Research Findings**:

Current `ExperimentResult.get_best_run()` in experiment.py:
- Uses `max(completed_runs, key=get_metric_value)` 
- Returns single RunResult
- No tie detection

**Decision**: Create new `BestRunInfo` dataclass:
```python
@dataclass
class BestRunInfo:
    run_id: int                    # Primary best run
    param_values: dict[str, Any]   # FR-015: Best hyperparameters
    score: float                   # compound_score or metric value
    score_type: str                # "compound_score" or metric name
    tied_run_ids: list[int]        # FR-016: All tied runs (empty if no ties)
```

Modify `ExperimentResult` to:
1. Compute best_run after all runs complete
2. Detect ties by collecting all runs with max score
3. Store `best_run: BestRunInfo | None`

**Alternatives Considered**:
1. Return list of RunResult for ties → Rejected: complicates API, just need IDs
2. Silent first-pick on ties → Rejected: user explicitly requested tie reporting

---

### R4: Dominant Weight Warning (FR-017)

**Question**: How to warn users when excluded metrics have >30% of weight?

**Research Findings**:

`compute_compound_score()` already calculates weight sums for renormalization. Can reuse this logic.

**Decision**: Create `check_weight_coverage()` function in `scoring.py`:
```python
def check_weight_coverage(
    weights: MetricWeights,
    excluded_metrics: list[str],
    threshold: float = 0.30,
) -> str | None:
    """Return warning message if excluded metrics > threshold of total weight."""
```

Call this before experiment runs when both weights and mode="light" are provided.

**Alternatives Considered**:
1. Check during compound_score computation → Rejected: too late, warning should be upfront
2. Raise error instead of warning → Rejected: spec says "warn", not "error"

---

### R5: CLI Integration Pattern

**Question**: Best practices for adding --weights and --mode to Click CLI?

**Research Findings**:

Current CLI in `metricate/cli/labricate.py`:
- Uses Click decorators (`@click.option`)
- Has `--include-metrics` and `--exclude-metrics` patterns already
- Creates `Experiment` instance and calls `.run()` or `.run_grid()`

**Decision**: Follow existing patterns:
```python
@click.option(
    "--weights", "-w",
    type=click.Path(exists=True),
    help="Path to weights JSON file for compound scoring",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["light", "heavy"]),
    default="heavy",
    help="Computation mode (default: heavy)",
)
```

**Alternatives Considered**:
1. Separate subcommand for weighted experiments → Rejected: same command, optional features

---

## Summary of Decisions

| Decision | Rationale |
|----------|-----------|
| Reuse `metricate.training.weights` entirely | Existing code handles load/validate/compute |
| New `modes.py` module | Separation of concerns, constitution compliance |
| New `scoring.py` module | Experiment-specific compound score + weight coverage |
| `BestRunInfo` dataclass | Clean representation of best run + ties |
| Warning for >30% excluded weight | Per spec FR-017, upfront user feedback |
| `weights` in `__init__`, `mode` in `run()` | Weights are constant, mode may vary per run |
