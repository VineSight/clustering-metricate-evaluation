# Quickstart: Implementing Labricate Weighted Evaluation

**Phase**: 1 (Design)  
**Date**: 2026-03-26

## Implementation Order

```
1. modes.py         (standalone, no dependencies)
2. scoring.py       (depends on: modes.py, weights.py)
3. experiment.py    (modify: integrate modes + scoring)
4. labricate CLI    (modify: add --weights, --mode)
5. tests            (parallel with each step)
```

## Step 1: Create modes.py

**File**: `metricate/labricate/core/modes.py`

```python
"""Computation mode definitions for Labricate experiments."""

from typing import Literal

from metricate.core.reference import METRIC_REFERENCE

ComputationMode = Literal["light", "heavy"]

def get_expensive_metrics() -> list[str]:
    """Return metrics with skip_large=True from METRIC_REFERENCE."""
    return [
        name for name, info in METRIC_REFERENCE.items() 
        if info.get("skip_large", False)
    ]

def apply_mode_exclusions(
    mode: ComputationMode,
    exclude_metrics: list[str] | None,
    include_metrics: list[str] | None = None,
) -> list[str] | None:
    """
    Compute final exclusion list based on mode and user preferences.
    
    Returns:
        Final exclusion list, or None if no exclusions.
    """
    if mode == "heavy":
        return exclude_metrics
    
    # Light mode: add expensive metrics to exclusions
    expensive = get_expensive_metrics()
    
    # Filter out any expensive metrics that user explicitly included
    if include_metrics:
        expensive = [m for m in expensive if m not in include_metrics]
    
    if not expensive and not exclude_metrics:
        return None
    
    # Merge with user exclusions
    final = set(exclude_metrics or [])
    final.update(expensive)
    return list(final) if final else None
```

**Tests**: `tests/unit/test_labricate_modes.py`
- Test `get_expensive_metrics()` returns exactly 6 metrics
- Test heavy mode returns exclude_metrics unchanged
- Test light mode adds expensive metrics
- Test include_metrics takes precedence

---

## Step 2: Create scoring.py

**File**: `metricate/labricate/core/scoring.py`

```python
"""Compound scoring utilities for Labricate experiments."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from metricate.training.weights import MetricWeights, compute_compound_score

if TYPE_CHECKING:
    from metricate.labricate.core.experiment import RunResult, BestRunInfo


class WeightCoverageWarning(UserWarning):
    """Warning when excluded metrics account for significant weight."""
    pass


def compute_run_scores(
    runs: list["RunResult"],
    weights: MetricWeights,
) -> None:
    """Compute compound_score for each completed run (mutates runs)."""
    for run in runs:
        if run.pipeline_result.status != "completed":
            continue
        
        # Convert metrics to dict with _norm suffix
        metrics_dict = {
            f"{m.name}_norm": m.value for m in run.metrics
        }
        
        score, _ = compute_compound_score(metrics_dict, weights, warn_on_missing=False)
        run.compound_score = score


def find_best_run(
    runs: list["RunResult"],
    weights: MetricWeights | None,
    best_metric: str = "Silhouette",
) -> "BestRunInfo | None":
    """Determine best run with tie detection."""
    from metricate.labricate.core.experiment import BestRunInfo
    
    completed = [r for r in runs if r.pipeline_result.status == "completed"]
    if not completed:
        return None
    
    # Determine score function
    if weights:
        score_type = "compound_score"
        def get_score(run):
            return run.compound_score or 0.0
    else:
        score_type = best_metric
        def get_score(run):
            for m in run.metrics:
                if m.name == best_metric:
                    return m.value
            return 0.0
    
    # Find max score
    max_score = max(get_score(r) for r in completed)
    
    # Find all runs with max score (tie detection)
    best_runs = [r for r in completed if get_score(r) == max_score]
    
    primary = best_runs[0]
    tied_ids = [r.run_id for r in best_runs[1:]]  # Exclude primary
    
    return BestRunInfo(
        run_id=primary.run_id,
        param_values=primary.param_values.copy(),
        score=max_score,
        score_type=score_type,
        tied_run_ids=tied_ids,
    )


def check_weight_coverage(
    weights: MetricWeights,
    excluded_metrics: list[str],
    threshold: float = 0.30,
) -> str | None:
    """Check if excluded metrics account for significant weight."""
    if not excluded_metrics:
        return None
    
    coefficients = weights.coefficients
    
    # Calculate total absolute weight
    total_weight = sum(abs(v) for v in coefficients.values())
    if total_weight == 0:
        return None
    
    # Calculate excluded weight (match _norm suffix)
    excluded_norm = {f"{m}_norm" for m in excluded_metrics}
    excluded_weight = sum(
        abs(v) for k, v in coefficients.items() 
        if k in excluded_norm
    )
    
    pct = excluded_weight / total_weight
    if pct > threshold:
        pct_display = int(pct * 100)
        excluded_with_weight = [
            m for m in excluded_metrics 
            if f"{m}_norm" in coefficients
        ]
        return (
            f"Warning: Excluded metrics {excluded_with_weight} account for "
            f"{pct_display}% of weight coefficients. Compound scores may be unreliable."
        )
    
    return None
```

**Tests**: `tests/unit/test_labricate_scoring.py`
- Test `compute_run_scores` sets compound_score
- Test `find_best_run` with weights returns highest compound_score
- Test `find_best_run` without weights uses metric
- Test tie detection returns all tied IDs
- Test `check_weight_coverage` triggers at >30%

---

## Step 3: Modify experiment.py

**File**: `metricate/labricate/core/experiment.py`

### 3.1 Add imports

```python
from metricate.labricate.core.modes import ComputationMode, apply_mode_exclusions
from metricate.labricate.core.scoring import (
    BestRunInfo,
    WeightCoverageWarning,
    check_weight_coverage,
    compute_run_scores,
    find_best_run,
)
from metricate.training.weights import MetricWeights, load_weights, validate_weights_schema
```

### 3.2 Add BestRunInfo dataclass

```python
@dataclass
class BestRunInfo:
    """Information about the best-performing experiment run."""
    run_id: int
    param_values: dict[str, Any]
    score: float
    score_type: str
    tied_run_ids: list[int]
    
    def __str__(self) -> str:
        base = f"Best run: run_id={self.run_id} ({self.score_type}={self.score:.3f})"
        if self.tied_run_ids:
            tied = ", ".join(f"run_id={rid}" for rid in self.tied_run_ids)
            base += f" (tied with {tied})"
        return base
```

### 3.3 Modify RunResult

```python
@dataclass
class RunResult:
    run_id: int
    param_values: dict[str, Any]
    pipeline_result: PipelineResult
    metrics: list[MetricResult]
    compound_score: float | None = None  # NEW
```

### 3.4 Modify ExperimentResult

```python
@dataclass
class ExperimentResult:
    # ... existing fields ...
    best_run: BestRunInfo | None = None  # NEW
    
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for run in self.runs:
            row = {"run_id": run.run_id}
            row.update(run.param_values)
            row["n_clusters"] = run.pipeline_result.n_clusters
            row["n_noise"] = run.pipeline_result.n_noise
            row["status"] = run.pipeline_result.status
            for metric in run.metrics:
                row[metric.name] = metric.value
            
            # NEW: Add compound_score and is_best_run
            if run.compound_score is not None:
                row["compound_score"] = run.compound_score
            if self.best_run:
                row["is_best_run"] = (
                    run.run_id == self.best_run.run_id or
                    run.run_id in self.best_run.tied_run_ids
                )
            
            rows.append(row)
        return pd.DataFrame(rows)
```

### 3.5 Modify Experiment.__init__

```python
def __init__(
    self,
    embeddings: ...,
    config: ...,
    name: ...,
    output_dir: ...,
    output_format: ...,
    pipeline: ...,
    weights: str | Path | dict[str, Any] | None = None,  # NEW
) -> None:
    # ... existing code ...
    
    # NEW: Load and validate weights
    self._weights: MetricWeights | None = None
    if weights is not None:
        if isinstance(weights, (str, Path)):
            self._weights = load_weights(weights)
        elif isinstance(weights, dict):
            is_valid, errors = validate_weights_schema(weights)
            if not is_valid:
                raise ValueError(f"Invalid weights: {'; '.join(errors)}")
            self._weights = MetricWeights(
                coefficients=weights["coefficients"],
                bias=weights["bias"],
                **weights.get("metadata", {}),
            )
```

### 3.6 Modify run() method

Add parameters:
```python
def run(
    self,
    # ... existing params ...
    mode: ComputationMode = "heavy",  # NEW
    best_metric: str = "Silhouette",  # NEW
) -> ExperimentResult:
```

Before the run loop:
```python
# NEW: Apply mode-based exclusions
final_exclude = apply_mode_exclusions(mode, exclude_metrics, include_metrics)

# NEW: Check weight coverage warning
if self._weights and final_exclude:
    warning = check_weight_coverage(self._weights, final_exclude)
    if warning:
        warnings.warn(warning, WeightCoverageWarning)
        if verbose:
            print(f"⚠ {warning}")
```

After the run loop, before creating ExperimentResult:
```python
# NEW: Compute compound scores if weights provided
if self._weights:
    compute_run_scores(runs, self._weights)

# NEW: Find best run
best_run = find_best_run(runs, self._weights, best_metric)

# Print best run if verbose
if verbose and best_run:
    print(f"\n  {best_run}")
```

---

## Step 4: Modify CLI

**File**: `metricate/cli/labricate.py`

Add options:
```python
@click.option(
    "--weights", "-w",
    type=click.Path(exists=True),
    default=None,
    help="Path to weights JSON file for compound scoring",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["light", "heavy"]),
    default="heavy",
    help="Computation mode: light (fast) or heavy (comprehensive)",
)
```

Pass to Experiment:
```python
exp = Experiment(
    embeddings=embeddings,
    config=config,
    name=name,
    output_dir=output_dir,
    output_format=output_format,
    weights=weights,  # NEW
)

result = exp.run(
    # ... existing args ...
    mode=mode,  # NEW
)
```

---

## Step 5: Update __init__.py exports

**File**: `metricate/labricate/__init__.py`

```python
from metricate.labricate.core.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentSummary,
    PipelineResult,
    RunResult,
    BestRunInfo,  # NEW
)
from metricate.labricate.core.modes import ComputationMode  # NEW
```

---

## Validation Checklist

After implementation, verify:

- [ ] `Experiment(weights="path.json")` loads and validates weights
- [ ] `exp.run(mode="light")` excludes 6 expensive metrics
- [ ] `result.best_run` contains param_values (FR-015)
- [ ] Ties are reported in `best_run.tied_run_ids` (FR-016)
- [ ] Warning appears when excluded weight >30% (FR-017)
- [ ] `result.to_dataframe()` has compound_score, is_best_run columns
- [ ] CLI `--weights` and `--mode` work as documented
- [ ] All existing tests still pass (SC-005)
