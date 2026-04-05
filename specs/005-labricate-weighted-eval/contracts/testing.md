# Testing Contract: Labricate Weighted Evaluation

**Version**: 1.0  
**Date**: 2026-03-26

## Test File Structure

```text
tests/
├── unit/
│   ├── test_labricate_modes.py       # NEW: Mode filtering tests
│   ├── test_labricate_scoring.py     # NEW: Scoring + best_run tests
│   └── test_labricate_weights.py     # NEW: Weights integration tests
└── integration/
    └── test_labricate_weighted_experiment.py  # NEW: End-to-end tests
```

---

## Unit Tests: modes.py

**File**: `tests/unit/test_labricate_modes.py`

### TC-001: get_expensive_metrics returns correct set

```python
def test_get_expensive_metrics_returns_six_metrics():
    """Verify exactly 6 metrics with skip_large=True."""
    metrics = get_expensive_metrics()
    
    assert len(metrics) == 6
    assert set(metrics) == {
        "Gamma", "G-plus", "Tau", 
        "Point-Biserial", "McClain-Rao", "NIVA"
    }
```

### TC-002: Heavy mode preserves exclude_metrics

```python
def test_heavy_mode_preserves_user_exclusions():
    """Heavy mode should not add any exclusions."""
    user_exclude = ["Silhouette", "Davies-Bouldin"]
    
    result = apply_mode_exclusions("heavy", user_exclude)
    
    assert result == user_exclude
```

### TC-003: Light mode adds expensive metrics

```python
def test_light_mode_adds_expensive_metrics():
    """Light mode should exclude expensive metrics."""
    result = apply_mode_exclusions("light", None)
    
    assert "Gamma" in result
    assert "Tau" in result
    assert len(result) == 6
```

### TC-004: Light mode merges with user exclusions

```python
def test_light_mode_merges_exclusions():
    """Light mode should merge with user's exclude list."""
    user_exclude = ["Silhouette"]
    
    result = apply_mode_exclusions("light", user_exclude)
    
    assert "Silhouette" in result
    assert "Gamma" in result
    assert len(result) == 7  # 6 expensive + 1 user
```

### TC-005: Include_metrics takes precedence

```python
def test_include_metrics_overrides_light_mode():
    """User's include_metrics should override light mode exclusions."""
    result = apply_mode_exclusions(
        "light", 
        exclude_metrics=None,
        include_metrics=["Gamma", "Tau"]
    )
    
    assert "Gamma" not in result
    assert "Tau" not in result
    assert "G-plus" in result  # Still excluded
```

---

## Unit Tests: scoring.py

**File**: `tests/unit/test_labricate_scoring.py`

### TC-006: compute_run_scores sets compound_score

```python
def test_compute_run_scores_sets_compound_score():
    """Verify compound_score is computed for completed runs."""
    weights = MetricWeights(
        coefficients={"Silhouette_norm": 0.5, "Davies-Bouldin_norm": -0.3},
        bias=0.2
    )
    runs = [make_run_result(run_id=1, silhouette=0.8, davies_bouldin=0.4)]
    
    compute_run_scores(runs, weights)
    
    assert runs[0].compound_score is not None
    assert 0 <= runs[0].compound_score <= 1
```

### TC-007: compute_run_scores skips failed runs

```python
def test_compute_run_scores_skips_failed_runs():
    """Failed runs should not have compound_score."""
    weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)
    runs = [make_failed_run_result(run_id=1)]
    
    compute_run_scores(runs, weights)
    
    assert runs[0].compound_score is None
```

### TC-008: find_best_run with weights uses compound_score

```python
def test_find_best_run_uses_compound_score():
    """Best run should be highest compound_score when weights provided."""
    weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)
    runs = [
        make_run_result(run_id=1, compound_score=0.7),
        make_run_result(run_id=2, compound_score=0.9),
        make_run_result(run_id=3, compound_score=0.8),
    ]
    
    best = find_best_run(runs, weights)
    
    assert best.run_id == 2
    assert best.score == 0.9
    assert best.score_type == "compound_score"
```

### TC-009: find_best_run without weights uses metric

```python
def test_find_best_run_uses_metric_without_weights():
    """Best run should use specified metric when no weights."""
    runs = [
        make_run_result(run_id=1, silhouette=0.6),
        make_run_result(run_id=2, silhouette=0.8),
    ]
    
    best = find_best_run(runs, weights=None, best_metric="Silhouette")
    
    assert best.run_id == 2
    assert best.score_type == "Silhouette"
```

### TC-010: find_best_run detects ties (FR-016)

```python
def test_find_best_run_detects_ties():
    """Ties should be reported in tied_run_ids."""
    runs = [
        make_run_result(run_id=1, compound_score=0.8),
        make_run_result(run_id=2, compound_score=0.9),
        make_run_result(run_id=3, compound_score=0.9),  # Tie with run 2
    ]
    weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)
    
    best = find_best_run(runs, weights)
    
    assert best.run_id == 2  # First with max score
    assert 3 in best.tied_run_ids
```

### TC-011: find_best_run returns param_values (FR-015)

```python
def test_find_best_run_includes_param_values():
    """Best run must include hyperparameter values."""
    runs = [make_run_result(
        run_id=1, 
        param_values={"hdbscan.min_cluster_size": 15},
        compound_score=0.9
    )]
    weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)
    
    best = find_best_run(runs, weights)
    
    assert best.param_values == {"hdbscan.min_cluster_size": 15}
```

### TC-012: find_best_run returns None for no completed runs

```python
def test_find_best_run_returns_none_when_all_failed():
    """No completed runs should return None."""
    runs = [make_failed_run_result(run_id=1)]
    
    best = find_best_run(runs, weights=None)
    
    assert best is None
```

### TC-013: check_weight_coverage triggers warning at threshold (FR-017)

```python
def test_check_weight_coverage_warns_above_threshold():
    """Warning when excluded metrics > 30% of weight."""
    weights = MetricWeights(
        coefficients={
            "Gamma_norm": 0.4,      # Will be excluded
            "Silhouette_norm": 0.6,
        },
        bias=0
    )
    
    warning = check_weight_coverage(weights, ["Gamma"], threshold=0.30)
    
    assert warning is not None
    assert "40%" in warning
    assert "Gamma" in warning
```

### TC-014: check_weight_coverage no warning below threshold

```python
def test_check_weight_coverage_no_warning_below_threshold():
    """No warning when excluded metrics < 30% of weight."""
    weights = MetricWeights(
        coefficients={
            "Gamma_norm": 0.2,      # Will be excluded
            "Silhouette_norm": 0.8,
        },
        bias=0
    )
    
    warning = check_weight_coverage(weights, ["Gamma"], threshold=0.30)
    
    assert warning is None
```

---

## Unit Tests: Weights Integration

**File**: `tests/unit/test_labricate_weights.py`

### TC-015: Experiment accepts weights path

```python
def test_experiment_accepts_weights_path(tmp_path):
    """Experiment should load weights from file path."""
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({
        "coefficients": {"Silhouette_norm": 1.0},
        "bias": 0.0
    }))
    
    exp = Experiment(
        embeddings=np.random.randn(100, 10),
        config={"clustering_algorithm": "hdbscan"},
        weights=str(weights_file)
    )
    
    assert exp._weights is not None
    assert exp._weights.coefficients["Silhouette_norm"] == 1.0
```

### TC-016: Experiment accepts weights dict

```python
def test_experiment_accepts_weights_dict():
    """Experiment should accept weights as dict."""
    exp = Experiment(
        embeddings=np.random.randn(100, 10),
        config={"clustering_algorithm": "hdbscan"},
        weights={
            "coefficients": {"Silhouette_norm": 0.5},
            "bias": 0.3
        }
    )
    
    assert exp._weights is not None
    assert exp._weights.bias == 0.3
```

### TC-017: Experiment validates weights schema (FR-002)

```python
def test_experiment_validates_weights_schema():
    """Invalid weights should raise ValueError with clear message."""
    with pytest.raises(ValueError, match="coefficients"):
        Experiment(
            embeddings=np.random.randn(100, 10),
            config={"clustering_algorithm": "hdbscan"},
            weights={"bias": 0.5}  # Missing coefficients
        )
```

### TC-018: Experiment without weights works unchanged (SC-005)

```python
def test_experiment_without_weights_unchanged():
    """Existing behavior should work when weights not provided."""
    exp = Experiment(
        embeddings=np.random.randn(100, 10),
        config={"clustering_algorithm": "hdbscan"}
    )
    
    assert exp._weights is None
```

---

## Unit Tests: ExperimentResult

**File**: `tests/unit/test_labricate_weights.py` (continued)

### TC-019: to_dataframe includes compound_score (FR-008)

```python
def test_to_dataframe_includes_compound_score():
    """DataFrame should have compound_score column when weights used."""
    result = make_experiment_result(with_compound_scores=True)
    
    df = result.to_dataframe()
    
    assert "compound_score" in df.columns
    assert df["compound_score"].notna().all()
```

### TC-020: to_dataframe includes is_best_run (FR-009)

```python
def test_to_dataframe_includes_is_best_run():
    """DataFrame should have is_best_run boolean column."""
    result = make_experiment_result(with_best_run=True)
    
    df = result.to_dataframe()
    
    assert "is_best_run" in df.columns
    assert df["is_best_run"].sum() >= 1  # At least one best
```

### TC-021: is_best_run handles ties

```python
def test_is_best_run_handles_ties():
    """Multiple rows should be marked as best when tied."""
    result = make_experiment_result(
        with_best_run=True,
        tied_run_ids=[2, 3]  # Run 1 best, tied with 2 and 3
    )
    
    df = result.to_dataframe()
    
    assert df["is_best_run"].sum() == 3
```

---

## Integration Tests

**File**: `tests/integration/test_labricate_weighted_experiment.py`

### TC-022: Full weighted experiment flow

```python
def test_weighted_experiment_end_to_end(tmp_path, sample_embeddings):
    """Complete experiment with weights produces expected output."""
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({
        "coefficients": {
            "Silhouette_norm": 0.3,
            "Davies-Bouldin_norm": -0.2,
            "Calinski-Harabasz_norm": 0.2,
        },
        "bias": 0.3
    }))
    
    exp = Experiment(
        embeddings=sample_embeddings,
        config={"clustering_algorithm": "hdbscan", "hdbscan": {"min_cluster_size": 10}},
        weights=str(weights_file)
    )
    
    result = exp.run(
        param="hdbscan.min_cluster_size",
        values=[5, 10, 15],
        verbose=False
    )
    
    # Verify compound scores computed
    for run in result.runs:
        if run.pipeline_result.status == "completed":
            assert run.compound_score is not None
    
    # Verify best_run populated
    assert result.best_run is not None
    assert result.best_run.score_type == "compound_score"
    assert "hdbscan.min_cluster_size" in result.best_run.param_values
```

### TC-023: Light mode faster than heavy (SC-002)

```python
@pytest.mark.slow
def test_light_mode_faster_than_heavy(large_embeddings):
    """Light mode should be at least 30% faster on large datasets."""
    exp = Experiment(
        embeddings=large_embeddings,  # 5000+ samples
        config={"clustering_algorithm": "hdbscan", "hdbscan": {"min_cluster_size": 10}}
    )
    
    # Time heavy mode
    start = time.time()
    exp.run(param="hdbscan.min_cluster_size", values=[10, 15], mode="heavy", verbose=False)
    heavy_time = time.time() - start
    
    # Time light mode
    start = time.time()
    exp.run(param="hdbscan.min_cluster_size", values=[10, 15], mode="light", verbose=False)
    light_time = time.time() - start
    
    # Assert at least 30% faster
    assert light_time < heavy_time * 0.7
```

### TC-024: Weight coverage warning displayed

```python
def test_weight_coverage_warning_displayed(tmp_path, capfd):
    """Warning should print when excluded metrics dominate weights."""
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({
        "coefficients": {
            "Gamma_norm": 0.5,       # Excluded in light mode
            "Silhouette_norm": 0.3,
        },
        "bias": 0.2
    }))
    
    exp = Experiment(
        embeddings=np.random.randn(100, 10),
        config={"clustering_algorithm": "hdbscan"},
        weights=str(weights_file)
    )
    
    with pytest.warns(WeightCoverageWarning):
        exp.run(
            param="hdbscan.min_cluster_size",
            values=[5, 10],
            mode="light",
            verbose=True
        )
```

---

## CLI Tests

**File**: `tests/unit/test_labricate_cli.py` (extend existing)

### TC-025: CLI --weights option accepted

```python
def test_cli_weights_option(runner, tmp_path):
    """CLI should accept --weights option."""
    weights_file = tmp_path / "weights.json"
    weights_file.write_text('{"coefficients": {"Silhouette_norm": 1}, "bias": 0}')
    
    result = runner.invoke(cli, [
        "labricate", "experiment",
        "-e", "embeddings.csv",
        "-c", "config.json",
        "-p", "hdbscan.min_cluster_size",
        "-v", "5,10",
        "--weights", str(weights_file)
    ])
    
    assert result.exit_code == 0 or "compound_score" in result.output
```

### TC-026: CLI --mode option accepted

```python
def test_cli_mode_option(runner):
    """CLI should accept --mode light/heavy."""
    result = runner.invoke(cli, [
        "labricate", "experiment",
        "-e", "embeddings.csv",
        "-c", "config.json",
        "-p", "hdbscan.min_cluster_size",
        "-v", "5,10",
        "--mode", "light"
    ])
    
    # Should not error on option parsing
    assert "invalid choice" not in result.output.lower()
```

### TC-027: CLI output includes best_run (FR-012)

```python
def test_cli_json_output_includes_best_run(runner, tmp_path):
    """JSON output should have best_run field."""
    output_dir = tmp_path / "output"
    
    runner.invoke(cli, [
        "labricate", "experiment",
        "-e", "embeddings.csv",
        "-c", "config.json",
        "-p", "hdbscan.min_cluster_size",
        "-v", "5,10",
        "-o", str(output_dir),
        "--format", "json"
    ])
    
    json_files = list(output_dir.glob("*.json"))
    if json_files:
        data = json.loads(json_files[0].read_text())
        assert "best_run" in data
```

---

## Test Fixtures

```python
# conftest.py additions

@pytest.fixture
def sample_embeddings():
    """Small embeddings for quick tests."""
    return np.random.randn(100, 10)

@pytest.fixture
def large_embeddings():
    """Large embeddings for performance tests."""
    return np.random.randn(5000, 50)

@pytest.fixture
def sample_weights():
    """Standard weights for testing."""
    return MetricWeights(
        coefficients={
            "Silhouette_norm": 0.3,
            "Davies-Bouldin_norm": -0.2,
            "Calinski-Harabasz_norm": 0.2,
        },
        bias=0.3
    )

def make_run_result(
    run_id: int,
    param_values: dict | None = None,
    silhouette: float = 0.5,
    davies_bouldin: float = 1.0,
    compound_score: float | None = None,
    status: str = "completed",
) -> RunResult:
    """Factory for RunResult in tests."""
    ...

def make_failed_run_result(run_id: int) -> RunResult:
    """Factory for failed RunResult."""
    ...

def make_experiment_result(
    with_compound_scores: bool = False,
    with_best_run: bool = False,
    tied_run_ids: list[int] | None = None,
) -> ExperimentResult:
    """Factory for ExperimentResult in tests."""
    ...
```

---

## Test Summary

| Category | Tests | Coverage |
|----------|-------|----------|
| modes.py | TC-001 to TC-005 | 5 tests |
| scoring.py | TC-006 to TC-014 | 9 tests |
| Weights integration | TC-015 to TC-021 | 7 tests |
| Integration | TC-022 to TC-024 | 3 tests |
| CLI | TC-025 to TC-027 | 3 tests |
| **Total** | **TC-001 to TC-027** | **27 tests** |

### Requirement Traceability

| Requirement | Test Coverage |
|-------------|---------------|
| FR-001 (weights param) | TC-015, TC-016 |
| FR-002 (validation) | TC-017 |
| FR-003 (compound_score) | TC-006, TC-007 |
| FR-004 (best_run) | TC-008 to TC-012 |
| FR-005 (mode param) | TC-002 to TC-005 |
| FR-006 (light excludes) | TC-001, TC-003 |
| FR-008 (df compound_score) | TC-019 |
| FR-009 (df is_best_run) | TC-020, TC-021 |
| FR-015 (param_values) | TC-011 |
| FR-016 (ties) | TC-010, TC-021 |
| FR-017 (coverage warning) | TC-013, TC-014, TC-024 |
| SC-002 (30% faster) | TC-023 |
| SC-005 (unchanged) | TC-018 |
