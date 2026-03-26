# Research: Labricate - Hyperparameter Experimentation Framework

**Date**: March 22, 2026 (Updated)  
**Feature**: 004-labricate  
**Purpose**: Resolve technical unknowns before implementation

---

## 0. BERTopic Library Integration (Primary Approach)

### Decision
Use BERTopic library as the default pipeline instead of direct UMAP + HDBSCAN.

### Rationale
- Mature, well-tested UMAP → HDBSCAN pipeline (v0.17.4, 7.5k GitHub stars)
- Modular architecture - accepts custom UMAP/HDBSCAN models
- Active development and community support
- Handles edge cases (noise points, empty clusters) gracefully

### Key Configuration
```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# Create custom sub-models with Labricate config
umap_model = UMAP(
    n_neighbors=config["umap"]["n_neighbors"],
    n_components=config["umap"]["n_components"],
    min_dist=config["umap"]["min_dist"],
    metric=config["umap"]["metric"],
    low_memory=config["umap"].get("low_memory", False),
    random_state=config.get("random_seed", 42)
)

hdbscan_model = HDBSCAN(
    min_cluster_size=config["hdbscan"]["min_cluster_size"],
    min_samples=config["hdbscan"]["min_samples"],
    cluster_selection_method=config["hdbscan"]["cluster_selection_method"],
    metric=config["hdbscan"].get("metric", "euclidean"),
    prediction_data=True
)

# Create BERTopic with custom models, skip topic representation
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,  # Skip for speed
    verbose=True
)

# Fit with pre-computed embeddings + placeholder docs
n_samples = embeddings.shape[0]
placeholder_docs = [""] * n_samples
topics, _ = topic_model.fit_transform(placeholder_docs, embeddings=embeddings)

# Extract outputs for Metricate
labels = topic_model.topics_  # Cluster assignments
reduced_embeddings = topic_model.umap_model.embedding_  # UMAP-reduced embeddings
```

### Output Extraction
| Output | BERTopic Attribute | Description |
|--------|-------------------|-------------|
| Cluster labels | `topic_model.topics_` | 1D array, -1 = noise |
| Reduced embeddings | `topic_model.umap_model.embedding_` | 2D array after UMAP |
| Topic info | `topic_model.get_topic_info()` | Only if representation enabled |

### Topic Representation (Optional)
By default, skip c-TF-IDF for speed. Enable via config:
```python
if config.get("enable_topic_representation", False):
    # Keep default representation behavior
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True
    )
```

---

## 1. UMAP Integration (via BERTopic)

### Decision
Use `umap-learn` library with explicit random_state parameter for reproducibility.

### Rationale
- `umap-learn` is the standard Python implementation, well-maintained
- Supports all required parameters: `n_neighbors`, `n_components`, `min_dist`, `metric`, `repulsion_strength`
- `random_state` parameter enables reproducible results when seed is provided
- Already commonly used with BERTopic pipelines

### Alternatives Considered
- **cuML UMAP** (GPU): Rejected - adds CUDA dependency, not essential for research tool
- **openTSNE**: Rejected - different algorithm, not what BERTopic uses

### Implementation Notes
```python
from umap import UMAP

umap_model = UMAP(
    n_neighbors=config["umap"]["n_neighbors"],
    n_components=config["umap"]["n_components"],
    min_dist=config["umap"]["min_dist"],
    metric=config["umap"]["metric"],
    repulsion_strength=config["umap"]["repulsion_strength"],
    low_memory=config["umap"].get("low_memory", False),  # NEW: large dataset support
    random_state=config.get("random_seed", 42)
)
reduced = umap_model.fit_transform(embeddings)
```

---

## 2. HDBSCAN Integration

### Decision
Use `hdbscan` library with `prediction_data=True` for label assignment.

### Rationale
- Standard implementation, actively maintained
- Supports required parameters: `min_cluster_size`, `min_samples`, `cluster_selection_method`, `metric`
- Can handle noise points (label=-1) which Metricate already handles
- No native random_state, but results are deterministic given same input

### Alternatives Considered
- **scikit-learn HDBSCAN** (sklearn 1.3+): Considered but `hdbscan` library has more features
- **OPTICS**: Rejected - different algorithm characteristics

### Implementation Notes
```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=config["hdbscan"]["min_cluster_size"],
    min_samples=config["hdbscan"]["min_samples"],
    cluster_selection_method=config["hdbscan"]["cluster_selection_method"],
    metric=config["hdbscan"].get("metric", "euclidean"),  # NEW: distance metric
    prediction_data=True
)
labels = clusterer.fit_predict(reduced_embeddings)
```

---

## 3. K-Means Integration

### Decision
Use `sklearn.cluster.KMeans` with random_state parameter.

### Rationale
- Standard scikit-learn implementation
- Simple interface, well-tested
- `random_state` enables reproducibility
- Already a project dependency via Metricate

### Implementation Notes
```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=config["kmeans"]["n_clusters"],
    random_state=config.get("random_seed", 42),
    n_init=10
)
labels = kmeans.fit_predict(reduced_embeddings)
```

---

## 4. Dot-Notation Path Resolution

### Decision
Implement custom path resolver using string split + recursive dict access.

### Rationale
- Simple to implement (< 30 lines)
- No external dependency needed
- Supports validation with clear error messages

### Alternatives Considered
- **python-dotenv style**: Overkill for nested dicts
- **jmespath**: Too powerful, adds dependency
- **glom**: Adds dependency for simple use case

### Implementation Notes
```python
def resolve_path(config: dict, path: str) -> tuple[dict, str]:
    """Returns (parent_dict, key) for a dot-notation path."""
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current:
            raise ValueError(f"Invalid path: '{path}' - section '{part}' not found")
        current = current[part]
    final_key = parts[-1]
    if final_key not in current:
        raise ValueError(f"Invalid path: '{path}' - key '{final_key}' not found")
    return current, final_key

def set_param(config: dict, path: str, value) -> dict:
    """Set a parameter value using dot notation."""
    config = copy.deepcopy(config)
    parent, key = resolve_path(config, path)
    parent[key] = value
    return config
```

---

## 5. Checkpoint/Resume Strategy

### Decision
Use JSON checkpoint file with run status tracking.

### Rationale
- Simple file format, human-readable
- Easy to implement save/load
- Can be manually edited if needed
- Matches project's NO_ORM principle

### Checkpoint Schema
```json
{
  "experiment_id": "exp_20260318_143022",
  "total_runs": 12,
  "completed_runs": [0, 1, 2, 3, 4],
  "failed_runs": [],
  "current_run": 5,
  "results": {
    "0": {"config": {...}, "metrics": {...}, "status": "completed"},
    "1": {"config": {...}, "metrics": {...}, "status": "completed"}
  }
}
```

### Resume Logic
1. Check for `checkpoint.json` in output directory
2. If exists, compare config hash with checkpoint's stored hash
3. If config mismatch:
   - Warn user about config change
   - Require `--force` flag to start fresh
   - Without `--force`, abort with error
4. If config matches, prompt user: "Resume from run 5/12? [y/N]"
5. If yes, load completed results and skip to `current_run`
6. If no, delete checkpoint and start fresh

---

## 6. Parallel Execution Strategy

### Decision
Use `concurrent.futures.ProcessPoolExecutor` for parallelism.

### Rationale
- Built into Python standard library
- ProcessPool avoids GIL issues with numpy/UMAP
- Simple API for map-style parallel execution
- Easy to limit workers via `max_workers` parameter

### Alternatives Considered
- **multiprocessing.Pool**: Lower-level, ProcessPoolExecutor is cleaner
- **joblib**: Adds dependency, used by sklearn but not needed here
- **Ray**: Overkill for local parallelism

### Worker Count Capping
```python
import os

def get_effective_workers(requested: int) -> int:
    cpu_count = os.cpu_count() or 1
    if requested > cpu_count:
        import warnings
        warnings.warn(
            f"Requested {requested} workers exceeds CPU count ({cpu_count}). "
            f"Capping at {cpu_count}."
        )
        return cpu_count
    return requested
```

### Implementation Notes
```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_parallel(configs: list[dict], n_workers: int):
    n_workers = get_effective_workers(n_workers)  # Cap at CPU count
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single, cfg): i for i, cfg in enumerate(configs)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"error": str(e)}
    return results
```

---

## 7. Visualization Library

### Decision
Use `matplotlib` for basic charts.

### Rationale
- Already in scientific Python stack
- Sufficient for line charts and basic heatmaps
- No additional dependencies
- Can save to PNG/PDF without display

### Alternatives Considered
- **Plotly**: Interactive but adds dependency, overkill for static charts
- **seaborn**: Built on matplotlib, could use for styling but not required
- **Altair**: Declarative but adds dependency

### Chart Types
1. **Single-param experiment**: Line chart (metric value vs parameter value)
2. **Grid search**: Heatmap (2D parameter grid, color = metric value)

---

## 8. Progress Bar Library

### Decision
Use `tqdm` for progress bars.

### Rationale
- De facto standard for Python progress bars
- Simple API: `for item in tqdm(items)`
- Works in notebooks and terminals
- Lightweight dependency

### Implementation Notes
```python
from tqdm import tqdm

for i, config in enumerate(tqdm(configs, desc="Running experiments")):
    tqdm.write(f"Run {i+1}/{len(configs)}: {param_path}={config[param_path]}")
    result = run_pipeline(embeddings, config)
```

---

## 9. Metricate Integration

### Decision
Direct import using existing `metricate.evaluate()` function.

### Rationale
- Labricate is a submodule of metricate package
- Direct import avoids subprocess overhead
- Can pass metric filtering options directly
- Access to structured result objects

### Implementation Notes
```python
from metricate import evaluate

# Convert pipeline output to DataFrame for Metricate
df = pd.DataFrame({
    "cluster_id": labels,
    **{f"dim_{i}": reduced_embeddings[:, i] for i in range(reduced_embeddings.shape[1])}
})

# Ground truth handling: if provided, supervised metrics are automatically included
result = evaluate(
    df,
    label_col="cluster_id",
    ground_truth=ground_truth_labels,  # None if not provided (unsupervised only)
    exclude=exclude_metrics,  # Optional filtering
    force_all=False
)
```

---

## 10. CLI Framework

### Decision
Extend existing `metricate` CLI with subcommand.

### Rationale
- Metricate already uses Click for CLI
- Adding subcommand maintains single entry point
- Consistent with constitution (single package)

### Command Structure
```bash
metricate labricate experiment \
    --embeddings data.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values 5,10,15,20 \
    --output-dir ./results \
    --output-format json \
    --workers 4 \
    --resume
```

---

## Dependencies Summary

| Dependency | Version | Purpose | New? |
|------------|---------|---------|------|
| bertopic | >=0.15 | Topic modeling pipeline (UMAP + HDBSCAN) | NEW |
| scikit-learn | >=1.0 | K-Means, utilities | Existing |
| numpy | >=1.20 | Array operations | Existing |
| pandas | >=1.3 | DataFrames | Existing |
| matplotlib | >=3.5 | Visualization | Existing |
| tqdm | >=4.60 | Progress bars | NEW |
| click | >=8.0 | CLI | Existing |

> **Note**: BERTopic is installed without embedding backends (`pip install bertopic`). 
> Users provide pre-computed embeddings, so sentence-transformers is not required.
> BERTopic transitively installs umap-learn and hdbscan.
