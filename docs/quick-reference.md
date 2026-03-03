# Quick Reference

A one-page cheat sheet for Metricate.

---

## Python API

```python
import metricate

# Evaluate
result = metricate.evaluate("clustering.csv")
print(result.to_table())

# Compare
comparison = metricate.compare("v1.csv", "v2.csv")
print(f"Winner: {comparison.winner}")

# Degrade
result = metricate.degrade("data.csv", "./output/")

# Train weights
result = metricate.train_weights("training.csv")
result.save_weights("weights.json")

# Use weights
weights = metricate.load_weights("weights.json")
result = metricate.evaluate("data.csv", weights=weights)
print(f"Score: {result.compound_score:.3f}")

# List options
metricate.list_metrics()
metricate.list_degradations()
```

---

## CLI Commands

```bash
# Evaluate
metricate evaluate clustering.csv
metricate evaluate data.csv --format json -o results.json
metricate evaluate data.csv --weights weights.json

# Compare
metricate compare v1.csv v2.csv
metricate compare a.csv b.csv --name-a "Old" --name-b "New"

# Train
metricate train training.csv -o weights.json
metricate train data.csv -o weights.json --auto-alpha -r lasso

# Degrade
metricate degrade data.csv ./output/
metricate degrade data.csv ./output/ --types label_swap_random,noise_injection

# List
metricate list metrics
metricate list degradations

# Web UI
metricate web --port 5000
```

---

## Key Options

| Option | Python | CLI | Description |
|--------|--------|-----|-------------|
| Exclude metrics | `exclude=["Gamma"]` | `--exclude Gamma` | Skip specific metrics |
| Force all | `force_all=True` | `--force-all` | Compute O(n²) on large data |
| Label column | `label_col="cluster"` | `--label-col cluster` | Specify label column |
| Embeddings | `embedding_cols=["dim_0"]` | `--embedding-cols dim_0` | Specify embedding columns |
| Weights | `weights=weights` | `--weights file.json` | Use learned weights |

---

## Metrics Summary

| Tier | Count | Complexity | Example |
|------|-------|------------|---------|
| Original | 6 | O(n) - O(n²) | Silhouette, Davies-Bouldin |
| Tier 1 | 6 | O(n·d) | Ball-Hall, R-squared |
| Tier 2 | 14 | O(n²) | Gamma, Tau, CS Index |
| Tier 3 | 5 | O(n·d²) | Banfield-Raftery, NIVA |
| External | 4 | O(n) | Adjusted Rand Index |

**Direction:**
- ↑ Higher is better: Silhouette, Calinski-Harabasz, Dunn Index, etc.
- ↓ Lower is better: Davies-Bouldin, SSE, S_Dbw, etc.

---

## Degradation Types

| Category | Types |
|----------|-------|
| **Labels** | `label_swap_random`, `label_swap_neighboring`, `label_swap_distant` |
| **Structure** | `merge_random`, `merge_nearest`, `merge_farthest`, `split_random`, `split_largest`, `split_loosest` |
| **Points** | `noise_injection`, `random_removal`, `core_removal`, `boundary_reassignment` |
| **Clusters** | `remove_smallest_clusters`, `remove_largest_clusters`, `remove_tightest_clusters` |
| **Embeddings** | `embedding_perturbation`, `centroid_displacement` |

**Levels:** `5pct`, `10pct`, `25pct`, `50pct`

---

## Input Format

**Minimum CSV:**
```csv
cluster_id,dim_0,dim_1,dim_2
0,0.1,0.2,0.3
0,0.15,0.25,0.35
1,0.8,0.9,1.0
```

**String embeddings (auto-parsed):**
```csv
cluster_id,embedding
0,"[0.1, 0.2, 0.3]"
1,"[0.8, 0.9, 1.0]"
```

---

## Training Data Format

```csv
clustering_name,Silhouette_norm,Davies-Bouldin_norm,...,quality_score
original,0.95,0.88,...,1.0
degraded_10pct,0.85,0.75,...,0.8
```

---

## Large Dataset Auto-Skip (>50k rows)

These O(n²) metrics are skipped by default:
- Gamma, Tau, Point-Biserial, G-plus, McClain-Rao, NIVA

Override with `force_all=True` / `--force-all`

---

## Result Objects

**EvaluationResult:**
```python
result.metrics          # dict of metric values
result.compound_score   # if weights provided
result.to_table()       # formatted string
result.to_dataframe()   # pandas DataFrame
result.to_json()        # JSON string
```

**ComparisonResult:**
```python
comparison.winner           # winning clustering name
comparison.wins             # {"A": n, "B": m, "Tie": t}
comparison.metric_winners   # metric → winner mapping
comparison.weighted_winner  # if weights provided
```

---

## Training Result

```python
result.cv_scores              # {"r2_mean": 0.85, "mae_mean": 0.12, ...}
result.feature_importance     # [(metric, weight), ...]
result.sanity_check_passed    # True/False
result.weights                # MetricWeights object
result.save_weights("w.json") # Save to file
```
