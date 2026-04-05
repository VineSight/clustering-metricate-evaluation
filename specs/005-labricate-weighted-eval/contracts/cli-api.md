# CLI API Contract: Labricate Weighted Evaluation

**Version**: 1.0  
**Date**: 2026-03-26

## Command: metricate labricate experiment

### New Options

```bash
metricate labricate experiment [OPTIONS]
```

#### --weights / -w (NEW - FR-010)

```
--weights, -w PATH
    Path to weights JSON file for compound scoring.
    
    The weights file must follow metricate's weights schema:
    {
      "version": "1.0",
      "coefficients": {
        "Silhouette_norm": 0.15,
        "Davies-Bouldin_norm": -0.12,
        ...
      },
      "bias": 0.5,
      "metadata": { ... }
    }
    
    When provided:
    - Each run includes a compound_score (0-1)
    - best_run is determined by highest compound_score
    - Output includes compound_score column in CSV/JSON
```

#### --mode / -m (NEW - FR-011)

```
--mode, -m [light|heavy]
    Computation mode for metrics. Default: heavy
    
    light: Fast mode, excludes expensive O(n²) metrics:
           - Gamma, G-plus, Tau, Point-Biserial, McClain-Rao, NIVA
           Recommended for: Quick iteration, large datasets (5k+ points)
    
    heavy: Comprehensive mode, computes all 34 metrics.
           Recommended for: Final evaluation, smaller datasets
```

### Updated Examples

#### Single-parameter with weights

```bash
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    -p "hdbscan.min_cluster_size" \
    -v "5,10,15,20" \
    --weights weights.json
```

Output includes:
```
✓ Experiment complete: 4/4 runs succeeded
  Best run: run_id=3 (compound_score=0.847)
    hdbscan.min_cluster_size=15
```

#### Light mode for quick iteration

```bash
metricate labricate experiment \
    -e large_embeddings.csv \
    -c config.json \
    -p "hdbscan.min_cluster_size" \
    -v "5,10,15,20,25,30" \
    --mode light
```

Output notes:
```
Running in LIGHT mode (excluding 6 expensive metrics)
...
```

#### Combined weights + light mode

```bash
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    -p "hdbscan.min_cluster_size" \
    -v "5,10,15,20" \
    --weights weights.json \
    --mode light
```

Warning output (if applicable):
```
⚠ Warning: Excluded metrics ['Gamma', 'Tau'] account for 35% of weight 
  coefficients. Compound scores may be unreliable.
```

#### Grid search with weights

```bash
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    --grid \
    --params "hdbscan.min_cluster_size=5,10,15" \
    --params "hdbscan.min_samples=3,5" \
    --weights weights.json
```

### Output Format Changes

#### JSON Output (FR-012)

```json
{
  "experiment_id": "exp_20260326_143022",
  "experiment_name": "hdbscan_sweep",
  "best_run": {
    "run_id": 3,
    "param_values": {
      "hdbscan.min_cluster_size": 15
    },
    "score": 0.847,
    "score_type": "compound_score",
    "tied_run_ids": []
  },
  "runs": [
    {
      "run_id": 1,
      "param_values": {"hdbscan.min_cluster_size": 5},
      "compound_score": 0.723,
      "metrics": { ... }
    },
    ...
  ],
  "summary": { ... }
}
```

#### CSV Output

New columns when weights provided:
- `compound_score`: float
- `is_best_run`: boolean (1/0)

```csv
run_id,hdbscan.min_cluster_size,n_clusters,Silhouette,Davies-Bouldin,...,compound_score,is_best_run
1,5,12,0.45,1.23,...,0.723,0
2,10,8,0.52,0.98,...,0.801,0
3,15,6,0.61,0.87,...,0.847,1
4,20,5,0.58,0.91,...,0.812,0
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments |
| 2 | Weights file not found |
| 3 | Weights validation failed |
| 4 | All runs failed |

### Verbose Output (-v)

With `--verbose` flag, additional output:
```
✓ Experiment complete: 4/4 runs succeeded
  Total duration: 2m 34s
  
  Best configuration:
    run_id: 3
    hdbscan.min_cluster_size: 15
    compound_score: 0.847
    
  (No ties detected)
```

With ties:
```
  Best configuration:
    run_id: 3 (tied with run_id: 7)
    hdbscan.min_cluster_size: 15
    compound_score: 0.847
```
