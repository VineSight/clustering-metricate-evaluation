#!/usr/bin/env python3
"""
Generate training dataset from 1247315 baseline and merge with existing results.

This script:
1. Generates degraded versions of narrative_dataset_model_1247315_with_reduced.csv
2. Evaluates metrics on all degraded versions
3. Merges results with existing *_metrics_results.csv files
4. Filters out excluded degradation types (default exclusions)

Usage:
    python generate_merged_training.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import metricate
from metricate.degradation import DEFAULT_DEGRADATION_TYPES, EXCLUDED_DEGRADATION_TYPES
from metricate.training.normalize import normalize_metrics, get_internal_metric_names
from metricate.core.reference import METRIC_REFERENCE

# ============================================================================
# CONFIGURATION
# ============================================================================

# New baseline dataset
BASELINE_CSV = "datasets/narrative_dataset_model_1247315_with_reduced.csv"

# Output directory for degraded datasets from 1247315
OUTPUT_DIR = Path("baseline_comparison_outputs/1247315")

# Column specifications for the 1247315 dataset
LABEL_COL = "cluster_id"
EMBEDDING_COLS = ["reduced_embedding"]

# Existing results to merge with
EXISTING_RESULTS = [
    "baseline_comparison_outputs/1303134/1303134_metrics_results.csv",
    "baseline_comparison_outputs/1304526/1304526_metrics_results.csv",
    "baseline_comparison_outputs/1305111/1305111_metrics_results.csv",
]

# Output paths
MERGED_OUTPUT = Path("training_merged")

# Degradation levels to use
LEVELS = ["5pct", "10pct", "25pct", "50pct"]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("GENERATING MERGED TRAINING DATASET")
    print("=" * 70)
    
    # Print excluded degradation types for reference
    print(f"\nExcluded degradation types (filtered out):")
    for t in EXCLUDED_DEGRADATION_TYPES:
        print(f"  - {t}")
    
    print(f"\nIncluded degradation types:")
    for t in DEFAULT_DEGRADATION_TYPES:
        print(f"  - {t}")
    
    # Step 1: Generate degraded versions from new baseline
    print("\n" + "=" * 70)
    print("[STEP 1] Generating degradations from 1247315 baseline")
    print("=" * 70)
    print(f"Input: {BASELINE_CSV}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Types: {len(DEFAULT_DEGRADATION_TYPES)} (excluding {len(EXCLUDED_DEGRADATION_TYPES)} problematic types)")
    print(f"Levels: {LEVELS}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    degraded_dir = OUTPUT_DIR / "degraded_datasets"
    degraded_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate degradations (only default types, excluding problematic ones)
    result = metricate.degrade(
        BASELINE_CSV,
        str(degraded_dir),
        types=DEFAULT_DEGRADATION_TYPES,
        levels=LEVELS,
        label_col=LABEL_COL,
        embedding_cols=EMBEDDING_COLS,
        random_seed=42,
        visualize=False,
    )
    
    print(f"\n✓ Generated {len(result.degradations)} degraded datasets")
    if result.warnings:
        for w in result.warnings:
            print(f"  Warning: {w}")
    
    # Step 2: Evaluate metrics on baseline and all degraded versions
    print("\n" + "=" * 70)
    print("[STEP 2] Evaluating metrics on all versions")
    print("=" * 70)
    
    records = []
    
    # Evaluate baseline
    print(f"Evaluating baseline: {BASELINE_CSV}")
    baseline_eval = metricate.evaluate(
        BASELINE_CSV,
        label_col=LABEL_COL,
        embedding_cols=EMBEDDING_COLS,
        force_all=True,  # Skip O(n²) metrics on large dataset
    )
    
    # Convert baseline to record
    baseline_record = {
        "type": "baseline",
        "level": 0.0,
        "filename": "baseline",
        "n_posts": baseline_eval.metadata.get("n_samples", 0),
        "n_noise_points": 0,
        "n_clusters": baseline_eval.metadata.get("n_clusters", 0),
        "n_samples": baseline_eval.metadata.get("n_samples", 0),
        "n_samples_total": baseline_eval.metadata.get("n_samples", 0),
    }
    
    for m in baseline_eval.metrics:
        baseline_record[m.metric] = m.value
    
    records.append(baseline_record)
    print(f"  ✓ Baseline: {baseline_record['n_clusters']} clusters, {baseline_record['n_samples']} samples")
    
    # Evaluate each degraded version
    total = len(result.degradations)
    for i, degradation in enumerate(result.degradations, 1):
        deg_type = degradation.type
        deg_level = degradation.level
        filepath = degradation.filepath
        
        print(f"[{i}/{total}] Evaluating: {deg_type} @ {deg_level}")
        
        try:
            eval_result = metricate.evaluate(
                filepath,
                label_col=LABEL_COL,
                embedding_cols=EMBEDDING_COLS,
                force_all=True,
            )
            
            # Parse level to float
            level_map = {"5pct": 0.05, "10pct": 0.10, "25pct": 0.25, "50pct": 0.50}
            level_float = level_map.get(deg_level, 0.0)
            
            record = {
                "type": deg_type,
                "level": level_float,
                "filename": Path(filepath).name,
                "n_posts": eval_result.metadata.get("n_samples", 0),
                "n_noise_points": 0,
                "n_clusters": eval_result.metadata.get("n_clusters", 0),
                "n_samples": eval_result.metadata.get("n_samples", 0),
                "n_samples_total": eval_result.metadata.get("n_samples", 0),
            }
            
            for m in eval_result.metrics:
                record[m.metric] = m.value
            
            records.append(record)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Create DataFrame from new results
    df_new = pd.DataFrame(records)
    df_new["clustering_name"] = "1247315"
    
    # Save new results
    new_results_path = OUTPUT_DIR / "1247315_metrics_results.csv"
    df_new.to_csv(new_results_path, index=False)
    print(f"\n✓ Saved new results to: {new_results_path}")
    print(f"  Rows: {len(df_new)}")
    
    # Step 3: Load and merge with existing results
    print("\n" + "=" * 70)
    print("[STEP 3] Merging with existing results")
    print("=" * 70)
    
    all_dfs = [df_new]
    
    for csv_path in EXISTING_RESULTS:
        path = Path(csv_path)
        if not path.exists():
            print(f"  ✗ Not found: {csv_path}")
            continue
        
        clustering_id = path.parent.name
        df = pd.read_csv(csv_path)
        df["clustering_name"] = clustering_id
        all_dfs.append(df)
        print(f"  ✓ Loaded {len(df)} rows from {clustering_id}")
    
    # Concatenate all
    df_merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total merged (before filtering): {len(df_merged)} rows")
    
    # Step 4: Filter out excluded degradation types
    print("\n" + "=" * 70)
    print("[STEP 4] Filtering out excluded degradation types")
    print("=" * 70)
    
    before = len(df_merged)
    df_filtered = df_merged[~df_merged["type"].isin(EXCLUDED_DEGRADATION_TYPES)]
    removed = before - len(df_filtered)
    
    print(f"  Removed {removed} rows with excluded types")
    print(f"  Remaining: {len(df_filtered)} rows")
    
    # Show breakdown by type
    print(f"\n  Degradation type counts after filtering:")
    type_counts = df_filtered["type"].value_counts()
    for t, count in type_counts.items():
        print(f"    {t}: {count}")
    
    # Show breakdown by clustering
    print(f"\n  Rows per clustering:")
    cluster_counts = df_filtered["clustering_name"].value_counts()
    for c, count in cluster_counts.items():
        print(f"    {c}: {count}")
    
    # Step 5: Normalize metrics
    print("\n" + "=" * 70)
    print("[STEP 5] Normalizing metrics")
    print("=" * 70)
    
    # Identify metric columns (exclude metadata columns and external metrics)
    metadata_cols = {
        "type", "level", "filename", "n_posts", "n_noise_points", 
        "n_clusters", "n_samples", "n_samples_total", "clustering_name",
        "source_file", "quality_score", "Adjusted Rand Index", 
        "Van Dongen", "Variation of Information", "Omega"
    }
    
    # Get metric columns from the data that are in METRIC_REFERENCE
    metric_cols = [
        col for col in df_filtered.columns 
        if col not in metadata_cols 
        and col in METRIC_REFERENCE
        and not col.endswith("_norm")
    ]
    
    print(f"  Found {len(metric_cols)} metrics to normalize:")
    for col in metric_cols[:5]:
        direction = METRIC_REFERENCE.get(col, {}).get("direction", "higher")
        print(f"    - {col} ({'↑' if direction == 'higher' else '↓'})")
    if len(metric_cols) > 5:
        print(f"    ... and {len(metric_cols) - 5} more")
    
    # Apply percentile normalization (handles direction automatically)
    df_normalized = normalize_metrics(df_filtered, metric_cols)
    
    norm_cols = [c for c in df_normalized.columns if c.endswith("_norm")]
    print(f"\n  ✓ Created {len(norm_cols)} normalized columns (suffix: _norm)")
    print(f"    Normalization: percentile rank [0, 1], 1 = best")
    
    # Step 6: Save merged output
    print("\n" + "=" * 70)
    print("[STEP 6] Saving merged training dataset")
    print("=" * 70)
    
    MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Save both raw and normalized versions
    merged_path = MERGED_OUTPUT / "training_data_merged.csv"
    df_filtered.to_csv(merged_path, index=False)
    
    normalized_path = MERGED_OUTPUT / "training_data_normalized.csv"
    df_normalized.to_csv(normalized_path, index=False)
    
    print(f"✓ Saved merged training data to:")
    print(f"    {merged_path} (raw metrics)")
    print(f"    {normalized_path} (with normalized columns)")
    print(f"  Total rows: {len(df_filtered)}")
    print(f"  Clusterings: {df_filtered['clustering_name'].nunique()}")
    print(f"  Degradation types: {df_filtered['type'].nunique()}")
    print(f"  Metric columns: {len(metric_cols)} raw + {len(norm_cols)} normalized")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Files created:
  - {new_results_path} ({len(df_new)} rows) - new baseline results
  - {merged_path} ({len(df_filtered)} rows) - raw metrics
  - {normalized_path} ({len(df_normalized)} rows) - with _norm columns

Merged from:
  - 1247315 (new baseline): {len(df_new)} rows
  - 1303134: {len(all_dfs[1]) if len(all_dfs) > 1 else 0} rows
  - 1304526: {len(all_dfs[2]) if len(all_dfs) > 2 else 0} rows  
  - 1305111: {len(all_dfs[3]) if len(all_dfs) > 3 else 0} rows

Normalization:
  - Method: Percentile rank → [0, 1]
  - Direction: "lower is better" metrics inverted (so 1 = best always)
  - Columns: {len(norm_cols)} metrics normalized (suffix: _norm)

Excluded degradation types (filtered out):
  {', '.join(EXCLUDED_DEGRADATION_TYPES)}

Included degradation types:
  {', '.join(DEFAULT_DEGRADATION_TYPES)}
""")


if __name__ == "__main__":
    main()
