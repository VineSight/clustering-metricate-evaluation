#!/usr/bin/env python3
"""
Training script using real baseline comparison results.

This script demonstrates how to:
1. Fuse multiple results CSVs together
2. Prepare training data (add quality_score, normalize metrics)
3. Train metric weights using the metricate.training module

Usage:
    python train_from_results.py
    python train_from_results.py --regularization lasso
    python train_from_results.py --auto-alpha
    python train_from_results.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Import from the metricate package
from metricate.training import train_weights, export_weights, TrainingResult
from metricate.training.normalize import normalize_metrics, get_internal_metric_names
from metricate.core.reference import METRIC_REFERENCE
from metricate.degradation import EXCLUDED_DEGRADATION_TYPES, DEFAULT_DEGRADATION_TYPES

# ============================================================================
# CONFIGURATION - Adjust these as needed
# ============================================================================

# Paths to the results CSVs
RESULTS_CSVS = [
    "baseline_comparison_outputs/1303134/1303134_metrics_results.csv",
    "baseline_comparison_outputs/1304526/1304526_metrics_results.csv",
    "baseline_comparison_outputs/1305111/1305111_metrics_results.csv",
]

# Output paths
OUTPUT_DIR = Path("training_real_data")

# External metrics to exclude (need ground truth)
EXTERNAL_METRICS = ["Adjusted Rand Index", "Van Dongen", "Variation of Information", "Omega"]


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def load_and_fuse_csvs(csv_paths: list[str | Path]) -> pd.DataFrame:
    """
    Load multiple results CSVs and fuse them together.
    
    Each CSV gets a clustering_name based on its source folder.
    """
    all_dfs = []
    
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping...")
            continue
            
        # Extract clustering ID from path (e.g., "1303134")
        clustering_id = path.parent.name
        
        df = pd.read_csv(path)
        df["clustering_name"] = str(clustering_id)  # Must be string for learner
        df["source_file"] = str(path)
        
        print(f"  Loaded {len(df)} rows from {clustering_id}")
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No CSV files found!")
    
    fused = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total fused: {len(fused)} rows")
    
    return fused


def compute_quality_score(row: pd.Series) -> float:
    """
    Compute quality score based on degradation type and level.
    
    Logic:
    - baseline (type='baseline') → 1.0
    - label_swap_* at level L → 1.0 - L  (e.g., 5% swap → 0.95)
    - random_removal at level L → 1.0 - (L * 0.5)  (less severe)
    - cluster operations → based on number of clusters affected
    - merge/split operations → similar to removal
    """
    deg_type = row.get("type", "baseline")
    level = row.get("level", 0.0)
    
    # Baseline is always perfect
    if deg_type == "baseline" or level == 0.0:
        return 1.0
    
    # Label swap operations - most impactful on quality
    if "label_swap" in deg_type:
        # label_swap_distant is most severe, then random, then neighboring
        if "distant" in deg_type:
            severity = 1.2  # 20% more severe
        elif "random" in deg_type:
            severity = 1.0
        else:  # neighboring
            severity = 0.8  # 20% less severe (neighbors are more similar)
        return max(0.0, 1.0 - (level * severity))
    
    # Random removal - moderate impact
    if "random_removal" in deg_type:
        return max(0.0, 1.0 - (level * 0.5))
    
    # Cluster removal operations
    if "remove_" in deg_type:
        # Level here is number of clusters removed
        n_clusters = row.get("n_clusters", 20)
        impact = level / max(n_clusters, 1)
        return max(0.0, 1.0 - impact)
    
    # Merge operations - moderate to low impact
    if "merge_" in deg_type:
        n_clusters = row.get("n_clusters", 20)
        impact = level / max(n_clusters, 1) * 0.5
        return max(0.0, 1.0 - impact)
    
    # Split operations - low impact
    if "split_" in deg_type:
        return max(0.0, 1.0 - (level * 0.1))
    
    # Fallback
    return max(0.0, 1.0 - level)


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the fused data for training:
    1. Add quality_score column
    2. Identify metric columns
    3. Normalize all metrics using metricate's normalize_metrics()
    """
    df = df.copy()
    
    # Ensure clustering_name is string (required by metricate.training.learner)
    if "clustering_name" in df.columns:
        df["clustering_name"] = df["clustering_name"].astype(str)
    
    # Add quality_score
    print("\n  Computing quality scores...")
    df["quality_score"] = df.apply(compute_quality_score, axis=1)
    
    # Identify metric columns (exclude metadata and external metrics)
    metadata_cols = {
        "type", "level", "filename", "n_posts", "n_noise_points", 
        "n_clusters", "n_samples", "n_samples_total", "clustering_name",
        "source_file", "quality_score"
    }
    
    # Get metric columns from the data that are also in METRIC_REFERENCE
    all_metric_names = set(METRIC_REFERENCE.keys())
    metric_cols = [
        col for col in df.columns 
        if col not in metadata_cols 
        and col not in EXTERNAL_METRICS
        and not col.endswith("_norm")
        and col in all_metric_names
    ]
    
    print(f"  Found {len(metric_cols)} internal metrics to normalize:")
    for col in metric_cols[:5]:
        direction = METRIC_REFERENCE.get(col, {}).get("direction", "higher")
        print(f"    - {col} ({'↑' if direction == 'higher' else '↓'})")
    print(f"    ... and {len(metric_cols) - 5} more")
    
    # Use metricate's normalize_metrics function
    print("\n  Normalizing metrics using metricate.training.normalize...")
    df = normalize_metrics(df, metric_cols)
    
    # Count normalized columns
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    print(f"  Created {len(norm_cols)} normalized columns")
    
    return df


def print_training_result(result: TrainingResult, top_n: int = 15) -> None:
    """Print training results in a nice format."""
    print(f"\n  Results:")
    print(f"    Regularization: {result.weights.regularization.upper()}")
    print(f"    Alpha: {result.weights.alpha}")
    print(f"    Non-zero coefficients: {result.weights.non_zero_count}/{len(result.weights.coefficients)}")
    print(f"    Bias (intercept): {result.weights.bias:.4f}")
    
    if result.cv_scores:
        print(f"\n  Cross-validation:")
        print(f"    R² = {result.cv_scores.get('r2_mean', 0):.4f} ± {result.cv_scores.get('r2_std', 0):.4f}")
        print(f"    RMSE = {result.cv_scores.get('rmse_mean', 0):.4f} ± {result.cv_scores.get('rmse_std', 0):.4f}")
        print(f"    MAE = {result.cv_scores.get('mae_mean', 0):.4f} ± {result.cv_scores.get('mae_std', 0):.4f}")
    
    print(f"\n  Top {top_n} features by |coefficient|:")
    print("  " + "-" * 50)
    
    for i, (name, coef) in enumerate(result.feature_importance[:top_n], 1):
        display_name = name.replace("_norm", "")
        direction = "+" if coef > 0 else "-"
        print(f"    {i:2}. {display_name:30} {direction}{abs(coef):.4f}")
    
    if result.zeroed_metrics:
        print(f"\n  Zeroed metrics (Lasso eliminated): {len(result.zeroed_metrics)}")
        for name in result.zeroed_metrics[:5]:
            print(f"    - {name.replace('_norm', '')}")
        if len(result.zeroed_metrics) > 5:
            print(f"    ... and {len(result.zeroed_metrics) - 5} more")
    
    if not result.sanity_check_passed:
        print(f"\n  ⚠️  Sanity check failed for: {result.sanity_failures}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train metric weights from baseline comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_from_results.py                    # Default: Ridge with alpha=1.0
  python train_from_results.py --regularization lasso  # Lasso for feature selection
  python train_from_results.py --auto-alpha       # Auto-select best alpha via CV
  python train_from_results.py --alpha 0.1        # Custom alpha value
  python train_from_results.py --no-cv            # Skip cross-validation

Uses metricate.training module functions:
  - normalize_metrics() for percentile normalization
  - train_weights() for Ridge/Lasso regression
  - export_weights() for saving learned coefficients
""",
    )
    
    # Training flags (these map to train_weights() parameters)
    parser.add_argument(
        "--regularization", "-r",
        choices=["ridge", "lasso"],
        default="ridge",
        help="Regularization type: 'ridge' (L2, keeps all features) or 'lasso' (L1, feature selection). Default: ridge"
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=1.0,
        help="Regularization strength. Higher = more regularization. Default: 1.0"
    )
    parser.add_argument(
        "--auto-alpha",
        action="store_true",
        help="Automatically select best alpha using cross-validation"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        help="Alpha values to try when --auto-alpha is set"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation (faster, but no generalization estimate)"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV folds. Default: 5"
    )
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="Skip sanity check (original > degraded for each clustering)"
    )
    parser.add_argument(
        "--exclude-problematic",
        action="store_true",
        help=f"Exclude degradation types that often improve metrics: {EXCLUDED_DEGRADATION_TYPES}"
    )
    
    # Output flags
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for results. Default: {OUTPUT_DIR}"
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=15,
        help="Number of top features to display. Default: 15"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING METRIC WEIGHTS FROM REAL DATA")
    print("Using metricate.training module")
    print("=" * 60)
    
    # Step 1: Load and fuse CSVs
    print("\n[1] Loading and fusing results CSVs...")
    df_fused = load_and_fuse_csvs(RESULTS_CSVS)
    
    # Optional: Filter out problematic degradation types
    if args.exclude_problematic:
        print(f"\n  Filtering out excluded degradation types: {EXCLUDED_DEGRADATION_TYPES}")
        before = len(df_fused)
        df_fused = df_fused[~df_fused["type"].isin(EXCLUDED_DEGRADATION_TYPES)]
        print(f"  Removed {before - len(df_fused)} rows, {len(df_fused)} remaining")
    
    # Step 2: Prepare training data
    print("\n[2] Preparing training data...")
    df_prepared = prepare_training_data(df_fused)
    
    # Save intermediate files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fused_path = args.output_dir / "fused_results.csv"
    training_path = args.output_dir / "training_data_normalized.csv"
    weights_path = args.output_dir / "learned_weights.json"
    
    df_fused.to_csv(fused_path, index=False)
    # Ensure clustering_name stays string when reloaded by train_weights
    # Add a prefix to force string type  
    df_prepared["clustering_name"] = "model_" + df_prepared["clustering_name"].astype(str)
    df_prepared.to_csv(training_path, index=False)
    print(f"\n  Saved fused data to: {fused_path}")
    print(f"  Saved training data to: {training_path}")
    
    # Step 3: Train weights using metricate.training.train_weights()
    print("\n[3] Training weights using metricate.training.train_weights()...")
    print(f"  Parameters:")
    print(f"    regularization = '{args.regularization}'")
    print(f"    alpha = {args.alpha}")
    print(f"    auto_alpha = {args.auto_alpha}")
    print(f"    run_cv = {not args.no_cv}")
    print(f"    cv_splits = {args.cv_splits}")
    print(f"    run_sanity_check = {not args.no_sanity_check}")
    
    result: TrainingResult = train_weights(
        training_path,
        regularization=args.regularization,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        alphas=args.alphas,
        run_cv=not args.no_cv,
        cv_splits=args.cv_splits,
        run_sanity_check=not args.no_sanity_check,
    )
    
    # Step 4: Display and save results
    print("\n[4] Results:")
    print_training_result(result, args.top_features)
    
    # Save using metricate's export_weights()
    export_weights(result, weights_path)
    print(f"\n  Weights saved to: {weights_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Summary
    print(f"""
Summary:
  - Input: {len(RESULTS_CSVS)} results CSVs fused ({len(df_fused)} total rows)
  - Quality scores: baseline=1.0, degraded=0.0-0.95
  - Features: {len(result.weights.coefficients)} normalized metrics
  - Non-zero weights: {result.weights.non_zero_count}/{len(result.weights.coefficients)}
  - Regularization: {result.weights.regularization.upper()} (α={result.weights.alpha})
  - Sanity check: {'✓ passed' if result.sanity_check_passed else '✗ failed'}
""")
    
    if result.cv_scores:
        print(f"  Cross-validation:")
        print(f"    R² = {result.cv_scores.get('r2_mean', 0):.4f} ± {result.cv_scores.get('r2_std', 0):.4f}")


if __name__ == "__main__":
    main()
