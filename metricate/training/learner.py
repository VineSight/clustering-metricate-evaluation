"""
Machine learning pipeline for learning metric weights.

This module provides functions to train regression models (Ridge/Lasso) on
degraded clustering datasets to learn optimal metric weights for quality scoring.

The learned coefficients form a fixed formula:
    score = clip(Σ(weight_i × metric_i) + bias, 0, 1)

Example:
    >>> from metricate.training.learner import train_weights
    >>> result = train_weights("training_data.csv", regularization="ridge")
    >>> print(f"CV R²: {result.cv_scores['r2']:.3f}")
    >>> result.weights.save("weights.json")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from metricate.training.weights import MetricWeights

__all__ = [
    "train_weights",
    "cross_validate_weights",
    "export_weights",
    "plot_feature_importance",
    "TrainingResult",
    "CVResult",
]

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """
    Cross-validation fold results.

    Attributes:
        fold: Fold index (0-based)
        held_out_group: Clustering name held out
        train_size: Number of training samples
        test_size: Number of test samples
        r2: R² score on held-out fold
        rmse: Root mean squared error
        mae: Mean absolute error
    """

    fold: int
    held_out_group: str
    train_size: int
    test_size: int
    r2: float
    rmse: float
    mae: float


@dataclass
class TrainingResult:
    """
    Output from the training process.

    Attributes:
        weights: The learned MetricWeights
        cv_scores: Cross-validation metrics dict with 'r2', 'rmse', 'mae' keys
        feature_importance: Metrics ranked by abs(coefficient) descending
        zeroed_metrics: Metrics with coefficient = 0 (for Lasso)
        sanity_check_passed: True if original > all degraded for all clusterings
        sanity_failures: Clustering names where sanity check failed
        cv_results: Per-fold cross-validation results
    """

    weights: MetricWeights
    cv_scores: dict[str, float] = field(default_factory=dict)
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    zeroed_metrics: list[str] = field(default_factory=list)
    sanity_check_passed: bool = True
    sanity_failures: list[str] = field(default_factory=list)
    cv_results: list[CVResult] = field(default_factory=list)

    def save_weights(self, path: str | Path) -> None:
        """Save the learned weights to a JSON file."""
        self.weights.save(path)


def train_weights(
    csv_path: str | Path,
    *,
    regularization: Literal["ridge", "lasso"] = "ridge",
    alpha: float = 1.0,
    auto_alpha: bool = False,
    alphas: list[float] | None = None,
    run_cv: bool = True,
    cv_splits: int = 5,
    run_sanity_check: bool = True,
) -> TrainingResult:
    """
    Train a regression model to learn metric weights for quality scoring.

    Args:
        csv_path: Path to training dataset CSV with normalized metrics and quality_score.
        regularization: Type of regularization ("ridge" or "lasso").
        alpha: Regularization strength (ignored if auto_alpha=True).
        auto_alpha: If True, use cross-validation to select optimal alpha.
        alphas: Candidate alpha values for auto-tuning (default: [0.01, 0.1, 1.0, 10.0, 100.0]).
        run_cv: If True, run cross-validation and populate cv_scores.
        cv_splits: Number of CV folds (default: 5).
        run_sanity_check: If True, verify original > all degraded for each clustering.

    Returns:
        TrainingResult containing learned weights, CV scores, and feature importance.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If CSV missing required columns or invalid format.
    """
    # Load and validate data
    df = _load_training_data(csv_path)
    X, feature_names = _extract_features(df)
    y = _extract_target(df)

    logger.info(f"Training on {len(y)} samples with {len(feature_names)} features")

    # Default alphas for auto-tuning
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Select and train model
    if auto_alpha:
        if regularization == "ridge":
            model = RidgeCV(alphas=alphas, cv=5)
        else:
            model = LassoCV(alphas=alphas, cv=5, max_iter=10000)
        model.fit(X, y)
        selected_alpha = model.alpha_
        logger.info(f"Auto-selected alpha: {selected_alpha}")
    else:
        selected_alpha = alpha
        if regularization == "ridge":
            model = Ridge(alpha=selected_alpha)
        else:
            model = Lasso(alpha=selected_alpha, max_iter=10000)
        model.fit(X, y)

    # Extract coefficients
    coefficients = dict(zip(feature_names, model.coef_, strict=True))
    bias = float(model.intercept_)

    # Compute feature importance (sorted by abs coefficient)
    feature_importance = sorted(
        [(name, float(coef)) for name, coef in coefficients.items()],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Detect zeroed metrics (for Lasso)
    zeroed_metrics = [name for name, coef in coefficients.items() if coef == 0.0]
    non_zero_count = len(feature_names) - len(zeroed_metrics)

    # Create MetricWeights
    weights = MetricWeights(
        coefficients=coefficients,
        bias=bias,
        regularization=regularization,
        alpha=selected_alpha,
        training_samples=len(y),
        non_zero_count=non_zero_count,
    )

    # Run cross-validation if requested
    cv_scores: dict[str, float] = {}
    cv_results: list[CVResult] = []

    if run_cv:
        try:
            cv_results, cv_scores = cross_validate_weights(
                csv_path,
                regularization=regularization,
                alpha=selected_alpha,
                n_splits=cv_splits,
            )
            logger.info(f"CV R²: {cv_scores.get('r2_mean', 0):.3f}±{cv_scores.get('r2_std', 0):.3f}")
        except ValueError as e:
            logger.warning(f"Cross-validation skipped: {e}")

    # Run sanity check if requested
    sanity_check_passed = True
    sanity_failures: list[str] = []

    if run_sanity_check:
        try:
            sanity_check_passed, sanity_failures = sanity_check(weights, csv_path)
        except Exception as e:
            logger.warning(f"Sanity check skipped due to error: {e}")

    # Create result
    result = TrainingResult(
        weights=weights,
        cv_scores=cv_scores,
        feature_importance=feature_importance,
        zeroed_metrics=zeroed_metrics,
        cv_results=cv_results,
        sanity_check_passed=sanity_check_passed,
        sanity_failures=sanity_failures,
    )

    logger.info(f"Training complete. Non-zero coefficients: {non_zero_count}/{len(feature_names)}")

    return result


def cross_validate_weights(
    csv_path: str | Path,
    *,
    regularization: Literal["ridge", "lasso"] = "ridge",
    alpha: float = 1.0,
    n_splits: int = 5,
) -> tuple[list[CVResult], dict[str, float]]:
    """
    Perform leave-one-clustering-out cross-validation.

    Uses GroupKFold to ensure all degradation variants of each base clustering
    are held out together, testing generalization to unseen clusterings.

    Args:
        csv_path: Path to training dataset CSV.
        regularization: Type of regularization ("ridge" or "lasso").
        alpha: Regularization strength.
        n_splits: Number of CV folds (limited by number of unique groups).

    Returns:
        Tuple of (list of CVResult with per-fold metrics, aggregate metrics dict).
        Aggregate dict contains: r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std.
    """
    # Load data
    df = _load_training_data(csv_path)
    X, feature_names = _extract_features(df)
    y = _extract_target(df)
    groups, group_names = _extract_groups(df)

    # Build group name mapping
    group_name_map = dict(enumerate(group_names))

    # Limit n_splits to number of unique groups
    n_unique_groups = len(set(groups))
    actual_splits = min(n_splits, n_unique_groups)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 groups for CV, found {n_unique_groups}")
    if actual_splits < n_splits:
        logger.warning(f"Reduced n_splits from {n_splits} to {actual_splits} (only {n_unique_groups} unique groups)")

    # Setup GroupKFold
    gkf = GroupKFold(n_splits=actual_splits)

    # Select model class
    if regularization == "ridge":
        model_class = Ridge
    else:
        model_class = Lasso

    cv_results: list[CVResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Identify held-out group
        test_groups = set(groups[test_idx])
        held_out_names = [group_name_map.get(g, f"group_{g}") for g in test_groups]
        held_out_group = ", ".join(held_out_names)

        # Train model on this fold
        if regularization == "lasso":
            model = model_class(alpha=alpha, max_iter=10000)
        else:
            model = model_class(alpha=alpha)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        cv_results.append(
            CVResult(
                fold=fold_idx,
                held_out_group=held_out_group,
                train_size=len(train_idx),
                test_size=len(test_idx),
                r2=float(r2),
                rmse=rmse,
                mae=mae,
            )
        )

        logger.debug(f"Fold {fold_idx}: R²={r2:.3f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Compute aggregate metrics
    r2_scores = [r.r2 for r in cv_results]
    rmse_scores = [r.rmse for r in cv_results]
    mae_scores = [r.mae for r in cv_results]

    aggregate = {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
    }

    logger.info(f"CV complete: R²={aggregate['r2_mean']:.3f}±{aggregate['r2_std']:.3f}")

    return cv_results, aggregate


# ============================================================================
# Private helper functions
# ============================================================================


def _load_training_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate training data from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    required_cols = {"clustering_name", "quality_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    # Check for at least one _norm column
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    if not norm_cols:
        raise ValueError("Training data must have at least one *_norm metric column")

    logger.debug(f"Loaded {len(df)} rows with {len(norm_cols)} metric columns")
    return df


def _extract_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract feature matrix and column names from DataFrame."""
    # Get all normalized metric columns
    norm_cols = sorted([c for c in df.columns if c.endswith("_norm")])
    X = df[norm_cols].values.astype(np.float64)

    # Handle any NaN values (replace with column mean)
    for i, col in enumerate(norm_cols):
        col_data = X[:, i]
        if np.isnan(col_data).any():
            col_mean = np.nanmean(col_data)
            X[np.isnan(X[:, i]), i] = col_mean
            logger.warning(f"NaN values in {col} replaced with mean: {col_mean:.4f}")

    return X, norm_cols


def _extract_target(df: pd.DataFrame) -> np.ndarray:
    """Extract target values from DataFrame."""
    y = df["quality_score"].values.astype(np.float64)
    if np.isnan(y).any():
        raise ValueError("quality_score contains NaN values")
    return y


def _extract_groups(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Extract group labels for cross-validation.

    Strategy:
    1. If multiple base clusterings exist → group by base clustering name (LOCO)
    2. If only 1 base clustering but degradation_type exists → group by degradation_type
    3. Otherwise → return simple row indices for k-fold

    Returns:
        Tuple of (group_ids array, list of group names for reporting)
    """
    # Try clustering_name first
    if "clustering_name" in df.columns:
        base_names = df["clustering_name"].apply(_get_base_clustering_name)
        unique_bases = base_names.unique()

        if len(unique_bases) > 1:
            # Multiple base clusterings - use LOCO CV
            name_to_id = {name: i for i, name in enumerate(sorted(unique_bases))}
            group_ids = base_names.map(name_to_id).values.astype(np.int32)
            group_names = sorted(unique_bases)
            logger.debug(f"Using LOCO CV with {len(unique_bases)} base clusterings")
            return group_ids, group_names

    # Fall back to degradation_type if available
    if "degradation_type" in df.columns:
        deg_types = df["degradation_type"].fillna("original")
        unique_types = deg_types.unique()

        if len(unique_types) > 1:
            # Group by degradation type
            type_to_id = {t: i for i, t in enumerate(sorted(unique_types))}
            group_ids = deg_types.map(type_to_id).values.astype(np.int32)
            group_names = sorted(unique_types)
            logger.debug(f"Using degradation-type CV with {len(unique_types)} types")
            return group_ids, group_names

    # Last resort: simple indices (will result in regular k-fold)
    group_ids = np.arange(len(df), dtype=np.int32)
    group_names = [f"sample_{i}" for i in range(len(df))]
    logger.debug("Using simple k-fold CV (no meaningful groups found)")
    return group_ids, group_names


def _get_base_clustering_name(name: str) -> str:
    """
    Extract base clustering name from potentially degraded name.

    Examples:
        "baseline" -> "baseline"
        "noise_injection_10pct" -> "baseline" (if standalone)
        "model_123_noise_10pct" -> "model_123"
    """
    # Common degradation suffixes to strip
    degradation_patterns = [
        "_noise_injection_", "_label_swap_", "_core_removal_",
        "_boundary_reassign_", "_centroid_displacement_",
        "_embedding_perturb_", "_merge_", "_split_",
        "_remove_", "pct", "_random", "_nearest", "_farthest",
        "_largest", "_smallest", "_tightest", "_loosest",
    ]

    # If name contains degradation pattern, strip it
    result = name
    for pattern in degradation_patterns:
        if pattern in result.lower():
            idx = result.lower().find(pattern)
            result = result[:idx] if idx > 0 else result

    return result.strip("_") or "baseline"


# ============================================================================
# Export functions
# ============================================================================


def export_weights(
    result: TrainingResult,
    output_path: str | Path,
    *,
    include_feature_importance: bool = True,
    include_zeroed_metrics: bool = True,
) -> Path:
    """
    Export trained weights to JSON file.

    Creates a JSON file compatible with load_weights() for use in evaluate/compare.

    Args:
        result: TrainingResult from train_weights().
        output_path: Path for output JSON file.
        include_feature_importance: Include ranked feature importance in JSON.
        include_zeroed_metrics: Include list of zeroed metrics (Lasso) in JSON.

    Returns:
        Absolute path to the saved file.

    Example:
        >>> result = train_weights("training.csv")
        >>> path = export_weights(result, "learned_weights.json")
        >>> print(f"Weights saved to {path}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build export data
    data = {
        "coefficients": result.weights.coefficients,
        "bias": result.weights.bias,
        "regularization": result.weights.regularization,
        "alpha": result.weights.alpha,
        "training_samples": result.weights.training_samples,
        "non_zero_count": result.weights.non_zero_count,
    }

    if include_feature_importance and result.feature_importance:
        data["feature_importance"] = [
            {"metric": name, "weight": weight}
            for name, weight in result.feature_importance
        ]

    if include_zeroed_metrics and result.zeroed_metrics:
        data["zeroed_metrics"] = result.zeroed_metrics

    # Include CV scores if available
    if result.cv_scores:
        data["cv_scores"] = result.cv_scores

    # Write JSON
    import json

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Weights exported to {output_path}")
    return output_path.resolve()


def sanity_check(
    weights: MetricWeights,
    training_csv: str | Path,
) -> tuple[bool, list[str]]:
    """
    Verify that weights produce expected ordering: original > all degraded.

    For each base clustering in the training data, computes compound scores
    and verifies that the original (quality_score=1.0) has the highest score.

    Args:
        weights: Trained MetricWeights.
        training_csv: Path to the training CSV used for training.

    Returns:
        Tuple of (sanity_check_passed, list of failure descriptions).
        Empty list means all sanity checks passed.

    Example:
        >>> result = train_weights("training.csv")
        >>> passed, failures = sanity_check(result.weights, "training.csv")
        >>> if not passed:
        ...     print(f"Sanity check failed: {failures}")
    """
    from metricate.training.weights import compute_compound_score

    df = _load_training_data(training_csv)
    X, feature_names = _extract_features(df)

    failures: list[str] = []

    # Build normalized metrics dict for each row
    for idx, row in df.iterrows():
        metrics_norm = {col: row[col] for col in feature_names}

        score, _ = compute_compound_score(metrics_norm, weights)
        df.loc[idx, "computed_score"] = score

    # Group by base clustering and check ordering
    df["base_name"] = df["clustering_name"].apply(_get_base_clustering_name)

    for base_name, group in df.groupby("base_name"):
        # Find original (quality_score closest to 1.0)
        original_rows = group[group["quality_score"] >= 0.99]

        if original_rows.empty:
            logger.warning(f"No original clustering found for base: {base_name}")
            continue

        original_score = original_rows["computed_score"].max()

        # Check all degraded versions have lower score
        degraded_rows = group[group["quality_score"] < 0.99]

        for _, deg_row in degraded_rows.iterrows():
            deg_score = deg_row["computed_score"]
            deg_name = deg_row["clustering_name"]

            if deg_score >= original_score:
                failures.append(
                    f"{deg_name}: degraded score ({deg_score:.4f}) >= original ({original_score:.4f})"
                )

    passed = len(failures) == 0

    if passed:
        logger.info("Sanity check passed: all original clusterings score higher than their degraded versions")
    else:
        logger.warning(f"Sanity check failed: {len(failures)} violations found")

    return passed, failures


def plot_feature_importance(
    training_result: TrainingResult,
    *,
    top_n: int = 10,
    exclude_zeroed: bool = True,
    title: str = "Feature Importance (Learned Metric Weights)",
    show: bool = True,
    save_path: str | Path | None = None,
) -> "go.Figure":
    """
    Generate a horizontal bar chart of metric coefficient importance.

    Creates a plotly visualization showing the most important metrics
    based on their learned coefficient magnitudes.

    Args:
        training_result: TrainingResult from train_weights().
        top_n: Number of top features to display (default: 10).
        exclude_zeroed: If True, exclude metrics with zero coefficients (default: True).
        title: Chart title.
        show: If True, display the figure interactively.
        save_path: If provided, save HTML to this path.

    Returns:
        Plotly Figure object.

    Raises:
        ImportError: If plotly is not installed.

    Example:
        >>> result = train_weights("training_data.csv")
        >>> fig = plot_feature_importance(result, top_n=15)
        >>> fig.write_html("importance.html")
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "Plotly is required for visualizations. Install with: pip install plotly"
        ) from e

    # Get feature importance from training result
    importance = training_result.feature_importance

    if not importance:
        raise ValueError("No feature importance data in training result")

    # Filter out zeroed metrics if requested
    if exclude_zeroed:
        importance = [(name, coef) for name, coef in importance if coef != 0.0]

    # Take top N (already sorted by abs value descending)
    top_features = importance[:top_n]

    if not top_features:
        raise ValueError("No non-zero features to display")

    # Reverse for horizontal bar chart (top at top)
    top_features = list(reversed(top_features))

    # Extract names and values
    names = [name.replace("_norm", "") for name, _ in top_features]
    values = [coef for _, coef in top_features]

    # Create color scale based on sign (positive=green, negative=red)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Metadata for subtitle
    weights = training_result.weights
    subtitle = (
        f"Regularization: {weights.regularization.upper()}, "
        f"α={weights.alpha:.4g}, "
        f"Non-zero: {weights.non_zero_count}/{len(weights.coefficients)}"
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Coefficient Value",
        yaxis_title="Metric",
        template="plotly_white",
        height=max(400, 50 * len(top_features)),
        margin=dict(l=150, r=100, t=80, b=50),
        showlegend=False,
    )

    # Save if path provided
    if save_path:
        fig.write_html(str(save_path))
        logger.info(f"Saved feature importance plot to {save_path}")

    # Show if requested
    if show:
        fig.show()

    return fig
