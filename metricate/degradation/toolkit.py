"""
Degradation toolkit for generating systematically degraded clusterings.

This module provides the degrade() function to generate degraded versions
of clustered datasets with various degradation types and levels.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Degradation type definitions
DEGRADATION_TYPES = {
    "Label Manipulation": [
        "label_swap_random",
        "label_swap_neighboring",
        "label_swap_distant",
    ],
    "Cluster Structure": [
        "merge_random",
        "merge_nearest",
        "merge_farthest",
        "split_random",
        "split_largest",
        "split_loosest",
    ],
    "Point Manipulation": [
        "noise_injection",
        "random_removal",
        "core_removal",
        "boundary_reassignment",
    ],
    "Cluster Removal": [
        "remove_smallest_clusters",
        "remove_largest_clusters",
        "remove_tightest_clusters",
    ],
    "Embedding Manipulation": [
        "embedding_perturbation",
        "centroid_displacement",
    ],
}

ALL_DEGRADATION_TYPES = [t for types in DEGRADATION_TYPES.values() for t in types]

# Degradation types excluded by default (can still be used explicitly)
EXCLUDED_DEGRADATION_TYPES = [
    "split_loosest",
    "split_random",
    "random_removal",
    "remove_largest_clusters",
    "remove_smallest_clusters",
    "merge_nearest",
]

# Default degradation types (excludes problematic/redundant types)
DEFAULT_DEGRADATION_TYPES = [t for t in ALL_DEGRADATION_TYPES if t not in EXCLUDED_DEGRADATION_TYPES]

DEFAULT_LEVELS = ["5pct", "10pct", "25pct", "50pct"]
LEVEL_FRACTIONS = {
    "5pct": 0.05,
    "10pct": 0.10,
    "25pct": 0.25,
    "50pct": 0.50,
}


@dataclass
class DegradationConfig:
    """Configuration for degradation generation."""

    types: list[str] = field(default_factory=lambda: DEFAULT_DEGRADATION_TYPES.copy())
    levels: list[str] = field(default_factory=lambda: DEFAULT_LEVELS.copy())
    random_seed: int = 42
    generate_visualizations: bool = True

    def __post_init__(self):
        """Validate config."""
        # Validate types
        invalid_types = [t for t in self.types if t not in ALL_DEGRADATION_TYPES]
        if invalid_types:
            raise ValueError(f"Unknown degradation types: {invalid_types}")

        # Validate levels
        invalid_levels = [l for l in self.levels if l not in LEVEL_FRACTIONS]
        if invalid_levels:
            raise ValueError(
                f"Unknown levels: {invalid_levels}. Valid: {list(LEVEL_FRACTIONS.keys())}"
            )


@dataclass
class DegradationEntry:
    """Information about a single generated degradation."""

    type: str
    level: str
    filename: str
    filepath: str
    n_rows: int
    original_rows: int
    change_description: str


@dataclass
class DegradationResult:
    """Result of degradation generation."""

    output_dir: str
    degradations: list[DegradationEntry]
    manifest_path: str
    index_html_path: str | None = None
    visualizations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def csv_files(self) -> list[str]:
        """List of generated CSV file paths."""
        return [d.filepath for d in self.degradations]

    def summary(self) -> str:
        """Generate a text summary of the degradation results."""
        lines = [
            "Degradation Generation Complete",
            "=" * 40,
            f"Output directory: {self.output_dir}",
            f"Total degradations: {len(self.degradations)}",
            f"Manifest: {self.manifest_path}",
        ]
        if self.index_html_path:
            lines.append(f"Index HTML: {self.index_html_path}")
        if self.visualizations:
            lines.append(f"Visualizations: {len(self.visualizations)}")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _import_degrader():
    """Import ClusteringDegrader from local degradation_toolkit module."""
    try:
        from metricate.degradation.degradation_toolkit import ClusteringDegrader

        return ClusteringDegrader
    except ImportError:
        # If not found, use a minimal implementation
        return None


def degrade(
    csv_path: str,
    output_dir: str = "./degraded_output",
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    random_seed: int = 42,
    visualize: bool = True,
) -> DegradationResult:
    """
    Generate degraded versions of a clustered dataset.

    Creates multiple degraded versions of the input dataset using various
    degradation strategies at different intensity levels. Optionally generates
    HTML visualizations for each degradation.

    Args:
        csv_path: Path to input CSV with clustering data
        output_dir: Directory to write degraded datasets
        label_col: Column containing cluster labels (auto-detected if None)
        embedding_cols: Columns containing embeddings (auto-detected if None)
        types: Degradation types to apply (all if None)
        levels: Degradation levels (default: 5pct, 10pct, 25pct, 50pct)
        random_seed: Random seed for reproducibility
        visualize: Whether to generate HTML visualizations

    Returns:
        DegradationResult with information about generated files

    Examples:
        >>> result = degrade("clustering.csv", "./output/")
        >>> print(result.summary())

        >>> result = degrade(
        ...     "data.csv",
        ...     output_dir="./degraded/",
        ...     types=["label_swap_random", "noise_injection"],
        ...     levels=["10pct", "25pct"],
        ... )
    """
    import pandas as pd

    # Load the raw CSV to detect columns
    df = pd.read_csv(csv_path)

    # Auto-detect label column
    if label_col is None:
        label_candidates = [
            "cluster_id",
            "cluster",
            "label",
            "labels",
            "class",
            "group",
            "new_cluster",
            "cluster_label",
            "assignment",
            "group_id",
        ]
        for candidate in label_candidates:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError("Could not auto-detect label column. Please specify --label-col.")

    # Auto-detect embedding column(s)
    if embedding_cols is None:
        # First check for string-encoded embedding columns
        emb_candidates = ["reduced_embedding", "embedding", "embeddings"]
        for candidate in emb_candidates:
            if candidate in df.columns:
                embedding_cols = [candidate]
                break

        # If not found, look for dimensional columns (umap_*, dim_*, x_*, etc.)
        if embedding_cols is None:
            dim_cols = [
                col
                for col in df.columns
                if any(
                    col.lower().startswith(prefix)
                    for prefix in [
                        "umap_",
                        "tsne_",
                        "pca_",
                        "dim_",
                        "x_",
                        "emb_",
                        "component_",
                        "reduced_",
                    ]
                )
            ]
            if len(dim_cols) >= 2:
                embedding_cols = sorted(dim_cols)

        if embedding_cols is None:
            raise ValueError(
                "Could not auto-detect embedding column. Please specify --embedding-cols."
            )

    # For ClusteringDegrader, we need to use the first column name
    # If we have dimensional columns, we'll pass None and let it auto-detect
    embedding_col = embedding_cols[0] if len(embedding_cols) == 1 else None

    # Create config
    config = DegradationConfig(
        types=types or ALL_DEGRADATION_TYPES,
        levels=levels or DEFAULT_LEVELS.copy(),
        random_seed=random_seed,
        generate_visualizations=visualize,
    )

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    degradations = []
    warnings = []

    # Try to use the full ClusteringDegrader
    ClusteringDegrader = _import_degrader()

    if ClusteringDegrader is not None:
        # Use the full degrader
        degrader = ClusteringDegrader(
            csv_path,
            cluster_col=label_col,
            embedding_col=embedding_col,
            random_seed=random_seed,
        )

        original_rows = len(degrader.df_original)

        # Generate degradations
        for deg_type in config.types:
            for level in config.levels:
                fraction = LEVEL_FRACTIONS[level]

                try:
                    degraded_df = _apply_degradation(degrader, deg_type, fraction, random_seed)

                    # Generate filename
                    filename = f"{deg_type}_{level}.csv"
                    filepath = output_path / filename

                    # Save
                    _save_degraded(degraded_df, filepath)

                    # Record
                    degradations.append(
                        DegradationEntry(
                            type=deg_type,
                            level=level,
                            filename=filename,
                            filepath=str(filepath),
                            n_rows=len(degraded_df),
                            original_rows=original_rows,
                            change_description=_get_change_description(
                                deg_type, level, original_rows, len(degraded_df)
                            ),
                        )
                    )

                except Exception as e:
                    warnings.append(f"Failed to generate {deg_type}_{level}: {e}")
    else:
        # Fallback: simple degradation without full toolkit
        warnings.append("ClusteringDegrader not available, using simplified degradation")
        original_rows = len(df)

        for deg_type in config.types:
            for level in config.levels:
                fraction = LEVEL_FRACTIONS[level]

                try:
                    degraded_df = _simple_degradation(
                        df, label_col, deg_type, fraction, random_seed
                    )

                    filename = f"{deg_type}_{level}.csv"
                    filepath = output_path / filename

                    degraded_df.to_csv(filepath, index=False)

                    degradations.append(
                        DegradationEntry(
                            type=deg_type,
                            level=level,
                            filename=filename,
                            filepath=str(filepath),
                            n_rows=len(degraded_df),
                            original_rows=original_rows,
                            change_description=_get_change_description(
                                deg_type, level, original_rows, len(degraded_df)
                            ),
                        )
                    )

                except Exception as e:
                    warnings.append(f"Failed to generate {deg_type}_{level}: {e}")

    # Generate manifest
    manifest = _generate_manifest(csv_path, output_dir, degradations, config)
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate visualizations
    visualizations = []
    index_html_path = None

    if visualize and degradations:
        try:
            from metricate.degradation.visualize import generate_index, generate_visualizations

            visualizations = generate_visualizations(degradations, output_path, original_csv_path=csv_path)
            index_html_path = generate_index(degradations, output_path)

        except ImportError:
            warnings.append("Visualization module not available")
        except Exception as e:
            warnings.append(f"Failed to generate visualizations: {e}")

    return DegradationResult(
        output_dir=str(output_path),
        degradations=degradations,
        manifest_path=str(manifest_path),
        index_html_path=index_html_path,
        visualizations=visualizations,
        warnings=warnings,
    )


def _apply_degradation(degrader, deg_type: str, fraction: float, seed: int) -> pd.DataFrame:
    """Apply a specific degradation type using ClusteringDegrader."""
    np.random.seed(seed)

    # Map degradation types to methods
    if deg_type == "label_swap_random":
        return degrader.label_swap(fraction=fraction, swap_type="random")
    elif deg_type == "label_swap_neighboring":
        return degrader.label_swap(fraction=fraction, swap_type="neighboring")
    elif deg_type == "label_swap_distant":
        return degrader.label_swap(fraction=fraction, swap_type="distant")
    elif deg_type == "merge_random":
        n_merges = max(1, int(fraction * 10))
        return degrader.merge_clusters(n_merges=n_merges, merge_type="random")
    elif deg_type == "merge_nearest":
        n_merges = max(1, int(fraction * 10))
        return degrader.merge_clusters(n_merges=n_merges, merge_type="nearest")
    elif deg_type == "merge_farthest":
        n_merges = max(1, int(fraction * 10))
        return degrader.merge_clusters(n_merges=n_merges, merge_type="farthest")
    elif deg_type == "split_random":
        n_splits = max(1, int(fraction * 10))
        return degrader.split_clusters(n_splits=n_splits, split_type="random")
    elif deg_type == "split_largest":
        n_splits = max(1, int(fraction * 10))
        return degrader.split_clusters(n_splits=n_splits, split_type="largest")
    elif deg_type == "split_loosest":
        n_splits = max(1, int(fraction * 10))
        return degrader.split_clusters(n_splits=n_splits, split_type="loosest")
    elif deg_type == "noise_injection":
        n_noise = max(10, int(len(degrader.df_original) * fraction))
        return degrader.add_noise_points(n_noise=n_noise)
    elif deg_type == "random_removal":
        return degrader.random_removal(fraction=fraction)
    elif deg_type == "core_removal":
        return degrader.remove_core_points(fraction=fraction)
    elif deg_type == "boundary_reassignment":
        return degrader.boundary_reassignment(fraction=fraction)
    elif deg_type == "remove_smallest_clusters":
        n_clusters = max(1, int(fraction * 10))
        return degrader.remove_tight_clusters(n_clusters=n_clusters, criteria="smallest")
    elif deg_type == "remove_largest_clusters":
        n_clusters = max(1, int(fraction * 10))
        return degrader.remove_tight_clusters(n_clusters=n_clusters, criteria="largest")
    elif deg_type == "remove_tightest_clusters":
        n_clusters = max(1, int(fraction * 10))
        return degrader.remove_tight_clusters(n_clusters=n_clusters, criteria="tightest")
    elif deg_type == "embedding_perturbation":
        return degrader.embedding_perturbation(noise_scale=fraction)
    elif deg_type == "centroid_displacement":
        return degrader.centroid_displacement(displacement_scale=fraction)
    else:
        raise ValueError(f"Unknown degradation type: {deg_type}")


def _simple_degradation(
    df: pd.DataFrame, label_col: str, deg_type: str, fraction: float, seed: int
) -> pd.DataFrame:
    """Simple fallback degradation without full toolkit."""
    np.random.seed(seed)
    df = df.copy()

    if deg_type.startswith("label_swap"):
        # Simple random label swap
        n_swap = int(len(df) * fraction)
        swap_indices = np.random.choice(df.index, size=n_swap, replace=False)
        unique_labels = df[label_col].unique()
        for idx in swap_indices:
            current = df.loc[idx, label_col]
            other_labels = [l for l in unique_labels if l != current and l != -1]
            if other_labels:
                df.loc[idx, label_col] = np.random.choice(other_labels)
        return df

    elif deg_type == "random_removal":
        n_remove = int(len(df) * fraction)
        drop_indices = np.random.choice(df.index, size=n_remove, replace=False)
        return df.drop(drop_indices).reset_index(drop=True)

    else:
        # For other types, just return the original with a warning
        return df


def _save_degraded(df: pd.DataFrame, filepath: Path):
    """Save degraded dataframe, dropping internal columns."""
    df_to_save = df.drop(columns=["_parsed_embedding"], errors="ignore")
    df_to_save.to_csv(filepath, index=False)


def _get_change_description(deg_type: str, level: str, original_rows: int, new_rows: int) -> str:
    """Generate human-readable description of the degradation."""
    fraction = LEVEL_FRACTIONS[level]
    pct = int(fraction * 100)

    if deg_type.startswith("label_swap"):
        swap_type = deg_type.replace("label_swap_", "")
        return f"Swapped {pct}% of cluster labels ({swap_type} strategy)"
    elif deg_type.startswith("merge"):
        merge_type = deg_type.replace("merge_", "")
        return f"Merged {max(1, int(fraction * 10))} cluster pairs ({merge_type})"
    elif deg_type.startswith("split"):
        split_type = deg_type.replace("split_", "")
        return f"Split {max(1, int(fraction * 10))} clusters ({split_type})"
    elif deg_type == "noise_injection":
        return f"Added {int(original_rows * fraction)} noise points"
    elif deg_type == "random_removal":
        return f"Removed {pct}% of rows randomly ({original_rows - new_rows} rows)"
    elif deg_type == "core_removal":
        return f"Removed {pct}% of core points from each cluster"
    elif deg_type == "boundary_reassignment":
        return f"Reassigned {pct}% of boundary points"
    elif deg_type.startswith("remove_"):
        criteria = deg_type.replace("remove_", "").replace("_clusters", "")
        return f"Removed {max(1, int(fraction * 10))} {criteria} clusters"
    elif deg_type == "embedding_perturbation":
        return f"Added {pct}% noise to embeddings"
    elif deg_type == "centroid_displacement":
        return f"Displaced points {pct}% toward wrong centroids"
    else:
        return f"Applied {deg_type} at {level} level"


def _generate_manifest(
    input_path: str,
    output_dir: str,
    degradations: list[DegradationEntry],
    config: DegradationConfig,
) -> dict:
    """Generate manifest.json with metadata."""
    return {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "input_file": input_path,
        "output_directory": output_dir,
        "config": {
            "types": config.types,
            "levels": config.levels,
            "random_seed": config.random_seed,
        },
        "degradations": [
            {
                "type": d.type,
                "level": d.level,
                "filename": d.filename,
                "n_rows": d.n_rows,
                "original_rows": d.original_rows,
                "description": d.change_description,
            }
            for d in degradations
        ],
        "summary": {
            "total_degradations": len(degradations),
            "types_used": list({d.type for d in degradations}),
            "levels_used": list({d.level for d in degradations}),
        },
    }
