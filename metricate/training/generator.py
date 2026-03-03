"""
Training dataset generation from clustering CSVs.

This module provides the core functions to generate labeled training datasets
by degrading good clusterings and calculating metrics.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

import metricate
from metricate.training.normalize import (
    get_internal_metric_names,
    normalize_metrics,
)
from metricate.training.result import TrainingDataResult

logger = logging.getLogger(__name__)

# Quality score mapping for degradation levels
LEVEL_TO_QUALITY_SCORE = {
    None: 1.00,  # Original
    "5pct": 0.95,
    "10pct": 0.90,
    "25pct": 0.75,
    "50pct": 0.50,
    "75pct": 0.25,
    "100pct": 0.00,
}

# External metrics to always exclude
EXTERNAL_METRICS = ["ARI", "Van Dongen", "VI", "Omega"]


def _extract_topic_from_filename(filename: str) -> str:
    """Extract topic from filename.

    Attempts to extract a meaningful topic from the filename.
    Falls back to the filename stem if no pattern matches.

    Args:
        filename: The filename (without path).

    Returns:
        Extracted topic string.
    """
    stem = Path(filename).stem

    # Common patterns: topic_clustering.csv, topic_dataset.csv
    for suffix in ["_clustering", "_dataset", "_clusters", "_data"]:
        if suffix in stem.lower():
            idx = stem.lower().find(suffix)
            return stem[:idx] if idx > 0 else stem

    return stem


def _evaluation_to_record(
    eval_result: Any,
    clustering_name: str,
    topic: str,
    quality: int,
    quality_score: float,
    degradation_type: str | None,
    degradation_level: str | None,
) -> dict:
    """Convert an EvaluationResult to a record dict.

    Args:
        eval_result: The EvaluationResult from metricate.evaluate().
        clustering_name: Source filename (without extension).
        topic: Topic/category of the clustering.
        quality: Binary label (1=original, 0=degraded).
        quality_score: Continuous quality score (0-1).
        degradation_type: Type of degradation applied (None for originals).
        degradation_level: Intensity level (None for originals).

    Returns:
        Record dictionary with all fields.
    """
    record = {
        "clustering_name": clustering_name,
        "topic": topic,
        "n_clusters": eval_result.metadata.get("n_clusters", 0),
        "n_samples": eval_result.metadata.get("n_samples", 0),
        "quality": quality,
        "quality_score": quality_score,
        "degradation_type": degradation_type,
        "degradation_level": degradation_level,
    }

    # Add metric values
    computed_count = 0
    failed_metrics = []

    for metric_value in eval_result.metrics:
        metric_name = metric_value.metric
        value = metric_value.value

        record[metric_name] = value

        if value is not None and metric_value.computed:
            computed_count += 1
        elif not metric_value.computed or value is None:
            failed_metrics.append(metric_name)

    record["metrics_computed"] = computed_count
    record["metrics_failed"] = ",".join(failed_metrics) if failed_metrics else ""

    return record


def generate_training_data(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    topic: str | None = None,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> TrainingDataResult:
    """Generate a training dataset from a single clustering CSV.

    Args:
        csv_path: Path to input clustering CSV.
        output_dir: Directory for degraded CSVs and outputs.
        types: Degradation types to apply (None = all 19).
        levels: Degradation levels (None = ["5pct", "10pct", "25pct", "50pct"]).
        exclude: Metrics to exclude from calculation.
        force_all: Compute O(n²) metrics on large datasets.
        topic: Manual topic assignment (None = extract from filename).
        random_seed: Seed for reproducibility.
        label_col: Cluster label column (auto-detected if None).
        embedding_cols: Embedding columns (auto-detected if None).

    Returns:
        TrainingDataResult with records for original + all degraded versions.

    Raises:
        FileNotFoundError: If csv_path doesn't exist.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    # Determine exclusions (always exclude external metrics)
    all_exclude = set(EXTERNAL_METRICS)
    if exclude:
        all_exclude.update(exclude)
    exclude_list = list(all_exclude)

    # Extract topic and name
    clustering_name = csv_path.stem
    if topic is None:
        topic = _extract_topic_from_filename(csv_path.name)

    result = TrainingDataResult(
        metadata={
            "source_file": str(csv_path),
            "output_dir": str(output_dir),
            "types": types,
            "levels": levels,
            "random_seed": random_seed,
        }
    )

    records = []

    # Step 1: Evaluate original clustering
    print(f"[1/?] Evaluating original: {csv_path.name}")
    logger.info(f"Evaluating original clustering: {csv_path}")
    try:
        eval_result = metricate.evaluate(
            csv_path,
            exclude=exclude_list,
            force_all=force_all,
            label_col=label_col,
            embedding_cols=embedding_cols,
        )

        original_record = _evaluation_to_record(
            eval_result=eval_result,
            clustering_name=clustering_name,
            topic=topic,
            quality=1,
            quality_score=1.0,
            degradation_type=None,
            degradation_level=None,
        )
        records.append(original_record)

    except Exception as e:
        result.errors.append(f"Failed to evaluate original: {e}")
        logger.error(f"Failed to evaluate original {csv_path}: {e}")
        result.records = records
        return result

    # Step 2: Generate degraded versions
    print(f"Generating degraded versions...")
    logger.info(f"Generating degraded versions in {output_dir}")
    try:
        degrade_result = metricate.degrade(
            str(csv_path),  # Convert to string for JSON serialization
            str(output_dir),
            levels=levels,
            types=types,
            visualize=False,  # Skip visualizations for training data
            label_col=label_col,
            embedding_cols=embedding_cols,
            random_seed=random_seed,
        )
    except Exception as e:
        result.errors.append(f"Failed to generate degradations: {e}")
        logger.error(f"Failed to degrade {csv_path}: {e}")
        result.records = records
        return result

    # Step 3: Evaluate each degraded version
    total = len(degrade_result.degradations) + 1  # +1 for original
    for i, degradation in enumerate(degrade_result.degradations, start=2):
        degraded_path = degradation.filepath
        deg_type = degradation.type
        deg_level = degradation.level

        print(f"[{i}/{total}] Evaluating: {deg_type} @ {deg_level}")
        logger.info(f"Evaluating degraded: {deg_type} @ {deg_level}")

        try:
            eval_result = metricate.evaluate(
                degraded_path,
                exclude=exclude_list,
                force_all=force_all,
                label_col=label_col,
                embedding_cols=embedding_cols,
            )

            quality_score = LEVEL_TO_QUALITY_SCORE.get(deg_level, 0.5)

            degraded_record = _evaluation_to_record(
                eval_result=eval_result,
                clustering_name=clustering_name,
                topic=topic,
                quality=0,
                quality_score=quality_score,
                degradation_type=deg_type,
                degradation_level=deg_level,
            )
            records.append(degraded_record)

        except Exception as e:
            result.warnings.append(f"Failed to evaluate {deg_type}@{deg_level}: {e}")
            logger.warning(f"Failed to evaluate {degraded_path}: {e}")

    # Step 4: Apply percentile normalization
    print(f"Normalizing metrics across {len(records)} records...")
    if records:
        df = pd.DataFrame(records)
        metric_cols = get_internal_metric_names()
        # Only normalize columns that exist in the dataframe
        metric_cols = [c for c in metric_cols if c in df.columns]
        df = normalize_metrics(df, metric_cols)
        records = df.to_dict("records")

    result.records = records
    result.metadata["n_degradations_attempted"] = len(degrade_result.degradations)
    result.metadata["n_degradations_successful"] = len(records) - 1  # Minus original

    # Auto-save training data to output directory
    training_csv_path = output_dir / "training_data.csv"
    result.to_csv(training_csv_path)
    result.metadata["training_data_path"] = str(training_csv_path)

    print(f"✓ Done! Generated {len(records)} records ({result.metadata['n_degradations_successful']} degraded)")
    print(f"✓ Saved to: {training_csv_path}")
    return result


def generate_training_data_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    topic_mapping: dict[str, str] | None = None,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> TrainingDataResult:
    """Generate training data from all CSVs in a directory.

    Args:
        input_dir: Directory containing clustering CSVs.
        output_dir: Directory for degraded CSVs and outputs.
        topic_mapping: Dict mapping filename to topic.
        types: Degradation types to apply (None = all 19).
        levels: Degradation levels (None = ["5pct", "10pct", "25pct", "50pct"]).
        exclude: Metrics to exclude from calculation.
        force_all: Compute O(n²) metrics on large datasets.
        random_seed: Seed for reproducibility.
        label_col: Cluster label column (auto-detected if None).
        embedding_cols: Embedding columns (auto-detected if None).

    Returns:
        TrainingDataResult with combined records from all files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    topic_mapping = topic_mapping or {}

    result = TrainingDataResult(
        metadata={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "n_input_files": len(csv_files),
            "types": types,
            "levels": levels,
            "random_seed": random_seed,
        }
    )

    all_records = []

    for i, csv_path in enumerate(csv_files):
        logger.info(f"Processing file {i + 1}/{len(csv_files)}: {csv_path.name}")

        # Get topic from mapping or extract from filename
        topic = topic_mapping.get(csv_path.name)

        # Create subdirectory for this file's degradations
        file_output_dir = output_dir / csv_path.stem

        try:
            file_result = generate_training_data(
                csv_path=csv_path,
                output_dir=file_output_dir,
                types=types,
                levels=levels,
                exclude=exclude,
                force_all=force_all,
                topic=topic,
                random_seed=random_seed,
                label_col=label_col,
                embedding_cols=embedding_cols,
            )

            # Collect records (without normalization - we'll renormalize at the end)
            all_records.extend(file_result.records)
            result.warnings.extend(file_result.warnings)
            result.errors.extend(file_result.errors)

        except Exception as e:
            result.errors.append(f"Failed to process {csv_path.name}: {e}")
            logger.error(f"Failed to process {csv_path}: {e}")

    # Re-apply percentile normalization across the entire combined dataset
    if all_records:
        # First, remove existing _norm columns to avoid duplication
        df = pd.DataFrame(all_records)
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        df = df.drop(columns=norm_cols, errors="ignore")

        # Apply normalization across full dataset
        metric_cols = get_internal_metric_names()
        metric_cols = [c for c in metric_cols if c in df.columns]
        df = normalize_metrics(df, metric_cols)
        all_records = df.to_dict("records")

    result.records = all_records
    result.metadata["n_files_processed"] = (
        len(csv_files) - len([e for e in result.errors if "Failed to process" in e])
    )

    return result
