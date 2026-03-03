"""
Generate training dataset from the 17 cluster narrative dataset.

This script creates degraded versions of the 17 cluster dataset,
evaluates metrics on each version, and produces a training dataset
suitable for ML quality scoring.
"""

import metricate
from pathlib import Path

# Input/output paths
INPUT_CSV = "datasets/narrative_dataset_17clusters_full.csv"
OUTPUT_DIR = "training_17clusters"

# Degradation configuration - use all types and common levels
LEVELS = ["5pct", "10pct", "25pct", "50pct"]
TYPES = None  # Use all 19 degradation types

# Column specifications for 17 cluster dataset
# Use the string-encoded reduced_embedding column (10-dimensional)
LABEL_COL = "new_cluster"
EMBEDDING_COLS = ["reduced_embedding"]

print("=" * 60)
print("Training Dataset Generation - 17 Cluster Dataset")
print("=" * 60)
print(f"Input: {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}")
print(f"Levels: {LEVELS}")
print(f"Types: all 19 degradation types")
print(f"Label column: {LABEL_COL}")
print(f"Embedding columns: {EMBEDDING_COLS}")
print("=" * 60)

# Generate training data
result = metricate.generate_training_data(
    INPUT_CSV,
    OUTPUT_DIR,
    types=TYPES,
    levels=LEVELS,
    topic="narrative_17clusters",
    random_seed=42,
    force_all=False,  # Skip O(n²) metrics on large dataset
    label_col=LABEL_COL,
    embedding_cols=EMBEDDING_COLS,
)

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(result.summary())

# Display DataFrame info
df = result.to_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Show quality distribution
print(f"\nQuality distribution:")
print(df.groupby(['quality', 'quality_score']).size().reset_index(name='count'))

# Show degradation type breakdown
print(f"\nDegradation type counts:")
deg_counts = df['degradation_type'].value_counts()
print(deg_counts)

# Save summary stats
output_path = Path(OUTPUT_DIR)
stats_path = output_path / "training_stats.txt"
with open(stats_path, "w") as f:
    f.write("Training Dataset Statistics\n")
    f.write("=" * 40 + "\n\n")
    f.write(result.summary())
    f.write(f"\n\nDataFrame shape: {df.shape}\n")
    f.write(f"\nColumns:\n{chr(10).join(df.columns)}\n")
    f.write(f"\nQuality distribution:\n{df.groupby(['quality', 'quality_score']).size().to_string()}\n")
    f.write(f"\nDegradation type counts:\n{deg_counts.to_string()}\n")

print(f"\n✓ Stats saved to: {stats_path}")
print(f"✓ Training data saved to: {output_path / 'training_data.csv'}")
