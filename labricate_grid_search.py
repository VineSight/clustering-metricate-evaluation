"""
Grid Search using Labricate Framework

This script runs a grid search over:
- UMAP dimensions (n_components): 10 to 50
- K-Means clusters (n_clusters): 5 to 30

Dataset: narrative_dataset_17clusters_full.csv
"""

import matplotlib.pyplot as plt

from metricate.labricate import Experiment
from metricate.labricate.output.visualization import plot_heatmap

# Configuration
DATASET_PATH = "/Users/alonneduva/Desktop/MindINT/Research/research/degrading-clustering-dataset/datasets/narrative_dataset_17clusters_full.csv"
OUTPUT_DIR = "./experiments/grid_search_umap_kmeans"
WEIGHTS_PATH = "/Users/alonneduva/Desktop/MindINT/Research/research/degrading-clustering-dataset/training_merged/learned_weights_pairwise.json"

# Define parameter ranges
UMAP_DIMS = list(range(10, 31, 1))  # 10, 11, 12, ..., 30 (21 values)
K_CLUSTERS = list(range(10, 61, 1))   # 10, 11, 12, ..., 60 (51 values)

print(f"Loading dataset from: {DATASET_PATH}")
print(f"Using learned weights from: {WEIGHTS_PATH}")
print(f"UMAP dimensions to test: {UMAP_DIMS}")
print(f"K-Means clusters to test: {K_CLUSTERS}")
print(f"Total combinations: {len(UMAP_DIMS) * len(K_CLUSTERS)}")

# Base configuration for K-Means clustering
config = {
    "random_seed": 42,
    "umap": {
        "n_neighbors": 15,
        "n_components": 10,  # Will be varied in grid search
        "min_dist": 0.0,
        "metric": "cosine",
    },
    "clustering_algorithm": "kmeans",  # Using K-Means
    "kmeans": {
        "n_clusters": 10,  # Will be varied in grid search
    },
    "enable_topic_representation": False,
    "calculate_probabilities": False,
}

print("\n" + "="*60)
print("Starting Grid Search Experiment")
print("="*60)

# Create experiment - labricate automatically loads embeddings from CSV
# It looks for dim_* or embedding_* columns, or falls back to numeric columns
exp = Experiment(
    embeddings=DATASET_PATH,  # Pass CSV path directly
    config=config,
    name="umap_kmeans_k_saturation_grid_search",
    output_dir=OUTPUT_DIR,
    output_format="csv",  # Save as both JSON and CSV
    weights=WEIGHTS_PATH,  # Use learned weights for compound scoring
)

# Run grid search
result = exp.run_grid(
    params={
        "umap.n_components": UMAP_DIMS,
        "kmeans.n_clusters": K_CLUSTERS,
    },
    n_workers=4,  # Use parallel workers for speed
    error_handling="continue",  # Continue even if some runs fail
    verbose=True,
)

# Display results
print("\n" + "="*60)
print("Grid Search Results")
print("="*60)

# Convert to DataFrame
results_df = result.to_dataframe()

# Display key metrics
display_cols = [
    "umap.n_components",
    "kmeans.n_clusters",
    "n_clusters",
    "compound_score",  # Weighted score from learned weights
    "Silhouette",
    "Davies-Bouldin",
    "Calinski-Harabasz",
]

# Add supervised metrics if available in results
if "ARI" in results_df.columns:
    display_cols.extend(["ARI", "NMI", "V-Measure"])

# Filter to available columns
display_cols = [c for c in display_cols if c in results_df.columns]
print("\nResults Summary:")
print(results_df[display_cols].to_string())

# Find best configurations
print("\n" + "="*60)
print("Best Configurations")
print("="*60)

# Best by Compound Score (from learned weights - this is the main metric!)
if result.best_run is not None:
    print("\n★ Best by Compound Score (Learned Weights):")
    print(f"  Run ID: {result.best_run.run_id}")
    print(f"  UMAP n_components: {result.best_run.param_values.get('umap.n_components')}")
    print(f"  K-Means n_clusters: {result.best_run.param_values.get('kmeans.n_clusters')}")
    print(f"  Compound Score: {result.best_run.score:.4f}")

# Best by Silhouette (higher is better)
if "Silhouette" in results_df.columns:
    best_silhouette = result.get_best_run("Silhouette")
    print("\nBest by Silhouette Score:")
    print(f"  UMAP n_components: {best_silhouette.param_values.get('umap.n_components')}")
    print(f"  K-Means n_clusters: {best_silhouette.param_values.get('kmeans.n_clusters')}")

# Best by Davies-Bouldin (lower is better)
if "Davies-Bouldin" in results_df.columns:
    best_db = result.get_best_run("Davies-Bouldin")
    print("\nBest by Davies-Bouldin Index:")
    print(f"  UMAP n_components: {best_db.param_values.get('umap.n_components')}")
    print(f"  K-Means n_clusters: {best_db.param_values.get('kmeans.n_clusters')}")

# Best by ARI if available (higher is better)
if "ARI" in results_df.columns:
    best_ari = result.get_best_run("ARI")
    print("\nBest by Adjusted Rand Index (ARI):")
    print(f"  UMAP n_components: {best_ari.param_values.get('umap.n_components')}")
    print(f"  K-Means n_clusters: {best_ari.param_values.get('kmeans.n_clusters')}")

print("\n" + "="*60)
print("Experiment Summary")
print("="*60)
print(result.summary)

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "="*60)
print("Generating Visualizations")
print("="*60)

# Create heatmaps for key metrics using labricate's plot_heatmap
metrics_to_plot = [
    ("compound_score", "seismic"),      # Higher is better (weighted score)
    ("Silhouette", "seismic"),           # Higher is better
    ("Davies-Bouldin", "seismic_r"),     # Lower is better (reversed colormap)
    ("Calinski-Harabasz", "seismic"),    # Higher is better
    ("Negentropy", "seismic_r"),             # lower is better
    ("Log_Det_Ratio", "seismic"),          # higher is better
    ("Trace_WiB", "seismic"),             # Higher is better
]

for metric, cmap in metrics_to_plot:
    try:
        output_path = f"{OUTPUT_DIR}/{metric.lower().replace('-', '_')}_heatmap.png"
        fig = plot_heatmap(
            result,
            metric=metric,
            param_x="umap.n_components",
            param_y="kmeans.n_clusters",
            output_path=output_path,
            title=f"{metric.replace('_', ' ').title()} by UMAP Dimensions and K-Clusters",
            cmap=cmap,
        )
        print(f"  Saved: {output_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  Could not create heatmap for {metric}: {e}")

print("\nDone!")
