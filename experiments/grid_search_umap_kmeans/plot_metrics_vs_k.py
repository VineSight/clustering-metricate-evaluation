"""Line plots of multiple metrics vs K (n_clusters) using labricate visualization style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load results
results_path = Path(__file__).parent / "results.csv"
df = pd.read_csv(results_path)

# Filter to completed runs only
df = df[df["status"] == "completed"].copy()

# Define metrics to plot (selecting some key ones for readability)
# Group 1: Normalized/bounded metrics (similar scales)
metrics_group1 = ["Silhouette", "R-squared", "Wemmert-Gancarski", "Gamma"]
# Group 2: Lower is better metrics
metrics_group2 = ["Davies-Bouldin", "Ray-Turi", "COP", "S_Dbw"]
# Group 3: compound score
metrics_group3 = ["compound_score"]

output_dir = Path(__file__).parent / "lineplots"
output_dir.mkdir(exist_ok=True)


def plot_metrics_vs_k(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    output_name: str,
    aggregate: bool = True,
    figsize: tuple[float, float] = (12, 7),
) -> plt.Figure:
    """
    Plot multiple metrics vs K.
    
    Args:
        df: DataFrame with results
        metrics: List of metric column names to plot
        title: Plot title
        output_name: Filename for saving
        aggregate: If True, average across n_components values
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if aggregate:
        # Group by K and average across n_components
        grouped = df.groupby("kmeans.n_clusters")[metrics].mean()
        k_values = grouped.index.values
        
        for metric in metrics:
            ax.plot(
                k_values, 
                grouped[metric].values, 
                marker="o", 
                linewidth=2, 
                markersize=6,
                label=metric
            )
    else:
        # Plot separate lines for each n_components value
        n_components_values = sorted(df["umap.n_components"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(n_components_values)))
        
        for metric in metrics:
            for i, n_comp in enumerate(n_components_values):
                subset = df[df["umap.n_components"] == n_comp].sort_values("kmeans.n_clusters")
                ax.plot(
                    subset["kmeans.n_clusters"],
                    subset[metric],
                    marker="o",
                    linewidth=1.5,
                    markersize=4,
                    alpha=0.7,
                    label=f"{metric} (n_comp={n_comp})"
                )
    
    ax.set_xlabel("K (Number of Clusters)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / output_name}")
    
    return fig


def plot_metrics_with_n_components_lines(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_name: str,
    figsize: tuple[float, float] = (12, 7),
) -> plt.Figure:
    """
    Plot a single metric vs K with separate lines for each n_components.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_components_values = sorted(df["umap.n_components"].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_components_values)))
    
    for i, n_comp in enumerate(n_components_values):
        subset = df[df["umap.n_components"] == n_comp].sort_values("kmeans.n_clusters")
        ax.plot(
            subset["kmeans.n_clusters"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors[i],
            label=f"n_components={n_comp}"
        )
    
    ax.set_xlabel("K (Number of Clusters)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / output_name}")
    
    return fig


if __name__ == "__main__":
    print(f"Loaded {len(df)} completed runs")
    print(f"K values: {sorted(df['kmeans.n_clusters'].unique())}")
    print(f"n_components values: {sorted(df['umap.n_components'].unique())}")
    
    # Plot 1: Quality metrics (higher is better) - aggregated
    plot_metrics_vs_k(
        df,
        metrics_group1,
        title="Quality Metrics vs K (Averaged across n_components)",
        output_name="quality_metrics_vs_k.png",
        aggregate=True,
    )
    
    # Plot 2: Lower-is-better metrics - aggregated
    plot_metrics_vs_k(
        df,
        metrics_group2,
        title="Separation Metrics vs K (Lower is Better)",
        output_name="separation_metrics_vs_k.png",
        aggregate=True,
    )
    
    # Plot 3: Compound score vs K
    plot_metrics_vs_k(
        df,
        metrics_group3,
        title="Compound Score vs K",
        output_name="compound_score_vs_k.png",
        aggregate=True,
    )
    
    # Plot 4: Silhouette with n_components breakdown
    plot_metrics_with_n_components_lines(
        df,
        metric="Silhouette",
        title="Silhouette Score vs K (by n_components)",
        output_name="silhouette_by_n_components.png",
    )
    
    # Plot 5: Compound score with n_components breakdown
    plot_metrics_with_n_components_lines(
        df,
        metric="compound_score",
        title="Compound Score vs K (by n_components)",
        output_name="compound_score_by_n_components.png",
    )
    
    # Plot 6: Davies-Bouldin with n_components breakdown
    plot_metrics_with_n_components_lines(
        df,
        metric="Davies-Bouldin",
        title="Davies-Bouldin Index vs K (by n_components)",
        output_name="davies_bouldin_by_n_components.png",
    )
    
    plt.show()
