"""
Clustering Degradation Toolkit

This module provides various methods to systematically degrade a well-clustered dataset
to study how clustering quality metrics respond to different types of degradation.

Usage:
    from degradation_toolkit import ClusteringDegrader

    degrader = ClusteringDegrader('narrative_dataset_model_1247315_with_reduced.csv')

    # Apply degradation
    degraded_df = degrader.random_removal(fraction=0.2)
    degraded_df = degrader.label_swap(fraction=0.1)
    # ... etc

    # Or generate a full degradation suite
    degrader.generate_degradation_suite(output_dir='degraded_datasets')
"""

import ast
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


class ClusteringDegrader:
    """
    A toolkit for systematically degrading clustered datasets to study
    how clustering quality metrics respond to various types of degradation.
    """

    def __init__(
        self,
        csv_path: str,
        cluster_col: str = "cluster_id",
        embedding_col: str = "reduced_embedding",
        random_seed: int = 42,
    ):
        """
        Initialize the degrader with a dataset.

        Args:
            csv_path: Path to the CSV file
            cluster_col: Name of the cluster column
            embedding_col: Name of the embedding column to use for distance calculations
            random_seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.cluster_col = cluster_col
        self.embedding_col = embedding_col
        self.random_seed = random_seed

        # Load data
        self.df_original = pd.read_csv(csv_path)
        self._parse_embeddings()

        # Compute cluster statistics
        self._compute_cluster_stats()

        np.random.seed(random_seed)

    def _parse_embeddings(self):
        """Parse embedding strings into numpy arrays."""

        def parse_embedding(emb_str):
            if isinstance(emb_str, str):
                return np.array(ast.literal_eval(emb_str))
            return emb_str

        self.df_original["_parsed_embedding"] = self.df_original[self.embedding_col].apply(
            parse_embedding
        )

    def _compute_cluster_stats(self):
        """Compute statistics for each cluster (centroid, spread, etc.)."""
        self.cluster_stats = {}

        # Exclude noise cluster (-1) from statistics
        valid_clusters = self.df_original[self.df_original[self.cluster_col] != -1]

        for cluster_id in valid_clusters[self.cluster_col].unique():
            cluster_mask = valid_clusters[self.cluster_col] == cluster_id
            cluster_embeddings = np.stack(
                valid_clusters.loc[cluster_mask, "_parsed_embedding"].values
            )

            centroid = cluster_embeddings.mean(axis=0)

            # Calculate intra-cluster distances (spread/tightness)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            self.cluster_stats[cluster_id] = {
                "centroid": centroid,
                "mean_distance": distances.mean(),
                "std_distance": distances.std(),
                "max_distance": distances.max(),
                "size": len(cluster_embeddings),
                "tightness_score": 1 / (1 + distances.mean()),  # Higher = tighter
            }

    def _get_df_copy(self) -> pd.DataFrame:
        """Get a fresh copy of the original dataframe."""
        return self.df_original.copy()

    # ==================== DEGRADATION METHODS ====================

    def random_removal(self, fraction: float = 0.1) -> pd.DataFrame:
        """
        Remove a random fraction of posts from the dataset.

        Args:
            fraction: Fraction of posts to remove (0.0 to 1.0)

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()
        n_remove = int(len(df) * fraction)
        indices_to_drop = np.random.choice(df.index, size=n_remove, replace=False)
        return df.drop(indices_to_drop).reset_index(drop=True)

    def remove_tight_clusters(
        self, n_clusters: int = 1, criteria: Literal["tightest", "smallest", "largest"] = "tightest"
    ) -> pd.DataFrame:
        """
        Remove entire clusters based on specified criteria.

        Args:
            n_clusters: Number of clusters to remove
            criteria: 'tightest' (most cohesive), 'smallest', or 'largest'

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()

        # Sort clusters by criteria
        if criteria == "tightest":
            sorted_clusters = sorted(
                self.cluster_stats.keys(),
                key=lambda x: self.cluster_stats[x]["tightness_score"],
                reverse=True,
            )
        elif criteria == "smallest":
            sorted_clusters = sorted(
                self.cluster_stats.keys(), key=lambda x: self.cluster_stats[x]["size"]
            )
        elif criteria == "largest":
            sorted_clusters = sorted(
                self.cluster_stats.keys(), key=lambda x: self.cluster_stats[x]["size"], reverse=True
            )
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

        clusters_to_remove = sorted_clusters[:n_clusters]
        print(f"Removing clusters: {clusters_to_remove}")

        return df[~df[self.cluster_col].isin(clusters_to_remove)].reset_index(drop=True)

    def label_swap(
        self,
        fraction: float = 0.1,
        swap_type: Literal["random", "neighboring", "distant"] = "random",
    ) -> pd.DataFrame:
        """
        Swap cluster labels of a fraction of posts.

        Args:
            fraction: Fraction of posts to swap labels for
            swap_type: 'random' (any cluster), 'neighboring' (nearby in embedding space),
                      'distant' (far in embedding space)

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()

        # Only consider non-noise points
        valid_mask = df[self.cluster_col] != -1
        valid_indices = df[valid_mask].index.tolist()

        n_swap = int(len(valid_indices) * fraction)
        swap_indices = np.random.choice(valid_indices, size=n_swap, replace=False)

        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]

        for idx in swap_indices:
            current_cluster = df.loc[idx, self.cluster_col]

            if swap_type == "random":
                # Pick any other cluster randomly
                other_clusters = [c for c in all_clusters if c != current_cluster]
                new_cluster = np.random.choice(other_clusters)

            elif swap_type == "neighboring":
                # Pick the nearest cluster by centroid distance
                current_centroid = self.cluster_stats[current_cluster]["centroid"]
                distances = {}
                for c in all_clusters:
                    if c != current_cluster:
                        distances[c] = np.linalg.norm(
                            current_centroid - self.cluster_stats[c]["centroid"]
                        )
                new_cluster = min(distances, key=distances.get)

            elif swap_type == "distant":
                # Pick the farthest cluster by centroid distance
                current_centroid = self.cluster_stats[current_cluster]["centroid"]
                distances = {}
                for c in all_clusters:
                    if c != current_cluster:
                        distances[c] = np.linalg.norm(
                            current_centroid - self.cluster_stats[c]["centroid"]
                        )
                new_cluster = max(distances, key=distances.get)

            df.loc[idx, self.cluster_col] = new_cluster

        return df

    def merge_clusters(
        self, n_merges: int = 1, merge_type: Literal["random", "nearest", "farthest"] = "nearest"
    ) -> pd.DataFrame:
        """
        Merge pairs of clusters into one (simulates over-clustering correction or errors).

        Args:
            n_merges: Number of cluster pairs to merge
            merge_type: 'random', 'nearest' (by centroid), 'farthest' (by centroid)

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()
        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]

        for _ in range(n_merges):
            if len(all_clusters) < 2:
                break

            if merge_type == "random":
                pair = np.random.choice(all_clusters, size=2, replace=False)

            elif merge_type == "nearest":
                min_dist = float("inf")
                pair = None
                for i, c1 in enumerate(all_clusters):
                    for c2 in all_clusters[i + 1 :]:
                        dist = np.linalg.norm(
                            self.cluster_stats[c1]["centroid"] - self.cluster_stats[c2]["centroid"]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            pair = (c1, c2)

            elif merge_type == "farthest":
                max_dist = 0
                pair = None
                for i, c1 in enumerate(all_clusters):
                    for c2 in all_clusters[i + 1 :]:
                        dist = np.linalg.norm(
                            self.cluster_stats[c1]["centroid"] - self.cluster_stats[c2]["centroid"]
                        )
                        if dist > max_dist:
                            max_dist = dist
                            pair = (c1, c2)

            # Merge: assign all points from second cluster to first
            df.loc[df[self.cluster_col] == pair[1], self.cluster_col] = pair[0]
            all_clusters.remove(pair[1])
            print(f"Merged cluster {pair[1]} into cluster {pair[0]}")

        return df

    def split_clusters(
        self, n_splits: int = 1, split_type: Literal["random", "largest", "loosest"] = "largest"
    ) -> pd.DataFrame:
        """
        Split clusters into two random halves (simulates under-clustering).

        Args:
            n_splits: Number of clusters to split
            split_type: 'random', 'largest' (by size), 'loosest' (by spread)

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()
        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]
        max_cluster_id = max(all_clusters)

        if split_type == "random":
            clusters_to_split = np.random.choice(
                all_clusters, size=min(n_splits, len(all_clusters)), replace=False
            )
        elif split_type == "largest":
            clusters_to_split = sorted(
                all_clusters, key=lambda x: self.cluster_stats[x]["size"], reverse=True
            )[:n_splits]
        elif split_type == "loosest":
            clusters_to_split = sorted(
                all_clusters, key=lambda x: self.cluster_stats[x]["tightness_score"]
            )[:n_splits]

        for cluster_id in clusters_to_split:
            cluster_indices = df[df[self.cluster_col] == cluster_id].index.tolist()

            # Split randomly in half
            half = len(cluster_indices) // 2
            np.random.shuffle(cluster_indices)
            indices_to_reassign = cluster_indices[:half]

            # Assign new cluster ID
            max_cluster_id += 1
            df.loc[indices_to_reassign, self.cluster_col] = max_cluster_id
            print(f"Split cluster {cluster_id} -> new cluster {max_cluster_id}")

        return df

    def boundary_reassignment(self, fraction: float = 0.2) -> pd.DataFrame:
        """
        Reassign boundary points (those furthest from their cluster centroid) to wrong clusters.

        Args:
            fraction: Fraction of each cluster's boundary points to reassign

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()
        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]

        for cluster_id in all_clusters:
            cluster_mask = df[self.cluster_col] == cluster_id
            cluster_indices = df[cluster_mask].index.tolist()

            if len(cluster_indices) < 2:
                continue

            # Calculate distances to centroid
            centroid = self.cluster_stats[cluster_id]["centroid"]
            distances = []
            for idx in cluster_indices:
                emb = df.loc[idx, "_parsed_embedding"]
                distances.append((idx, np.linalg.norm(emb - centroid)))

            # Sort by distance (furthest first)
            distances.sort(key=lambda x: x[1], reverse=True)

            # Reassign the top fraction
            n_reassign = max(1, int(len(distances) * fraction))
            boundary_indices = [d[0] for d in distances[:n_reassign]]

            # Reassign to random other cluster
            other_clusters = [c for c in all_clusters if c != cluster_id]
            for idx in boundary_indices:
                df.loc[idx, self.cluster_col] = np.random.choice(other_clusters)

        return df

    def add_noise_points(self, n_noise: int = 100) -> pd.DataFrame:
        """
        Add synthetic noise points with random cluster assignments.
        Creates new points by perturbing existing embeddings.

        Args:
            n_noise: Number of noise points to add

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()

        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]

        # Sample random points and perturb them
        sample_indices = np.random.choice(df.index, size=n_noise, replace=True)

        new_rows = []
        for i, idx in enumerate(sample_indices):
            row = df.loc[idx].copy()

            # Perturb embedding significantly
            original_emb = row["_parsed_embedding"]
            noise = np.random.randn(*original_emb.shape) * np.std(original_emb) * 2
            new_emb = original_emb + noise

            # Update row
            row["post_id"] = f"synthetic_noise_{i}"
            row["_parsed_embedding"] = new_emb
            row[self.embedding_col] = str(new_emb.tolist())
            row[self.cluster_col] = np.random.choice(all_clusters)  # Random cluster

            new_rows.append(row)

        noise_df = pd.DataFrame(new_rows)
        return pd.concat([df, noise_df], ignore_index=True)

    def embedding_perturbation(self, noise_scale: float = 0.1) -> pd.DataFrame:
        """
        Add Gaussian noise to all embeddings (keeps labels unchanged).
        Tests if metrics correctly use embedding information.

        Args:
            noise_scale: Scale of noise relative to embedding std

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()

        # Calculate global embedding std
        all_embeddings = np.stack(df["_parsed_embedding"].values)
        global_std = np.std(all_embeddings)

        new_embeddings = []
        for emb in df["_parsed_embedding"]:
            noise = np.random.randn(*emb.shape) * global_std * noise_scale
            new_embeddings.append(emb + noise)

        df["_parsed_embedding"] = new_embeddings
        df[self.embedding_col] = [str(emb.tolist()) for emb in new_embeddings]

        return df

    def remove_core_points(self, fraction: float = 0.2) -> pd.DataFrame:
        """
        Remove points closest to cluster centroids (destroys cluster identity).

        Args:
            fraction: Fraction of each cluster's core points to remove

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()
        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]

        indices_to_drop = []

        for cluster_id in all_clusters:
            cluster_mask = df[self.cluster_col] == cluster_id
            cluster_indices = df[cluster_mask].index.tolist()

            if len(cluster_indices) < 2:
                continue

            # Calculate distances to centroid
            centroid = self.cluster_stats[cluster_id]["centroid"]
            distances = []
            for idx in cluster_indices:
                emb = df.loc[idx, "_parsed_embedding"]
                distances.append((idx, np.linalg.norm(emb - centroid)))

            # Sort by distance (closest first)
            distances.sort(key=lambda x: x[1])

            # Remove the top fraction (closest to centroid)
            n_remove = max(1, int(len(distances) * fraction))
            indices_to_drop.extend([d[0] for d in distances[:n_remove]])

        return df.drop(indices_to_drop).reset_index(drop=True)

    def centroid_displacement(self, displacement_scale: float = 0.5) -> pd.DataFrame:
        """
        Move points towards the wrong cluster centroids.

        Args:
            displacement_scale: How much to move towards wrong centroid (0=none, 1=fully)

        Returns:
            Degraded DataFrame
        """
        df = self._get_df_copy()
        all_clusters = [c for c in df[self.cluster_col].unique() if c != -1]

        new_embeddings = []
        new_embedding_strs = []

        for idx in df.index:
            if df.loc[idx, self.cluster_col] == -1:
                new_embeddings.append(df.loc[idx, "_parsed_embedding"])
                new_embedding_strs.append(df.loc[idx, self.embedding_col])
                continue

            current_cluster = df.loc[idx, self.cluster_col]
            current_emb = df.loc[idx, "_parsed_embedding"]

            # Pick a random wrong centroid
            other_clusters = [c for c in all_clusters if c != current_cluster]
            wrong_cluster = np.random.choice(other_clusters)
            wrong_centroid = self.cluster_stats[wrong_cluster]["centroid"]

            # Move towards wrong centroid
            new_emb = current_emb + displacement_scale * (wrong_centroid - current_emb)

            new_embeddings.append(new_emb)
            new_embedding_strs.append(str(new_emb.tolist()))

        df["_parsed_embedding"] = new_embeddings
        df[self.embedding_col] = new_embedding_strs

        return df

    # ==================== UTILITY METHODS ====================

    def save_degraded(self, df: pd.DataFrame, output_path: str, format: str = "csv"):
        """Save degraded dataframe, dropping internal columns.
        
        Args:
            df: DataFrame to save
            output_path: Path to save to (extension will be adjusted based on format)
            format: 'csv' or 'parquet' (parquet is ~5-10x smaller)
        """
        df_to_save = df.drop(columns=["_parsed_embedding"], errors="ignore")
        
        output_path = Path(output_path)
        if format == "parquet":
            # Change extension to .parquet if needed
            if output_path.suffix == ".csv":
                output_path = output_path.with_suffix(".parquet")
            # Use fastparquet engine (pyarrow 23 has pandas compatibility issues)
            df_to_save.to_parquet(output_path, index=False, compression="snappy", engine="fastparquet")
        else:
            df_to_save.to_csv(output_path, index=False)
        print(f"Saved degraded dataset to: {output_path}")

    def generate_degradation_suite(
        self, output_dir: str = "degraded_datasets", levels: list[float] = None, format: str = "parquet"
    ):
        """
        Generate a comprehensive suite of degraded datasets at various levels.

        Args:
            output_dir: Directory to save degraded datasets
            levels: Degradation levels to apply (fractions)
            format: 'csv' or 'parquet' (parquet is ~5-10x smaller, recommended)
        """
        if levels is None:
            levels = [0.05, 0.1, 0.25, 0.5]
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        ext = ".parquet" if format == "parquet" else ".csv"

        degradations = []

        # 1. Random removal at various levels
        for level in levels:
            df = self.random_removal(fraction=level)
            filename = f"random_removal_{int(level * 100)}pct{ext}"
            self.save_degraded(df, output_path / filename, format=format)
            degradations.append(
                {"type": "random_removal", "level": level, "filename": filename, "n_posts": len(df)}
            )

        # 2. Label swapping at various levels
        for level in levels:
            for swap_type in ["random", "neighboring", "distant"]:
                df = self.label_swap(fraction=level, swap_type=swap_type)
                filename = f"label_swap_{swap_type}_{int(level * 100)}pct{ext}"
                self.save_degraded(df, output_path / filename, format=format)
                degradations.append(
                    {
                        "type": f"label_swap_{swap_type}",
                        "level": level,
                        "filename": filename,
                        "n_posts": len(df),
                    }
                )

        # 3. Remove tight clusters
        for n in [1, 2, 3, 5]:
            for criteria in ["tightest", "smallest", "largest"]:
                df = self.remove_tight_clusters(n_clusters=n, criteria=criteria)
                filename = f"remove_{criteria}_{n}_clusters{ext}"
                self.save_degraded(df, output_path / filename, format=format)
                degradations.append(
                    {
                        "type": f"remove_{criteria}_clusters",
                        "level": n,
                        "filename": filename,
                        "n_posts": len(df),
                    }
                )

        # 4. Merge clusters
        for n in [1, 2, 3, 5]:
            for merge_type in ["nearest", "farthest", "random"]:
                np.random.seed(self.random_seed)  # Reset for reproducibility
                df = self.merge_clusters(n_merges=n, merge_type=merge_type)
                filename = f"merge_{merge_type}_{n}_pairs{ext}"
                self.save_degraded(df, output_path / filename, format=format)
                degradations.append(
                    {
                        "type": f"merge_{merge_type}",
                        "level": n,
                        "filename": filename,
                        "n_posts": len(df),
                    }
                )

        # 5. Split clusters
        for n in [1, 2, 3, 5]:
            for split_type in ["largest", "loosest", "random"]:
                np.random.seed(self.random_seed)
                df = self.split_clusters(n_splits=n, split_type=split_type)
                filename = f"split_{split_type}_{n}_clusters{ext}"
                self.save_degraded(df, output_path / filename, format=format)
                degradations.append(
                    {
                        "type": f"split_{split_type}",
                        "level": n,
                        "filename": filename,
                        "n_posts": len(df),
                    }
                )

        # 6. Boundary reassignment
        for level in levels:
            df = self.boundary_reassignment(fraction=level)
            filename = f"boundary_reassign_{int(level * 100)}pct{ext}"
            self.save_degraded(df, output_path / filename, format=format)
            degradations.append(
                {
                    "type": "boundary_reassignment",
                    "level": level,
                    "filename": filename,
                    "n_posts": len(df),
                }
            )

        # 7. Core removal
        for level in levels:
            df = self.remove_core_points(fraction=level)
            filename = f"core_removal_{int(level * 100)}pct{ext}"
            self.save_degraded(df, output_path / filename, format=format)
            degradations.append(
                {"type": "core_removal", "level": level, "filename": filename, "n_posts": len(df)}
            )

        # 8. Embedding perturbation
        for level in [0.1, 0.25, 0.5, 1.0]:
            df = self.embedding_perturbation(noise_scale=level)
            filename = f"embedding_perturb_{int(level * 100)}pct{ext}"
            self.save_degraded(df, output_path / filename, format=format)
            degradations.append(
                {
                    "type": "embedding_perturbation",
                    "level": level,
                    "filename": filename,
                    "n_posts": len(df),
                }
            )

        # 9. Add noise points
        for n in [100, 500, 1000, 2000]:
            df = self.add_noise_points(n_noise=n)
            filename = f"noise_injection_{n}_points{ext}"
            self.save_degraded(df, output_path / filename, format=format)
            degradations.append(
                {"type": "noise_injection", "level": n, "filename": filename, "n_posts": len(df)}
            )

        # 10. Centroid displacement
        for level in [0.1, 0.25, 0.5, 0.75]:
            df = self.centroid_displacement(displacement_scale=level)
            filename = f"centroid_displacement_{int(level * 100)}pct{ext}"
            self.save_degraded(df, output_path / filename, format=format)
            degradations.append(
                {
                    "type": "centroid_displacement",
                    "level": level,
                    "filename": filename,
                    "n_posts": len(df),
                }
            )

        # Save manifest
        manifest = pd.DataFrame(degradations)
        manifest.to_csv(output_path / "manifest.csv", index=False)
        print(f"\nGenerated {len(degradations)} degraded datasets")
        print(f"Manifest saved to: {output_path / 'manifest.csv'}")

        return manifest


# Quick test / demo
if __name__ == "__main__":
    import sys

    csv_path = "narrative_dataset_model_1247315_with_reduced.csv"

    if not Path(csv_path).exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("CLUSTERING DEGRADATION TOOLKIT - DEMO")
    print("=" * 60)

    degrader = ClusteringDegrader(csv_path)

    print(f"\nOriginal dataset: {len(degrader.df_original)} posts")
    print(f"Clusters: {degrader.df_original['cluster_id'].nunique()}")

    print("\n" + "=" * 60)
    print("CLUSTER STATISTICS")
    print("=" * 60)

    stats_df = pd.DataFrame(
        [
            {
                "cluster_id": cid,
                "size": stats["size"],
                "mean_dist": round(stats["mean_distance"], 4),
                "tightness": round(stats["tightness_score"], 4),
            }
            for cid, stats in degrader.cluster_stats.items()
        ]
    ).sort_values("tightness", ascending=False)

    print("\nTop 10 tightest clusters:")
    print(stats_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("SAMPLE DEGRADATIONS")
    print("=" * 60)

    # Demo each degradation type
    print("\n1. Random removal (20%):")
    df = degrader.random_removal(0.2)
    print(f"   Posts: {len(degrader.df_original)} -> {len(df)}")

    print("\n2. Label swap (10%, random):")
    df = degrader.label_swap(0.1, "random")
    print(f"   Posts: {len(df)} (labels changed)")

    print("\n3. Remove tightest cluster:")
    df = degrader.remove_tight_clusters(1, "tightest")
    print(f"   Posts: {len(degrader.df_original)} -> {len(df)}")

    print("\n4. Merge nearest clusters:")
    df = degrader.merge_clusters(1, "nearest")
    print(
        f"   Clusters: {degrader.df_original['cluster_id'].nunique()} -> {df['cluster_id'].nunique()}"
    )

    print("\n5. Split largest cluster:")
    df = degrader.split_clusters(1, "largest")
    print(
        f"   Clusters: {degrader.df_original['cluster_id'].nunique()} -> {df['cluster_id'].nunique()}"
    )

    print("\n" + "=" * 60)
    print("To generate full degradation suite, run:")
    print("  degrader.generate_degradation_suite('degraded_datasets')")
    print("=" * 60)
