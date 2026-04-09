### This script samples all clusters for each of the 12 clustering methods,

from __future__ import annotations

from pathlib import Path
from os import environ
import re

import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS2_DIR = r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2"


REVIEW_REASON_PRIORITY: list[tuple[str, str, set[int] | None]] = [
    ("umap_3d_kmeans", "in_set", {6, 15, 20, 21, 22}),
    ("umap_3d_hdbscan", "in_set", {0, 1, 3, 5}),
    ("tsne_3d_hdbscan", "in_set", {0, 1}),
    ("umap_20d_hdbscan", "in_set", {0, 1, 3, 4}),
    ("tsne_20d_hdbscan", "all_non_noise", None),
    ("pca_3d_hdbscan", "in_set", set(range(0, 25))),
]


def normalize_path(path_str: str) -> Path:
    """Support both Windows and WSL-style execution for D:\\... paths."""
    p = Path(path_str)
    if p.exists():
        return p

    if re.match(r"^[A-Za-z]:\\", path_str):
        drive = path_str[0].lower()
        suffix = path_str[2:].replace("\\", "/")
        wsl_candidate = Path(f"/mnt/{drive}{suffix}")
        if wsl_candidate.exists():
            return wsl_candidate

    return p


def get_embeddings2_dir() -> Path:
    env_embeddings2 = environ.get("POSIDONIA_EMBEDDINGS2_DIR")
    if env_embeddings2:
        return normalize_path(env_embeddings2)

    env_dataset = environ.get("POSIDONIA_DATASET_DIR")
    if env_dataset:
        return normalize_path(env_dataset).parent

    return normalize_path(DEFAULT_EMBEDDINGS2_DIR)


def parse_cluster(value: object) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    try:
        if "." in text:
            number = float(text)
            if not number.is_integer():
                return None
            return int(number)
        return int(text)
    except ValueError:
        return None


def matches_rule(rule: str, cluster_value: int | None, allowed: set[int] | None) -> bool:
    if cluster_value is None:
        return False

    if rule == "in_set":
        if allowed is None:
            return False
        return cluster_value in allowed

    if rule == "all_non_noise":
        return cluster_value != -1

    return False


def first_review_selection(row: pd.Series) -> tuple[str, int | None]:
    to_review_value = str(row.get("ToReview", "")).strip().lower()
    if to_review_value not in {"true", "1", "yes", "y", "t"}:
        return "", None

    for method, rule, allowed in REVIEW_REASON_PRIORITY:
        cluster_value = parse_cluster(row.get(method, pd.NA))
        if matches_rule(rule, cluster_value, allowed):
            return method, cluster_value
    return "", None


def main() -> None:
    embeddings2_dir = get_embeddings2_dir()
    diagnostics_dir = embeddings2_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Load the full diagnostics CSV
    full_csv = diagnostics_dir / "all_methods_cluster_diagnostics.csv"
    if not full_csv.exists():
        raise FileNotFoundError(f"Missing full diagnostics CSV: {full_csv}")

    df_full = pd.read_csv(str(full_csv))

    # Define the 12 method columns (exclude filename, ToReview, Prediction)
    method_columns = [
        "umap_3d_kmeans",
        "tsne_3d_kmeans",
        "pca_3d_kmeans",
        "pca_256D_kmeans",
        "umap_3d_hdbscan",
        "tsne_3d_hdbscan",
        "umap_3d_silhouette",
        "tsne_3d_silhouette",
        "pca_3d_hdbscan",
        "pca_256D_hdbscan",
        "umap_20d_hdbscan",
        "tsne_20d_hdbscan",
    ]

    # Collect indices of rows to sample (union across all method-cluster pairs)
    selected_indices = set()
    rng = np.random.default_rng(42)

    for method in method_columns:
        # Get unique clusters (excluding NaN)
        clusters = df_full[method].dropna().unique()
        
        for cluster_id in sorted(clusters):
            # Get rows with this cluster in this method
            mask = df_full[method] == cluster_id
            matching_indices = np.where(mask)[0]
            
            # Sample up to 100 rows
            n_samples = min(100, len(matching_indices))
            sampled = rng.choice(matching_indices, size=n_samples, replace=False)
            selected_indices.update(sampled.tolist())

    # Create output dataframe with selected rows
    selected_list = sorted(list(selected_indices))
    df_sample = df_full.iloc[selected_list].reset_index(drop=True)

    # Convert cluster columns to integers (nullable Int64 preserves NaN)
    for method in method_columns:
        df_sample[method] = df_sample[method].astype('Int64')

    # Keep review selection columns as final columns for downstream scripts.
    if "ReviewReason" in df_sample.columns:
        df_sample = df_sample.drop(columns=["ReviewReason"])
    if "ReviewCluster" in df_sample.columns:
        df_sample = df_sample.drop(columns=["ReviewCluster"])

    review_selection = df_sample.apply(first_review_selection, axis=1, result_type="expand")
    review_selection.columns = ["ReviewReason", "ReviewCluster"]

    df_sample["ReviewReason"] = review_selection["ReviewReason"]
    df_sample["ReviewCluster"] = pd.to_numeric(
        review_selection["ReviewCluster"], errors="coerce"
    ).astype("Int64")

    # Save the sampled CSV
    output_csv = diagnostics_dir / "sample_100_per_cluster.csv"
    df_sample.to_csv(output_csv, index=False)

    print(f"Wrote clustered samples CSV: {output_csv}")
    print(f"Total rows: {len(df_sample)}")
    print(f"ToReview=True: {int(df_sample['ToReview'].sum())}")
    print(f"ToReview=False: {int((df_sample['ToReview'] == False).sum())}")
    
    # Print summary per method
    print("\nSamples per method:")
    for method in method_columns:
        clusters = df_sample[method].dropna().unique()
        n_clusters = len(clusters)
        n_rows_with_method = df_sample[method].notna().sum()
        print(f"  {method}: {n_clusters} clusters, {n_rows_with_method} rows")


if __name__ == "__main__":
    main()
