"""Build the 5th-approach review CSV from the full diagnostics file.

This approach keeps only the chosen clusters from the review notebooks,
samples up to 100 rows per selected cluster, and writes a CSV with the same
column layout as sample_100_per_cluster.csv.
"""

from __future__ import annotations

from os import environ
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS2_DIR = r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2"


REVIEW_RULES: dict[str, tuple[str, set[int] | None]] = {
    "umap_3d_kmeans": ("in_set", {6, 15, 20, 21, 22}),
    "umap_3d_hdbscan": ("in_set", {0, 1, 2, 4, 5}),
    "tsne_3d_hdbscan": ("in_set", {1, 2}),
    "umap_20d_hdbscan": ("in_set", {0, 1, 2, 4, 5}),
    "tsne_20d_hdbscan": ("all_non_noise", None),
    "pca_3d_hdbscan": ("in_set", set(range(0, 25))),
}


REVIEW_PRIORITY: list[str] = [
    "umap_3d_kmeans",
    "umap_3d_hdbscan",
    "tsne_3d_hdbscan",
    "umap_20d_hdbscan",
    "tsne_20d_hdbscan",
    "pca_3d_hdbscan",
]


DISPLAY_NAMES: dict[str, str] = {
    "umap_3d_kmeans": "umap kmeans",
    "umap_3d_hdbscan": "umap hdbscan",
    "tsne_3d_hdbscan": "tsne hdbscan",
    "umap_20d_hdbscan": "umap 20d hdbscan",
    "tsne_20d_hdbscan": "tsne 20d hdbscan",
    "pca_3d_hdbscan": "pca hdbscan",
}


# Keep all rare UMAP-HDBSCAN clusters, but cap the dominant cluster 3 spillover.
UMAP3D_HDBSCAN_PRIORITY_CLUSTERS: set[int] = {0, 1, 2, 4, 5}
UMAP3D_HDBSCAN_BACKGROUND_CLUSTER = 3
UMAP3D_HDBSCAN_BACKGROUND_MAX = 300


def normalize_path(path_str: str) -> Path:
    p = Path(path_str)
    try:
        if p.exists():
            return p
    except OSError:
        pass

    if re.match(r"^[A-Za-z]:\\", path_str):
        drive = path_str[0].lower()
        suffix = path_str[2:].replace("\\", "/")
        wsl_candidate = Path(f"/mnt/{drive}{suffix}")
        try:
            if wsl_candidate.exists():
                return wsl_candidate
        except OSError:
            pass

    return p


def get_embeddings2_dir() -> Path:
    env_embeddings2 = environ.get("POSIDONIA_EMBEDDINGS2_DIR")
    if env_embeddings2:
        return normalize_path(env_embeddings2)

    env_dataset = environ.get("POSIDONIA_DATASET_DIR")
    if env_dataset:
        return normalize_path(env_dataset).parent

    local_base = Path(__file__).resolve().parent
    if (local_base / "diagnostics" / "all_methods_cluster_diagnostics.csv").exists():
        return local_base

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
    for method in REVIEW_PRIORITY:
        rule, allowed = REVIEW_RULES[method]
        cluster_value = parse_cluster(row.get(method, pd.NA))
        if matches_rule(rule, cluster_value, allowed):
            return method, cluster_value
    return "", None


def count_method_candidates(df: pd.DataFrame, method: str) -> int:
    rule, allowed = REVIEW_RULES[method]
    matched = 0
    for cluster_id in df[method].dropna().unique():
        cluster_value = parse_cluster(cluster_id)
        if matches_rule(rule, cluster_value, allowed):
            matched += int((df[method] == cluster_value).sum())
    return matched


def apply_review_focus_filter(df_sample: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if "umap_3d_hdbscan" not in df_sample.columns:
        return df_sample

    umap_hdbscan = pd.to_numeric(df_sample["umap_3d_hdbscan"], errors="coerce").astype("Int64")

    priority_mask = umap_hdbscan.isin(list(UMAP3D_HDBSCAN_PRIORITY_CLUSTERS)).fillna(False)
    background_mask = (umap_hdbscan == UMAP3D_HDBSCAN_BACKGROUND_CLUSTER).fillna(False)
    other_mask = ~(priority_mask | background_mask)

    keep_priority_idx = np.where(priority_mask.to_numpy())[0]
    keep_other_idx = np.where(other_mask.to_numpy())[0]
    background_idx = np.where(background_mask.to_numpy())[0]

    if len(background_idx) > UMAP3D_HDBSCAN_BACKGROUND_MAX:
        keep_background_idx = rng.choice(background_idx, size=UMAP3D_HDBSCAN_BACKGROUND_MAX, replace=False)
    else:
        keep_background_idx = background_idx

    keep_idx = np.unique(np.concatenate([keep_priority_idx, keep_background_idx, keep_other_idx]))
    return df_sample.iloc[keep_idx].reset_index(drop=True)


def main() -> None:
    embeddings2_dir = get_embeddings2_dir()
    diagnostics_dir = embeddings2_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    full_csv = diagnostics_dir / "all_methods_cluster_diagnostics.csv"
    if not full_csv.exists():
        fallback_csv = Path(__file__).resolve().parent / "all_methods_cluster_diagnostics.csv"
        if fallback_csv.exists():
            full_csv = fallback_csv
        else:
            raise FileNotFoundError(f"Missing full diagnostics CSV: {full_csv}")

    df_full = pd.read_csv(str(full_csv))

    if "filename" not in df_full.columns:
        raise ValueError("Expected 'filename' in all_methods_cluster_diagnostics.csv")

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

    selected_indices: set[int] = set()
    rng = np.random.default_rng(42)

    candidate_counts = {
        method: count_method_candidates(df_full, method) for method in REVIEW_PRIORITY
    }

    for method in REVIEW_PRIORITY:
        rule, allowed = REVIEW_RULES[method]
        for cluster_id in sorted(df_full[method].dropna().unique()):
            cluster_value = parse_cluster(cluster_id)
            if not matches_rule(rule, cluster_value, allowed):
                continue

            mask = df_full[method] == cluster_value
            matching_indices = np.where(mask)[0]
            n_samples = min(100, len(matching_indices))
            if n_samples == 0:
                continue
            sampled = rng.choice(matching_indices, size=n_samples, replace=False)
            selected_indices.update(sampled.tolist())

    selected_list = sorted(selected_indices)
    df_sample = df_full.iloc[selected_list].reset_index(drop=True)
    df_sample = apply_review_focus_filter(df_sample, rng)

    for method in method_columns:
        df_sample[method] = pd.to_numeric(df_sample[method], errors="coerce").astype("Int64")

    review_reason_series = []
    review_cluster_series = []
    for _, row in df_sample.iterrows():
        reason, cluster_value = first_review_selection(row)
        review_reason_series.append(reason)
        review_cluster_series.append(cluster_value)

    df_sample["ToReview"] = True
    df_sample["Prediction"] = ""
    if "ReviewReason" in df_sample.columns:
        df_sample = df_sample.drop(columns=["ReviewReason"])
    if "ReviewCluster" in df_sample.columns:
        df_sample = df_sample.drop(columns=["ReviewCluster"])
    df_sample["ReviewReason"] = review_reason_series
    df_sample["ReviewCluster"] = pd.Series(review_cluster_series, dtype="Int64")

    owned_counts: dict[str, int] = {method: 0 for method in REVIEW_PRIORITY}
    for reason in review_reason_series:
        if reason in owned_counts:
            owned_counts[reason] += 1

    output_csv = diagnostics_dir / "5th_approach_sample_100_per_cluster.csv"
    df_sample.to_csv(output_csv, index=False)

    print(f"Wrote 5th approach CSV: {output_csv}")
    print(f"Rows: {len(df_sample)}")
    print(f"ToReview=True: {int(df_sample['ToReview'].sum())}")
    for method in REVIEW_PRIORITY:
        label = DISPLAY_NAMES.get(method, method.replace("_", " "))
        print(f"{label}: {candidate_counts[method]} candidates, {owned_counts[method]} owned")


if __name__ == "__main__":
    main()
