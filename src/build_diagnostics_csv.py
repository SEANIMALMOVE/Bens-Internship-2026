from __future__ import annotations

from pathlib import Path
from os import environ
import re

import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS2_DIR = r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2"


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


def load_labels_full(label_path: Path, n_rows: int, subset_idx_path: Path | None = None) -> pd.Series:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels file: {label_path}")

    labels = np.load(str(label_path))

    if subset_idx_path is None:
        if len(labels) != n_rows:
            raise ValueError(f"Length mismatch for {label_path}: got {len(labels)}, expected {n_rows}")
        return pd.Series(labels, dtype="Int64")

    if not subset_idx_path.exists():
        raise FileNotFoundError(f"Missing subset index file: {subset_idx_path}")

    subset_idx = np.load(str(subset_idx_path))
    if len(labels) != len(subset_idx):
        raise ValueError(
            f"Subset length mismatch for {label_path} and {subset_idx_path}: "
            f"{len(labels)} vs {len(subset_idx)}"
        )

    full = np.full(n_rows, pd.NA, dtype=object)
    full[subset_idx] = labels
    return pd.Series(full, dtype="Int64")


def remap_to_zero_based(series: pd.Series) -> pd.Series:
    valid = series.dropna().astype(int)
    if valid.empty:
        return series.astype("Int64")

    unique = sorted(valid.unique().tolist())
    mapping = {old: new for new, old in enumerate(unique)}
    remapped = series.map(lambda x: mapping.get(int(x), pd.NA) if pd.notna(x) else pd.NA)
    return remapped.astype("Int64")


def parse_best_k(silhouette_txt: Path) -> tuple[int, int]:
    if not silhouette_txt.exists():
        raise FileNotFoundError(f"Missing silhouette best-k file: {silhouette_txt}")

    lines = [line.strip() for line in silhouette_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = int(value.strip())

    if "best_k_umap" not in values or "best_k_tsne" not in values:
        raise ValueError(f"Could not parse best_k_umap/best_k_tsne from {silhouette_txt}")

    return values["best_k_umap"], values["best_k_tsne"]


def mark_review(rule: str, labels: pd.Series, allowed: set[int] | None = None) -> pd.Series:
    if rule == "in_set":
        if allowed is None:
            raise ValueError("allowed set is required for in_set rule")
        return labels.map(lambda x: pd.notna(x) and int(x) in allowed)

    if rule == "all_non_noise":
        return labels.map(lambda x: pd.notna(x) and int(x) != -1)

    return pd.Series(False, index=labels.index)


def main() -> None:
    embeddings2_dir = get_embeddings2_dir()
    dataset_dir = embeddings2_dir / "dataset"
    npy_dir = dataset_dir / "npy_files"
    diagnostics_dir = embeddings2_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "unlabeled_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    manifest_df = pd.read_csv(str(manifest_path))
    if "audio_path" not in manifest_df.columns:
        raise ValueError("Expected column 'audio_path' in unlabeled_manifest.csv")

    n_rows = len(manifest_df)
    result_df = pd.DataFrame(
        {
            "filename": manifest_df["audio_path"].apply(lambda p: Path(str(p)).name),
        }
    )

    umap_tsne_dir = npy_dir / "umap_and_tsne"
    umap_tsne_20d_dir = npy_dir / "umap_and_tsne_20d"
    pca_3d_dir = npy_dir / "pca"
    pca_256d_dir = npy_dir / "PCA_256D"

    best_k_umap, best_k_tsne = parse_best_k(umap_tsne_dir / "silhouette_best_k.txt")

    raw_labels: dict[str, pd.Series] = {
        "umap_3d_kmeans": load_labels_full(umap_tsne_dir / "umap_kmeans_labels_k25.npy", n_rows),
        "tsne_3d_kmeans": load_labels_full(
            umap_tsne_dir / "tsne_subset_kmeans_labels_k25.npy",
            n_rows,
            subset_idx_path=umap_tsne_dir / "tsne_subset_indices.npy",
        ),
        "pca_3d_kmeans": load_labels_full(pca_3d_dir / "pca_kmeans_labels_k25.npy", n_rows),
        "pca_256D_kmeans": load_labels_full(pca_256d_dir / "pca_kmeans_labels_k25.npy", n_rows),
        "umap_3d_hdbscan": load_labels_full(
            umap_tsne_dir / "umap_hdbscan_labels_mcs80_ms10.npy",
            n_rows,
        ),
        "tsne_3d_hdbscan": load_labels_full(
            umap_tsne_dir / "tsne_subset_hdbscan_labels_mcs80_ms10.npy",
            n_rows,
            subset_idx_path=umap_tsne_dir / "tsne_subset_indices.npy",
        ),
        "umap_3d_silhouette": load_labels_full(
            umap_tsne_dir / f"umap_kmeans_labels_k{best_k_umap}.npy",
            n_rows,
        ),
        "tsne_3d_silhouette": load_labels_full(
            umap_tsne_dir / f"tsne_subset_kmeans_labels_k{best_k_tsne}.npy",
            n_rows,
            subset_idx_path=umap_tsne_dir / "tsne_subset_indices.npy",
        ),
        "pca_3d_hdbscan": load_labels_full(
            pca_3d_dir / "pca_hdbscan_labels_mcs80_ms10.npy",
            n_rows,
        ),
        "pca_256D_hdbscan": load_labels_full(
            pca_256d_dir / "pca_hdbscan_labels_mcs80_ms10.npy",
            n_rows,
        ),
        "umap_20d_hdbscan": load_labels_full(
            umap_tsne_20d_dir / "umap_20d_hdbscan_labels_mcs80_ms10.npy",
            n_rows,
        ),
        "tsne_20d_hdbscan": load_labels_full(
            umap_tsne_20d_dir / "tsne_20d_subset_hdbscan_labels_mcs80_ms10.npy",
            n_rows,
            subset_idx_path=umap_tsne_20d_dir / "tsne_subset_indices_20d.npy",
        ),
    }

    # Keep review rules aligned with the notes written in visualize/cluster notebooks.
    review_rules = {
        "umap_3d_kmeans": ("in_set", {6, 15, 20, 21, 22}),
        "tsne_3d_kmeans": ("none", None),
        "pca_3d_kmeans": ("none", None),
        "pca_256D_kmeans": ("none", None),
        "umap_3d_hdbscan": ("in_set", {0, 1, 3, 5}),
        "tsne_3d_hdbscan": ("in_set", {0, 1}),
        "umap_3d_silhouette": ("none", None),
        "tsne_3d_silhouette": ("none", None),
        "pca_3d_hdbscan": ("in_set", set(range(0, 25))),
        "pca_256D_hdbscan": ("none", None),
        "umap_20d_hdbscan": ("in_set", {0, 1, 3, 4}),
        "tsne_20d_hdbscan": ("all_non_noise", None),
    }

    to_review = pd.Series(False, index=result_df.index)

    for method_name, labels in raw_labels.items():
        result_df[method_name] = remap_to_zero_based(labels)

        rule_name, allowed = review_rules[method_name]
        to_review = to_review | mark_review(rule_name, labels, allowed)

    result_df["ToReview"] = to_review
    result_df["Prediction"] = ""

    output_csv = diagnostics_dir / "all_methods_cluster_diagnostics.csv"
    result_df.to_csv(output_csv, index=False)

    print(f"Wrote diagnostics CSV: {output_csv}")
    print(f"Rows: {len(result_df)}")
    print(f"ToReview=True: {int(result_df['ToReview'].sum())}")


if __name__ == "__main__":
    main()
