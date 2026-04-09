"""Audit sampled cluster CSV rows against source clustering outputs.

This script compares filenames in diagnostics/sample_100_per_cluster.csv with
the underlying clustering assignments from .npy label files (and optional
subsample CSVs where available).

Example:
    python check_sample_vs_sources.py --method umap_3d_hdbscan --cluster 3
"""

from __future__ import annotations

import argparse
from os import environ
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS2_DIR = r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2"


def normalize_path(path_str: str) -> Path:
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


def get_embeddings2_dir(cli_value: str | None = None) -> Path:
    if cli_value:
        return normalize_path(cli_value)

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


def method_spec(npy_dir: Path) -> dict[str, dict[str, Path | None]]:
    umap_tsne_dir = npy_dir / "umap_and_tsne"
    umap_tsne_20d_dir = npy_dir / "umap_and_tsne_20d"
    pca_3d_dir = npy_dir / "pca"
    pca_256d_dir = npy_dir / "PCA_256D"

    return {
        "umap_3d_kmeans": {
            "label": umap_tsne_dir / "umap_kmeans_labels_k25.npy",
            "subset": None,
            "subsample_csv": umap_tsne_dir / "subsample_umap_kmeans_k25.csv",
        },
        "tsne_3d_kmeans": {
            "label": umap_tsne_dir / "tsne_subset_kmeans_labels_k25.npy",
            "subset": umap_tsne_dir / "tsne_subset_indices.npy",
            "subsample_csv": None,
        },
        "pca_3d_kmeans": {
            "label": pca_3d_dir / "pca_kmeans_labels_k25.npy",
            "subset": None,
            "subsample_csv": None,
        },
        "pca_256D_kmeans": {
            "label": pca_256d_dir / "pca_kmeans_labels_k25.npy",
            "subset": None,
            "subsample_csv": None,
        },
        "umap_3d_hdbscan": {
            "label": umap_tsne_dir / "umap_hdbscan_labels_mcs80_ms10.npy",
            "subset": None,
            "subsample_csv": umap_tsne_dir / "subsample_umap_hdbscan_mcs80_ms10.csv",
        },
        "tsne_3d_hdbscan": {
            "label": umap_tsne_dir / "tsne_subset_hdbscan_labels_mcs80_ms10.npy",
            "subset": umap_tsne_dir / "tsne_subset_indices.npy",
            "subsample_csv": umap_tsne_dir / "subsample_tsne_hdbscan_mcs80_ms10.csv",
        },
        "umap_3d_silhouette": {
            "label": umap_tsne_dir / "umap_kmeans_labels_k5.npy",
            "subset": None,
            "subsample_csv": None,
        },
        "tsne_3d_silhouette": {
            "label": umap_tsne_dir / "tsne_subset_kmeans_labels_k5.npy",
            "subset": umap_tsne_dir / "tsne_subset_indices.npy",
            "subsample_csv": None,
        },
        "pca_3d_hdbscan": {
            "label": pca_3d_dir / "pca_hdbscan_labels_mcs80_ms10.npy",
            "subset": None,
            "subsample_csv": pca_3d_dir / "subsample_pca_hdbscan_mcs80_ms10.csv",
        },
        "pca_256D_hdbscan": {
            "label": pca_256d_dir / "pca_hdbscan_labels_mcs80_ms10.npy",
            "subset": None,
            "subsample_csv": None,
        },
        "umap_20d_hdbscan": {
            "label": umap_tsne_20d_dir / "umap_20d_hdbscan_labels_mcs80_ms10.npy",
            "subset": None,
            "subsample_csv": umap_tsne_20d_dir / "subsample_umap_20d_hdbscan_mcs80_ms10.csv",
        },
        "tsne_20d_hdbscan": {
            "label": umap_tsne_20d_dir / "tsne_20d_subset_hdbscan_labels_mcs80_ms10.npy",
            "subset": umap_tsne_20d_dir / "tsne_subset_indices_20d.npy",
            "subsample_csv": umap_tsne_20d_dir / "subsample_tsne_20d_hdbscan_mcs80_ms10.csv",
        },
    }


def parse_cluster_like(value: object) -> int | None:
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


def remap_to_zero_based(series: pd.Series) -> pd.Series:
    valid = series.dropna().astype(int)
    if valid.empty:
        return series.astype("Int64")

    unique = sorted(valid.unique().tolist())
    mapping = {old: new for new, old in enumerate(unique)}
    remapped = series.map(lambda x: mapping.get(int(x), pd.NA) if pd.notna(x) else pd.NA)
    return remapped.astype("Int64")


def cluster_counts(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if values.empty:
        return pd.Series(dtype="int64")
    return values.value_counts().sort_index()


def plot_cluster_distribution(
    source_counts: pd.Series,
    sample_counts: pd.Series,
    target_cluster: int,
    output_path: Path,
    method: str,
) -> None:
    all_clusters = sorted(set(source_counts.index.tolist()) | set(sample_counts.index.tolist()))
    if not all_clusters:
        raise ValueError("No cluster values available to plot")

    source_values = [int(source_counts.get(cluster, 0)) for cluster in all_clusters]
    sample_values = [int(sample_counts.get(cluster, 0)) for cluster in all_clusters]

    width = 1200
    height = 700
    margin_left = 80
    margin_right = 30
    margin_top = 70
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_value = max(source_values + sample_values) if (source_values or sample_values) else 1
    max_value = max(max_value, 1)

    def scale_value(value: int) -> float:
        return (value / max_value) * plot_height

    bar_group_width = plot_width / len(all_clusters)
    bar_width = min(24.0, bar_group_width * 0.3)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#1f2937">Cluster distribution for {method}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Blue = source labels, orange = sample CSV</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1"/>'
    ]

    tick_count = 5
    for tick in range(tick_count + 1):
        tick_value = int(round(max_value * tick / tick_count))
        y = margin_top + plot_height - scale_value(tick_value)
        svg_parts.append(f'<line x1="{margin_left - 4}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="#374151" stroke-width="1"/>')
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{tick_value}</text>'
        )

    for index, cluster in enumerate(all_clusters):
        group_x = margin_left + index * bar_group_width
        source_height = scale_value(source_values[index])
        sample_height = scale_value(sample_values[index])
        source_x = group_x + bar_group_width * 0.18
        sample_x = group_x + bar_group_width * 0.52
        base_y = margin_top + plot_height

        svg_parts.append(
            f'<rect x="{source_x:.2f}" y="{base_y - source_height:.2f}" width="{bar_width:.2f}" height="{source_height:.2f}" fill="#4C78A8"/>'
        )
        svg_parts.append(
            f'<rect x="{sample_x:.2f}" y="{base_y - sample_height:.2f}" width="{bar_width:.2f}" height="{sample_height:.2f}" fill="#F58518"/>'
        )
        svg_parts.append(
            f'<text x="{group_x + bar_group_width / 2:.2f}" y="{base_y + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{cluster}</text>'
        )

        if cluster == target_cluster:
            svg_parts.append(
                f'<line x1="{group_x + bar_group_width / 2:.2f}" y1="{margin_top}" x2="{group_x + bar_group_width / 2:.2f}" y2="{base_y}" stroke="#E45756" stroke-width="1.5" stroke-dasharray="5,4"/>'
            )
            svg_parts.append(
                f'<text x="{group_x + bar_group_width / 2:.2f}" y="{margin_top - 8}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#E45756">cluster {target_cluster}</text>'
            )

    legend_y = height - 28
    svg_parts.append(f'<rect x="{margin_left}" y="{legend_y}" width="14" height="14" fill="#4C78A8"/>')
    svg_parts.append(
        f'<text x="{margin_left + 20}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="12" fill="#374151">Source labels</text>'
    )
    svg_parts.append(f'<rect x="{margin_left + 140}" y="{legend_y}" width="14" height="14" fill="#F58518"/>')
    svg_parts.append(
        f'<text x="{margin_left + 160}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="12" fill="#374151">Sample CSV</text>'
    )
    svg_parts.append("</svg>")

    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sample CSV against source cluster files")
    parser.add_argument("--embeddings2-dir", default=None, help="Override Embeddings_2 directory")
    parser.add_argument("--method", default="umap_3d_hdbscan", help="Method column to audit")
    parser.add_argument("--cluster", type=int, default=3, help="Cluster id to inspect")
    parser.add_argument("--max-show", type=int, default=20, help="Max filenames to print for diffs")
    parser.add_argument("--plot", action="store_true", help="Save a cluster distribution plot")
    parser.add_argument("--plot-path", default=None, help="Optional output path for the plot PNG")
    args = parser.parse_args()

    embeddings2_dir = get_embeddings2_dir(args.embeddings2_dir)
    dataset_dir = embeddings2_dir / "dataset"
    npy_dir = dataset_dir / "npy_files"
    diagnostics_dir = embeddings2_dir / "diagnostics"

    manifest_path = dataset_dir / "unlabeled_manifest.csv"
    sample_csv = diagnostics_dir / "sample_100_per_cluster.csv"
    all_methods_csv = diagnostics_dir / "all_methods_cluster_diagnostics.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    if not sample_csv.exists():
        raise FileNotFoundError(f"Missing sample CSV: {sample_csv}")

    manifest_df = pd.read_csv(str(manifest_path))
    if "audio_path" not in manifest_df.columns:
        raise ValueError("Expected 'audio_path' in unlabeled_manifest.csv")
    all_filenames = manifest_df["audio_path"].map(lambda p: Path(str(p)).name)

    specs = method_spec(npy_dir)
    if args.method not in specs:
        available = ", ".join(sorted(specs.keys()))
        raise ValueError(f"Unknown method '{args.method}'. Available: {available}")

    spec = specs[args.method]
    labels = load_labels_full(
        label_path=normalize_path(str(spec["label"])),
        n_rows=len(manifest_df),
        subset_idx_path=normalize_path(str(spec["subset"])) if spec["subset"] is not None else None,
    )

    labels = remap_to_zero_based(labels)
    source_mask = labels == args.cluster
    source_filenames = set(all_filenames[source_mask.fillna(False)].tolist())

    sample_df = pd.read_csv(str(sample_csv))
    if "filename" not in sample_df.columns:
        raise ValueError("Expected 'filename' in sample_100_per_cluster.csv")
    if args.method not in sample_df.columns:
        raise ValueError(f"Method column not present in sample CSV: {args.method}")

    sample_cluster = pd.to_numeric(sample_df[args.method], errors="coerce").astype("Int64")
    sample_mask = sample_cluster == args.cluster
    sample_filenames = set(sample_df.loc[sample_mask.fillna(False), "filename"].astype(str).tolist())

    source_cluster_counts = cluster_counts(labels)
    sample_cluster_counts = cluster_counts(sample_df[args.method])
    source_total = int(source_cluster_counts.sum())
    sample_total = int(sample_cluster_counts.sum())
    source_target = int(source_cluster_counts.get(args.cluster, 0))
    sample_target = int(sample_cluster_counts.get(args.cluster, 0))

    source_share = (source_target / source_total) if source_total else 0.0
    sample_share = (sample_target / sample_total) if sample_total else 0.0
    lift = (sample_share / source_share) if source_share else float("inf")

    all_methods_count = None
    if all_methods_csv.exists():
        full_df = pd.read_csv(str(all_methods_csv), usecols=["filename", args.method])
        full_cluster = pd.to_numeric(full_df[args.method], errors="coerce").astype("Int64")
        all_methods_count = int((full_cluster == args.cluster).sum())

    subsample_count = None
    subsample_filenames: set[str] | None = None
    subsample_csv = spec["subsample_csv"]
    if subsample_csv is not None:
        subsample_path = normalize_path(str(subsample_csv))
        if subsample_path.exists():
            sub_df = pd.read_csv(str(subsample_path))
            if "cluster" in sub_df.columns and "audio_path" in sub_df.columns:
                sub_cluster = pd.to_numeric(sub_df["cluster"], errors="coerce").astype("Int64")
                sub_mask = sub_cluster == args.cluster
                subsample_filenames = set(
                    sub_df.loc[sub_mask.fillna(False), "audio_path"].map(lambda p: Path(str(p)).name).tolist()
                )
                subsample_count = len(subsample_filenames)

    sample_not_in_source = sorted(sample_filenames - source_filenames)
    source_not_in_sample = sorted(source_filenames - sample_filenames)

    print(f"Embeddings dir: {embeddings2_dir}")
    print(f"Method: {args.method}")
    print(f"Cluster: {args.cluster}")
    print("-")
    print(f"Count in source labels (.npy): {len(source_filenames)}")
    print(f"Count in sample_100_per_cluster.csv: {len(sample_filenames)}")
    print(f"Target cluster in source labels: {source_target} / {source_total} ({source_share:.4%})")
    print(f"Target cluster in sample CSV: {sample_target} / {sample_total} ({sample_share:.4%})")
    print(f"Lift (sample share / source share): {lift:.3f}")
    if all_methods_count is not None:
        print(f"Count in all_methods_cluster_diagnostics.csv: {all_methods_count}")
    if subsample_count is not None:
        print(f"Count in subsample CSV for same method/cluster: {subsample_count}")
    print("-")
    print(f"Sample filenames missing from source labels: {len(sample_not_in_source)}")
    print(f"Source filenames not present in sample CSV: {len(source_not_in_sample)}")

    if sample_not_in_source:
        print("\nExample sample filenames missing from source:")
        for name in sample_not_in_source[: args.max_show]:
            print(f"  {name}")

    if source_not_in_sample:
        print("\nExample source filenames not in sample:")
        for name in source_not_in_sample[: args.max_show]:
            print(f"  {name}")

    if args.plot:
        plot_path = Path(args.plot_path) if args.plot_path else diagnostics_dir / f"{args.method}_cluster_distribution.svg"
        plot_cluster_distribution(
            source_counts=source_cluster_counts,
            sample_counts=sample_cluster_counts,
            target_cluster=args.cluster,
            output_path=plot_path,
            method=args.method,
        )
        print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
