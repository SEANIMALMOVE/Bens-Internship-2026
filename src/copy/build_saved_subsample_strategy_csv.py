"""Build consolidated CSVs from saved per-method subsample strategy files.

Outputs:
1) saved_subsample_strategy_long.csv: all rows from chosen method subsamples.
2) saved_subsample_strategy_selected.csv: only rows matching review rules.
"""

from __future__ import annotations

from pathlib import Path
from os import environ
import csv
import re


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


def get_embeddings2_dir() -> Path:
    env_embeddings2 = environ.get("POSIDONIA_EMBEDDINGS2_DIR")
    if env_embeddings2:
        return normalize_path(env_embeddings2)

    env_dataset = environ.get("POSIDONIA_DATASET_DIR")
    if env_dataset:
        return normalize_path(env_dataset).parent

    return normalize_path(DEFAULT_EMBEDDINGS2_DIR)


def parse_cluster(value: object) -> int | None:
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


def matches_rule(rule: str, cluster_value: int | None, clusters: object) -> bool:
    if cluster_value is None:
        return False

    if rule == "in_set":
        if clusters is None:
            return False
        allowed = set(int(c) for c in clusters)
        return cluster_value in allowed

    if rule == "all_non_noise":
        return cluster_value != -1

    return False


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    embeddings2_dir = get_embeddings2_dir()
    npy_dir = embeddings2_dir / "dataset" / "npy_files"
    diagnostics_dir = embeddings2_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Priority order used in downstream Strategy B assignment.
    method_specs: list[dict[str, object]] = [
        {
            "priority": 1,
            "method": "umap_3d_kmeans",
            "rule": "in_set",
            "clusters": [6, 15, 20, 21, 22],
            "subsample_csv": npy_dir / "umap_and_tsne" / "subsample_umap_kmeans_k25.csv",
        },
        {
            "priority": 2,
            "method": "umap_3d_hdbscan",
            "rule": "in_set",
            "clusters": [0, 1, 3, 5],
            "subsample_csv": npy_dir / "umap_and_tsne" / "subsample_umap_hdbscan_mcs80_ms10.csv",
        },
        {
            "priority": 3,
            "method": "tsne_3d_hdbscan",
            "rule": "in_set",
            "clusters": [0, 1],
            "subsample_csv": npy_dir / "umap_and_tsne" / "subsample_tsne_hdbscan_mcs80_ms10.csv",
        },
        {
            "priority": 4,
            "method": "umap_20d_hdbscan",
            "rule": "in_set",
            "clusters": [0, 1, 3, 4],
            "subsample_csv": npy_dir / "umap_and_tsne_20d" / "subsample_umap_20d_hdbscan_mcs80_ms10.csv",
        },
        {
            "priority": 5,
            "method": "tsne_20d_hdbscan",
            "rule": "all_non_noise",
            "clusters": None,
            "subsample_csv": npy_dir / "umap_and_tsne_20d" / "subsample_tsne_20d_hdbscan_mcs80_ms10.csv",
        },
        {
            "priority": 6,
            "method": "pca_3d_hdbscan",
            "rule": "in_set",
            "clusters": list(range(0, 25)),
            "subsample_csv": npy_dir / "pca" / "subsample_pca_hdbscan_mcs80_ms10.csv",
        },
    ]

    long_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for spec in method_specs:
        priority = int(spec["priority"])
        method = str(spec["method"])
        rule = str(spec["rule"])
        clusters = spec["clusters"]
        subsample_csv = normalize_path(str(spec["subsample_csv"]))

        if not subsample_csv.exists():
            raise FileNotFoundError(f"Missing subsample file for {method}: {subsample_csv}")

        with subsample_csv.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for source_row, row in enumerate(reader, start=2):
                audio_path = str(row.get("audio_path", "")).strip()
                filename = Path(audio_path).name if audio_path else ""
                cluster_value = parse_cluster(row.get("cluster", ""))
                selected_for_review = matches_rule(rule, cluster_value, clusters)

                out = {
                    "priority": priority,
                    "method": method,
                    "rule": rule,
                    "cluster": cluster_value if cluster_value is not None else "",
                    "selected_for_review": selected_for_review,
                    "audio_path": audio_path,
                    "filename": filename,
                    "embedding_path": str(row.get("embedding_path", "")).strip(),
                    "reduced_embedding_filepath": str(row.get("reduced_embedding_filepath", "")).strip(),
                    "reduced_embeddings_idx": str(row.get("reduced_embeddings_idx", "")).strip(),
                    "source_subsample_csv": str(subsample_csv),
                    "source_row": source_row,
                }

                long_rows.append(out)
                if selected_for_review:
                    selected_rows.append(out)

    # Assign one review reason per filename based on highest-priority matching method.
    reason_by_filename: dict[str, tuple[int, str]] = {}
    for row in selected_rows:
        filename = str(row.get("filename", "")).strip()
        if filename == "":
            continue

        priority = int(row.get("priority", 10**9))
        method = str(row.get("method", "")).strip()
        previous = reason_by_filename.get(filename)
        if previous is None or priority < previous[0]:
            reason_by_filename[filename] = (priority, method)

    for row in long_rows:
        filename = str(row.get("filename", "")).strip()
        if filename in reason_by_filename:
            row["ReviewReason"] = reason_by_filename[filename][1]
        else:
            row["ReviewReason"] = ""

    long_csv = diagnostics_dir / "saved_subsample_strategy_long.csv"
    selected_csv = diagnostics_dir / "saved_subsample_strategy_selected.csv"

    fields = [
        "priority",
        "method",
        "rule",
        "cluster",
        "selected_for_review",
        "audio_path",
        "filename",
        "embedding_path",
        "reduced_embedding_filepath",
        "reduced_embeddings_idx",
        "source_subsample_csv",
        "source_row",
        "ReviewReason",
    ]

    write_csv(long_csv, long_rows, fields)
    write_csv(selected_csv, selected_rows, fields)

    print(f"Wrote: {long_csv}")
    print(f"Rows (all subsamples): {len(long_rows)}")
    print(f"Wrote: {selected_csv}")
    print(f"Rows (selected by rules): {len(selected_rows)}")


if __name__ == "__main__":
    main()
