"""Create Strategy B ownership from saved subsamples, then copy audio files.

Workflow:
1) Build assignment CSV from method-specific subsample CSVs.
2) Copy each audio exactly once to the chosen owner method/cluster folder.
"""

from __future__ import annotations

from pathlib import Path
import csv
import re
import shutil


CSV_PATH = Path(
    r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\diagnostics\sample_100_per_cluster.csv"
)
SOURCE_DIR = Path(r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\5seg")
DEST_ROOT = Path(
    r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\diagnostics\ToReview_2026_04_07"
)
SUBSAMPLE_BASE_DIR = Path(
    r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\dataset\npy_files"
)

FILENAME_COLUMN = "filename"
TO_REVIEW_COLUMN = "ToReview"
USE_TO_REVIEW_ONLY = True

# Method order defines ownership priority when an audio matches multiple methods.
# Only methods with reviewable rules are included (skip all rule="none" methods).
METHOD_RULE_PRIORITY: list[dict[str, object]] = [
    {
        "method": "umap_3d_kmeans",
        "rule": "in_set",
        "clusters": [6, 15, 20, 21, 22],
        "subsample_csv": SUBSAMPLE_BASE_DIR / "umap_and_tsne" / "subsample_umap_kmeans_k25.csv",
    },
    {
        "method": "umap_3d_hdbscan",
        "rule": "in_set",
        "clusters": [0, 1, 3, 5],
        "subsample_csv": SUBSAMPLE_BASE_DIR / "umap_and_tsne" / "subsample_umap_hdbscan_mcs80_ms10.csv",
    },
    {
        "method": "tsne_3d_hdbscan",
        "rule": "in_set",
        "clusters": [0, 1],
        "subsample_csv": SUBSAMPLE_BASE_DIR / "umap_and_tsne" / "subsample_tsne_hdbscan_mcs80_ms10.csv",
    },
    {
        "method": "umap_20d_hdbscan",
        "rule": "in_set",
        "clusters": [0, 1, 3, 4],
        "subsample_csv": SUBSAMPLE_BASE_DIR / "umap_and_tsne_20d" / "subsample_umap_20d_hdbscan_mcs80_ms10.csv",
    },
    {
        "method": "tsne_20d_hdbscan",
        "rule": "all_non_noise",
        "clusters": None,
        "subsample_csv": SUBSAMPLE_BASE_DIR / "umap_and_tsne_20d" / "subsample_tsne_20d_hdbscan_mcs80_ms10.csv",
    },
    {
        "method": "pca_3d_hdbscan",
        "rule": "in_set",
        "clusters": list(range(0, 25)),
        "subsample_csv": SUBSAMPLE_BASE_DIR / "pca" / "subsample_pca_hdbscan_mcs80_ms10.csv",
    },
]


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_path(path: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p

    s = str(path)
    if re.match(r"^[A-Za-z]:\\", s):
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        wsl = Path(f"/mnt/{drive}{rest}")
        if wsl.exists() or str(wsl).startswith("/mnt/"):
            return wsl

    return p


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

    if rule == "none":
        return False

    if rule == "in_set":
        if clusters is None:
            return False
        allowed = set(int(c) for c in clusters)
        return cluster_value in allowed

    if rule == "all_non_noise":
        return cluster_value != -1

    return False


def filename_from_audio_path(audio_path: str) -> str:
    return Path(str(audio_path).strip()).name


def source_audio_from_row(audio_path: str, filename: str, source_dir: Path) -> Path:
    direct = normalize_path(Path(audio_path))
    if direct.exists():
        return direct
    return source_dir / filename


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    source_dir = normalize_path(SOURCE_DIR)
    dest_root = normalize_path(DEST_ROOT)

    dest_root.mkdir(parents=True, exist_ok=True)
    reports_dir = dest_root / "_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Phase 1: collect candidate memberships from saved subsample files.
    candidates_by_file: dict[str, list[dict[str, object]]] = {}
    memberships: list[dict[str, object]] = []
    unique_memberships: set[tuple[str, str, int]] = set()
    scanned_rows = 0

    for priority, spec in enumerate(METHOD_RULE_PRIORITY):
        method = str(spec["method"])
        rule = str(spec["rule"])
        clusters = spec["clusters"]
        subsample_csv = normalize_path(Path(spec["subsample_csv"]))

        if not subsample_csv.exists():
            raise FileNotFoundError(f"Missing subsample CSV for {method}: {subsample_csv}")

        with subsample_csv.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                scanned_rows += 1
                audio_path = str(row.get("audio_path", "")).strip()
                filename = filename_from_audio_path(audio_path)
                if filename == "":
                    continue

                cluster_value = parse_cluster(row.get("cluster", ""))
                if not matches_rule(rule, cluster_value, clusters):
                    continue

                cluster_int = int(cluster_value)
                key = (filename, method, cluster_int)
                if key in unique_memberships:
                    continue
                unique_memberships.add(key)

                candidate = {
                    "priority": priority,
                    "method": method,
                    "cluster": cluster_int,
                    "rule": rule,
                    "audio_path": audio_path,
                    "subsample_csv": str(subsample_csv),
                    "source_row": row_idx,
                }
                candidates_by_file.setdefault(filename, []).append(candidate)

                memberships.append(
                    {
                        "filename": filename,
                        "method": method,
                        "cluster": cluster_int,
                        "rule": rule,
                        "audio_path": audio_path,
                        "subsample_csv": str(subsample_csv),
                        "source_row": row_idx,
                        "is_owner": False,
                    }
                )

    # Phase 2: build assignment CSV first.
    decisions: list[dict[str, object]] = []
    membership_owner_keys: set[tuple[str, str, int]] = set()

    for filename in sorted(candidates_by_file.keys()):
        candidates = sorted(candidates_by_file[filename], key=lambda x: int(x["priority"]))
        owner = candidates[0]

        owner_method = str(owner["method"])
        owner_cluster = int(owner["cluster"])
        owner_rule = str(owner["rule"])
        owner_audio_path = str(owner["audio_path"])
        owner_source_path = source_audio_from_row(owner_audio_path, filename, source_dir)

        membership_owner_keys.add((filename, owner_method, owner_cluster))

        decisions.append(
            {
                "filename": filename,
                "status": "assigned",
                "owner_method": owner_method,
                "owner_cluster": owner_cluster,
                "owner_rule": owner_rule,
                "owner_reason": (
                    f"matched {len(candidates)} candidates from saved subsamples; "
                    f"picked highest-priority method={owner_method}"
                ),
                "candidate_count": len(candidates),
                "candidates": "|".join(
                    f"{c['method']}:{c['cluster']}({c['rule']})" for c in candidates
                ),
                "owner_audio_path": owner_audio_path,
                "source_path": str(owner_source_path),
                "dest_path": "",
                "copy_status": "pending",
            }
        )

    for membership in memberships:
        key = (str(membership["filename"]), str(membership["method"]), int(membership["cluster"]))
        membership["is_owner"] = key in membership_owner_keys

    assignments_csv = reports_dir / "strategy_b_subsample_assignments.csv"
    memberships_csv = reports_dir / "strategy_b_subsample_memberships.csv"

    write_csv(
        assignments_csv,
        decisions,
        [
            "filename",
            "status",
            "owner_method",
            "owner_cluster",
            "owner_rule",
            "owner_reason",
            "candidate_count",
            "candidates",
            "owner_audio_path",
            "source_path",
            "dest_path",
            "copy_status",
        ],
    )

    write_csv(
        memberships_csv,
        memberships,
        [
            "filename",
            "method",
            "cluster",
            "rule",
            "audio_path",
            "subsample_csv",
            "source_row",
            "is_owner",
        ],
    )

    # Phase 3: copy from assignment decisions.
    copied = 0
    missing_source = 0
    for decision in decisions:
        filename = str(decision["filename"])
        owner_method = str(decision["owner_method"])
        owner_cluster = int(decision["owner_cluster"])
        src = normalize_path(Path(str(decision["source_path"])))

        dest_dir = dest_root / owner_method / f"cluster_{owner_cluster:02d}"
        dst = dest_dir / filename

        if src.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            decision["copy_status"] = "copied"
            decision["dest_path"] = str(dst)
        else:
            missing_source += 1
            decision["copy_status"] = "missing_source"
            decision["dest_path"] = ""

    copy_results_csv = reports_dir / "strategy_b_subsample_copy_results.csv"
    write_csv(
        copy_results_csv,
        decisions,
        [
            "filename",
            "status",
            "owner_method",
            "owner_cluster",
            "owner_rule",
            "owner_reason",
            "candidate_count",
            "candidates",
            "owner_audio_path",
            "source_path",
            "dest_path",
            "copy_status",
        ],
    )

    print(f"Rows scanned across subsample CSVs: {scanned_rows}")
    print(f"Assigned unique files: {len(decisions)}")
    print(f"Copied: {copied}")
    print(f"Missing source files: {missing_source}")
    print(f"Assignments CSV: {assignments_csv}")
    print(f"Memberships CSV: {memberships_csv}")
    print(f"Copy results CSV: {copy_results_csv}")


if __name__ == "__main__":
    main()
