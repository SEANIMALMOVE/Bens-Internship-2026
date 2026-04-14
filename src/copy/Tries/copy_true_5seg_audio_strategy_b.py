"""Create Strategy B ownership from sample_100_per_cluster, then copy audio files.

This mode does NOT impose a per-cluster sampling cap.

Workflow:
1) Build assignment CSV from sample_100_per_cluster.csv.
2) Copy each audio exactly once to the chosen owner method/cluster folder.
"""

from __future__ import annotations

from pathlib import Path
import csv
import re
import shutil

DIAGNOSTICS_CSV_CANDIDATES = [
    Path(r"C:\home\ben\Bens-Internship-2026\src\copy\sample_100_per_cluster.csv"),
    Path(
        r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\diagnostics\sample_100_per_cluster.csv"
    ),
]
SOURCE_DIR = Path(r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\5seg")
DEST_ROOT = Path(
    r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\diagnostics\ToReview_2026_04_08"
)

FILENAME_COLUMN = "filename"
TO_REVIEW_COLUMN = "ToReview"
USE_TO_REVIEW_ONLY = True

# Method order defines ownership priority when an audio matches multiple methods.
# Only methods with reviewable rules are included (skip all rule="none" methods).
METHOD_RULE_PRIORITY: list[dict[str, object]] = [
    {"method": "umap_3d_kmeans", "rule": "in_set", "clusters": [6, 15, 20, 21, 22]},
    {"method": "umap_3d_hdbscan", "rule": "in_set", "clusters": [0, 1, 3, 5]},
    {"method": "tsne_3d_hdbscan", "rule": "in_set", "clusters": [0, 1]},
    {"method": "umap_20d_hdbscan", "rule": "in_set", "clusters": [0, 1, 3, 4]},
    {"method": "tsne_20d_hdbscan", "rule": "all_non_noise", "clusters": None},
    {"method": "pca_3d_hdbscan", "rule": "in_set", "clusters": list(range(0, 25))},
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


def source_audio_from_filename(filename: str, source_dir: Path) -> Path:
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
    diagnostics_csv = None
    for candidate in DIAGNOSTICS_CSV_CANDIDATES:
        normalized = normalize_path(candidate)
        if normalized.exists():
            diagnostics_csv = normalized
            break

    dest_root.mkdir(parents=True, exist_ok=True)
    reports_dir = dest_root / "_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if diagnostics_csv is None:
        candidate_text = " | ".join(str(p) for p in DIAGNOSTICS_CSV_CANDIDATES)
        raise FileNotFoundError(f"Missing base CSV. Tried: {candidate_text}")

    # Phase 1: collect candidate memberships from sample_100_per_cluster base CSV.
    candidates_by_file: dict[str, list[dict[str, object]]] = {}
    memberships: list[dict[str, object]] = []
    unique_memberships: set[tuple[str, str, int]] = set()
    scanned_rows = 0

    with diagnostics_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            scanned_rows += 1

            filename = Path(str(row.get(FILENAME_COLUMN, "")).strip()).name
            if filename == "":
                continue

            to_review = is_true(str(row.get(TO_REVIEW_COLUMN, "")))
            if USE_TO_REVIEW_ONLY and not to_review:
                continue

            for priority, spec in enumerate(METHOD_RULE_PRIORITY):
                method = str(spec["method"])
                rule = str(spec["rule"])
                clusters = spec["clusters"]

                cluster_value = parse_cluster(row.get(method, ""))
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
                    "csv_row": row_idx,
                }
                candidates_by_file.setdefault(filename, []).append(candidate)

                memberships.append(
                    {
                        "filename": filename,
                        "method": method,
                        "cluster": cluster_int,
                        "rule": rule,
                        "csv_row": row_idx,
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
        owner_source_path = source_audio_from_filename(filename, source_dir)

        membership_owner_keys.add((filename, owner_method, owner_cluster))

        decisions.append(
            {
                "filename": filename,
                "status": "assigned",
                "owner_method": owner_method,
                "owner_cluster": owner_cluster,
                "owner_rule": owner_rule,
                "owner_reason": (
                    f"matched {len(candidates)} candidates from sample_100_per_cluster; "
                    f"picked highest-priority method={owner_method}"
                ),
                "candidate_count": len(candidates),
                "candidates": "|".join(
                    f"{c['method']}:{c['cluster']}({c['rule']})" for c in candidates
                ),
                "source_path": str(owner_source_path),
                "dest_path": "",
                "copy_status": "pending",
            }
        )

    for membership in memberships:
        key = (str(membership["filename"]), str(membership["method"]), int(membership["cluster"]))
        membership["is_owner"] = key in membership_owner_keys

    assignments_csv = reports_dir / "strategy_b_sample100_assignments.csv"
    memberships_csv = reports_dir / "strategy_b_sample100_memberships.csv"

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
            "csv_row",
            "is_owner",
        ],
    )

    # Phase 2b: build overlap verification reports.
    method_summary: list[dict[str, object]] = []
    methods = [str(spec["method"]) for spec in METHOD_RULE_PRIORITY]
    owned_by_method: dict[str, set[str]] = {method: set() for method in methods}
    candidates_by_method: dict[str, set[str]] = {method: set() for method in methods}

    for membership in memberships:
        filename = str(membership["filename"])
        method = str(membership["method"])
        candidates_by_method[method].add(filename)
        if bool(membership["is_owner"]):
            owned_by_method[method].add(filename)

    for method in methods:
        method_summary.append(
            {
                "method": method,
                "candidate_files": len(candidates_by_method[method]),
                "owned_files": len(owned_by_method[method]),
                "owner_rate": (
                    len(owned_by_method[method]) / len(candidates_by_method[method])
                    if candidates_by_method[method]
                    else 0
                ),
            }
        )

    method_overlap_rows: list[dict[str, object]] = []
    for left_method in methods:
        left_files = candidates_by_method[left_method]
        for right_method in methods:
            right_files = candidates_by_method[right_method]
            method_overlap_rows.append(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "shared_files": len(left_files & right_files),
                }
            )

    summary_csv = reports_dir / "strategy_b_sample100_method_summary.csv"
    overlap_csv = reports_dir / "strategy_b_sample100_method_overlap_matrix.csv"

    write_csv(
        summary_csv,
        method_summary,
        ["method", "candidate_files", "owned_files", "owner_rate"],
    )

    write_csv(
        overlap_csv,
        method_overlap_rows,
        ["left_method", "right_method", "shared_files"],
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

    copy_results_csv = reports_dir / "strategy_b_sample100_copy_results.csv"
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
            "source_path",
            "dest_path",
            "copy_status",
        ],
    )

    print(f"Rows scanned in base CSV: {scanned_rows}")
    print(f"Assigned unique files: {len(decisions)}")
    print(f"Copied: {copied}")
    print(f"Missing source files: {missing_source}")
    print(f"Assignments CSV: {assignments_csv}")
    print(f"Memberships CSV: {memberships_csv}")
    print(f"Copy results CSV: {copy_results_csv}")
    print(f"Method summary CSV: {summary_csv}")
    print(f"Method overlap CSV: {overlap_csv}")


if __name__ == "__main__":
    main()
