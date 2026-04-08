### copy the 5-second audio files that were marked as "ToReview" in the diagnostics CSV to a separate folder for manual review. 

from pathlib import Path
import csv
import shutil
import re


CSV_PATH = Path(r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\diagnostics\sample_100_per_cluster.csv")
SOURCE_DIR = Path(r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\5seg")
DEST_DIR = Path(r"D:\Posidonia Soundscapes\Fondeo 1_Formentera Ille Espardell\Embeddings_2\diagnostics\ToReview_2026_03_25")

BOOL_COLUMN = "ToReview"
FILENAME_COLUMN = "filename"


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


def main() -> None:
    csv_path = normalize_path(CSV_PATH)
    source_dir = normalize_path(SOURCE_DIR)
    dest_dir = normalize_path(DEST_DIR)

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not is_true(row.get(BOOL_COLUMN, "")):
                continue

            name = Path(row[FILENAME_COLUMN]).name
            src = source_dir / name
            dst = dest_dir / name

            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1

    print(f"Copied: {copied}")
    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
