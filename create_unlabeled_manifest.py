#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def add_audio_props(df, sr=None):
    import librosa  # lazy import
    durations = []
    sample_rates = []
    for p in df["audio_path"]:
        y, sr_loaded = librosa.load(p, sr=sr, mono=True)
        durations.append(librosa.get_duration(y=y, sr=sr_loaded))
        sample_rates.append(sr_loaded)
    df["duration_sec"] = durations
    df["sample_rate"] = sample_rates
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="Path to index CSV (audio_path,embedding_path).")
    ap.add_argument("--manifest-out", default="../data/dataset/unlabeled_manifest.csv",
                    help="Where to save the manifest CSV.")
    ap.add_argument("--embeddings-out", default="../data/dataset/unlabeled_embeddings.npy",
                    help="Where to save stacked embeddings.")
    ap.add_argument("--add-audio-props", action="store_true",
                    help="Also load audio to add duration/sample_rate (slower).")
    args = ap.parse_args()

    df = pd.read_csv(args.index, header=0, low_memory=False)
    
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"Initial rows: {len(df)}")

    # Normalize audio column name so downstream code always uses audio_path
    audio_candidates = ["audio_path", "original_audio", "segment_path"]
    audio_col = next((c for c in audio_candidates if c in df.columns), None)
    if audio_col is None:
        raise KeyError(f"Expected one of {audio_candidates} in the CSV for audio paths.")
    if audio_col != "audio_path":
        df["audio_path"] = df[audio_col]
        print(f"Using '{audio_col}' as audio_path")

    # Clean empty/NaN rows and strip whitespace
    df = df.replace({r"^\s*$": np.nan}, regex=True)
    df["embedding_path"] = df["embedding_path"].astype(str).str.strip()
    df["audio_path"] = df["audio_path"].astype(str).str.strip()
    
    # Drop rows with empty/missing embedding_path or audio_path
    df = df.dropna(subset=["embedding_path", "audio_path"])
    df = df[(df["embedding_path"] != "") & (df["audio_path"] != "")]
    
    print(f"After cleaning: {len(df)} rows")
    print(f"First 3 embedding paths:")
    for i, p in enumerate(df["embedding_path"].head(3)):
        exists = os.path.isfile(p)
        print(f"  [{i}] exists={exists}: {p}")

    rows = []
    emb_list = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings"):
        emb_path = row["embedding_path"]
        try:
            arr = np.load(emb_path, allow_pickle=False)
        except Exception as e:
            print(f"Skip (load error): {emb_path} ({e})")
            skipped += 1
            continue

        if arr.ndim == 1 and arr.shape[0] == 1536:
            rows.append(row)
            emb_list.append(arr.astype(np.float32))
        elif arr.ndim == 3 and arr.shape[-1] == 1536:
            emb = arr.mean(axis=(0, 1)).astype(np.float32)
            rows.append(row)
            emb_list.append(emb)
        else:
            print(f"Skip (shape {arr.shape}): {emb_path}")
            skipped += 1

    if not emb_list:
        raise RuntimeError("No valid embeddings found.")

    # Quick sanity check before saving
    print(f"\nSanity check before saving:")
    print(f"  Rows collected: {len(rows)}")
    print(f"  Embeddings collected: {len(emb_list)}")
    print(f"  First embedding shape: {emb_list[0].shape if emb_list else 'N/A'}")
    print(f"  All embeddings same shape: {len(set(e.shape for e in emb_list)) == 1}")

    manifest = pd.DataFrame(rows).reset_index(drop=True)
    manifest["file_name"] = manifest["audio_path"].apply(lambda p: os.path.basename(p))
    manifest["embedding_dim"] = [1536] * len(manifest)

    if args.add_audio_props:
        manifest = add_audio_props(manifest)

    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.embeddings_out), exist_ok=True)

    manifest.to_csv(args.manifest_out, index=False)
    np.save(args.embeddings_out, np.stack(emb_list, axis=0))

    print(f"Saved manifest: {args.manifest_out}")
    print(f"Saved embeddings: {args.embeddings_out}")
    print(f"Total kept: {len(manifest)}, skipped: {skipped}")

if __name__ == "__main__":
    main()