#!/usr/bin/env python3
"""
perch_extract_embeddings.py
PerchV2 CPU embedding extractor with simplified logic:

- Assumes input audio ≈ 5 s.
- If audio ≈ 5 s → produce one embedding: <basename>.npy
- If audio < 5 s → zero-pad → <basename>.npy
- If audio > 5 s → extract single centered 5-s window → <basename>_startsec_endsec.npy
- No nested folder mirrors; all embeddings go to: output/embeddings/
- Optional: save 5 s segment WAVs
- Optional: diagnostics (spectrograms, t-SNE, distance heatmap)
"""


import os
import sys
import math
import argparse
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import time
import subprocess
import tempfile
import shutil
import audioread

from tqdm import tqdm
import tensorflow_hub as hub

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# --------------------------
# CONFIG
# --------------------------
# PERCH_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
# CPU version (required for non-GPU machines)
# PERCH_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
PERCH_MODEL_URL = "models/perch_v2/2"
# PERCH_MODEL_URL = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"


TARGET_SR = 32000
WINDOW_S = 5
WINDOW_SAMPLES = TARGET_SR * WINDOW_S

ROUGH_FIVE_SEC_MIN = TARGET_SR * 4.5
ROUGH_FIVE_SEC_MAX = TARGET_SR * 5.5


# --------------------------
# UTILITIES
# --------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_audio_files(root, exts=(".wav", ".flac", ".mp3", ".ogg")):
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]


def load_audio_safe(path, sr=TARGET_SR, mono=True):
    """Try to load audio with librosa/soundfile; if that fails (no backend or unsupported format),
    try to convert the file to a temporary WAV using `ffmpeg` (if available) and load that.

    Raises the original exception if no fallback is available or conversion fails.
    """
    try:
        return librosa.load(path, sr=sr, mono=mono)
    except Exception as orig_err:
        # If ffmpeg is available, try converting to WAV and load that
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_name = tmp.name
        tmp.close()

        cmd = [ffmpeg_path, "-y", "-i", str(path), "-ar", str(sr), "-ac", "1", tmp_name]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            y, sr_ret = librosa.load(tmp_name, sr=sr, mono=mono)
            return y, sr_ret
        except Exception:
            # conversion failed — re-raise original error for clarity
            raise orig_err
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass


def load_perch_model():
    print("Loading PerchV2 CPU model...")
    return hub.load(PERCH_MODEL_URL)


# --------------------------
# AUDIO WINDOW SELECTION
# --------------------------
def extract_single_window(path):
    """
    Returns:
        waveform (np.array, 5 s),
        output_name_suffix (string to append to basename)
    """
    y, sr = load_audio_safe(path, sr=TARGET_SR)
    n = len(y)

    # Case 1: 5 s audio (± tolerance)
    if ROUGH_FIVE_SEC_MIN <= n <= ROUGH_FIVE_SEC_MAX:
        y5 = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        y5[:min(n, WINDOW_SAMPLES)] = y[:WINDOW_SAMPLES]
        return y5, ""

    # Case 2: shorter than 5 s → zero pad
    if n < WINDOW_SAMPLES:
        y5 = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        y5[:n] = y
        return y5, ""

    # Case 3: longer than 5 s → take middle window
    mid = n // 2
    start = max(0, mid - WINDOW_SAMPLES // 2)
    end = start + WINDOW_SAMPLES
    if end > n:
        end = n
        start = end - WINDOW_SAMPLES
    y5 = y[start:end]
    return y5.astype(np.float32)#, f"_{start}_{end}"


# --------------------------
# MODEL INFERENCE
# --------------------------
import tensorflow as tf

def get_embedding(model, waveform_1d):
    inp = tf.convert_to_tensor(waveform_1d[np.newaxis, :], dtype=tf.float32)
    outs = model.signatures['serving_default'](inputs=inp)
    emb = outs['embedding'].numpy()[0]
    return emb.astype(np.float32)


# --------------------------
# EXTRA: diagnostics
# --------------------------
def save_spectrogram_preview(waveform, sr, outfile):
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=1024, hop_length=256)
    Sdb = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6, 4))
    plt.imshow(Sdb, aspect='auto', origin='lower')
    plt.title("Spectrogram preview")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def diagnostics_tsne_and_heatmap(emb_paths, outdir, max_points=5000):
    ensure_dir(outdir)
    if len(emb_paths) > max_points:
        emb_paths = random.sample(emb_paths, max_points)

    X = np.stack([np.load(p) for p in emb_paths], axis=0)

    # 2D t-SNE
    reducer = TSNE(n_components=2, init='pca', random_state=42)
    X_tsne = reducer.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.4)
    plt.title("t-SNE of embeddings")
    plt.tight_layout()
    plt.savefig(outdir / "tsne.png")
    plt.close()

    # 2D PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.4)
    plt.title("PCA of embeddings")
    plt.tight_layout()
    plt.savefig(outdir / "pca.png")
    plt.close()

    D = pairwise_distances(X, metric='euclidean')
    plt.figure(figsize=(8, 6))
    plt.imshow(D, cmap="viridis", aspect="auto")
    plt.title("Pairwise distances")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outdir / "heatmap.png")
    plt.close()


# --------------------------
# MAIN EXTRACTION
# --------------------------
def extract_embeddings(input_dir, out_dir, model,
                       save_segments=False,
                       enable_visuals=False,
                       max_previews=5):
    files = list_audio_files(input_dir)
    total_files = len(files)
    print(f"Found {total_files} audio files.")

    # We'll mirror the input folder structure under `out_dir`.
    # Per-file destination dirs are created on-the-fly below.
    ensure_dir(Path(out_dir))

    # Diagnostics spectrograms folder (global) when visuals are enabled
    diag_spec_dir = None
    if enable_visuals:
        diag_spec_dir = Path(out_dir) / "diagnostics" / "spectrograms"
        ensure_dir(diag_spec_dir)

    rows = []
    preview_count = 0

    skipped = 0
    processed = 0
    failed = 0
    failed_files = []

    start_time = time.time()

    for file in tqdm(files, desc="Embedding"):
        # Compute path relative to input root and mirror it under out_dir
        rel = Path(file).relative_to(Path(input_dir))
        dest_dir = Path(out_dir) / rel.parent
        ensure_dir(dest_dir)

        basename = Path(file).stem

        # Extract 5s window first (fast compared to model inference)
        try:
            seg, suffix = extract_single_window(file)
        except Exception as ex:
            failed += 1
            failed_files.append({
                "original_audio": file,
                "stage": "load",
                "error": str(ex)
            })
            continue

        emb_path = dest_dir / f"{basename}{suffix}.npy"

        # If embedding already exists, skip heavy work
        if emb_path.exists():
            skipped += 1
            continue

        # Compute embedding and save
        try:
            emb = get_embedding(model, seg)
        except Exception as ex:
            # Check if this is a CUDA/platform incompatibility error
            error_str = str(ex)
            if "platform CPU is not among the platforms required" in error_str or "CUDA" in error_str:
                print(f"\n❌ FATAL ERROR: Model requires CUDA but CPU-only mode is active.")
                print(f"Error: {error_str}")
                print("Stopping execution.")
                sys.exit(1)
            failed += 1
            failed_files.append({
                "original_audio": file,
                "stage": "inference",
                "error": str(ex)
            })
            continue

        try:
            np.save(emb_path, emb)
        except Exception as ex:
            failed += 1
            failed_files.append({
                "original_audio": file,
                "stage": "save_embedding",
                "error": str(ex)
            })
            continue

        seg_path = ""
        if save_segments:
            seg_path = dest_dir / f"{basename}{suffix}.wav"
            sf.write(seg_path, seg, TARGET_SR)

        rows.append({
            "original_audio": file,
            "embedding_path": str(emb_path),
            "segment_path": str(seg_path),
        })

        processed += 1

        # Save spectrogram preview into the diagnostics folder when requested
        if enable_visuals and preview_count < max_previews:
            # Create a unique filename based on the input file's relative path
            # e.g. sub/dir/file -> sub__dir__file.png
            rel_no_ext = rel.with_suffix("").as_posix()
            safe_name = rel_no_ext.replace('/', '__') + f"{suffix}.png"
            out_img = diag_spec_dir / safe_name
            save_spectrogram_preview(seg, TARGET_SR, out_img)
            preview_count += 1

    end_time = time.time()
    elapsed = end_time - start_time

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "index.csv", index=False)
    print(f"Index saved to: {Path(out_dir) / 'index.csv'}")

    # Write failed files log if any
    diag_dir = Path(out_dir) / "diagnostics"
    ensure_dir(diag_dir)
    if failed_files:
        failed_df = pd.DataFrame(failed_files)
        failed_log = diag_dir / "failed_files.csv"
        failed_df.to_csv(failed_log, index=False)
        print(f"Wrote failed files log to: {failed_log}")

    if enable_visuals and len(df) > 10:
        print("Generating t-SNE and heatmap…")
        diag_dir = Path(out_dir) / "diagnostics"
        # set max point all samples
        max_points = len(df) if len(df) > 5000 else 500
        diagnostics_tsne_and_heatmap(df["embedding_path"].tolist(), diag_dir, max_points=max_points)

    # Summary
    def _format_secs(s):
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:d}h {m:d}m {sec:d}s" if h else f"{m:d}m {sec:d}s"

    print("")
    print("Summary:")
    print(f"- Total files found: {total_files}")
    print(f"- Already processed (skipped due to existing embeddings): {skipped}")
    print(f"- Newly processed: {processed}")
    print(f"- Failed/skipped due to errors: {failed}")
    print(f"- Total processing time: {elapsed:.2f} seconds ({_format_secs(elapsed)})")

    print("Done.")
    return df


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--save-segments", action="store_true")
    parser.add_argument("--enable-visuals", action="store_true")
    parser.add_argument("--max-previews", type=int, default=5)

    args = parser.parse_args()
    model = load_perch_model()

    extract_embeddings(
        args.input,
        args.output,
        model,
        save_segments=args.save_segments,
        enable_visuals=args.enable_visuals,
        max_previews=args.max_previews
    )


if __name__ == "__main__":
    main()