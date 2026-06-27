# src-mhealth/extract_mhealth_mantis_embeddings.py
# Usage:
#   python src-mhealth/extract_mhealth_mantis_embeddings.py


import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


def resize_time_axis(X, target_len, mode):
    if X.ndim != 3:
        raise ValueError(f"Expected X [N,T,C], got {X.shape}")

    n, t, c = X.shape

    if t == target_len:
        return X.astype(np.float32)

    if mode == "repeat":
        if target_len % t == 0:
            factor = target_len // t
            return np.repeat(X, factor, axis=1).astype(np.float32)

        print("[WARN] repeat ratio is non-integer; falling back to interpolation.", flush=True)
        mode = "interpolate"

    if mode == "interpolate":
        old_idx = np.linspace(0.0, 1.0, t)
        new_idx = np.linspace(0.0, 1.0, target_len)
        out = np.empty((n, target_len, c), dtype=np.float32)

        for i in range(n):
            for ch in range(c):
                out[i, :, ch] = np.interp(new_idx, old_idx, X[i, :, ch])

        return out

    if mode == "pad":
        if t > target_len:
            return X[:, :target_len, :].astype(np.float32)

        pad_len = target_len - t
        return np.pad(
            X,
            ((0, 0), (0, pad_len), (0, 0)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32)

    if mode == "crop":
        if t < target_len:
            raise ValueError("crop mode requires input length >= target_len")
        return X[:, :target_len, :].astype(np.float32)

    raise ValueError(f"Unknown resize mode: {mode}")


def extract_embeddings(X_mantis, batch_size, device):
    print("Loading MANTIS...", flush=True)

    network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
    trainer = MantisTrainer(device=device, network=network)

    all_embeddings = []
    n = len(X_mantis)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X_mantis[start:end].astype(np.float32)

        print(f"Embedding batch {start}:{end} / {n}", flush=True)
        t0 = time.time()

        with torch.no_grad():
            emb = trainer.transform(xb)

        print(f"Finished batch {start}:{end} in {time.time() - t0:.2f}s", flush=True)

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()

        all_embeddings.append(emb.astype(np.float32))

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_mhealth")
    parser.add_argument("--x-name", type=str, default="X.npy")
    parser.add_argument("--out-name", type=str, default="X_mantis.npy")
    parser.add_argument("--target-len", type=int, default=512)
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="repeat",
        choices=["repeat", "interpolate", "pad", "crop"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    X_PATH = data_dir / args.x_name
    Y_PATH = data_dir / "y.npy"
    SUBJECTS_PATH = data_dir / "subjects.npy"
    OUT_PATH = data_dir / args.out_name

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    subjects = np.load(SUBJECTS_PATH).astype(np.int64)

    print("Loaded:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)

    X_resized = resize_time_axis(X, target_len=args.target_len, mode=args.resize_mode)
    print("After time adaptation:", X_resized.shape)

    X_mantis_input = np.transpose(X_resized, (0, 2, 1)).astype(np.float32)
    print("MANTIS input:", X_mantis_input.shape)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("device:", device, flush=True)

    embeddings = extract_embeddings(
        X_mantis=X_mantis_input,
        batch_size=args.batch_size,
        device=device,
    )

    np.save(OUT_PATH, embeddings)

    meta_path = data_dir / "mhealth_mantis_metadata.txt"
    with open(meta_path, "w") as f:
        f.write("dataset=MHEALTH\n")
        f.write(f"resize_mode={args.resize_mode}\n")
        f.write(f"target_len={args.target_len}\n")
        f.write(f"raw_X_shape={X.shape}\n")
        f.write(f"mantis_input_shape={X_mantis_input.shape}\n")
        f.write(f"embeddings_shape={embeddings.shape}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"device={device}\n")

    print("\nSaved embedding file:")
    print(" ", OUT_PATH)
    print("Embeddings:", embeddings.shape)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()