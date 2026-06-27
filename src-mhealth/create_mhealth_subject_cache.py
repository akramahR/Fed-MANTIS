# Usage:
#   python src-mhealth/create_mhealth_subject_cache.py

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_mhealth")
    parser.add_argument("--emb-name", type=str, default="X_mantis.npy")
    parser.add_argument("--out-dir-name", type=str, default="mantis_subject_cache")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    X_PATH = data_dir / args.emb_name
    Y_PATH = data_dir / "y.npy"
    SUBJECTS_PATH = data_dir / "subjects.npy"
    OUT_DIR = data_dir / args.out_dir_name

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    subjects = np.load(SUBJECTS_PATH).astype(np.int64)

    if X.ndim != 2:
        raise ValueError(f"Expected X_mantis [N,D], got {X.shape}")

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            f"Length mismatch: X={len(X)}, y={len(y)}, subjects={len(subjects)}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_subjects = sorted(np.unique(subjects).astype(int).tolist())

    print("X_mantis:", X.shape)
    print("subjects:", unique_subjects)

    manifest = {
        "dataset": "MHEALTH",
        "X_mantis": str(X_PATH),
        "y": str(Y_PATH),
        "subjects": str(SUBJECTS_PATH),
        "embedding_shape": list(X.shape),
        "num_subjects": len(unique_subjects),
        "subjects": unique_subjects,
        "files": {},
    }

    for sid in unique_subjects:
        idx = subjects == sid

        X_sub = X[idx].astype(np.float32)
        y_sub = y[idx].astype(np.int64)

        out_path = OUT_DIR / f"subject_{sid}.npz"

        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{out_path} already exists. Use --overwrite to replace."
            )

        np.savez_compressed(
            out_path,
            X=X_sub,
            y=y_sub,
            sid=np.asarray(sid, dtype=np.int64),
        )

        labels, counts = np.unique(y_sub, return_counts=True)

        manifest["files"][str(sid)] = {
            "path": str(out_path),
            "num_windows": int(len(X_sub)),
            "label_counts": {
                str(int(k)): int(v)
                for k, v in zip(labels, counts)
            },
        }

        print(f"Saved {out_path} | X={X_sub.shape}, y={y_sub.shape}")

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:", manifest_path)
    print("Done.")


if __name__ == "__main__":
    main()