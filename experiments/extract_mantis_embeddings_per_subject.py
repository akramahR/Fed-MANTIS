import os
import numpy as np
import torch
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


# use for federated scenarios


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def extract_subject_embeddings(trainer, X_sub, batch_size=512):
    """
    X_sub: (N, 512, F)
    returns embeddings: (N, D) numpy
    """
    # Safety Check
    if X_sub.shape[1] != 512:
        raise ValueError(f"Subject data mismatch! Expected 512 time steps, got {X_sub.shape[1]}")

    # Mantis expects (N, C, T) => (N, 51, 512)
    X_t = np.transpose(X_sub, (0, 2, 1)).astype(np.float32)

    embs = []
    n = X_t.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X_t[start:end]  # (B, 51, 512)

        # trainer.transform handles the forward pass (frozen)
        eb = trainer.transform(xb)  # (B, D)
        embs.append(eb)

    return np.concatenate(embs, axis=0)


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUT_DIR = os.path.join(DATA_DIR, "mantis_subject_cache")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    # Load Mantis once
    model_name = "paris-noah/Mantis-8M"
    network = Mantis8M(device=device).from_pretrained(model_name)
    trainer = MantisTrainer(device=device, network=network)

    unique_subjects = np.unique(subjects).tolist()
    print("Subjects found:", unique_subjects)

    for sid in unique_subjects:
        sid = int(sid)
        out_path = os.path.join(OUT_DIR, f"subject_{sid}.npz")

        # Optional: Check if file exists, maybe you want to overwrite now that logic changed?
        # If you want to force re-write, comment out the next 3 lines:
        # if os.path.exists(out_path):
        #     print(f"[cache] exists (skipping): {out_path}")
        #     continue

        idx = np.where(subjects == sid)[0]
        X_sub = X[idx]
        y_sub = y[idx]

        print(f"\n[cache] extracting subject {sid} | windows={len(idx)}")

        # No resizing needed anymore!
        emb = extract_subject_embeddings(trainer, X_sub, batch_size=512)

        print(f"[cache] subject {sid} embeddings: {emb.shape}")

        # Store embeddings + labels
        np.savez_compressed(out_path, X=emb.astype(np.float32), y=y_sub.astype(np.int64), sid=sid)
        print(f"[cache] saved: {out_path}")


if __name__ == "__main__":
    main()