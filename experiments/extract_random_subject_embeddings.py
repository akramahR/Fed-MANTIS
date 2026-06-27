import os
import numpy as np

# ===== CONFIG =====
EMBED_DIM = 13056   # value is equal to MANTIS embedding dimension
RANDOM_SEED = 42
BATCH_SIZE = 512


def extract_random_embeddings(X_sub, embed_dim, rng, batch_size=512):

    if X_sub.shape[1] != 512:
        raise ValueError(f"Expected 512 time steps, got {X_sub.shape[1]}")

    n = X_sub.shape[0]
    embs = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start

        # Random Gaussian embeddings
        eb = rng.normal(size=(b, embed_dim)).astype(np.float32)
        embs.append(eb)

    return np.concatenate(embs, axis=0)


def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    OUT_DIR = os.path.join(DATA_DIR, "random_subject_cache")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

    rng = np.random.default_rng(RANDOM_SEED)

    unique_subjects = np.unique(subjects).tolist()
    print("Subjects found:", unique_subjects)

    for sid in unique_subjects:

        sid = int(sid)
        out_path = os.path.join(OUT_DIR, f"subject_{sid}.npz")

        idx = np.where(subjects == sid)[0]

        X_sub = X[idx]
        y_sub = y[idx]

        print(f"\n[random] extracting subject {sid} | windows={len(idx)}")

        emb = extract_random_embeddings(
            X_sub,
            embed_dim=EMBED_DIM,
            rng=rng,
            batch_size=BATCH_SIZE
        )

        print(f"[random] subject {sid} embeddings: {emb.shape}")

        np.savez_compressed(
            out_path,
            X=emb.astype(np.float32),
            y=y_sub.astype(np.int64),
            sid=sid
        )

        print(f"[random] saved: {out_path}")


if __name__ == "__main__":
    main()