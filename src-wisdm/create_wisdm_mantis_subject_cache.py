import os
import numpy as np


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(BASE_DIR, "data_wisdm")
    CACHE_DIR = os.path.join(DATA_DIR, "mantis_subject_cache")

    os.makedirs(CACHE_DIR, exist_ok=True)

    X_PATH = os.path.join(DATA_DIR, "X_mantis.npy")
    Y_PATH = os.path.join(DATA_DIR, "y.npy")
    SUBJECTS_PATH = os.path.join(DATA_DIR, "subjects.npy")

    if not os.path.exists(X_PATH):
        raise RuntimeError(f"Missing MANTIS embeddings: {X_PATH}. Run extract_wisdm_mantis_embeddings.py first.")

    if not os.path.exists(Y_PATH):
        raise RuntimeError(f"Missing labels: {Y_PATH}. Run preprocess_wisdm.py first.")

    if not os.path.exists(SUBJECTS_PATH):
        raise RuntimeError(f"Missing subjects: {SUBJECTS_PATH}. Run preprocess_wisdm.py first.")

    print("Loading WISDM MANTIS embeddings...")
    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    subjects = np.load(SUBJECTS_PATH).astype(np.int64)

    print("X_mantis:", X.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(subjects)={len(subjects)}"
        )

    unique_subjects = sorted(np.unique(subjects).tolist())
    print("Subjects:", unique_subjects)

    for sid in unique_subjects:
        idx = subjects == sid

        X_sub = X[idx]
        y_sub = y[idx]

        out_path = os.path.join(CACHE_DIR, f"subject_{int(sid)}.npz")

        np.savez_compressed(
            out_path,
            X=X_sub.astype(np.float32),
            y=y_sub.astype(np.int64),
            sid=int(sid),
        )

        print(f"Saved subject {int(sid)} | X={X_sub.shape}, y={y_sub.shape}")

    print("\nDone.")
    print("Cache directory:", CACHE_DIR)


if __name__ == "__main__":
    main()