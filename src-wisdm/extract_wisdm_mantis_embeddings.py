import os
import numpy as np
import torch

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer

# for centralized

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def resize_time_axis(X, target_len=512):
    """
    Resize time dimension from (N, T, C) to (N, target_len, C)
    using linear interpolation.
    """
    n, t, c = X.shape

    if t == target_len:
        return X.astype(np.float32)

    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, target_len)

    X_resized = np.empty((n, target_len, c), dtype=np.float32)

    for i in range(n):
        for ch in range(c):
            X_resized[i, :, ch] = np.interp(new_idx, old_idx, X[i, :, ch])

    return X_resized


def extract_mantis_embeddings(X, output_path, batch_size=512, model_name="paris-noah/Mantis-8M"):
    """
    X: (N, T, C), e.g. WISDM (57098, 128, 6)
    MANTIS input expected: (N, C, 512)
    """
    print("Original X shape:", X.shape)

    if X.shape[1] != 512:
        print(f"Resizing time axis from {X.shape[1]} to 512...")
        X = resize_time_axis(X, target_len=512)

    print("Resized X shape:", X.shape)

    network = Mantis8M(device=device).from_pretrained(model_name)
    trainer = MantisTrainer(device=device, network=network)

    X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)

    print("MANTIS input shape:", X_t.shape)
    print("Extracting MANTIS embeddings...")

    embeddings = []

    n = X_t.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X_t[start:end]

        print(f"Batch {start}:{end}")

        emb = trainer.transform(xb)
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)

    print("Embeddings shape:", embeddings.shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    print(f"Saved embeddings to: {output_path}")

    return embeddings


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(BASE_DIR, "data_wisdm")

    X_PATH = os.path.join(DATA_DIR, "X.npy")
    Y_PATH = os.path.join(DATA_DIR, "y.npy")
    SUBJECTS_PATH = os.path.join(DATA_DIR, "subjects.npy")
    OUT_PATH = os.path.join(DATA_DIR, "X_mantis.npy")

    print(f"Loading WISDM raw windows from: {X_PATH}")

    if not os.path.exists(X_PATH):
        raise RuntimeError(f"Missing X file: {X_PATH}. Run preprocess_wisdm.py first.")

    X = np.load(X_PATH).astype(np.float32)

    print("X:", X.shape)

    # Optional checks only
    if os.path.exists(Y_PATH):
        y = np.load(Y_PATH)
        print("y:", y.shape)

    if os.path.exists(SUBJECTS_PATH):
        subjects = np.load(SUBJECTS_PATH)
        print("subjects:", subjects.shape)

    extract_mantis_embeddings(
        X,
        output_path=OUT_PATH,
        batch_size=512,
    )

    print("Saved MANTIS embeddings:", OUT_PATH)


if __name__ == "__main__":
    main()