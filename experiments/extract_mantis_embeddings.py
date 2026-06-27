import numpy as np
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import os
import torch

# for centralized
# options: "loso", "random"
SPLIT_MODE = "random"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def extract_mantis_embeddings(
    X,
    output_path,
    model_name="paris-noah/Mantis-8M"
):
    # Validation check
    if X.shape[1] != 512:
        raise ValueError(
            f"Input mismatch! Expected seq_len=512, got {X.shape[1]}. "
            "Please re-run data loader."
        )

    network = Mantis8M(device=device).from_pretrained(model_name)
    trainer = MantisTrainer(device=device, network=network)

    # Input X: (N, 512, F)
    # MANTIS expects: (N, F, 512)
    X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)

    print("Extracting MANTIS embeddings (frozen)...")
    embeddings = trainer.transform(X_t)
    print("Embeddings shape:", embeddings.shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to: {output_path}")

    return embeddings


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if SPLIT_MODE == "loso":
        DATA_DIR = os.path.join(BASE_DIR, "data")
        X_PATH = os.path.join(DATA_DIR, "X.npy")
        OUT_PATH = os.path.join(DATA_DIR, "X_mantis.npy")

    elif SPLIT_MODE == "random":
        DATA_DIR = os.path.join(BASE_DIR, "data", "random")
        X_PATH = os.path.join(DATA_DIR, "X_random.npy")
        OUT_PATH = os.path.join(DATA_DIR, "X_mantis_random.npy")

    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    print(f"SPLIT_MODE: {SPLIT_MODE}")
    print(f"Loading X from: {X_PATH}")

    X = np.load(X_PATH)

    print(f"X shape: {X.shape}")
    extract_mantis_embeddings(X, output_path=OUT_PATH)