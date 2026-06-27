import os
import numpy as np
import torch
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    print("Loading X...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))

    print("Input shape:", X.shape)  # expected (N, 512, F)

    # Take a small sample to test
    sample = X[:32]

    if sample.shape[1] != 512:
        raise ValueError(f"Expected 512 timesteps, got {sample.shape[1]}")

    # MANTIS expects (N, C, T)
    sample_t = np.transpose(sample, (0, 2, 1)).astype(np.float32)

    print("Transposed shape:", sample_t.shape)

    print("Loading MANTIS model...")
    model_name = "paris-noah/Mantis-8M"
    network = Mantis8M(device=device).from_pretrained(model_name)
    trainer = MantisTrainer(device=device, network=network)

    print("Running embedding extraction...")

    with torch.no_grad():
        embeddings = trainer.transform(sample_t)

    print("Embeddings shape:", embeddings.shape)

    # Print embedding dimension clearly
    print("Embedding dimension D =", embeddings.shape[1])
    print("Single embedding:", embeddings[0].shape)


if __name__ == "__main__":
    main()