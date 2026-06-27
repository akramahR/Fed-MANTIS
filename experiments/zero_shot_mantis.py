import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# Load PAMAP2 data
# ===============================
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    print(f"Loaded PAMAP2 data: X={X.shape}, y={y.shape}")
    return X, y


# ===============================
# Extract Mantis embeddings
# ===============================
def extract_mantis_embeddings(X, model_name="paris-noah/Mantis-8M"):
    print(f"Loading Mantis model: {model_name}")
    network = Mantis8M(device=device)
    network = network.from_pretrained(model_name)
    trainer = MantisTrainer(device=device, network=network)

    # Mantis expects input length ≈512; resize if needed
    seq_len = 512
    if X.shape[1] != seq_len:
        print(f"Resizing sequence length from {X.shape[1]} → {seq_len}")
        X_resized = np.zeros((X.shape[0], seq_len, X.shape[2]), dtype=np.float32)
        for i in range(X.shape[0]):
            for f in range(X.shape[2]):
                X_resized[i, :, f] = np.interp(
                    np.linspace(0, X.shape[1]-1, seq_len),
                    np.arange(X.shape[1]),
                    X[i, :, f]
                )
        X = X_resized

    print("Extracting Mantis embeddings (zero-shot)...")
    # Mantis expects input as [batch, channels, seq_len]
    X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)  # (N, 51, 512)
    embeddings = trainer.transform(X_t)  # (N, embedding_dim)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


# ===============================
# t-SNE Visualization
# ===============================
def plot_tsne(embeddings, labels, title):
    print(f"Running t-SNE on {embeddings.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="random", random_state=42)
    X_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    X, y = load_data()

    emb_mantis = extract_mantis_embeddings(X)
    np.save("data/embeddings_mantis.npy", emb_mantis)

    plot_tsne(emb_mantis, y, "Mantis Zero-Shot Representations (PAMAP2)")
