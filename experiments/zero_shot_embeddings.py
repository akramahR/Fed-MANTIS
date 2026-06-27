import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel
from uni2ts.model.moirai import MoiraiModule
import os
from sklearn.preprocessing import StandardScaler


device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Load preprocessed PAMAP2 data
# ===============================
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    print(f"Loaded PAMAP2 data: X={X.shape}, y={y.shape}")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ===============================
# Raw embedding (baseline)
# ===============================
def extract_raw_embeddings(X):
    """Flatten raw sensor sequences into fixed-length vectors and normalize."""
    X_flat = X.view(X.shape[0], -1).cpu().numpy()
    X_norm = StandardScaler().fit_transform(X_flat)
    return X_norm


# ===============================
# Chronos embeddings
# ===============================
def extract_chronos_embeddings(X, model_name="amazon/chronos-t5-small"):
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeds = []
    with torch.no_grad():
        for batch_start in range(0, len(X), 32):
            batch = X[batch_start:batch_start + 32].to(device)  # [B, seq, features]
            batch_mean = batch.mean(dim=1)  # compress sequence (rough token mapping)
            emb = model.encoder.embed_tokens(batch_mean.int()).mean(dim=1)
            all_embeds.append(emb.cpu().numpy())
    return np.vstack(all_embeds)


# ===============================
# Moirai embeddings
# ===============================
def extract_moirai_embeddings(X, pretrained_model="Salesforce/moirai-1.0-R-small"):
    model = MoiraiModule.from_pretrained(pretrained_model).to(device)
    model.eval()
    d_model = model.d_model  # hidden dimension, typically 384

    # Feature projection: 51 → d_model
    projector = torch.nn.Linear(X.shape[2], d_model).to(device)

    all_embeds = []
    with torch.no_grad():
        for batch_start in range(0, len(X), 32):
            batch = X[batch_start:batch_start + 32].to(device)  # [B, 256, 51]
            batch_proj = projector(batch)                       # [B, 256, 384]
            outputs = model.encoder(batch_proj)                 # encoded sequence
            pooled = outputs.mean(dim=1)
            all_embeds.append(pooled.cpu().numpy())

    return np.vstack(all_embeds)


# ===============================
# Visualization
# ===============================
def plot_tsne(embeddings, labels, title):
    print(f"Running t-SNE on {embeddings.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                init="random", random_state=42)
    X_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                          cmap="tab10", s=8, alpha=0.8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    X, y = load_data()
    print("Unique activity labels:", np.unique(y), "Count:", len(np.unique(y)))

    # Raw flattened features
    print("\nExtracting raw feature embeddings...")
    emb_raw = extract_raw_embeddings(X)
    plot_tsne(emb_raw, y.numpy(), "Raw Sensor Space (PAMAP2)")

    # 2Moirai embeddings
    print("\nExtracting Moirai embeddings...")
    emb_moirai = extract_moirai_embeddings(X)
    plot_tsne(emb_moirai, y.numpy(), "Moirai Zero-Shot Representations (PAMAP2)")

    # Optional Chronos (commented if memory limited)
    # print("\nExtracting Chronos embeddings...")
    # emb_chronos = extract_chronos_embeddings(X)
    # plot_tsne(emb_chronos, y.numpy(), "Chronos Zero-Shot Representations (PAMAP2)")
