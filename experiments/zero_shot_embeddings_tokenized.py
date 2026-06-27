import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_quantized_data():
    Xq = np.load("../data/Xq.npy")  # quantized sequences
    y = np.load("../data/y.npy")
    print(f"Loaded quantized data: Xq={Xq.shape}, y={y.shape}")
    return torch.tensor(Xq, dtype=torch.long), torch.tensor(y, dtype=torch.long)
#FLATTENED
def extract_token_embeddings(Xq, model_name="amazon/chronos-t5-small"):
    """Extract embeddings from Chronos (encoder only) with progress tracking."""
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    batch_size = 8
    all_embeds = []

    print(f"Extracting embeddings using {model_name} on {device} ...")
    with torch.no_grad():
        for i in tqdm(range(0, len(Xq), batch_size), desc="Embedding batches", ncols=100):
            batch = Xq[i:i + batch_size].to(device)          # [B, 256, 51]
            batch = batch[:, ::4, :].reshape(batch.size(0), -1)  # downsample + flatten → [B, 3264]
            outputs = model.encoder(input_ids=batch)
            pooled = outputs.last_hidden_state.mean(dim=1)
            all_embeds.append(pooled.cpu().float().numpy())

    embeddings = np.vstack(all_embeds)
    print(f"✅ Extraction complete — embeddings shape: {embeddings.shape}")
    return embeddings




def plot_tsne(embeddings, labels, title):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

if __name__ == "__main__":
    Xq, y = load_quantized_data()
    embeddings = extract_token_embeddings(Xq)
    plot_tsne(embeddings, y.numpy(), "Chronos Tokenized Zero-Shot Embeddings (PAMAP2)")
