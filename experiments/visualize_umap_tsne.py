import numpy as np
import matplotlib.pyplot as plt
import umap
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ===============================
# Configs for "Aesthetic" Plots
# ===============================
# n_neighbors: Larger = Focus on Global Structure (15-50)
# min_dist:    Smaller = Tighter Clusters (0.1 is standard, 0.05 is tighter)
# metric:      'cosine' is CRITICAL for high-dim embeddings (like Mantis)
N_NEIGHBORS = 30
MIN_DIST = 0.1
METRIC = 'cosine'


def load_data():
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # Use your best embeddings (the Zero-Shot ones from the frozen model)
    # If you have X_mantis.npy, use that.
    X = np.load(os.path.join(DATA_DIR, "X_mantis.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    return X, y


def plot_umap():
    print("Loading Data...")
    X, y = load_data()

    # 1. Safety Check: Flatten if 3D
    if X.ndim > 2:
        print(f"Flattening data from {X.shape}...")
        X = X.reshape(X.shape[0], -1)

    # 2. Normalize (Crucial for UMAP/t-SNE)
    print("Normalizing...")
    X = StandardScaler().fit_transform(X)

    # 3. Run UMAP
    print(f"Running UMAP (Neighbors={N_NEIGHBORS}, MinDist={MIN_DIST}, Metric={METRIC})...")
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_components=2,
        metric=METRIC,
        random_state=42  # Fixed seed for reproducible poster images
    )
    embedding = reducer.fit_transform(X)

    # 4. Plotting
    print("Generating Plot...")
    plt.figure(figsize=(12, 10))

    # Define Activity Labels (Update this map if needed)
    # Assuming PAMAP2 standard IDs:
    PAMAP2_LABELS = {
        1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking",
        5: "Running", 6: "Cycling", 7: "Nordic Walk",
        12: "Asc Stairs", 13: "Desc Stairs", 16: "Vacuum",
        17: "Ironing", 24: "Rope Jump"
    }

    unique_y = np.unique(y)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_y)))

    for i, cls in enumerate(unique_y):
        mask = y == cls
        label_text = PAMAP2_LABELS.get(cls, f"Class {cls}")
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=label_text,
            s=10,  # Dot size
            alpha=0.7  # Transparency
        )

    plt.title("UMAP Projection of Mantis Embeddings (Zero-Shot)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Activities")
    plt.axis('off')  # Clean look for poster (removes box/axes)
    plt.tight_layout()

    # Save high-res
    save_path = "data/umap_poster_ready.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved UMAP to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_umap()