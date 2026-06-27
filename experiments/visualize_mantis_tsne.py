import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ===============================
# Activity Names (PAMAP2)
# ===============================
PAMAP2_LABELS = {
    0: "Lying",
    1: "Sitting",
    2: "Standing",
    3: "Walking",
    4: "Running",
    5: "Cycling",
    6: "Nordic Walking",
    7: "Ascending Stairs",
    8: "Descending Stairs",
    9: "Vacuum Cleaning",
    10: "Ironing",
    11: "Rope Jumping"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_raw():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    print(f"Raw X: {X.shape}, y: {y.shape}")
    return X, y


def load_mantis():
    path = os.path.join(DATA_DIR, "X_mantis.npy")
    print(f" Loading Mantis embeddings ")
    if not os.path.exists(path):
        raise FileNotFoundError("X_mantis.npy not found. Generate Mantis embeddings first.")
    X_mantis = np.load(path)
    print(f"Mantis embeddings: {X_mantis.shape}")
    return X_mantis


def plot_legends(ax, class_ids):
    """Adds a legend with activity names instead of numeric labels."""
    handles = []
    for class_id in class_ids:
        color = plt.cm.tab20(class_id % 20 / 20)
        label = PAMAP2_LABELS.get(class_id, f"Class {class_id}")
        handles.append(plt.Line2D([], [], marker='o', color=color, linestyle='', label=label))
    ax.legend(handles=handles, title="Activities", bbox_to_anchor=(1.05, 1), loc='upper left')


def tsne_and_plot(X, y, title, save_name):
    print(f"\n[{title}] Original input shape: {X.shape}")

    # 1) Flatten if needed
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # 2) Normalize
    X = StandardScaler().fit_transform(X)

    # 3) Subsample
    max_points = 10000
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub, y_sub = X, y

    # 4) Silhouette
    try:
        sil = silhouette_score(X_sub, y_sub)
        print(f"[{title}] Silhouette Score: {sil:.4f}")
    except:
        pass

    # 5) t-SNE
    print(f"[{title}] Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, init="pca", random_state=42)
    X_2d = tsne.fit_transform(X_sub)

    # 6) PLOT WITH CORRECT COLORS
    fig, ax = plt.subplots(figsize=(12, 10))  # Bigger size for legend

    unique_labels = np.unique(y_sub)
    # Get distinct colors from tab20
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label_id in enumerate(unique_labels):
        # Find all points for this specific activity
        mask = y_sub == label_id

        # Get the friendly name
        activity_name = PAMAP2_LABELS.get(label_id, f"Class {label_id}")

        # Plot ONLY this activity's points
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[colors[i]],  # Force this specific color
            label=activity_name,  # Auto-generates correct legend
            s=15,
            alpha=0.7
        )

    ax.set_title(title, fontsize=14)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

    ax.grid(alpha=0.3)

    # Legend is now guaranteed to be correct
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Activities")

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, f"{save_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"[{title}] Saved to {out_path}\n")

if __name__ == "__main__":
    # Load
    X_raw, y = load_raw()

    # 1) Raw Sensor Space
    tsne_and_plot(X_raw, y, "PAMAP2 Raw Sensor Space", "tsne_raw")

    # 2) Mantis Embeddings
    X_mantis = load_mantis()
    if X_mantis.ndim == 3:
        print("Pooling Mantis embeddings over time...")
        X_mantis = X_mantis.mean(axis=1)

    tsne_and_plot(X_mantis, y, "Mantis Embedding Space", "tsne_mantis")
