import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from scipy.spatial.distance import cdist

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
    if not os.path.exists(path):
        raise FileNotFoundError("X_mantis.npy not found.")
    X_mantis = np.load(path)
    print(f"Mantis embeddings: {X_mantis.shape}")
    return X_mantis


def analyze_class_separation(X, y, title="Embedding Space"):
    print(f"\n=== {title}: Class Separation Analysis ===")

    # Standardize
    X = StandardScaler().fit_transform(X)

    # Overall silhouette
    overall_sil = silhouette_score(X, y)
    print(f"Overall silhouette score: {overall_sil:.4f}")

    # Per-sample silhouette -> class-wise mean silhouette
    sil_samples = silhouette_samples(X, y)
    unique_labels = np.unique(y)

    class_silhouette = {}
    for label in unique_labels:
        class_silhouette[label] = sil_samples[y == label].mean()

    print("\nPer-class silhouette scores:")
    sorted_scores = sorted(class_silhouette.items(), key=lambda x: x[1], reverse=True)
    for label, score in sorted_scores:
        print(f"{PAMAP2_LABELS[label]:20s}: {score:.4f}")

    # Easiest and hardest
    print("\nMost separable activities:")
    for label, score in sorted_scores[:3]:
        print(f"  {PAMAP2_LABELS[label]} ({score:.4f})")

    print("\nLeast separable activities:")
    for label, score in sorted_scores[-3:]:
        print(f"  {PAMAP2_LABELS[label]} ({score:.4f})")

    return class_silhouette


def compute_centroid_distances(X, y):
    print("\n=== Pairwise Centroid Distances ===")
    X = StandardScaler().fit_transform(X)
    unique_labels = np.unique(y)

    centroids = []
    for label in unique_labels:
        centroids.append(X[y == label].mean(axis=0))
    centroids = np.vstack(centroids)

    dist_matrix = cdist(centroids, centroids, metric="euclidean")

    # Print closest pairs
    pairs = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            pairs.append((unique_labels[i], unique_labels[j], dist_matrix[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    print("\nClosest activity pairs (hardest to separate):")
    for a, b, d in pairs_sorted[:5]:
        print(f"  {PAMAP2_LABELS[a]} <-> {PAMAP2_LABELS[b]} : {d:.4f}")

    print("\nFarthest activity pairs (easiest to separate):")
    for a, b, d in pairs_sorted[-5:]:
        print(f"  {PAMAP2_LABELS[a]} <-> {PAMAP2_LABELS[b]} : {d:.4f}")

    return dist_matrix, unique_labels


def nearest_centroid_confusion(X, y):
    print("\n=== Nearest-Centroid Confusion ===")
    X = StandardScaler().fit_transform(X)
    unique_labels = np.unique(y)

    # Compute centroids
    centroids = []
    for label in unique_labels:
        centroids.append(X[y == label].mean(axis=0))
    centroids = np.vstack(centroids)

    # Assign each sample to nearest centroid
    distances = cdist(X, centroids, metric="euclidean")
    pred_idx = np.argmin(distances, axis=1)
    pred_labels = unique_labels[pred_idx]

    cm = confusion_matrix(y, pred_labels, labels=unique_labels)

    print("\nMost confused classes:")
    confusion_pairs = []
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((unique_labels[i], unique_labels[j], cm[i, j]))

    confusion_pairs = sorted(confusion_pairs, key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in confusion_pairs[:10]:
        print(f"  {PAMAP2_LABELS[true_label]} -> {PAMAP2_LABELS[pred_label]} : {count}")

    return cm, unique_labels


def plot_tsne_with_labels(X, y, title):
    X = StandardScaler().fit_transform(X)

    print(f"\nRunning t-SNE for {title}...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate=200,
        init="pca",
        random_state=42
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(12, 9))
    unique_labels = np.unique(y)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=15,
            alpha=0.7,
            c=[colors[i]],
            label=PAMAP2_LABELS[label]
        )

        # plot class centroid in 2D
        cx = X_2d[mask, 0].mean()
        cy = X_2d[mask, 1].mean()
        plt.text(cx, cy, PAMAP2_LABELS[label], fontsize=9, weight='bold')

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_raw, y = load_raw()

    X_mantis = load_mantis()
    if X_mantis.ndim == 3:
        X_mantis = X_mantis.mean(axis=1)

    # Analyze MANTIS
    analyze_class_separation(X_mantis, y, title="MANTIS Embeddings")
    compute_centroid_distances(X_mantis, y)
    nearest_centroid_confusion(X_mantis, y)
    plot_tsne_with_labels(X_mantis, y, "MANTIS Embedding Space (PAMAP2)")