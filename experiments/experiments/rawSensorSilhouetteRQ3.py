import os
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist

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
    11: "Rope Jumping",
}

# Script is inside experiments/, so project root is one folder up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_raw():
    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    print("Looking for data in:", DATA_DIR)
    print("X path:", x_path)

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Raw X: {X.shape}, y: {y.shape}")
    return X, y


def flatten_raw_windows(X_raw):
    """
    Converts raw sensor windows from:
        (N, 512, 51)
    to:
        (N, 26112)


    """
    X_flat = X_raw.reshape(X_raw.shape[0], -1).astype(np.float32)
    print(f"Flattened raw shape: {X_flat.shape}")
    return X_flat


def analyze_class_separation_direct(X, y, title="Raw Sensor Space"):
    print(f"\n=== {title}: Class Separation Analysis ===")

    overall_sil = silhouette_score(X, y, metric="euclidean")
    print(f"Overall silhouette score: {overall_sil:.4f}")

    sil_samples = silhouette_samples(X, y, metric="euclidean")
    unique_labels = np.unique(y)

    class_silhouette = {}

    print("\nPer-class silhouette scores:")
    for label in unique_labels:
        label = int(label)
        score = float(sil_samples[y == label].mean())
        class_silhouette[label] = score
        print(f"{PAMAP2_LABELS[label]:20s}: {score:.4f}")

    sorted_scores = sorted(class_silhouette.items(), key=lambda x: x[1], reverse=True)

    print("\nMost separable activities:")
    for label, score in sorted_scores[:3]:
        print(f"  {PAMAP2_LABELS[label]} ({score:.4f})")

    print("\nLeast separable activities:")
    for label, score in sorted_scores[-3:]:
        print(f"  {PAMAP2_LABELS[label]} ({score:.4f})")

    return class_silhouette, overall_sil


def compute_centroid_distances_direct(X, y):
    print("\n=== Pairwise Centroid Distances Directly in Raw Flattened Space ===")

    unique_labels = np.unique(y)

    centroids = []
    for label in unique_labels:
        centroids.append(X[y == label].mean(axis=0))

    centroids = np.vstack(centroids)
    dist_matrix = cdist(centroids, centroids, metric="euclidean")

    pairs = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            a = int(unique_labels[i])
            b = int(unique_labels[j])
            d = float(dist_matrix[i, j])
            pairs.append((a, b, d))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    print("\nClosest activity pairs:")
    for a, b, d in pairs_sorted[:8]:
        print(f"  {PAMAP2_LABELS[a]} <-> {PAMAP2_LABELS[b]}: {d:.4f}")

    print("\nFarthest activity pairs:")
    for a, b, d in pairs_sorted[-8:]:
        print(f"  {PAMAP2_LABELS[a]} <-> {PAMAP2_LABELS[b]}: {d:.4f}")

    return dist_matrix


if __name__ == "__main__":
    X_raw, y = load_raw()

    X_flat = flatten_raw_windows(X_raw)

    analyze_class_separation_direct(
        X_flat,
        y,
        title="Raw Sensor Windows after Flattening"
    )

    compute_centroid_distances_direct(X_flat, y)