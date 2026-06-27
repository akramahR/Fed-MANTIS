import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ===============================
# Labels
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

np.random.seed(42)

# ===============================
# Load data
# ===============================
X_raw = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

X_mantis = np.load(os.path.join(DATA_DIR, "X_mantis.npy"))
if X_mantis.ndim == 3:
    X_mantis = X_mantis.mean(axis=1)

print(f"Raw data shape: {X_raw.shape}")
print(f"MANTIS embedding shape: {X_mantis.shape}")
print(f"Labels shape: {y.shape}")

# ===============================
# Preprocess
# ===============================
def preprocess(X):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return StandardScaler().fit_transform(X)

X_raw = preprocess(X_raw)
X_mantis = preprocess(X_mantis)

# ===============================
# Subsample for speed (optional)
# ===============================
max_points = 5000
if len(X_raw) > max_points:
    idx = np.random.choice(len(X_raw), max_points, replace=False)
    X_raw = X_raw[idx]
    X_mantis = X_mantis[idx]
    y = y[idx]
    print(f"Subsampled to {max_points} points for t-SNE")

# ===============================
# t-SNE
# ===============================
def run_tsne(X, title):
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate=200,
        init="pca",
        random_state=42
    )
    return tsne.fit_transform(X)

X_raw_2d = run_tsne(X_raw, "Raw Sensor Space")
X_mantis_2d = run_tsne(X_mantis, "MANTIS Embedding Space")

# ===============================
# Plot settings
# ===============================
unique_labels = np.unique(y)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

fig, axes = plt.subplots(2, 1, figsize=(6, 11))

def plot_tsne(ax, X_2d, labels, title, show_xlabel=True):
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=0.7,
            c=[colors[i]]
        )

    ax.set_title(title, fontsize=11, pad=10)

    # Y label always
    ax.set_ylabel("t-SNE Component 2", fontsize=10)


    ax.set_xlabel("t-SNE Component 1", fontsize=10)

    ax.grid(alpha=0.3)

# Top plot (no x-label)
plot_tsne(axes[0], X_raw_2d, y, "(a) Raw Sensor Space", show_xlabel=False)

# Bottom plot (with x-label)
plot_tsne(axes[1], X_mantis_2d, y, "(b) MANTIS Embedding Space", show_xlabel=True)

# ===============================
# Shared legend
# ===============================
handles = []
for i, label in enumerate(unique_labels):
    handles.append(
        plt.Line2D(
            [],
            [],
            marker='o',
            linestyle='',
            color=colors[i],
            label=PAMAP2_LABELS[label]
        )
    )

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.02),
    title="Activities",
    fontsize=9,
    title_fontsize=10
)

fig.subplots_adjust(hspace=0.4, bottom=0.22)

# ===============================
# Layout
# ===============================
fig.subplots_adjust(hspace=0.4, bottom=0.18)

# ===============================
# Save
# ===============================
out_path = os.path.join(DATA_DIR, "tsne_comparison_vertical.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_path}")