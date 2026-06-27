import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMB_DIR = os.path.join(BASE_DIR, "results", "rq3_embeddings")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)


def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"], d["subjects"]


def preprocess(X):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return StandardScaler().fit_transform(X)


def run_tsne(X, title):
    print(f"Running t-SNE for {title} | shape={X.shape}")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate=200,
        init="pca",
        random_state=42,
    )
    return tsne.fit_transform(X)


def plot_tsne(ax, X_2d, y, title, colors, unique_labels):
    for i, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=8,
            alpha=0.65,
            c=[colors[i]],
        )

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("t-SNE Component 1", fontsize=9)
    ax.set_ylabel("t-SNE Component 2", fontsize=9)
    ax.grid(alpha=0.25)


# ===============================
# Load raw and frozen MANTIS
# ===============================
X_raw = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

X_mantis = np.load(os.path.join(DATA_DIR, "X_mantis.npy"))
if X_mantis.ndim == 3:
    X_mantis = X_mantis.mean(axis=1)

subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"))

print("Raw:", X_raw.shape)
print("MANTIS:", X_mantis.shape)
print("Labels:", y.shape)

# ===============================
# Load adapted out-of-fold embeddings
# ===============================
lowrank_path = os.path.join(
    EMB_DIR, "adapter_lowrank_128_r32_oof_adapted_embeddings.npz"
)
supcon_path = os.path.join(
    EMB_DIR, "supcon_lowrank_128_r32_oof_adapted_embeddings.npz"
)

X_lowrank, y_lowrank, subjects_lowrank = load_npz(lowrank_path)
X_supcon, y_supcon, subjects_supcon = load_npz(supcon_path)

print("Low-rank adapted:", X_lowrank.shape)
print("SupCon adapted:", X_supcon.shape)

# Sanity check: labels should match after concatenating LOSO folds
# If this fails, we will use the labels from each NPZ separately.
same_lowrank_labels = np.array_equal(y, y_lowrank)
same_supcon_labels = np.array_equal(y, y_supcon)

print("Low-rank labels match y.npy:", same_lowrank_labels)
print("SupCon labels match y.npy:", same_supcon_labels)

# ===============================
# Subsample same indices for all spaces
# ===============================
max_points = 5000
N = len(y)

if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
else:
    idx = np.arange(N)

X_raw_sub = X_raw[idx]
X_mantis_sub = X_mantis[idx]
y_sub = y[idx]

# For adapted embeddings, order should match if folds were appended in subject order.
# If labels do not match, sort adapted embeddings by subject is more complex.
# In normal case, your subjects loop is sorted, so this should match.
X_lowrank_sub = X_lowrank[idx]
X_supcon_sub = X_supcon[idx]

if not np.array_equal(y_sub, y_lowrank[idx]):
    print("[WARN] Low-rank labels do not match selected y indices. Using y_lowrank for plotting.")
    y_lowrank_sub = y_lowrank[idx]
else:
    y_lowrank_sub = y_sub

if not np.array_equal(y_sub, y_supcon[idx]):
    print("[WARN] SupCon labels do not match selected y indices. Using y_supcon for plotting.")
    y_supcon_sub = y_supcon[idx]
else:
    y_supcon_sub = y_sub

# ===============================
# Preprocess for t-SNE
# ===============================
X_raw_sub = preprocess(X_raw_sub)
X_mantis_sub = preprocess(X_mantis_sub)
X_lowrank_sub = preprocess(X_lowrank_sub)
X_supcon_sub = preprocess(X_supcon_sub)

# ===============================
# Run t-SNE
# ===============================
X_raw_2d = run_tsne(X_raw_sub, "Raw Sensor Space")
X_mantis_2d = run_tsne(X_mantis_sub, "Frozen MANTIS Embedding Space")
X_lowrank_2d = run_tsne(X_lowrank_sub, "Low-Rank Adapted Space")
X_supcon_2d = run_tsne(X_supcon_sub, "SupCon Adapted MANTIS Space")

# ===============================
# Plot
# ===============================
unique_labels = np.unique(y_sub)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_tsne(
    axes[0, 0],
    X_raw_2d,
    y_sub,
    "(a) Raw Sensor Space",
    colors,
    unique_labels,
)

plot_tsne(
    axes[0, 1],
    X_mantis_2d,
    y_sub,
    "(b) Frozen MANTIS Embedding Space",
    colors,
    unique_labels,
)

plot_tsne(
    axes[1, 0],
    X_lowrank_2d,
    y_lowrank_sub,
    "(c) Low-Rank Adapted Space",
    colors,
    unique_labels,
)

plot_tsne(
    axes[1, 1],
    X_supcon_2d,
    y_supcon_sub,
    "(d) SupCon Adapted MANTIS Space",
    colors,
    unique_labels,
)

handles = []
for i, label in enumerate(unique_labels):
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=colors[i],
            label=PAMAP2_LABELS[int(label)],
        )
    )

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
    title="Activities",
    fontsize=8,
    title_fontsize=9,
)

fig.subplots_adjust(hspace=0.28, wspace=0.22, bottom=0.14)

out_path = os.path.join(FIG_DIR, "tsne_rq3_four_stage.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_path}")