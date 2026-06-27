import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Results
# -----------------------------
models = [
    "CNN",
    "LSTM",
    "InceptionTime",
    "DeepConvLSTM",
    "MANTIS Linear",
    "MANTIS-SupCon",
]

random_acc = np.array([
    95.33,  # CNN
    93.46,  # LSTM
    96.26,  # InceptionTime
    96.26,  # DeepConvLSTM test/stable variant
    94.53,  # MANTIS Linear Probe
    96.26,  # MANTIS-SupCon 128/r32
])

loso_acc = np.array([
    82.07,  # CNN
    77.58,  # LSTM
    81.82,  # InceptionTime
    80.19,  # DeepConvLSTM
    93.13,  # MANTIS Linear Probe
    94.01,  # MANTIS-SupCon 128/r32
])

gaps = random_acc - loso_acc

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(10.8, 6.2))

bars_random = ax.bar(
    x - width / 2,
    random_acc,
    width,
    label="Random Split",
    color="#F2C94C",
    edgecolor="black",
    linewidth=0.6,
)

bars_loso = ax.bar(
    x + width / 2,
    loso_acc,
    width,
    label="LOSO",
    color="#27AE60",
    edgecolor="black",
    linewidth=0.6,
)

# Value labels inside bars
for bars in [bars_random, bars_loso]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 3.0,
            f"{height:.1f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

# Gap labels
gap_y = 109.5
for i, gap in enumerate(gaps):
    ax.text(
        x[i],
        gap_y,
        f"Gap {gap:.1f} pp",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#F7F7F7",
            edgecolor="black",
            linewidth=0.6,
        ),
    )

ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Model")
ax.set_title("Random-Split vs. LOSO Accuracy Comparison", pad=28)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right")

ax.set_ylim(0, 116)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    ncol=2,
    frameon=False,
)

ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# -----------------------------
# Save
# -----------------------------
os.makedirs("figures", exist_ok=True)

pdf_path = os.path.join("figures", "random_vs_loso_accuracy.pdf")
png_path = os.path.join("figures", "random_vs_loso_accuracy.png")

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()