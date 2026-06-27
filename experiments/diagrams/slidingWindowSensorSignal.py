import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Parameters
# -----------------------------
fs = 100
window_len = 512
step = 256
num_windows = 4

total_samples = window_len + step * (num_windows - 1)
samples = np.arange(total_samples)
t = samples / fs

# -----------------------------
# Create sensor-like signal
# -----------------------------
np.random.seed(7)

signal = (
    0.9 * np.sin(2 * np.pi * 1.1 * t)
    + 0.35 * np.sin(2 * np.pi * 2.8 * t + 0.8)
    + 0.15 * np.random.randn(len(t))
)

signal = (signal - signal.mean()) / signal.std()

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(13, 4.5))

ax.plot(samples, signal, linewidth=1.6)

y_min = signal.min() - 0.6
y_max = signal.max() + 0.7
rect_height = y_max - y_min

# Draw windows
for i in range(num_windows):
    start = i * step

    rect = Rectangle(
        (start, y_min),
        window_len,
        rect_height,
        linewidth=1.8,
        edgecolor="black",
        facecolor="none"
    )
    ax.add_patch(rect)



    ax.axvline(start, linestyle="--", linewidth=0.9, alpha=0.7)

# End boundary
ax.axvline(total_samples, linestyle="--", linewidth=0.9, alpha=0.7)

# -----------------------------
# Styling
# -----------------------------
ax.set_title(
    "",
    fontsize=15,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Time / samples")
ax.set_ylabel("Sensor value")

ax.set_xlim(-20, total_samples + 20)
ax.set_ylim(y_min - 0.2, y_max + 0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# Save outputs
plt.savefig("sliding_window_clean_no_overlap.png", dpi=300, bbox_inches="tight")
plt.savefig("sliding_window_clean_no_overlap.pdf", bbox_inches="tight")
plt.savefig("sliding_window_clean_no_overlap.svg", bbox_inches="tight")

plt.show()