import matplotlib.pyplot as plt
import numpy as np

# Data from your experiment
labels = ["Random + Linear", "Random + MLP", "MANTIS + Linear"]
accuracy = [9.84, 10.43, 91.98]

# Positions
x = np.arange(len(labels))

plt.figure(figsize=(6,4))
bars = plt.bar(x, accuracy)

# Labels
plt.ylabel("Accuracy (%)")
plt.title("Representation Ablation (LOSO)")
plt.xticks(x, labels, rotation=15)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.ylim(0, 100)
plt.tight_layout()

# Save figure for LaTeX
plt.savefig("representation_ablation.png", dpi=300, bbox_inches="tight")

plt.show()