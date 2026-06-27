import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Activity Names (PAMAP2) - Matching your previous dictionary
PAMAP2_LABELS = {
    0: "Lying", 1: "Sitting", 2: "Standing", 3: "Walking", 4: "Running",
    5: "Cycling", 6: "Nordic Walking", 7: "Ascending Stairs",
    8: "Descending Stairs", 9: "Vacuum Cleaning", 10: "Ironing", 11: "Rope Jumping"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    # Get unique labels present in the data
    unique_ids = np.unique(np.concatenate([y_true, y_pred]))
    tick_labels = [PAMAP2_LABELS.get(i, f"Class {i}") for i in unique_ids]

    cm = confusion_matrix(y_true, y_pred, labels=unique_ids)

    # Normalize by row (true labels) to see percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels)

    plt.title(f"{title} (Normalized)")
    plt.ylabel('True Activity')
    plt.xlabel('Predicted Activity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = os.path.join(DATA_DIR, "confusion_matrix_mantis.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Matrix saved to {save_path}")


if __name__ == "__main__":
    # 1. Load Data
    print("Loading Mantis embeddings...")
    X = np.load(os.path.join(DATA_DIR, "X_mantis.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    # 2. Preprocess (Global Average Pool if 3D)
    if X.ndim == 3:
        X = X.mean(axis=1)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Classifier
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = rf.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred,
                                target_names=[PAMAP2_LABELS[i] for i in np.unique(y)]))

    # 6. Visualize
    plot_confusion_matrix(y_test, y_pred, title="Mantis Embedding Performance")