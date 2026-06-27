import os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def evaluate_embeddings(X, y, name="Mantis"):
    print(f"\n--- Evaluating {name} Embeddings ---")

    # Standardize features (crucial for SVM)
    scaler = StandardScaler()

    # 1. Random Forest (Non-linear, robust to scaling)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 2. Linear SVM (Checks if the embeddings are linearly separable)
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(dual=False, max_iter=1000, random_state=42))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for clf, clf_name in [(rf, "Random Forest"), (svm, "Linear SVM")]:
        scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        print(f"{clf_name} Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")


if __name__ == "__main__":
    # Load labels
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    # Load and prep Mantis
    X_mantis = np.load(os.path.join(DATA_DIR, "X_mantis.npy"))
    if X_mantis.ndim == 3:
        X_mantis = X_mantis.mean(axis=1)  # Global Average Pooling

    evaluate_embeddings(X_mantis, y, "Mantis")

    # Optional: Load and prep Raw for comparison
    # X_raw = np.load(os.path.join(DATA_DIR, "X.npy")).reshape(len(y), -1)
    # evaluate_embeddings(X_raw, y, "Raw")