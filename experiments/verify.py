import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def verify_embeddings_FIXED():
    print("Loading data...")
    X_raw = np.load("data/X.npy")
    X_raw_flat = X_raw.reshape(X_raw.shape[0], -1)

    X_mantis = np.load("data/X_mantis.npy")
    y = np.load("data/y.npy")

    # Subset for speed
    idx = np.random.choice(len(y), 2000, replace=False)
    X_raw_sub = X_raw_flat[idx]
    X_mantis_sub = X_mantis[idx]
    y_sub = y[idx]

    print(f"Testing on {len(y_sub)} samples...")

    # Scale
    scaler = StandardScaler()
    Xr = scaler.fit_transform(X_raw_sub)
    Xm = scaler.fit_transform(X_mantis_sub)

    # ==========================================
    # THE FIX: Use the SAME random_state
    # ==========================================
    print("Splitting data (aligned)...")
    # Split Raw
    X_train_r, X_test_r, y_train, y_test = train_test_split(
        Xr, y_sub, test_size=0.2, random_state=42
    )

    # Split Mantis (Using SAME random_state=42 ensures indices match y_train)
    X_train_m, X_test_m, _, _ = train_test_split(
        Xm, y_sub, test_size=0.2, random_state=42
    )

    # Fit Raw
    print("Training Raw Logistic Regression...")
    clf_r = LogisticRegression(max_iter=200).fit(X_train_r, y_train)
    acc_r = clf_r.score(X_test_r, y_test)

    # Fit Mantis
    print("Training Mantis Logistic Regression...")
    clf_m = LogisticRegression(max_iter=200).fit(X_train_m, y_train)
    acc_m = clf_m.score(X_test_m, y_test)

    print(f"\nResults:")
    print(f"Raw Data Accuracy:    {acc_r * 100:.2f}%")
    print(f"Mantis Data Accuracy: {acc_m * 100:.2f}%")


if __name__ == "__main__":
    verify_embeddings_FIXED()