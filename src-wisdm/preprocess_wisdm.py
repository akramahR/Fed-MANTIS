import os
import re
import argparse
import numpy as np
import pandas as pd


# python src-wisdm\preprocess_wisdm.py --root wisdm-dataset --device watch --window 128 --stride 64

ACTIVITY_MAP = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}


def read_wisdm_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # WISDM lines often end with semicolon
            line = line.replace(";", "")

            parts = line.split(",")

            if len(parts) != 6:
                continue

            try:
                user = int(parts[0])
                activity = parts[1].strip()
                timestamp = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except ValueError:
                continue

            rows.append((user, activity, timestamp, x, y, z))

    return pd.DataFrame(rows, columns=["subject", "activity", "timestamp", "x", "y", "z"])


def extract_subject_id(filename):
    match = re.search(r"data_(\d+)_", filename)
    if not match:
        raise ValueError(f"Could not extract subject id from filename: {filename}")
    return int(match.group(1))


def load_sensor_folder(folder):
    dfs = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        df = read_wisdm_file(path)

        if len(df) == 0:
            print(f"Skipped empty/unreadable file: {path}")
            continue

        file_subject = extract_subject_id(filename)

        # Keep only rows matching filename subject, just in case
        df = df[df["subject"] == file_subject].copy()

        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid txt files found in {folder}")

    return pd.concat(dfs, ignore_index=True)


def align_accel_gyro(accel_df, gyro_df):
    merged_subjects = []

    for subject in sorted(accel_df["subject"].unique()):
        acc_sub = accel_df[accel_df["subject"] == subject].copy()
        gyr_sub = gyro_df[gyro_df["subject"] == subject].copy()

        if len(acc_sub) == 0 or len(gyr_sub) == 0:
            continue

        for activity in sorted(set(acc_sub["activity"]).intersection(set(gyr_sub["activity"]))):
            acc = acc_sub[acc_sub["activity"] == activity].copy()
            gyr = gyr_sub[gyr_sub["activity"] == activity].copy()

            acc = acc.sort_values("timestamp")
            gyr = gyr.sort_values("timestamp")

            # Timestamp-nearest merge within same subject/activity.
            merged = pd.merge_asof(
                acc,
                gyr,
                on="timestamp",
                direction="nearest",
                suffixes=("_acc", "_gyro"),
            )

            merged["subject"] = subject
            merged["activity"] = activity

            merged = merged[
                [
                    "subject",
                    "activity",
                    "timestamp",
                    "x_acc",
                    "y_acc",
                    "z_acc",
                    "x_gyro",
                    "y_gyro",
                    "z_gyro",
                ]
            ].dropna()

            merged_subjects.append(merged)

    if not merged_subjects:
        raise RuntimeError("Could not align accelerometer and gyroscope data.")

    return pd.concat(merged_subjects, ignore_index=True)


def create_windows(df, window_size=128, stride=64):
    X_list = []
    y_list = []
    subject_list = []

    activities = sorted(df["activity"].unique())
    activity_to_label = {act: idx for idx, act in enumerate(activities)}

    feature_cols = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]

    for subject in sorted(df["subject"].unique()):
        sub_df = df[df["subject"] == subject]

        for activity in sorted(sub_df["activity"].unique()):
            act_df = sub_df[sub_df["activity"] == activity].sort_values("timestamp")

            values = act_df[feature_cols].values.astype(np.float32)

            if len(values) < window_size:
                continue

            for start in range(0, len(values) - window_size + 1, stride):
                end = start + window_size
                window = values[start:end]

                X_list.append(window)
                y_list.append(activity_to_label[activity])
                subject_list.append(subject)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subject_list, dtype=np.int64)

    return X, y, subjects, activity_to_label


def normalize_per_subject(X, subjects):
    X_norm = X.copy()

    for subject in np.unique(subjects):
        idx = subjects == subject
        subject_data = X_norm[idx]

        mean = subject_data.mean(axis=(0, 1), keepdims=True)
        std = subject_data.std(axis=(0, 1), keepdims=True)
        std[std < 1e-6] = 1.0

        X_norm[idx] = (subject_data - mean) / std

    return X_norm


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="wisdm-dataset")
    parser.add_argument("--device", type=str, default="watch", choices=["watch", "phone"])
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="data_wisdm")
    args = parser.parse_args()

    accel_folder = os.path.join(args.root, "raw", args.device, "accel")
    gyro_folder = os.path.join(args.root, "raw", args.device, "gyro")

    print(f"Loading accelerometer from: {accel_folder}")
    accel_df = load_sensor_folder(accel_folder)

    print(f"Loading gyroscope from: {gyro_folder}")
    gyro_df = load_sensor_folder(gyro_folder)

    print("Aligning accelerometer and gyroscope...")
    merged_df = align_accel_gyro(accel_df, gyro_df)

    print("Creating windows...")
    X, y, subjects, activity_to_label = create_windows(
        merged_df,
        window_size=args.window,
        stride=args.stride,
    )

    print("Applying per-subject normalization...")
    X = normalize_per_subject(X, subjects)

    out_dir = os.path.join(BASE_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    activity_names = np.array(
        [activity for activity, _ in sorted(activity_to_label.items(), key=lambda x: x[1])]
    )

    np.save(os.path.join(out_dir, "X.npy"), X.astype(np.float32))
    np.save(os.path.join(out_dir, "y.npy"), y.astype(np.int64))
    np.save(os.path.join(out_dir, "subjects.npy"), subjects.astype(np.int64))
    np.save(os.path.join(out_dir, "activity_names.npy"), activity_names)

    print("\nSaved PAMAP-style WISDM files to:", out_dir)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subjects shape:", subjects.shape)
    print("subjects:", sorted(np.unique(subjects).tolist()))
    print("num classes:", len(np.unique(y)))
    print("activity_names:", activity_names)
    print("activity_to_label:", activity_to_label)


if __name__ == "__main__":
    main()