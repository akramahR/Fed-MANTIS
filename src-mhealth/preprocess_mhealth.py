# Usage:
#   python src-mhealth/preprocess_mhealth.py --root MHEALTHDATASET


import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVITY_NAMES = {
    1: "standing_still",
    2: "sitting_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_arms",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back",
}

# MHEALTH raw columns, 0-based:
# 0-2 chest acc
# 3-4 ECG
# 5-7 left ankle acc
# 8-10 left ankle gyro
# 11-13 left ankle mag
# 14-16 right arm acc
# 17-19 right arm gyro
# 20-22 right arm mag
# 23 label
MOTION_COLS_21 = [
    0, 1, 2,
    5, 6, 7,
    8, 9, 10,
    11, 12, 13,
    14, 15, 16,
    17, 18, 19,
    20, 21, 22,
]

ACC_GYRO_COLS_15 = [
    0, 1, 2,
    5, 6, 7,
    8, 9, 10,
    14, 15, 16,
    17, 18, 19,
]


def infer_subject_id(path: Path) -> int:
    match = re.search(r"subject[_-]?(\d+)", path.name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1])

    raise ValueError(f"Could not infer subject id from filename: {path.name}")


def find_subject_files(root: Path):
    candidates = []

    for pattern in ("*.log", "*.txt", "*.csv"):
        candidates.extend(root.rglob(pattern))

    subject_files = [
        p for p in candidates
        if "subject" in p.name.lower() and not p.name.startswith(".")
    ]

    subject_files = sorted(subject_files, key=lambda p: infer_subject_id(p))

    if not subject_files:
        raise FileNotFoundError(
            f"No subject files found under {root}. "
            "Expected files like mHealth_subject1.log."
        )

    return subject_files


def load_subject_file(path: Path, feature_set: str, keep_null: bool):
    sid = infer_subject_id(path)

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if df.shape[1] < 24:
        raise ValueError(f"{path} has {df.shape[1]} columns, expected at least 24.")

    labels_raw = df.iloc[:, 23].astype(int).to_numpy()

    if feature_set == "motion21":
        feature_cols = MOTION_COLS_21
    elif feature_set == "accgyro15":
        feature_cols = ACC_GYRO_COLS_15
    elif feature_set == "all23":
        feature_cols = list(range(23))
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    X_raw = df.iloc[:, feature_cols].astype(np.float32).to_numpy()

    if not keep_null:
        mask = labels_raw != 0
        X_raw = X_raw[mask]
        labels_raw = labels_raw[mask]

    return sid, X_raw, labels_raw


def make_windows_for_subject(sid, X_raw, labels_raw, window, stride):
    X_windows = []
    y_windows_raw = []
    subject_windows = []

    n = len(labels_raw)
    start = 0

    # Window inside contiguous same-label segments.
    # This prevents transition windows from mixing activities.
    while start < n:
        label = int(labels_raw[start])
        end = start + 1

        while end < n and int(labels_raw[end]) == label:
            end += 1

        seg_X = X_raw[start:end]

        if len(seg_X) >= window and label != 0:
            for i in range(0, len(seg_X) - window + 1, stride):
                X_windows.append(seg_X[i:i + window])
                y_windows_raw.append(label)
                subject_windows.append(sid)

        start = end

    return X_windows, y_windows_raw, subject_windows


def normalize_windowwise(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    return ((X - mean) / std).astype(np.float32)


def normalize_per_subject(X, subjects):
    Xn = X.copy().astype(np.float32)

    for sid in np.unique(subjects):
        idx = subjects == sid
        mean = Xn[idx].mean(axis=(0, 1), keepdims=True)
        std = Xn[idx].std(axis=(0, 1), keepdims=True) + 1e-6
        Xn[idx] = (Xn[idx] - mean) / std

    return Xn.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data_mhealth")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="motion21",
        choices=["motion21", "accgyro15", "all23"],
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="window",
        choices=["window", "subject", "none"],
    )
    parser.add_argument("--keep-null", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_files = find_subject_files(root)

    print("Found subject files:")
    for p in subject_files:
        print(" ", p)

    all_X = []
    all_y_raw = []
    all_subjects = []

    for path in subject_files:
        sid, X_raw, labels_raw = load_subject_file(
            path,
            feature_set=args.feature_set,
            keep_null=args.keep_null,
        )

        print(
            f"Subject {sid}: raw={X_raw.shape}, "
            f"labels={sorted(np.unique(labels_raw).tolist())}"
        )

        Xw, yw, sw = make_windows_for_subject(
            sid=sid,
            X_raw=X_raw,
            labels_raw=labels_raw,
            window=args.window,
            stride=args.stride,
        )

        print(f"  windows={len(Xw)}")

        all_X.extend(Xw)
        all_y_raw.extend(yw)
        all_subjects.extend(sw)

    if not all_X:
        raise RuntimeError("No windows created. Try smaller --window.")

    X = np.stack(all_X).astype(np.float32)
    y_raw = np.asarray(all_y_raw, dtype=np.int64)
    subjects = np.asarray(all_subjects, dtype=np.int64)

    raw_labels_sorted = sorted(np.unique(y_raw).tolist())
    raw_to_idx = {int(raw): i for i, raw in enumerate(raw_labels_sorted)}
    y = np.asarray([raw_to_idx[int(v)] for v in y_raw], dtype=np.int64)

    if args.normalization == "window":
        X = normalize_windowwise(X)
    elif args.normalization == "subject":
        X = normalize_per_subject(X, subjects)
    elif args.normalization == "none":
        X = X.astype(np.float32)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subjects.npy", subjects)
    np.save(out_dir / "y_raw.npy", y_raw)

    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump(raw_to_idx, f, indent=2)

    idx_names = {
        str(raw_to_idx[raw]): ACTIVITY_NAMES.get(raw, f"class_{raw_to_idx[raw]}")
        for raw in raw_labels_sorted
    }

    with open(out_dir / "mhealth_class_names.json", "w") as f:
        json.dump(idx_names, f, indent=2)

    meta = {
        "dataset": "MHEALTH",
        "sampling_rate_hz": 50,
        "window": args.window,
        "stride": args.stride,
        "feature_set": args.feature_set,
        "normalization": args.normalization,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "subjects_shape": list(subjects.shape),
        "subjects": sorted(np.unique(subjects).astype(int).tolist()),
        "raw_labels": raw_labels_sorted,
        "num_classes": len(raw_labels_sorted),
    }

    with open(out_dir / "mhealth_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved files:")
    print(" ", out_dir / "X.npy")
    print(" ", out_dir / "y.npy")
    print(" ", out_dir / "subjects.npy")
    print(" ", out_dir / "class_mapping.json")
    print("\nShapes:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)
    print("subjects:", sorted(np.unique(subjects).tolist()))
    print("classes:", len(raw_labels_sorted))
    print("feature_set:", args.feature_set)
    print("normalization:", args.normalization)


if __name__ == "__main__":
    main()