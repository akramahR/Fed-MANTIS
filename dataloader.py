import os
import json
import numpy as np
import pandas as pd

ACTIVITIES = {
    0: "other",
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "Nordic walking", 9: "watching TV",
    10: "computer work", 11: "car driving", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    18: "folding laundry", 19: "house cleaning", 20: "playing soccer",
    24: "rope jumping"
}

FEATURE_START_COL = 3  # PAMAP2: features start here
LABEL_COL = 1          # activity label
SUBJECT_COL_NAME = "subject_id"

def load_pamap2(path, exclude_subjects=None):
    """
    Loads all .dat files, attaches subject_id, and concatenates.
    """
    exclude_subjects = set(exclude_subjects or [])
    all_data = []

    files = sorted([f for f in os.listdir(path) if f.endswith(".dat")])
    for fname in files:
        subject_id = int("".join(filter(str.isdigit, fname)))
        if subject_id in exclude_subjects:
            print(f"Skipping subject {subject_id}")
            continue

        # Use sep instead of deprecated delim_whitespace
        df = pd.read_csv(os.path.join(path, fname), sep=r"\s+", engine="python", header=None)
        df[SUBJECT_COL_NAME] = subject_id
        all_data.append(df)

    if not all_data:
        raise RuntimeError("No PAMAP2 .dat files loaded. Check path and exclusions.")

    return pd.concat(all_data, ignore_index=True)

def _per_subject_ffill(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prevents cross-subject leakage during missing-value filling.
    """
    data = data.replace(-1, np.nan)
    # forward-fill within each subject only
    data = (
        data.groupby(SUBJECT_COL_NAME, sort=False, group_keys=False)
            .apply(lambda g: g.ffill())
    )
    return data

def _compute_stats(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)

def preprocess_pamap2(
    data: pd.DataFrame,
    seq_len=512,
    step=256,
    norm="per_subject",   # "per_subject" (recommended), "global" (leaky for LOSO), "none"
):
    """
    Creates windows per (subject, activity) without mixing labels.

    norm:
      - "per_subject": normalize each subject using its own mean/std (no LOSO leakage, FL-realistic)
      - "global": normalize using mean/std over all subjects (LEAKY for LOSO; disclose if used)
      - "none": no normalization here (normalize later inside LOSO fold)
    """
    if norm not in {"per_subject", "global", "none"}:
        raise ValueError("norm must be one of: per_subject, global, none")

    # 1) Clean missing safely (no cross-subject leakage)
    data = _per_subject_ffill(data)

    # 2) Remove class 0 early
    data = data[data.iloc[:, LABEL_COL] != 0].copy()

    # 3) Decide normalization stats
    feature_cols = list(range(FEATURE_START_COL, data.shape[1] - 1))  # exclude subject_id col
    valid_acts = set(ACTIVITIES.keys()) - {0}

    global_mean, global_std = None, None
    if norm == "global":
        X_all = data.iloc[:, feature_cols].values.astype(np.float32)
        global_mean, global_std = _compute_stats(X_all)

    # compute per-subject stats ONCE (subject-only, not subject+label)
    per_subject_stats = {}
    if norm == "per_subject":
        for sid, gsub in data.groupby(SUBJECT_COL_NAME, sort=True):
            Xsub = gsub.iloc[:, feature_cols].values.astype(np.float32)
            per_subject_stats[int(sid)] = _compute_stats(Xsub)

    # 4) Windowing per (subject, activity) (this is fine for boundaries)
    X_seq, y_seq, subj_seq = [], [], []
    grouped = data.groupby([SUBJECT_COL_NAME, data.iloc[:, LABEL_COL]], sort=True)

    for (sid, act), group in grouped:
        sid = int(sid)
        act = int(act)
        if act not in valid_acts:
            continue

        Xg = group.iloc[:, feature_cols].values.astype(np.float32)

        # ✅ FIX: normalize using subject stats (not activity-specific stats)
        if norm == "per_subject":
            mean, std = per_subject_stats[sid]
            Xg = (Xg - mean) / std
        elif norm == "global":
            Xg = (Xg - global_mean) / global_std
        elif norm == "none":
            pass

        n = len(Xg)
        if n < seq_len:
            continue

        for i in range(0, n - seq_len + 1, step):
            X_seq.append(Xg[i:i + seq_len])
            y_seq.append(act)   # keep RAW activity id for mapping
            subj_seq.append(sid)

    # 5) Remap labels to 0..K-1 (consistent for training)
    y_raw = np.array(y_seq, dtype=int)
    subjects = np.array(subj_seq, dtype=np.int64)

    classes = np.unique(y_raw)
    class_to_idx = {int(c): int(i) for i, c in enumerate(classes)}
    y_mapped = np.array([class_to_idx[int(c)] for c in y_raw], dtype=np.int64)

    X = np.array(X_seq, dtype=np.float32)
    return X, y_mapped, subjects, class_to_idx

if __name__ == "__main__":
    data = load_pamap2("PAMAP2_Dataset/Protocol", exclude_subjects=[109])

    # RECOMMENDED: per_subject normalization (no LOSO leakage, FL-realistic)
    X, y, subjects, class_to_idx = preprocess_pamap2(data, seq_len=512, step=256, norm="per_subject")

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Shape of subjects:", subjects.shape)
    print("Class mapping:", class_to_idx)

    os.makedirs("data", exist_ok=True)
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    np.save("data/subjects.npy", subjects)

    # Prefer JSON for mapping
    with open("data/class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # sanity: per-subject window counts
    for sid in np.unique(subjects):
        print("windows subject", sid, ":", int(np.sum(subjects == sid)))
