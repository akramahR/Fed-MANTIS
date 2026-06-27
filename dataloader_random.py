import os
import json
import numpy as np

from dataloader import load_pamap2, preprocess_pamap2

if __name__ == "__main__":
    data = load_pamap2("PAMAP2_Dataset/Protocol", exclude_subjects=[109])

    # Random-split comparison dataset:
    # use non-overlapping windows to reduce overlap leakage between train/test.
    X, y, subjects, class_to_idx = preprocess_pamap2(
        data,
        seq_len=512,
        step=512,
        norm="per_subject",
    )

    print("Random comparison dataset")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Shape of subjects:", subjects.shape)
    print("Class mapping:", class_to_idx)

    out_dir = os.path.join("data", "random")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_random.npy"), X)
    np.save(os.path.join(out_dir, "y_random.npy"), y)
    np.save(os.path.join(out_dir, "subjects_random.npy"), subjects)

    with open(os.path.join(out_dir, "class_mapping_random.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    for sid in np.unique(subjects):
        print("windows subject", sid, ":", int(np.sum(subjects == sid)))