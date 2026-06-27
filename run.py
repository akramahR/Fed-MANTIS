# run.py
# .venv\Scripts\activate
#
# Centralized raw-window baselines under switchable evaluation:
# - LOSO for strict unseen-subject evaluation
# - random split for comparison / sanity checks
# - No early stopping on the test split (prevents test-peeking)
# - Reproducible shuffling (seeded)
#
# Standardized logging:
# - Accuracy mean ± std
# - Macro-F1 mean ± std
# - Class-wise F1 and confusion matrix from the last evaluated fold/split

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from models.chronosFrozenMLP import ChronosFrozenMLP
from utils.dataset import PAMAP2Dataset
from utils.loso_split import loso_split
from utils.train import train_model

from models.cnn import CNNModel
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.inceptiontime import InceptionTimeModel
from models.deepConvLstmForFed import DeepConvLSTMModelTest

from dataclasses import dataclass


DATASET = "mhealth"  # options: "pamap2", "mhealth"


def get_data_dir(dataset):
    if dataset == "pamap2":
        return "data"
    elif dataset == "wisdm":
        return "data_wisdm"
    elif dataset == "mhealth":
        return "data_mhealth"
    else:
        raise ValueError(f"Unknown DATASET: {dataset}")
# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------------
# Configs
# -----------------------------

# For RQ1, use "loso".
# For RQ5/protocol-gap comparison, use "random".
SPLIT_MODE = "loso"   # options: "loso", "random"
TEST_SIZE = 0.2       # only used for random split


@dataclass
class CentralRawConfig:
    run_name: str
    optimizer: str = "sgd"
    lr: float = 1e-2
    epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 0.0


TUNING_PRESETS = [
    CentralRawConfig(run_name="sgd_lr1e2", optimizer="sgd", lr=1e-2, epochs=50),
    CentralRawConfig(run_name="sgd_lr1e3", optimizer="sgd", lr=1e-3, epochs=50),
    CentralRawConfig(run_name="adam_lr1e3", optimizer="adam", lr=1e-3, epochs=50),
    CentralRawConfig(run_name="adam_lr1e4", optimizer="adam", lr=1e-4, epochs=50),
]

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = get_data_dir(DATASET)

if SPLIT_MODE == "loso":
    X_PATH = os.path.join(DATA_DIR, "X.npy")
    Y_PATH = os.path.join(DATA_DIR, "y.npy")
    SUBJ_PATH = os.path.join(DATA_DIR, "subjects.npy")

elif SPLIT_MODE == "random":
    if DATASET == "pamap2":
        DATA_DIR = os.path.join("data", "random")
        X_PATH = os.path.join(DATA_DIR, "X_random.npy")
        Y_PATH = os.path.join(DATA_DIR, "y_random.npy")
        SUBJ_PATH = os.path.join(DATA_DIR, "subjects_random.npy")

    elif DATASET == "wisdm":
        raise NotImplementedError(
            "Random split for WISDM is not prepared yet. "
            "Because WISDM uses overlapping windows, random splitting must be done "
            "carefully before windowing or by grouping non-overlapping segments to avoid leakage. "
            "Use SPLIT_MODE='loso' for now."
        )

    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

else:
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")


# -----------------------------
# Load data
# -----------------------------
X = np.load(X_PATH)
y = np.load(Y_PATH)
subjects = np.load(SUBJ_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

if X.ndim != 3:
    raise ValueError(f"Expected X to be 3D [N, T, F], got {X.shape}")
if y.ndim != 1:
    raise ValueError(f"Expected y to be 1D [N], got {y.shape}")
if subjects.ndim != 1:
    raise ValueError(f"Expected subjects to be 1D [N], got {subjects.shape}")
if not (len(X) == len(y) == len(subjects)):
    raise ValueError(
        f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(subjects)={len(subjects)}"
    )

seq_len = int(X.shape[1])
input_dim = int(X.shape[2])
num_classes = int(len(np.unique(y)))
unique_subjects = np.unique(subjects)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Seq length (T): {seq_len}")
print(f"Input dim (F): {input_dim}")
print(f"Num classes: {num_classes}")
print(f"Unique subjects: {unique_subjects.tolist()}")


# -----------------------------
# Model factory
# -----------------------------
def build_cnn():
    return CNNModel(input_dim=input_dim, num_classes=num_classes)


def build_lstm():
    return LSTMModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2,
        bidirectional=True,
    )


def build_transformer():
    return TransformerModel(
        input_dim=input_dim,
        num_classes=num_classes,
    )



def build_inceptiontime():
    return InceptionTimeModel(
        input_dim=input_dim,
        num_classes=num_classes,
        bottleneck_channels=32,
        num_blocks=6,
        use_residual=True,
        dropout=0.0,
    )


def build_chronos():
    return ChronosFrozenMLP(
        num_classes=num_classes,
        num_features=input_dim,
        mlp_hidden_dims=(512, 256),
        dropout=0.3,
    )


def build_deepconvlstm_test():
    return DeepConvLSTMModelTest(
        input_dim=input_dim,
        num_classes=num_classes,
        conv_channels=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        pooling="mean_max",
    )



models = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "inceptiontime": build_inceptiontime,
    "deepconvlstm": build_deepconvlstm_test,


    # Optional/debug variants:
    # "transformer": build_transformer,
    # "chronos": build_chronos,

}


# -----------------------------
# Split helper
# -----------------------------
def random_split(X, y, batch_size=64, seed=42, test_size=0.2):
    """
    Window-level random split for comparison / sanity checks.
    This intentionally mixes subjects across train/test and should not
    be used as the main evaluation protocol for the paper.
    """
    idx = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = PAMAP2Dataset(X_train, y_train)
    test_ds = PAMAP2Dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: str, num_classes: int):
    """
    Standard evaluation used across experiment scripts.

    Returns:
        acc: float, percentage
        macro_f1: float, percentage
        class_f1: np.ndarray, percentage per class
        cm: np.ndarray, confusion matrix
    """
    model.eval()

    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()

        outputs = model(X_batch)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = 100.0 * float((y_true == y_pred).mean())
    macro_f1 = 100.0 * float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    class_f1 = 100.0 * f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return acc, macro_f1, class_f1, cm


# -----------------------------
# Main evaluation loop
# -----------------------------
final_results = {}

for cfg in TUNING_PRESETS:
    print("\n" + "=" * 120)
    print(f"TUNING RUN: {cfg.run_name}")
    print(cfg)
    print("=" * 120)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    for model_name, model_fn in models.items():
        full_name = f"{model_name}__{cfg.run_name}"

        print(f"\n===== MODEL: {full_name.upper()} =====")

        all_accuracies = []
        all_macro_f1s = []

        last_class_f1 = None
        last_cm = None

        if SPLIT_MODE == "loso":
            print("Evaluation mode: CENTRALIZED LOSO")

            for test_subject in unique_subjects:
                print(f"\n--- LOSO: Test Subject {int(test_subject)} ---")

                model = model_fn()

                train_loader, test_loader = loso_split(
                    X,
                    y,
                    subjects,
                    test_subject,
                    batch_size=cfg.batch_size,
                    seed=SEED,
                )

                trained_model = train_model(
                    model,
                    train_loader,
                    val_loader=None,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    device=device,
                    use_early_stopping=False,
                    optimizer_name=cfg.optimizer,
                    weight_decay=cfg.weight_decay,
                )

                acc, macro_f1, class_f1, cm = evaluate_full(
                    trained_model,
                    test_loader,
                    device,
                    num_classes,
                )

                all_accuracies.append(acc)
                all_macro_f1s.append(macro_f1)
                last_class_f1 = class_f1
                last_cm = cm

                print(
                    f"Subject {int(test_subject)} Accuracy: {acc:.2f}% | "
                    f"Macro-F1: {macro_f1:.2f}%"
                )

        elif SPLIT_MODE == "random":
            print("Evaluation mode: RANDOM SPLIT")

            model = model_fn()

            train_loader, test_loader = random_split(
                X,
                y,
                batch_size=cfg.batch_size,
                seed=SEED,
                test_size=TEST_SIZE,
            )

            trained_model = train_model(
                model,
                train_loader,
                val_loader=None,
                epochs=cfg.epochs,
                lr=cfg.lr,
                device=device,
                use_early_stopping=False,
                optimizer_name=cfg.optimizer,
                weight_decay=cfg.weight_decay,
            )

            acc, macro_f1, class_f1, cm = evaluate_full(
                trained_model,
                test_loader,
                device,
                num_classes,
            )

            all_accuracies.append(acc)
            all_macro_f1s.append(macro_f1)
            last_class_f1 = class_f1
            last_cm = cm

            print(
                f"✅ Random Split Accuracy: {acc:.2f}% | "
                f"Macro-F1: {macro_f1:.2f}%"
            )

        else:
            raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

        mean_acc = float(np.mean(all_accuracies))
        std_acc = float(np.std(all_accuracies))
        mean_f1 = float(np.mean(all_macro_f1s))
        std_f1 = float(np.std(all_macro_f1s))

        print(f"\n=== {full_name.upper()} Overall Performance ===")
        print(
            f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}% | "
            f"Macro-F1: {mean_f1:.2f}% ± {std_f1:.2f}%"
        )

        print("\nClass-wise F1 from last evaluated fold/split:")
        if last_class_f1 is not None:
            for i, f1v in enumerate(last_class_f1):
                print(f"  class_{i:02d}: {f1v:.2f}%")

        print("\nConfusion matrix from last evaluated fold/split:")
        if last_cm is not None:
            print(last_cm)

        final_results[full_name] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "optimizer": cfg.optimizer,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
        }

# -----------------------------
# Final results table
# -----------------------------
print(f"\n================== FINAL RESULTS ({SPLIT_MODE.upper()}) ==================")
print("Centralized raw-window hyperparameter tuning")
print(
    f"{'model':34s} | {'acc':>14s} | {'macroF1':>14s} | "
    f"{'optimizer':>9s} | {'lr':>10s} | {'epochs':>6s}"
)
print("-" * 100)

for model_name, r in sorted(
    final_results.items(),
    key=lambda x: x[1]["mean_acc"],
    reverse=True,
):
    print(
        f"{model_name:34s} | "
        f"{r['mean_acc']:6.2f}% ± {r['std_acc']:5.2f}% | "
        f"{r['mean_macro_f1']:6.2f}% ± {r['std_macro_f1']:5.2f}% | "
        f"{r['optimizer']:>9s} | "
        f"{r['lr']:10.1e} | "
        f"{r['epochs']:6d}"
    )

print("=" * 100 + "\n")

print("\n================== EXPERIMENT CONFIGURATION ==================")
print("Experiment type: Centralized raw-window baseline tuning")
print(f"Dataset: {DATASET}")
print(f"Split mode: {SPLIT_MODE}")
print(f"Seed: {SEED}")
print(f"Device: {device}")
print(f"Input shape: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
print(f"Sequence length: {seq_len}")
print(f"Input dimension: {input_dim}")
print(f"Number of classes: {num_classes}")
print(f"Subjects: {unique_subjects.tolist()}")
print(f"Models run: {list(models.keys())}")
print("Early stopping: False")
print("Tuning presets:")

for preset in TUNING_PRESETS:
    print(
        f"  {preset.run_name}: "
        f"optimizer={preset.optimizer}, "
        f"lr={preset.lr}, "
        f"epochs={preset.epochs}, "
        f"batch_size={preset.batch_size}, "
        f"weight_decay={preset.weight_decay}"
    )

print("==============================================================\n")