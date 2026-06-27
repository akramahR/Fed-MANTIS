# runMantisembeddings.py
# Usage:
#   python runMantisembeddings.py
#
# Centralized frozen MANTIS embeddings + adapter/head variants.
#
# This is the centralized counterpart of run_fedavg_mantis_adapter.py.
#   - linear_probe
#   - adapter_lowrank_128_r32
#   - adapter_lowrank_256_r64
#   - supcon_lowrank_128_r32
#   - supcon_lowrank_256_r64
#   - confusion_lowrank_256_r64


import os
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import silhouette_samples

from utils.dataset_embeddings import EmbeddingDataset
from utils.loso_split import loso_split
from models.adapted_mantis_head import AdaptedMantisClassifier

import copy

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

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Main config
# ============================================================

EPOCHS = 50
LR = 1e-2

SPLIT_MODE = "loso"   # options: "loso", "random"
TEST_SIZE = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# ============================================================
# Variant config
# ============================================================

@dataclass
class CentralAdapterConfig:
    # model
    use_adapter: bool = True
    adapter_type: str = "lowrank"
    adapter_dim: int = 128
    adapter_rank: int = 32
    adapter_bottleneck_dim: int = 64
    head_type: str = "linear"
    dropout: float = 0.3
    normalize_features_for_head: bool = False

    # training
    epochs: int = EPOCHS
    lr: float = LR
    optimizer: str = "sgd"
    weight_decay: float = 0.0
    batch_size: int = 64
    eval_batch_size: int = 1024

    # losses
    use_supcon_loss: bool = False
    lambda_supcon: float = 0.05

    use_confusion_loss: bool = False
    gamma_conf: float = 0.05
    confusion_margin: float = 5.0

    # confusion-pair discovery
    silhouette_threshold: float = 0.05
    confusion_pairs_mode: str = "auto"  # "auto" or "manual"
    manual_confusion_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # bookkeeping
    run_name: str = "central_adapter"


EXPERIMENTS: Dict[str, CentralAdapterConfig] = {
    "linear_probe": CentralAdapterConfig(
        run_name="linear_probe",
        use_adapter=False,
        head_type="linear",
        batch_size=64,
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "adapter_lowrank_128_r32": CentralAdapterConfig(
        run_name="adapter_lowrank_128_r32",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=128,
        adapter_rank=32,
        head_type="linear",
        dropout=0.3,
        batch_size=64,
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "adapter_lowrank_256_r64": CentralAdapterConfig(
        run_name="adapter_lowrank_256_r64",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=256,
        adapter_rank=64,
        head_type="linear",
        dropout=0.3,
        batch_size=64,
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    # Main method variant 1
    "supcon_lowrank_128_r32": CentralAdapterConfig(
        run_name="supcon_lowrank_128_r32",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=128,
        adapter_rank=32,
        head_type="linear",
        dropout=0.1,
        batch_size=128,
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=False,
    ),

    # Main method variant 2
    "supcon_lowrank_256_r64": CentralAdapterConfig(
        run_name="supcon_lowrank_256_r64",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=256,
        adapter_rank=64,
        head_type="linear",
        dropout=0.3,
        batch_size=128,
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=False,
    ),

    # Exploratory extension
    "confusion_lowrank_128_r32_m05": CentralAdapterConfig(
        run_name="confusion_lowrank_128_r32_m05",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=128,
        adapter_rank=32,
        head_type="linear",
        dropout=0.1,
        batch_size=128,          # use 256 if memory allows
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=True,
        gamma_conf=0.05,
        confusion_margin=0.5,
        silhouette_threshold=0.05,
        confusion_pairs_mode="auto",
    ),

    "confusion_lowrank_128_r32_m10": CentralAdapterConfig(
        run_name="confusion_lowrank_128_r32_m10",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=128,
        adapter_rank=32,
        head_type="linear",
        dropout=0.1,
        batch_size=128,
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=True,
        gamma_conf=0.05,
        confusion_margin=1.0,    # alternative margin
        silhouette_threshold=0.05,
        confusion_pairs_mode="auto",
    ),
}


# ============================================================
# Main experiments to run
# ============================================================

# SupCon low-rank is the main method.
# Plain low-rank adapters are ablations.
# Confusion-aware is exploratory.
"""
MODELS_TO_RUN = [
    "linear_probe",
    "adapter_lowrank_256_r64",
    "supcon_lowrank_256_r64",
]
"""

MODELS_TO_RUN = [
    "linear_probe",
    "adapter_lowrank_128_r32",
    "adapter_lowrank_256_r64",
    "supcon_lowrank_128_r32",
    "supcon_lowrank_256_r64",
    "confusion_lowrank_128_r32_m05",
    "confusion_lowrank_128_r32_m10",
]


TUNING_PRESETS = [
    {"suffix": "sgd_lr1e2", "optimizer": "sgd", "lr": 1e-2},
    {"suffix": "sgd_lr1e3", "optimizer": "sgd", "lr": 1e-3},
    {"suffix": "adam_lr1e3", "optimizer": "adam", "lr": 1e-3},
    {"suffix": "adam_lr1e4", "optimizer": "adam", "lr": 1e-4},
]
SUPCON_LAMBDAS = [0.01, 0.05, 0.10]
# ============================================================
# Paths
# ============================================================

DATA_DIR = get_data_dir(DATASET)

if SPLIT_MODE == "loso":
    X_PATH = os.path.join(DATA_DIR, "X_mantis.npy")
    Y_PATH = os.path.join(DATA_DIR, "y.npy")
    SUBJ_PATH = os.path.join(DATA_DIR, "subjects.npy")

elif SPLIT_MODE == "random":
    if DATASET == "pamap2":
        RANDOM_DIR = os.path.join(DATA_DIR, "random")
        X_PATH = os.path.join(RANDOM_DIR, "X_mantis_random.npy")
        Y_PATH = os.path.join(RANDOM_DIR, "y_random.npy")
        SUBJ_PATH = os.path.join(RANDOM_DIR, "subjects_random.npy")

    elif DATASET == "wisdm":
        raise NotImplementedError(
            "Random split for WISDM is not prepared yet. "
            "Because WISDM uses overlapping windows, random splitting must be done "
            "carefully before windowing or by grouped non-overlapping segments to avoid leakage. "
            "Use SPLIT_MODE='loso' for now."
        )

    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

else:
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")
# ============================================================
# Load data
# ============================================================

X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.int64)
subjects = np.load(SUBJ_PATH).astype(np.int64)

print("X_mantis shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

if X.ndim != 2:
    raise ValueError(
        f"Expected X_mantis to be 2D embeddings [N, D]. Got {X.shape}. "
        "If your embeddings are [N, T, D], pool them first."
    )

if y.ndim != 1:
    raise ValueError(f"Expected y to be 1D [N], got {y.shape}")

if subjects.ndim != 1:
    raise ValueError(f"Expected subjects to be 1D [N], got {subjects.shape}")

if not (len(X) == len(y) == len(subjects)):
    raise ValueError(
        f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(subjects)={len(subjects)}"
    )

input_dim = int(X.shape[1])
num_classes = int(len(np.unique(y)))
unique_subjects = np.unique(subjects)

print(f"Device: {device}")
print(f"Embedding dim (D): {input_dim}")
print(f"Num classes: {num_classes}")
print(f"Unique subjects: {unique_subjects.tolist()}")


# ============================================================
# Helpers
# ============================================================

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_size_mb(model: nn.Module) -> float:
    return count_trainable_params(model) * 4 / (1024 ** 2)


def pretty_mb(x: float) -> str:
    return f"{x:.3f} MB"


def load_class_names(data_dir: str, num_classes: int) -> List[str]:
    names_path = os.path.join(data_dir, "activity_names.npy")

    if os.path.exists(names_path):
        names = np.load(names_path, allow_pickle=True).tolist()

        if len(names) == num_classes:
            return [str(x) for x in names]

    return [f"class_{i}" for i in range(num_classes)]


class_names = load_class_names(DATA_DIR, num_classes)
print("Class names:", class_names)


# ============================================================
# Losses
# ============================================================

def supervised_contrastive_loss(features, labels, temperature: float = 0.1):
    """
    Supervised contrastive loss over adapted embeddings.

    features: [B, D]
    labels:   [B]
    """
    if features.size(0) <= 1:
        return features.new_tensor(0.0)

    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits = torch.div(torch.matmul(features, features.T), temperature)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    mask = mask * logits_mask

    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives_per_sample = mask.sum(dim=1)
    valid = positives_per_sample > 0

    if valid.sum() == 0:
        return features.new_tensor(0.0)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (
        positives_per_sample + 1e-12
    )

    loss = -mean_log_prob_pos[valid].mean()

    return loss


def confusion_margin_loss(
    features,
    labels,
    confusion_pairs,
    margin: float = 0.5,
    min_samples_per_class: int = 2,
):
    """
    Margin loss over batch-level class centroids.
    """
    if not confusion_pairs:
        return features.new_tensor(0.0)

    losses = []

    for a, b in confusion_pairs:
        mask_a = labels == int(a)
        mask_b = labels == int(b)

        if mask_a.sum() < min_samples_per_class or mask_b.sum() < min_samples_per_class:
            continue

        mu_a = features[mask_a].mean(dim=0)
        mu_b = features[mask_b].mean(dim=0)

        dist = torch.norm(mu_a - mu_b, p=2)
        losses.append(F.relu(margin - dist))

    if len(losses) == 0:
        return features.new_tensor(0.0)

    return torch.stack(losses).mean()


# ============================================================
# Confusion-pair discovery
# ============================================================

def compute_classwise_silhouette(X_np: np.ndarray, y_np: np.ndarray, num_classes: int):
    if len(np.unique(y_np)) < 2:
        return {i: np.nan for i in range(num_classes)}

    sil = silhouette_samples(X_np, y_np, metric="euclidean")

    out = {}

    for c in range(num_classes):
        idx = y_np == c

        if idx.sum() == 0:
            out[c] = np.nan
        else:
            out[c] = float(np.mean(sil[idx]))

    return out


def compute_nearest_classes(X_np: np.ndarray, y_np: np.ndarray, num_classes: int):
    centroids = {}

    for c in range(num_classes):
        idx = y_np == c

        if idx.sum() > 0:
            centroids[c] = X_np[idx].mean(axis=0)

    nearest = {}

    for c in centroids:
        best_cls = None
        best_dist = float("inf")

        for d in centroids:
            if c == d:
                continue

            dist = np.linalg.norm(centroids[c] - centroids[d])

            if dist < best_dist:
                best_dist = dist
                best_cls = d

        nearest[c] = (best_cls, float(best_dist))

    return nearest


def discover_confusion_pairs(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    num_classes: int,
    threshold: float,
    class_names: Optional[List[str]] = None,
):
    class_sil = compute_classwise_silhouette(X_train_np, y_train_np, num_classes)
    nearest = compute_nearest_classes(X_train_np, y_train_np, num_classes)

    pairs = set()

    print("\n================== FROZEN EMBEDDING ANALYSIS ==================")
    print(f"Silhouette threshold: {threshold}")
    print(
        f"{'class':>5s} | {'name':25s} | {'silhouette':>10s} | "
        f"{'nearest':25s} | {'dist':>10s}"
    )
    print("-" * 90)

    for c in range(num_classes):
        name = class_names[c] if class_names else f"class_{c}"
        sil = class_sil.get(c, np.nan)
        nearest_cls, dist = nearest.get(c, (None, np.nan))

        nearest_name = (
            class_names[nearest_cls]
            if class_names and nearest_cls is not None
            else str(nearest_cls)
        )

        print(
            f"{c:5d} | {name:25s} | {sil:10.4f} | "
            f"{nearest_name:25s} | {dist:10.4f}"
        )

        if nearest_cls is not None and not np.isnan(sil) and sil < threshold:
            a, b = sorted((int(c), int(nearest_cls)))
            pairs.add((a, b))

    pairs = sorted(list(pairs))

    print("\nAuto confusing pairs:")

    if not pairs:
        print("  None found.")
    else:
        for a, b in pairs:
            an = class_names[a] if class_names else f"class_{a}"
            bn = class_names[b] if class_names else f"class_{b}"
            print(f"  ({a}, {b}) -> {an} ↔ {bn}")

    print("===============================================================\n")

    return pairs


# ============================================================
# Model / data / training
# ============================================================

def build_model(cfg: CentralAdapterConfig):
    return AdaptedMantisClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        use_adapter=cfg.use_adapter,
        adapter_type=cfg.adapter_type,
        adapter_dim=cfg.adapter_dim,
        adapter_rank=cfg.adapter_rank,
        adapter_bottleneck_dim=cfg.adapter_bottleneck_dim,
        head_type=cfg.head_type,
        dropout=cfg.dropout,
        normalize_features_for_head=cfg.normalize_features_for_head,
    )


def random_split_embeddings(X_np, y_np, batch_size=64, seed=42, test_size=0.2):
    idx = np.arange(len(X_np))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_np,
    )

    X_train, y_train = X_np[train_idx], y_np[train_idx]
    X_test, y_test = X_np[test_idx], y_np[test_idx]

    train_ds = EmbeddingDataset(X_train, y_train)
    test_ds = EmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_idx, test_idx


def train_model_adapter(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: CentralAdapterConfig,
    confusion_pairs: List[Tuple[int, int]],
):
    model = model.to(device)
    model.train()

    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()

            optimizer.zero_grad()

            logits, feats = model(xb, return_features=True)

            loss_ce = ce_loss_fn(logits, yb)
            loss = loss_ce

            feats_for_metric = F.normalize(feats, dim=1)

            if cfg.use_supcon_loss:
                loss_supcon = supervised_contrastive_loss(feats_for_metric, yb)
                loss = loss + cfg.lambda_supcon * loss_supcon

            if cfg.use_confusion_loss:
                loss_conf = confusion_margin_loss(
                    feats_for_metric,
                    yb,
                    confusion_pairs,
                    margin=cfg.confusion_margin,
                    min_samples_per_class=2,
                )
                loss = loss + cfg.gamma_conf * loss_conf

            if torch.isnan(loss):
                print("[WARN] NaN loss. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)

            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())
            total_samples += int(yb.numel())

        train_loss = total_loss / max(1, total_samples)
        train_acc = 100.0 * total_correct / max(1, total_samples)

        print(
            f"Epoch [{epoch}/{cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}%"
        )

    print("✅ Training complete.\n")

    return model


@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, num_classes: int):
    model.eval()

    all_preds = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).long()

        logits = model(xb)

        if isinstance(logits, tuple):
            logits = logits[0]

        if hasattr(logits, "logits"):
            logits = logits.logits

        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

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


# ============================================================
# Main experiment loop
# ============================================================

all_results = {}

for exp_name in MODELS_TO_RUN:
    base_cfg = EXPERIMENTS[exp_name]

    lambda_values = [base_cfg.lambda_supcon]

    if base_cfg.use_supcon_loss:
        lambda_values = SUPCON_LAMBDAS

    for tune in TUNING_PRESETS:
        for lam in lambda_values:
            cfg = copy.deepcopy(base_cfg)

            cfg.optimizer = tune["optimizer"]
            cfg.lr = tune["lr"]

            if cfg.use_supcon_loss:
                cfg.lambda_supcon = lam
                tuned_name = f"{exp_name}__{tune['suffix']}__lam{lam}"
            else:
                tuned_name = f"{exp_name}__{tune['suffix']}"

            cfg.run_name = tuned_name

            print("\n" + "#" * 100)
            print(f"EXPERIMENT: {tuned_name}")
            print(cfg)
            print("#" * 100)

            all_accs = []
            all_macro_f1s = []

            model_mb_ref = None
            param_count_ref = None

            last_class_f1 = None
            last_cm = None

            if SPLIT_MODE == "loso":
                print("Evaluation mode: CENTRALIZED LOSO")

                for test_subject in unique_subjects:
                    print(f"\n--- LOSO: Test Subject {int(test_subject)} ---")

                    train_idx = np.where(subjects != test_subject)[0]

                    X_train_np = X[train_idx]
                    y_train_np = y[train_idx]

                    if cfg.use_confusion_loss:
                        if cfg.confusion_pairs_mode == "manual":
                            confusion_pairs = cfg.manual_confusion_pairs
                        else:
                            confusion_pairs = discover_confusion_pairs(
                                X_train_np,
                                y_train_np,
                                num_classes=num_classes,
                                threshold=cfg.silhouette_threshold,
                                class_names=class_names,
                            )
                    else:
                        confusion_pairs = []

                    model = build_model(cfg)

                    model_mb_ref = trainable_size_mb(model)
                    param_count_ref = count_trainable_params(model)

                    print(
                        f"Trainable size: {pretty_mb(model_mb_ref)} | "
                        f"Trainable params: {param_count_ref:,}"
                    )

                    train_loader, test_loader = loso_split(
                        X,
                        y,
                        subjects,
                        test_subject,
                        batch_size=cfg.batch_size,
                        seed=SEED,
                        dataset_cls=EmbeddingDataset,
                    )

                    trained_model = train_model_adapter(
                        model,
                        train_loader,
                        cfg,
                        confusion_pairs=confusion_pairs,
                    )

                    acc, macro_f1, class_f1, cm = evaluate_full(
                        trained_model,
                        test_loader,
                        num_classes,
                    )

                    all_accs.append(acc)
                    all_macro_f1s.append(macro_f1)

                    last_class_f1 = class_f1
                    last_cm = cm

                    print(
                        f"✅ Subject {int(test_subject)} Accuracy: {acc:.2f}% | "
                        f"Macro-F1: {macro_f1:.2f}%"
                    )

            elif SPLIT_MODE == "random":
                print("Evaluation mode: RANDOM SPLIT")

                train_loader, test_loader, train_idx, test_idx = random_split_embeddings(
                    X,
                    y,
                    batch_size=cfg.batch_size,
                    seed=SEED,
                    test_size=TEST_SIZE,
                )

                X_train_np = X[train_idx]
                y_train_np = y[train_idx]

                if cfg.use_confusion_loss:
                    if cfg.confusion_pairs_mode == "manual":
                        confusion_pairs = cfg.manual_confusion_pairs
                    else:
                        confusion_pairs = discover_confusion_pairs(
                            X_train_np,
                            y_train_np,
                            num_classes=num_classes,
                            threshold=cfg.silhouette_threshold,
                            class_names=class_names,
                        )
                else:
                    confusion_pairs = []

                model = build_model(cfg)

                model_mb_ref = trainable_size_mb(model)
                param_count_ref = count_trainable_params(model)

                print(
                    f"Trainable size: {pretty_mb(model_mb_ref)} | "
                    f"Trainable params: {param_count_ref:,}"
                )

                trained_model = train_model_adapter(
                    model,
                    train_loader,
                    cfg,
                    confusion_pairs=confusion_pairs,
                )

                acc, macro_f1, class_f1, cm = evaluate_full(
                    trained_model,
                    test_loader,
                    num_classes,
                )

                all_accs.append(acc)
                all_macro_f1s.append(macro_f1)

                last_class_f1 = class_f1
                last_cm = cm

                print(
                    f"Random Split Accuracy: {acc:.2f}% | "
                    f"Macro-F1: {macro_f1:.2f}%"
                )

            else:
                raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

            mean_acc = float(np.mean(all_accs))
            std_acc = float(np.std(all_accs))
            mean_f1 = float(np.mean(all_macro_f1s))
            std_f1 = float(np.std(all_macro_f1s))

            print("\n" + "-" * 100)
            print(
                f"[FINAL] {tuned_name} | "
                f"acc={mean_acc:.2f}% ± {std_acc:.2f}% | "
                f"macroF1={mean_f1:.2f}% ± {std_f1:.2f}%"
            )
            print(f"[SIZE ] trainable_size={pretty_mb(model_mb_ref or 0.0)}")
            print(f"[PARAM] trainable_params={param_count_ref}")
            print("-" * 100)

            print("\nClass-wise F1 from last evaluated fold/split:")

            if last_class_f1 is not None:
                for i, f1v in enumerate(last_class_f1):
                    cname = class_names[i] if class_names else f"class_{i}"
                    print(f"  {i:02d} {cname:25s}: {f1v:.2f}%")

            print("\nConfusion matrix from last evaluated fold/split:")

            if last_cm is not None:
                print(last_cm)

            all_results[tuned_name] = {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "model_mb": float(model_mb_ref or 0.0),
                "params": int(param_count_ref or 0),
                "optimizer": cfg.optimizer,
                "lr": cfg.lr,
                "lambda_supcon": cfg.lambda_supcon if cfg.use_supcon_loss else None,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
            }


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 120)
print(f"SUMMARY Centralized MANTIS Adapter Heads ({SPLIT_MODE.upper()})")
print("=" * 120)
print(
    f"{'experiment':55s} | {'acc':>14s} | {'macroF1':>14s} | "
    f"{'size':>10s} | {'params':>12s}"
)
print("-" * 120)

for name, r in sorted(all_results.items(), key=lambda x: x[1]["mean_acc"], reverse=True):
    print(
        f"{name:55s} | "
        f"{r['mean_acc']:6.2f}% ± {r['std_acc']:5.2f}% | "
        f"{r['mean_macro_f1']:6.2f}% ± {r['std_macro_f1']:5.2f}% | "
        f"{pretty_mb(r['model_mb']):>10s} | "
        f"{r['params']:12d}"
    )

print("=" * 120)

print("\n================== EXPERIMENT CONFIGURATION ==================")
print("Experiment type: Centralized MANTIS frozen-embedding adapter/head variants")
print(f"Split mode: {SPLIT_MODE}")
print(f"Default epochs: {EPOCHS}")
print(f"Default learning rate: {LR}")
print(f"Seed: {SEED}")
print(f"Device: {device}")
print(f"Embedding shape: X_mantis={X.shape}")
print(f"Labels shape: y={y.shape}")
print(f"Subjects shape: subjects={subjects.shape}")
print(f"Embedding dimension: {input_dim}")
print(f"Number of classes: {num_classes}")
print(f"Subjects: {unique_subjects.tolist()}")
print(f"Class names: {class_names}")
print(f"Models run: {MODELS_TO_RUN}")
print("Frozen backbone: MANTIS")
print("Centralized training: adapter/head only")
print("Early stopping: False")
print("==============================================================\n")