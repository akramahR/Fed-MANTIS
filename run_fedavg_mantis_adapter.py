# run_fedavg_mantis_adapter.py
# Run from project root:
#   python run_fedavg_mantis_adapter.py

import os
import copy
import json
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import silhouette_samples

from models.adapted_mantis_head import AdaptedMantisClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Config
# ============================================================

@dataclass
class AdapterFLConfig:
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 1e-2
    optimizer: str = "sgd"
    weight_decay: float = 0.0
    client_frac: float = 1.0
    batch_size: int = 64
    eval_batch_size: int = 1024
    seed: int = 0
    eval_every: int = 5

    # model
    use_adapter: bool = True
    adapter_type: str = "linear"
    adapter_dim: int = 512
    head_type: str = "linear"
    dropout: float = 0.3
    adapter_rank: int = 64
    adapter_bottleneck_dim: int = 64
    normalize_features_for_head: bool = False

    # losses
    use_supcon_loss: bool = False
    lambda_supcon: float = 0.1

    use_confusion_loss: bool = False
    gamma_conf: float = 0.1
    confusion_margin: float = 5.0

    # confusion-pair discovery
    silhouette_threshold: float = 0.05
    confusion_pairs_mode: str = "auto"  # "auto" or "manual"
    manual_confusion_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # run name
    run_name: str = "adapter_only"


# ============================================================
# Presets
# ============================================================

EXPERIMENTS: Dict[str, AdapterFLConfig] = {
    "baseline_linear": AdapterFLConfig(
        run_name="baseline_linear",
        use_adapter=False,
        head_type="linear",
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "adapter_linear_512": AdapterFLConfig(
        run_name="adapter_linear_512",
        use_adapter=True,
        adapter_type="linear",
        adapter_dim=512,
        head_type="linear",
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "adapter_lowrank_128_r32": AdapterFLConfig(
        run_name="adapter_lowrank_128_r32",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=128,
        adapter_rank=32,
        head_type="linear",
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "adapter_lowrank_256_r64": AdapterFLConfig(
        run_name="adapter_lowrank_256_r64",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=256,
        adapter_rank=64,
        head_type="linear",
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "supcon_lowrank_256_r64": AdapterFLConfig(
        run_name="supcon_lowrank_256_r64",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=256,
        adapter_rank=64,
        head_type="linear",
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=False,
    ),

    "confusion_lowrank_256_r64": AdapterFLConfig(
        run_name="confusion_lowrank_256_r64",
        use_adapter=True,
        adapter_type="lowrank",
        adapter_dim=256,
        adapter_rank=64,
        head_type="linear",
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=True,
        gamma_conf=0.05,
        confusion_margin=5.0,
        silhouette_threshold=0.05,
        confusion_pairs_mode="auto",
    ),

    "adapter_residual_b64": AdapterFLConfig(
        run_name="adapter_residual_b64",
        use_adapter=True,
        adapter_type="residual_bottleneck",
        adapter_bottleneck_dim=64,
        head_type="linear",
        use_supcon_loss=False,
        use_confusion_loss=False,
    ),

    "confusion_residual_b64": AdapterFLConfig(
        run_name="confusion_residual_b64",
        use_adapter=True,
        adapter_type="residual_bottleneck",
        adapter_bottleneck_dim=64,
        head_type="linear",
        use_supcon_loss=True,
        lambda_supcon=0.05,
        use_confusion_loss=True,
        gamma_conf=0.05,
        confusion_margin=1.0,
        silhouette_threshold=0.05,
        confusion_pairs_mode="auto",
    ),
    "supcon_lowrank_128_r32": AdapterFLConfig(
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

    "confusion_lowrank_128_r32_m05": AdapterFLConfig(
        run_name="confusion_lowrank_128_r32_m05",
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
        confusion_margin=0.5,
        silhouette_threshold=0.05,
        confusion_pairs_mode="auto",
    ),

    "confusion_lowrank_128_r32_m10": AdapterFLConfig(
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
        confusion_margin=1.0,
        silhouette_threshold=0.05,
        confusion_pairs_mode="auto",
    ),
}

# choose which experiments to run
"""
MODELS_TO_RUN = [
     "baseline_linear",
    "adapter_lowrank_256_r64",
    "supcon_lowrank_256_r64",
]

"""
MODELS_TO_RUN = [
    "baseline_linear",
    "adapter_lowrank_128_r32",
    "adapter_lowrank_256_r64",
    "supcon_lowrank_128_r32",
    "supcon_lowrank_256_r64",
    "confusion_lowrank_128_r32_m05",
    "confusion_lowrank_128_r32_m10",
]

TUNING_PRESETS = [
    {"suffix": "sgd_lr1e2_e1", "optimizer": "sgd", "lr": 1e-2, "local_epochs": 1},
    {"suffix": "sgd_lr1e3_e1", "optimizer": "sgd", "lr": 1e-3, "local_epochs": 1},
    {"suffix": "adam_lr1e3_e1", "optimizer": "adam", "lr": 1e-3, "local_epochs": 1},
    {"suffix": "adam_lr1e3_e3", "optimizer": "adam", "lr": 1e-3, "local_epochs": 3},
    {"suffix": "sgd_lr1e2_e3", "optimizer": "sgd", "lr": 1e-2, "local_epochs": 3},
    {"suffix": "sgd_lr1e3_e3", "optimizer": "sgd", "lr": 1e-3, "local_epochs": 3},
    {"suffix": "adam_lr1e4_e1", "optimizer": "adam", "lr": 1e-4, "local_epochs": 1},
    {"suffix": "adam_lr1e4_e3", "optimizer": "adam", "lr": 1e-4, "local_epochs": 3},
]

SUPCON_LAMBDAS = [0.01, 0.05, 0.10]

# ============================================================
# Helpers
# ============================================================

def state_dict_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for t in model.state_dict().values():
        if torch.is_tensor(t):
            total_bytes += t.numel() * t.element_size()
    return total_bytes / (1024 ** 2)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretty_mb(x: float) -> str:
    return f"{x:.3f} MB"


def load_subject_npz(cache_dir: str, sid: int):
    path = os.path.join(cache_dir, f"subject_{int(sid)}.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache file: {path}\n"
            "Re-run your per-subject MANTIS embedding extractor."
        )

    d = np.load(path)
    return torch.from_numpy(d["X"]).float(), torch.from_numpy(d["y"]).long()


def load_class_names(data_dir: str, num_classes: int) -> List[str]:
    mhealth_names_path = os.path.join(data_dir, "mhealth_class_names.json")

    if os.path.exists(mhealth_names_path):
        with open(mhealth_names_path, "r") as f:
            idx_to_name = json.load(f)

        return [
            idx_to_name.get(str(i), f"class_{i}")
            for i in range(num_classes)
        ]

    names_path = os.path.join(data_dir, "activity_names.npy")

    if os.path.exists(names_path):
        names = np.load(names_path, allow_pickle=True).tolist()
        if len(names) == num_classes:
            return [str(x) for x in names]

    return [f"class_{i}" for i in range(num_classes)]

# def load_class_names(data_dir: str, num_classes: int) -> List[str]:
#     activity_names = {
#         1: "lying",
#         2: "sitting",
#         3: "standing",
#         4: "walking",
#         5: "running",
#         6: "cycling",
#         7: "Nordic walking",
#         9: "watching TV",
#         10: "computer work",
#         11: "car driving",
#         12: "ascending stairs",
#         13: "descending stairs",
#         16: "vacuum cleaning",
#         17: "ironing",
#         18: "folding laundry",
#         19: "house cleaning",
#         20: "playing soccer",
#         24: "rope jumping",
#     }
#
#     mapping_path = os.path.join(data_dir, "class_mapping.json")
#
#     if not os.path.exists(mapping_path):
#         return [f"class_{i}" for i in range(num_classes)]
#
#     with open(mapping_path, "r") as f:
#         class_to_idx = json.load(f)
#
#     idx_to_name = [""] * num_classes
#
#     for raw_id_str, idx in class_to_idx.items():
#         raw_id = int(raw_id_str)
#         idx = int(idx)
#         idx_to_name[idx] = activity_names.get(raw_id, f"class_{idx}")
#
#     return idx_to_name


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

    # remove self-comparisons
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    mask = mask * logits_mask

    # numerical stability
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives_per_sample = mask.sum(dim=1)

    valid = positives_per_sample > 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives_per_sample + 1e-12)
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
    Batch-level centroid margin loss for selected class pairs.
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
# Silhouette and nearest-class analysis
# ============================================================

def compute_classwise_silhouette(X: np.ndarray, y: np.ndarray, num_classes: int):
    """
    Returns:
        class_silhouette: dict class_id -> average silhouette
    """
    if len(np.unique(y)) < 2:
        return {i: np.nan for i in range(num_classes)}

    sil = silhouette_samples(X, y, metric="euclidean")

    out = {}
    for c in range(num_classes):
        idx = y == c
        if idx.sum() == 0:
            out[c] = np.nan
        else:
            out[c] = float(np.mean(sil[idx]))

    return out


def compute_nearest_classes(X: np.ndarray, y: np.ndarray, num_classes: int):
    """
    Finds nearest class centroid for each class.
    """
    centroids = {}

    for c in range(num_classes):
        idx = y == c
        if idx.sum() > 0:
            centroids[c] = X[idx].mean(axis=0)

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
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    threshold: float,
    class_names: Optional[List[str]] = None,
):
    """
    Identify pairs where class-wise silhouette is below threshold and nearest class is another class.
    """
    class_sil = compute_classwise_silhouette(X, y, num_classes)
    nearest = compute_nearest_classes(X, y, num_classes)

    pairs = set()

    print("\n================== FROZEN EMBEDDING ANALYSIS ==================")
    print(f"Silhouette threshold: {threshold}")
    print(f"{'class':>5s} | {'name':25s} | {'silhouette':>10s} | {'nearest':25s} | {'dist':>10s}")
    print("-" * 90)

    for c in range(num_classes):
        name = class_names[c] if class_names else f"class_{c}"
        sil = class_sil.get(c, np.nan)
        nearest_cls, dist = nearest.get(c, (None, np.nan))
        nearest_name = class_names[nearest_cls] if class_names and nearest_cls is not None else str(nearest_cls)

        print(f"{c:5d} | {name:25s} | {sil:10.4f} | {nearest_name:25s} | {dist:10.4f}")

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

    return pairs, class_sil, nearest


@torch.no_grad()
def extract_features_from_model(model: nn.Module, X: torch.Tensor, batch_size: int):
    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False, drop_last=False)

    feats = []
    for (xb,) in loader:
        xb = xb.to(device).float()
        z = model.forward_features(xb)
        feats.append(z.detach().cpu())

    return torch.cat(feats, dim=0).numpy()


# ============================================================
# Federated Client
# ============================================================

class AdapterFLClient:
    def __init__(
        self,
        sid: int,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        confusion_pairs: List[Tuple[int, int]],
        cfg: AdapterFLConfig,
    ):
        self.sid = int(sid)
        self.n = int(len(X))
        self.loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.model: Optional[nn.Module] = None
        self.confusion_pairs = confusion_pairs
        self.cfg = cfg

    def set_model(self, model: nn.Module):
        self.model = model.to(device)

    def train_local(
            self,
            global_state,
            local_epochs: int,
            lr: float,
            optimizer: str,
            weight_decay: float = 0.0,
    ):
        assert self.model is not None, "Client model not set."

        self.model.load_state_dict(global_state)
        self.model.train()

        if optimizer == "sgd":
            opt = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif optimizer == "adam":
            opt = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        ce_loss_fn = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for xb, yb in self.loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()

                opt.zero_grad()

                logits, feats = self.model(xb, return_features=True)

                loss_ce = ce_loss_fn(logits, yb)
                loss = loss_ce

                feats_for_metric = F.normalize(feats, dim=1)

                if self.cfg.use_supcon_loss:
                    loss_supcon = supervised_contrastive_loss(feats_for_metric, yb)
                    loss = loss + self.cfg.lambda_supcon * loss_supcon

                if self.cfg.use_confusion_loss:
                    loss_conf = confusion_margin_loss(
                        feats_for_metric,
                        yb,
                        self.confusion_pairs,
                        margin=self.cfg.confusion_margin,
                        min_samples_per_class=2,
                    )
                    loss = loss + self.cfg.gamma_conf * loss_conf

                if torch.isnan(loss):
                    print(f"[WARN] NaN loss on client {self.sid}. Skipping batch.")
                    continue

                loss.backward()
                opt.step()

        return copy.deepcopy(self.model.state_dict()), self.n


# ============================================================
# FedAvg
# ============================================================

def fedavg_weighted(global_model: nn.Module, client_updates: List[Tuple[dict, int]]):
    total_n = sum(n for _, n in client_updates)
    assert total_n > 0, "No client samples provided."

    global_state = global_model.state_dict()
    ref_state = client_updates[0][0]

    new_state = {}

    for k, v in global_state.items():
        if torch.is_floating_point(v):
            new_state[k] = torch.zeros_like(v)
        else:
            new_state[k] = ref_state[k].clone()

    for state, n in client_updates:
        w = n / total_n
        for k in new_state:
            if torch.is_floating_point(new_state[k]):
                new_state[k] += state[k].to(new_state[k].dtype) * w

    global_model.load_state_dict(new_state)
    return global_model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_full(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int, num_classes: int):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False, drop_last=False)

    all_preds = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).long()

        logits = model(xb)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = 100.0 * float((y_true == y_pred).mean())
    macro_f1 = 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    class_f1 = 100.0 * f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return acc, macro_f1, class_f1, cm


# ============================================================
# LOSO Runner
# ============================================================

def build_model(input_dim: int, num_classes: int, cfg: AdapterFLConfig):
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


def run_loso_for_experiment(
    exp_name: str,
    cfg: AdapterFLConfig,
    cache_dir: str,
    subjects: List[int],
    num_classes: int,
    class_names: List[str],
):
    print("\n" + "#" * 100)
    print(f"EXPERIMENT: {exp_name}")
    print(cfg)
    print("#" * 100)

    final_accs = []
    final_macro_f1s = []
    comm_per_fold = []
    model_mb_ref = None
    param_count_ref = None

    last_class_f1 = None
    last_cm = None

    oof_features = []
    oof_labels = []
    oof_subjects = []

    for test_sid in subjects:
        train_sids = [s for s in subjects if s != test_sid]

        X0, _ = load_subject_npz(cache_dir, train_sids[0])
        input_dim = int(X0.shape[1])

        # Load all training embeddings for confusion-pair discovery
        X_train_np = []
        y_train_np = []

        for sid in train_sids:
            Xs, ys = load_subject_npz(cache_dir, sid)
            X_train_np.append(Xs.numpy())
            y_train_np.append(ys.numpy())

        X_train_np = np.concatenate(X_train_np, axis=0)
        y_train_np = np.concatenate(y_train_np, axis=0)

        if cfg.use_confusion_loss:
            if cfg.confusion_pairs_mode == "manual":
                confusion_pairs = cfg.manual_confusion_pairs
            else:
                confusion_pairs, _, _ = discover_confusion_pairs(
                    X_train_np,
                    y_train_np,
                    num_classes=num_classes,
                    threshold=cfg.silhouette_threshold,
                    class_names=class_names,
                )
        else:
            confusion_pairs = []

        global_model = build_model(input_dim, num_classes, cfg).to(device)

        model_mb = state_dict_size_mb(global_model)
        param_count = count_trainable_params(global_model)

        model_mb_ref = model_mb
        param_count_ref = param_count

        clients = []
        for sid in train_sids:
            Xs, ys = load_subject_npz(cache_dir, sid)
            c = AdapterFLClient(
                sid=sid,
                X=Xs,
                y=ys,
                batch_size=cfg.batch_size,
                confusion_pairs=confusion_pairs,
                cfg=cfg,
            )
            c.set_model(build_model(input_dim, num_classes, cfg))
            clients.append(c)

        Xtest, ytest = load_subject_npz(cache_dir, test_sid)

        print("\n" + "=" * 100)
        print(
            f"[LOSO-FL] exp={exp_name} | test_subject={test_sid} | "
            f"model_size={pretty_mb(model_mb)} | trainable_params={param_count}"
        )
        print("=" * 100)

        total_comm_mb = 0.0
        final_metrics = None

        for r in range(1, cfg.rounds + 1):
            if cfg.client_frac >= 1.0:
                selected = clients
            else:
                m = max(1, int(cfg.client_frac * len(clients)))
                selected = random.sample(clients, m)

            m = len(selected)
            round_comm = 2.0 * m * model_mb
            total_comm_mb += round_comm

            updates = []
            global_state = copy.deepcopy(global_model.state_dict())

            for c in selected:
                upd, n = c.train_local(
                    global_state,
                    local_epochs=cfg.local_epochs,
                    lr=cfg.lr,
                    optimizer=cfg.optimizer,
                    weight_decay=cfg.weight_decay,
                )
                updates.append((upd, n))

            fedavg_weighted(global_model, updates)

            if cfg.eval_every and (r == 1 or r % cfg.eval_every == 0 or r == cfg.rounds):
                acc, macro_f1, _, _ = evaluate_full(
                    global_model,
                    Xtest,
                    ytest,
                    cfg.eval_batch_size,
                    num_classes,
                )
                print(
                    f"Round {r:02d} | acc={acc:.2f}% | macroF1={macro_f1:.2f}% | "
                    f"round_comm={pretty_mb(round_comm)}"
                )
                final_metrics = (acc, macro_f1)

        acc, macro_f1, class_f1, cm = evaluate_full(
            global_model,
            Xtest,
            ytest,
            cfg.eval_batch_size,
            num_classes,
        )

        final_accs.append(acc)
        final_macro_f1s.append(macro_f1)
        comm_per_fold.append(total_comm_mb)

        last_class_f1 = class_f1
        last_cm = cm

        print(f"✅ Final held-out subject {test_sid} accuracy: {acc:.2f}% | macroF1: {macro_f1:.2f}%")

        # after-adaptation silhouette on held-out subject
        Xtest_feats = extract_features_from_model(global_model, Xtest, cfg.eval_batch_size)
        ytest_np = ytest.numpy()

        oof_features.append(Xtest_feats.astype(np.float32))
        oof_labels.append(ytest_np.astype(np.int64))
        oof_subjects.append(
            np.full((len(ytest_np),), int(test_sid), dtype=np.int64)
        )

        try:
            adapted_sil = compute_classwise_silhouette(Xtest_feats, ytest_np, num_classes)
            nearest_after = compute_nearest_classes(Xtest_feats, ytest_np, num_classes)

            print("\nAdapted embedding analysis on held-out subject:")
            for c in range(num_classes):
                cname = class_names[c] if class_names else f"class_{c}"
                sil = adapted_sil.get(c, np.nan)
                ncls, dist = nearest_after.get(c, (None, np.nan))
                nname = class_names[ncls] if class_names and ncls is not None else str(ncls)
                print(f"  class={c:02d} {cname:25s} | sil={sil:.4f} | nearest={nname:25s} | dist={dist:.4f}")
        except Exception as e:
            print(f"[WARN] Could not compute adapted silhouette for subject {test_sid}: {e}")

    mean_acc = float(np.mean(final_accs))
    std_acc = float(np.std(final_accs))
    mean_f1 = float(np.mean(final_macro_f1s))
    std_f1 = float(np.std(final_macro_f1s))
    avg_comm = float(np.mean(comm_per_fold))

    print("\n" + "-" * 100)
    print(f"[FINAL] {exp_name} | acc={mean_acc:.2f}% ± {std_acc:.2f}% | macroF1={mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"[COMM ] model_size={pretty_mb(model_mb_ref or 0.0)} | avg_comm/LOSO={pretty_mb(avg_comm)}")
    print(f"[PARAM] trainable_params={param_count_ref}")
    print("-" * 100)

    print("\nClass-wise F1 from last LOSO fold:")
    if last_class_f1 is not None:
        for i, f1v in enumerate(last_class_f1):
            cname = class_names[i] if class_names else f"class_{i}"
            print(f"  {i:02d} {cname:25s}: {f1v:.2f}%")

    print("\nConfusion matrix from last LOSO fold:")
    if last_cm is not None:
        print(last_cm)

    out_dir = os.path.join("results", "rq3_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    if len(oof_features) > 0:
        oof_features_np = np.concatenate(oof_features, axis=0)
        oof_labels_np = np.concatenate(oof_labels, axis=0)
        oof_subjects_np = np.concatenate(oof_subjects, axis=0)

        out_path = os.path.join(out_dir, f"{exp_name}_oof_adapted_embeddings.npz")

        np.savez_compressed(
            out_path,
            X=oof_features_np,
            y=oof_labels_np,
            subjects=oof_subjects_np,
            experiment=exp_name,
        )

        print(f"\n[SAVED] {out_path}")
        print(f"X={oof_features_np.shape}, y={oof_labels_np.shape}, subjects={oof_subjects_np.shape}")

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "model_mb": float(model_mb_ref or 0.0),
        "avg_comm": avg_comm,
        "params": int(param_count_ref or 0),
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "local_epochs": cfg.local_epochs,
        "lambda_supcon": cfg.lambda_supcon if cfg.use_supcon_loss else None,
    }


# ============================================================
# Main
# ============================================================

def main():
    DATA_DIR = get_data_dir(DATASET)
    cache_dir = os.path.join(DATA_DIR, "mantis_subject_cache")

    if not os.path.exists(cache_dir):
        raise RuntimeError(f"Missing cache directory: {cache_dir}")

    subjects_path = os.path.join(DATA_DIR, "subjects.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    if not os.path.exists(subjects_path):
        raise RuntimeError("Missing data/subjects.npy")
    if not os.path.exists(y_path):
        raise RuntimeError("Missing data/y.npy")

    subjects = np.unique(np.load(subjects_path)).tolist()
    num_classes = int(len(np.unique(np.load(y_path))))
    class_names = load_class_names(DATA_DIR, num_classes)

    print("Subjects:", subjects)
    print("Number of classes:", num_classes)
    print("Class names:", class_names)

    all_results = {}

    for exp_name in MODELS_TO_RUN:
        base_cfg = EXPERIMENTS[exp_name]

        lambda_values = SUPCON_LAMBDAS if base_cfg.use_supcon_loss else [base_cfg.lambda_supcon]

        for tune in TUNING_PRESETS:
            for lam in lambda_values:
                cfg = copy.deepcopy(base_cfg)

                cfg.optimizer = tune["optimizer"]
                cfg.lr = tune["lr"]
                cfg.local_epochs = tune["local_epochs"]

                if cfg.use_supcon_loss:
                    cfg.lambda_supcon = lam
                    tuned_name = f"{exp_name}__{tune['suffix']}__lam{lam}"
                else:
                    tuned_name = f"{exp_name}__{tune['suffix']}"

                cfg.run_name = tuned_name

                set_seed(cfg.seed)

                result = run_loso_for_experiment(
                    exp_name=tuned_name,
                    cfg=cfg,
                    cache_dir=cache_dir,
                    subjects=subjects,
                    num_classes=num_classes,
                    class_names=class_names,
                )

                all_results[tuned_name] = result

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(
        f"{'experiment':60s} | {'acc':>14s} | {'macroF1':>14s} | "
        f"{'size':>10s} | {'avg_comm/LOSO':>15s} | {'params':>12s}"
    )
    print("-" * 120)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]["mean_acc"], reverse=True):
        print(
            f"{name:60s} | "
            f"{r['mean_acc']:6.2f}% ± {r['std_acc']:5.2f}% | "
            f"{r['mean_macro_f1']:6.2f}% ± {r['std_macro_f1']:5.2f}% | "
            f"{pretty_mb(r['model_mb']):>10s} | "
            f"{pretty_mb(r['avg_comm']):>15s} | "
            f"{r['params']:12d}"
        )

    print("=" * 120)

    print("\n================== EXPERIMENT CONFIGURATION ==================")
    print("Experiment type: Confusion-aware contrastive federated MANTIS adaptation")
    print("Evaluation protocol: LOSO-FL")
    print(f"Data directory: {DATA_DIR}")
    print(f"Embedding cache directory: {cache_dir}")
    print(f"Subjects: {subjects}")
    print(f"Number of classes: {num_classes}")
    print(f"Models run: {MODELS_TO_RUN}")
    print("Frozen backbone: MANTIS")
    print("Trainable components: adapter + classifier head")
    print("Aggregation: weighted FedAvg")
    print("Tuning presets:")
    for preset in TUNING_PRESETS:
        print(
            f"  {preset['suffix']}: "
            f"optimizer={preset['optimizer']}, "
            f"lr={preset['lr']}, "
            f"local_epochs={preset['local_epochs']}"
        )

    print(f"SupCon lambdas tested: {SUPCON_LAMBDAS}")
    print("Communication accounting: downlink + uplink adapter/head weights")
    print("==============================================================\n")


if __name__ == "__main__":
    main()