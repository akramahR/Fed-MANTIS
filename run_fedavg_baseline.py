"""
run_fedavg_baseline.py

FedAvg baselines on PAMAP2 raw windows (LOSO-FL).

- Partitions data by subject => one subject is one client
- LOSO-FL: for each held-out subject, train on remaining subjects with FedAvg
- Tests on the held-out subject
- Full-model FedAvg baseline
- Standardized logging:
    - Accuracy mean ± std
    - Macro-F1 mean ± std
    - Model/update size
    - Average communication per LOSO fold
    - Trainable parameter count

Expected:
- data/X.npy:        (N, T, F)
- data/y.npy:        (N,)
- data/subjects.npy: (N,)
"""

import os
import copy
import random
from dataclasses import dataclass
from typing import Dict, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix

from models.cnn import CNNModel
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.inceptiontime import InceptionTimeModel
from models.deepConvLstmForFed import DeepConvLSTMModelTest


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

# -----------------------
# Helpers
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_dict_size_mb(model: nn.Module) -> float:
    """
    Communication size for full-model FedAvg.
    Uses state_dict tensors because full model weights/buffers are exchanged.
    """
    total_bytes = 0

    for t in model.state_dict().values():
        if torch.is_tensor(t):
            total_bytes += t.numel() * t.element_size()

    return total_bytes / (1024 ** 2)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretty_mb(x: float) -> str:
    return f"{x:.3f} MB"


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    num_classes: int,
):
    """
    Standard evaluation used across experiment scripts.

    Returns:
        acc: float, percentage
        macro_f1: float, percentage
        class_f1: np.ndarray, percentage per class
        cm: np.ndarray, confusion matrix
    """
    model.eval()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_preds = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).long()

        outputs = model(xb)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        preds = outputs.argmax(dim=1)

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


def fedavg_weighted(global_model: nn.Module, client_updates: List[Tuple[dict, int]]):
    """
    Weighted FedAvg that safely handles non-float buffers
    such as BatchNorm num_batches_tracked.
    """
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

        for k in new_state.keys():
            if torch.is_floating_point(new_state[k]):
                new_state[k] += state[k].to(new_state[k].dtype) * w

    global_model.load_state_dict(new_state)
    return global_model


# -----------------------
# Federated client
# -----------------------
class FLClient:
    def __init__(self, sid: int, X: torch.Tensor, y: torch.Tensor, batch_size: int = 64):
        self.sid = int(sid)
        self.n = int(len(X))

        ds = TensorDataset(X, y)

        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model: nn.Module | None = None

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


        loss_fn = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for xb, yb in self.loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()

                opt.zero_grad()
                logits = self.model(xb)

                if isinstance(logits, tuple):
                    logits = logits[0]
                if hasattr(logits, "logits"):
                    logits = logits.logits

                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        return copy.deepcopy(self.model.state_dict()), self.n


# -----------------------
# Config
# -----------------------
@dataclass
class FLConfig:
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 1e-2
    optimizer: str = "sgd"
    weight_decay: float = 0.0
    client_frac: float = 1.0
    batch_size: int = 64
    eval_batch_size: int = 256
    seed: int = 0
    eval_every: int = 5
    run_name: str = "default"


# -----------------------
# LOSO-FL runner
# -----------------------
def run_loso_fedavg_for_model(
    model_name: str,
    build_model: Callable[[int], nn.Module],
    num_classes: int,
    X_all: torch.Tensor,
    y_all: torch.Tensor,
    subjects_all: np.ndarray,
    unique_subjects: List[int],
    cfg: FLConfig,
):
    final_accs: List[float] = []
    final_macro_f1s: List[float] = []
    comm_per_fold_mb: List[float] = []

    model_copy_mb: float | None = None
    param_count_ref: int | None = None

    last_class_f1 = None
    last_cm = None

    for test_sid in unique_subjects:
        train_sids = [s for s in unique_subjects if s != test_sid]

        # Build clients from raw data
        clients: List[FLClient] = []

        for sid in train_sids:
            idx = np.where(subjects_all == sid)[0]
            Xs = X_all[idx]
            ys = y_all[idx]

            c = FLClient(
                sid=sid,
                X=Xs,
                y=ys,
                batch_size=cfg.batch_size,
            )

            c.set_model(build_model(num_classes))
            clients.append(c)

        # Held-out test subject
        idx_test = np.where(subjects_all == test_sid)[0]
        Xtest = X_all[idx_test]
        ytest = y_all[idx_test]

        # Global model
        global_model = build_model(num_classes).to(device)

        model_mb = state_dict_size_mb(global_model)
        param_count = count_trainable_params(global_model)

        model_copy_mb = model_mb
        param_count_ref = param_count

        print("\n" + "=" * 100)
        print(
            f"[LOSO-FedAvg] model={model_name} | test_subject={test_sid} | "
            f"model_copy={pretty_mb(model_mb)} | trainable_params={param_count:,}"
        )
        print("=" * 100)

        total_comm_mb = 0.0

        for r in range(1, cfg.rounds + 1):
            if cfg.client_frac >= 1.0:
                selected = clients
            else:
                m = max(1, int(cfg.client_frac * len(clients)))
                selected = random.sample(clients, m)

            m = len(selected)

            # Communication accounting: downlink + uplink full model copy.
            round_comm = 2.0 * m * model_mb
            total_comm_mb += round_comm

            global_state = copy.deepcopy(global_model.state_dict())

            updates = []

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
                acc_mid, macro_f1_mid, _, _ = evaluate_full(
                    global_model,
                    Xtest,
                    ytest,
                    batch_size=cfg.eval_batch_size,
                    num_classes=num_classes,
                )

                print(
                    f"Round {r:02d} | acc={acc_mid:.2f}% | "
                    f"macroF1={macro_f1_mid:.2f}% | "
                    f"round_comm={pretty_mb(round_comm)}"
                )

        final_acc, final_macro_f1, class_f1, cm = evaluate_full(
            global_model,
            Xtest,
            ytest,
            batch_size=cfg.eval_batch_size,
            num_classes=num_classes,
        )

        final_accs.append(final_acc)
        final_macro_f1s.append(final_macro_f1)
        comm_per_fold_mb.append(total_comm_mb)

        last_class_f1 = class_f1
        last_cm = cm

        print(
            f"Final held-out subject {test_sid} accuracy: {final_acc:.2f}% | "
            f"macroF1: {final_macro_f1:.2f}%"
        )

    mean_acc = float(np.mean(final_accs))
    std_acc = float(np.std(final_accs))
    mean_f1 = float(np.mean(final_macro_f1s))
    std_f1 = float(np.std(final_macro_f1s))
    mean_comm = float(np.mean(comm_per_fold_mb))

    print("\n" + "#" * 100)
    print(
        f"[FINAL] {model_name} | "
        f"acc={mean_acc:.2f}% ± {std_acc:.2f}% | "
        f"macroF1={mean_f1:.2f}% ± {std_f1:.2f}%"
    )
    print(
        f"[COMM ] {model_name} | model_copy={pretty_mb(model_copy_mb or 0.0)} | "
        f"avg_comm/LOSO={pretty_mb(mean_comm)}"
    )
    print(f"[PARAM] trainable_params={param_count_ref}")
    print("#" * 100)

    print("\nClass-wise F1 from last LOSO fold:")
    if last_class_f1 is not None:
        for i, f1v in enumerate(last_class_f1):
            print(f"  class_{i:02d}: {f1v:.2f}%")

    print("\nConfusion matrix from last LOSO fold:")
    if last_cm is not None:
        print(last_cm)

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "model_mb": float(model_copy_mb or 0.0),
        "avg_comm": mean_comm,
        "params": int(param_count_ref or 0),
    }


# -----------------------
# Main
# -----------------------
def main():
    TUNING_PRESETS = [
        FLConfig(run_name="sgd_lr1e2_e1", optimizer="sgd", lr=1e-2, local_epochs=1),
        FLConfig(run_name="sgd_lr1e3_e1", optimizer="sgd", lr=1e-3, local_epochs=1),
        FLConfig(run_name="adam_lr1e3_e1", optimizer="adam", lr=1e-3, local_epochs=1),
        FLConfig(run_name="adam_lr1e3_e3", optimizer="adam", lr=1e-3, local_epochs=3),
        FLConfig(run_name="sgd_lr1e2_e3", optimizer="sgd", lr=1e-2, local_epochs=3),
        FLConfig(run_name="sgd_lr1e3_e3", optimizer="sgd", lr=1e-3, local_epochs=3),
        FLConfig(run_name="adam_lr1e4_e1", optimizer="adam", lr=1e-4, local_epochs=1),
        FLConfig(run_name="adam_lr1e4_e3", optimizer="adam", lr=1e-4, local_epochs=3),
    ]


    DATA_DIR = get_data_dir(DATASET)

    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    s_path = os.path.join(DATA_DIR, "subjects.npy")

    X = np.load(X_path).astype(np.float32)          # (N, T, F)
    y = np.load(y_path).astype(np.int64)          # (N,)
    subjects = np.load(s_path).astype(np.int64)   # (N,)

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    seq_len = int(X_t.shape[1])
    input_dim = int(X_t.shape[2])
    num_classes = int(len(np.unique(y)))
    unique_subjects = np.unique(subjects).tolist()

    print(
        f"X: {tuple(X_t.shape)} | seq_len={seq_len} | "
        f"input_dim={input_dim} | classes={num_classes}"
    )
    print(f"subjects: {unique_subjects}")

    models: Dict[str, Callable[[int], nn.Module]] = {
        "cnn": lambda nc: CNNModel(
            input_dim=input_dim,
            num_classes=nc,
        ),

        "lstm": lambda nc: LSTMModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=nc,
            dropout=0.2,
            bidirectional=True,
        ),

        "inceptiontime": lambda nc: InceptionTimeModel(
            input_dim=input_dim,
            num_classes=nc,
            bottleneck_channels=32,
            num_blocks=6,
            use_residual=True,
            dropout=0.0,
        ),


        # Optional/debug variants:
        "transformer": lambda nc: TransformerModel(
            input_dim=input_dim,
            num_classes=nc,
        ),

        "deepconvlstm": lambda nc: DeepConvLSTMModelTest(
            input_dim=input_dim,
            num_classes=nc,
            conv_channels=32,
            hidden_dim=64,
            num_layers=1,
            dropout=0.1,
            bidirectional=False,
            pooling="mean_max",
        ),
    }


    MODELS_TO_RUN = [
        "cnn",
        "lstm",
        "inceptiontime",
        "deepconvlstm",
    ]

    results = {}

    for cfg in TUNING_PRESETS:
        print("\n" + "=" * 120)
        print(f"TUNING RUN: {cfg.run_name}")
        print(cfg)
        print("=" * 120)

        set_seed(cfg.seed)

        for name in MODELS_TO_RUN:
            full_name = f"{name}__{cfg.run_name}"

            result = run_loso_fedavg_for_model(
                model_name=full_name,
                build_model=models[name],
                num_classes=num_classes,
                X_all=X_t,
                y_all=y_t,
                subjects_all=subjects,
                unique_subjects=unique_subjects,
                cfg=cfg,
            )

            results[full_name] = result

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(
        f"{'model':32s} | {'acc':>14s} | {'macroF1':>14s} | "
        f"{'size':>10s} | {'avg_comm/LOSO':>15s} | {'params':>12s}"
    )
    print("-" * 120)

    for name, r in sorted(results.items(), key=lambda x: x[1]["mean_acc"], reverse=True):
        print(
            f"{name:32s} | "
            f"{r['mean_acc']:6.2f}% ± {r['std_acc']:5.2f}% | "
            f"{r['mean_macro_f1']:6.2f}% ± {r['std_macro_f1']:5.2f}% | "
            f"{pretty_mb(r['model_mb']):>10s} | "
            f"{pretty_mb(r['avg_comm']):>15s} | "
            f"{r['params']:12d}"
        )

    print("=" * 120)

    print("\n================== EXPERIMENT CONFIGURATION ==================")
    print("Experiment type: Federated raw-window baseline tuning")
    print("Evaluation protocol: LOSO-FL")
    print(f"Dataset: {DATASET}")
    print(f"Device: {device}")
    print(f"Input shape: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
    print(f"Sequence length: {seq_len}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Subjects: {unique_subjects}")
    print(f"Models run: {MODELS_TO_RUN}")
    print("Client definition: one subject = one client")
    print("Aggregation: weighted FedAvg")
    print("Communication accounting: downlink + uplink model weights")
    print("Tuning presets:")

    for preset in TUNING_PRESETS:
        print(
            f"  {preset.run_name}: "
            f"optimizer={preset.optimizer}, "
            f"lr={preset.lr}, "
            f"local_epochs={preset.local_epochs}, "
            f"rounds={preset.rounds}, "
            f"batch_size={preset.batch_size}, "
            f"eval_every={preset.eval_every}"
        )

    print("==============================================================\n")


if __name__ == "__main__":
    main()
