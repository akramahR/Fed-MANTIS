import torch
import torch.nn as nn
from copy import deepcopy

def _get_logits(outputs):
    # Handle tuple outputs (e.g., (logits, ...))
    if isinstance(outputs, tuple):
        return outputs[0]
    # Handle HF outputs with .logits
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs

def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=10,
    lr=1e-3,
    device=None,
    patience=5,
    use_early_stopping=True,
    optimizer_name="sgd",
    weight_decay=0.0,
):
    """
    Generic training loop with optional early stopping.

    For LOSO experiments, early stopping should use a validation split from
    the training subjects rather than the held-out test subject.

    optimizer_name:
        "sgd"  -> SGD(momentum=0.9)
        "adam" -> Adam
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    print(
        f"Optimizer: {optimizer_name} | "
        f"lr={lr} | weight_decay={weight_decay}"
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).long()

            optimizer.zero_grad()
            outputs = _get_logits(model(X_batch))

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

        train_acc = 100 * total_correct / total_samples
        train_loss = total_loss / total_samples

        if val_loader is None:
            print(
                f"Epoch [{epoch}/{epochs}]  "
                f"Train Loss: {train_loss:.4f}  "
                f"Train Acc: {train_acc:.2f}%"
            )
            continue

        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.2f}%"
        )

        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}\n")
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    print("Training complete.\n")
    return model

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).long()

            outputs = _get_logits(model(X_batch))

            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    acc = 100 * total_correct / total_samples
    return acc, avg_loss
