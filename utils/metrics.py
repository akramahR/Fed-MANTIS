import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_metrics(y_true, y_pred, average="macro"):
    """
    Compute classification metrics.
    Args:
        y_true: torch.Tensor or numpy array (ground truth labels)
        y_pred: torch.Tensor or numpy array (predicted labels)
        average: averaging method for multiclass ('macro', 'micro', 'weighted')
    Returns:
        dict of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
