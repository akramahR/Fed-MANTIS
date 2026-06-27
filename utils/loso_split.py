import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import PAMAP2Dataset

def loso_split(X, y, subjects, test_subject, batch_size=64, seed=42, dataset_cls=None):
    """
    Leave-One-Subject-Out (LOSO) split
    dataset_cls: Dataset class to wrap (X, y).
      - default PAMAP2Dataset for [N, T, F]
      - use EmbeddingDataset for [N, D]
    """
    if dataset_cls is None:
        dataset_cls = PAMAP2Dataset

    train_idx = subjects != test_subject
    test_idx = subjects == test_subject

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_dataset = dataset_cls(X_train, y_train)
    test_dataset = dataset_cls(X_test, y_test)

    # reproducible shuffle
    g = None
    try:
        import torch
        g = torch.Generator()
        g.manual_seed(seed)
    except Exception:
        g = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
