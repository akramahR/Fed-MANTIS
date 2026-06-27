import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """
    For precomputed embeddings.
    X: [N, D]
    y: [N]
    Returns:
      x: torch.FloatTensor [D]
      y: torch.LongTensor scalar
    """
    def __init__(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Expected X shape [N, D], got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y shape [N], got {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)} vs len(y)={len(y)}")

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()   # [D]
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y
