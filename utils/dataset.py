import numpy as np
import torch
from torch.utils.data import Dataset

class PAMAP2Dataset(Dataset):
    """
    Returns:
      X: torch.FloatTensor [T, F]
      y: torch.LongTensor scalar
    """
    def __init__(self, X, y, transform=None):
        # Keep as numpy to avoid copying entire array into torch up front
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"Expected X shape [N, T, F], got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y shape [N], got {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)} vs len(y)={len(y)}")

        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.X[idx]).float()   # [T, F]
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
