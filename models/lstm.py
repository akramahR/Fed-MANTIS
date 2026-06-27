# models/lstm.py
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 51,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,          # expects [B, T, F]
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Safety: if it looks like [B, F, T], fix to [B, T, F]
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            # e.g. [B, 51, 256] -> [B, 256, 51]
            x = x.permute(0, 2, 1)

        # LSTM output: [B, T, hidden_dim * num_directions]
        out, _ = self.lstm(x)

        # Global average pooling over time dimension
        # out_mean: [B, hidden_dim * num_directions]
        out_mean = out.mean(dim=1)

        logits = self.fc(out_mean)
        return logits
