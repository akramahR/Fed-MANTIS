# models/transformer.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        # pe[:T] -> [T, d_model], unsqueeze -> [1, T, d_model] for broadcasting
        return x + self.pe[:T].unsqueeze(0)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()

        # project input features (channels) -> hidden size
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,  # expect [B, T, H]
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [B, T, F]  (batch, time, features)
        """
        x = self.embedding(x)        # [B, T, H]
        x = self.pos_encoder(x)      # [B, T, H]
        x = self.transformer(x)      # [B, T, H]

        # Global average pooling over time
        x = x.mean(dim=1)            # [B, H]

        out = self.fc(x)             # [B, num_classes]
        return out
