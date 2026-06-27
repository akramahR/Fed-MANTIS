import torch
import torch.nn as nn

class DeepConvLSTMModelTest(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 51,
        conv_channels: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: str = "mean_max",
    ):
        super().__init__()

        self.pooling = pooling
        self.bidirectional = bidirectional

        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)

        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(conv_channels)

        self.conv4 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(conv_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        factor = 2 if bidirectional else 1
        pooled_factor = 2 if pooling == "mean_max" else 1
        self.fc = nn.Linear(hidden_dim * factor * pooled_factor, num_classes)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, T, F] or [B, F, T], got {x.shape}")

        if x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)

        if self.pooling == "last":
            out = out[:, -1, :]
        elif self.pooling == "mean":
            out = out.mean(dim=1)
        elif self.pooling == "mean_max":
            out_mean = out.mean(dim=1)
            out_max = out.max(dim=1).values
            out = torch.cat([out_mean, out_max], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.fc(out)
        return logits