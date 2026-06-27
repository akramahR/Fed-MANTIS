import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_dim=51, num_classes=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        return self.net(x)
