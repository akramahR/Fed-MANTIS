# models/inceptiontime.py
import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """
    InceptionTime-style block for 1D time series.

    Input:  [B, C, T]
    Output: [B, 4 * bottleneck_channels, T]
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 32,
        kernel_sizes=(9, 19, 39),
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.use_bottleneck = use_bottleneck and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            conv_in_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_in_channels = in_channels

        self.conv1 = nn.Conv1d(
            conv_in_channels,
            bottleneck_channels,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            conv_in_channels,
            bottleneck_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
            bias=False,
        )
        self.conv3 = nn.Conv1d(
            conv_in_channels,
            bottleneck_channels,
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2,
            bias=False,
        )

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
        )

        out_channels = bottleneck_channels * 4
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b = self.bottleneck(x)

        y1 = self.conv1(x_b)
        y2 = self.conv2(x_b)
        y3 = self.conv3(x_b)
        y4 = self.pool_conv(x)

        out = torch.cat([y1, y2, y3, y4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.proj = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.relu(self.proj(x) + out)


class InceptionTimeModel(nn.Module):
    """
    Drop-in InceptionTime baseline for HAR pipeline.

    Expected input:
      x: [B, T, F]  (preferred)
         or [B, F, T] (auto-fixed)

    Output:
      logits: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 51,
        num_classes: int = 12,
        bottleneck_channels: int = 32,
        num_blocks: int = 6,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.blocks = nn.ModuleList()

        in_channels = input_dim
        residual_pairs = []

        # Build blocks and record residual connections every 3 blocks
        for i in range(num_blocks):
            if i % 3 == 0:
                residual_in_channels = in_channels

            block = InceptionBlock(
                in_channels=in_channels,
                bottleneck_channels=bottleneck_channels,
                kernel_sizes=(9, 19, 39),
                use_bottleneck=True,
            )
            self.blocks.append(block)

            in_channels = block.out_channels

            if use_residual and i % 3 == 2:
                residual_pairs.append((residual_in_channels, in_channels))

        self.shortcuts = nn.ModuleList(
            [ResidualShortcut(in_ch, out_ch) for in_ch, out_ch in residual_pairs]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F] or [B, F, T]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, T, F] or [B, F, T], got {x.shape}")

        # Convert [B, T, F] -> [B, F, T] if needed
        if x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)

        shortcut_idx = 0

        for i, block in enumerate(self.blocks):
            if i % 3 == 0:
                residual_input = x

            x = block(x)

            if self.use_residual and i % 3 == 2:
                x = self.shortcuts[shortcut_idx](residual_input, x)
                shortcut_idx += 1

        x = self.global_pool(x).squeeze(-1)  # [B, C]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits