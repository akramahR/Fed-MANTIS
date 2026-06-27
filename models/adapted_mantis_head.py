import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    """
    Simple projection adapter:
        z: [B, input_dim] -> [B, adapter_dim]

    """

    def __init__(self, input_dim: int, adapter_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.net(z)


class MLPAdapter(nn.Module):
    """
    Larger MLP adapter:
        z: [B, input_dim] -> [B, adapter_dim]

    """

    def __init__(
        self,
        input_dim: int,
        adapter_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.net(z)


class LowRankAdapter(nn.Module):
    """
    Low-rank projection adapter:
        z: [B, input_dim] -> [B, rank] -> [B, adapter_dim]

    """

    def __init__(
        self,
        input_dim: int,
        adapter_dim: int = 512,
        rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, rank, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rank, adapter_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.net(z)


class ResidualBottleneckAdapter(nn.Module):
    """
    Residual bottleneck adapter:
        z: [B, input_dim]
        down: input_dim -> bottleneck_dim
        up: bottleneck_dim -> input_dim
        output: [B, input_dim]

    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, input_dim, bias=False)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = nn.LayerNorm(input_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, z):
        residual = z

        out = self.down(z)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up(out)

        out = residual + out
        out = self.norm(out)

        return out


class IdentityAdapter(nn.Module):
    """
    Explicit identity adapter for no-adapter baseline.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z


class AdaptedMantisClassifier(nn.Module):
    """
    Frozen MANTIS embeddings -> optional trainable adapter -> classifier head.

    Input:
        z: [B, D] frozen MANTIS embedding

    Output:
        logits: [B, num_classes]
        adapted embedding:
            - [B, adapter_dim] for linear/mlp/lowrank adapters
            - [B, input_dim] for residual_bottleneck or no adapter
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_adapter: bool = True,
        adapter_type: str = "linear",
        adapter_dim: int = 512,
        adapter_rank: int = 64,
        adapter_bottleneck_dim: int = 64,
        head_type: str = "linear",
        dropout: float = 0.3,
        normalize_features_for_head: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_adapter = use_adapter
        self.adapter_type = adapter_type
        self.head_type = head_type
        self.normalize_features_for_head = normalize_features_for_head

        if not use_adapter:
            self.adapter = IdentityAdapter()
            head_input_dim = input_dim

        else:
            if adapter_type == "linear":
                self.adapter = LinearAdapter(
                    input_dim=input_dim,
                    adapter_dim=adapter_dim,
                )
                head_input_dim = adapter_dim

            elif adapter_type == "mlp":
                self.adapter = MLPAdapter(
                    input_dim=input_dim,
                    adapter_dim=adapter_dim,
                    hidden_dim=512,
                    dropout=dropout,
                )
                head_input_dim = adapter_dim

            elif adapter_type == "lowrank":
                self.adapter = LowRankAdapter(
                    input_dim=input_dim,
                    adapter_dim=adapter_dim,
                    rank=adapter_rank,
                    dropout=dropout,
                )
                head_input_dim = adapter_dim

            elif adapter_type == "residual_bottleneck":
                self.adapter = ResidualBottleneckAdapter(
                    input_dim=input_dim,
                    bottleneck_dim=adapter_bottleneck_dim,
                    dropout=dropout,
                    use_layernorm=True,
                )
                head_input_dim = input_dim

            else:
                raise ValueError(
                    f"Unknown adapter_type: {adapter_type}. "
                    "Expected one of: linear, mlp, lowrank, residual_bottleneck."
                )

        if head_type == "linear":
            self.classifier = nn.Linear(head_input_dim, num_classes)

        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(head_input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )

        else:
            raise ValueError(
                f"Unknown head_type: {head_type}. "
                "Expected one of: linear, mlp."
            )

    def forward_features(self, z):
        """
        Returns adapted features.

        """
        return self.adapter(z)

    def forward(self, z, return_features: bool = False):
        z_adapted = self.forward_features(z)

        if self.normalize_features_for_head:
            logits_input = F.normalize(z_adapted, dim=1)
        else:
            logits_input = z_adapted

        logits = self.classifier(logits_input)

        if return_features:
            return logits, z_adapted

        return logits