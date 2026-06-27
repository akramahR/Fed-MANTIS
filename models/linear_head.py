import torch
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Linear Probing = Single Affine Transformation (Wx + b)
        # This is mathematically equivalent to Logistic Regression
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)