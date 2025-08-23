import torch
import torch.nn as nn


class DiagonalFeatureScaler(nn.Module):
    """
    Learnable per-feature scaling: y = x * s
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class NumericEmbedding(nn.Module):
    """
    Optional numeric embedding: project per-feature to small dim, GELU, then concat and project.
    """

    def __init__(self, num_features: int, embed_dim: int, out_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.per_feature = nn.Linear(1, embed_dim)
        self.activation = nn.GELU()
        self.proj = nn.Linear(num_features * embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        b, f = x.shape
        x_expanded = x.view(b, f, 1)
        z = self.per_feature(x_expanded)
        z = self.activation(z)
        z = z.reshape(b, f * self.embed_dim)
        return self.proj(z)


