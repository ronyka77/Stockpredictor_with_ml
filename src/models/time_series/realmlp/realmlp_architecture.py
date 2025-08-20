import torch
import torch.nn as nn
from typing import List, Optional

from .realmlp_layers import DiagonalFeatureScaler, NumericEmbedding


class RealMLPModule(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = "gelu",
        dropout: float = 0.1,
        batch_norm: bool = True,
        use_diagonal: bool = True,
        use_numeric_embedding: bool = True,
        numeric_embedding_dim: int = 16,
        numeric_embedding_out_dim: Optional[int] = None,
        num_categories: Optional[int] = None,
        cat_embed_dim: int = 32,
        embedding_dropout: float = 0.1,
        output_size: int = 1) -> None:
        
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.use_diagonal = use_diagonal
        self.use_numeric_embedding = use_numeric_embedding
        self.numeric_embedding_dim = numeric_embedding_dim
        self.numeric_embedding_out_dim = numeric_embedding_out_dim or input_size

        # Categorical embedding config
        self.num_categories = num_categories
        self.cat_embed_dim = cat_embed_dim
        self.embedding_dropout_rate = embedding_dropout

        act = activation.lower()
        if act == "gelu":
            self.activation = nn.GELU()
        elif act == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        current_dim = input_size

        if self.use_numeric_embedding:
            self.num_embed = NumericEmbedding(
                input_size, numeric_embedding_dim, self.numeric_embedding_out_dim
            )
            current_dim = self.numeric_embedding_out_dim
        else:
            self.num_embed = None

        if self.use_diagonal:
            self.diag = DiagonalFeatureScaler(current_dim)
        else:
            self.diag = None

        # Optional categorical (ticker) embedding
        if self.num_categories is not None and self.num_categories > 0:
            self.cat_embedding = nn.Embedding(self.num_categories, self.cat_embed_dim)
            self.cat_embed_dropout = nn.Dropout(p=self.embedding_dropout_rate)
            trunk_input_dim = current_dim + self.cat_embed_dim
        else:
            self.cat_embedding = None
            self.cat_embed_dropout = None
            trunk_input_dim = current_dim

        dims = [trunk_input_dim] + hidden_sizes
        self.linear_blocks = nn.ModuleList()
        self.bn_blocks: Optional[nn.ModuleList] = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.linear_blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                self.bn_blocks.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropouts.append(nn.Dropout(p=dropout))

        self.out = nn.Linear(hidden_sizes[-1], output_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_num: torch.Tensor, cat_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Numeric path
        z = x_num
        if self.num_embed is not None:
            z = self.num_embed(z)
        if self.diag is not None:
            z = self.diag(z)

        # Categorical path (optional)
        if self.cat_embedding is not None:
            # Fallback: if categorical indices are not provided during inference/logging,
            # use a zero (OOV) index vector to preserve expected trunk input dimensionality
            if cat_idx is None:
                cat_idx = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            if cat_idx is not None:
                # Ensure 1D indices
                if cat_idx.dim() > 1:
                    cat_idx_flat = cat_idx.view(cat_idx.size(0))
                else:
                    cat_idx_flat = cat_idx
                e = self.cat_embedding(cat_idx_flat)
                if self.cat_embed_dropout is not None:
                    e = self.cat_embed_dropout(e)
                z = torch.cat([z, e], dim=1)

        # Trunk
        for i, lin in enumerate(self.linear_blocks):
            z = lin(z)
            if self.batch_norm and self.bn_blocks is not None:
                z = self.bn_blocks[i](z)
            z = self.activation(z)
            z = self.dropouts[i](z)
        return self.out(z)


