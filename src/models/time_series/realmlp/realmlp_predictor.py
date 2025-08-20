from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np

from src.models.time_series.base_pytorch_model import PyTorchBasePredictor
from src.models.time_series.realmlp.realmlp_architecture import RealMLPModule
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealMLPPredictor(PyTorchBasePredictor):
    def __init__(self, model_name: str = "RealMLP", config: Optional[Dict[str, Any]] = None, threshold_evaluator=None):
        super().__init__(model_name=model_name, config=config or {}, threshold_evaluator=threshold_evaluator)

    def _create_model(self) -> nn.Module:
        cfg = self.config
        input_size = cfg.get("input_size")
        if input_size is None:
            raise ValueError("Config must include 'input_size'")
        hidden_sizes = cfg.get("hidden_sizes", [512, 256, 128, 64])
        activation = cfg.get("activation", "gelu")
        dropout = cfg.get("dropout", 0.1)
        batch_norm = cfg.get("batch_norm", True)
        use_diagonal = cfg.get("use_diagonal", True)
        use_numeric_embedding = cfg.get("use_numeric_embedding", True)
        numeric_embedding_dim = cfg.get("numeric_embedding_dim", 16)

        num_categories = self.config.get("num_categories")  # ticker vocab size incl. OOV
        cat_embed_dim = self.config.get("cat_embed_dim", 32)
        embedding_dropout = self.config.get("embedding_dropout", 0.1)

        return RealMLPModule(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            use_diagonal=use_diagonal,
            use_numeric_embedding=use_numeric_embedding,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_embedding_out_dim=hidden_sizes[0] if use_numeric_embedding else input_size,
            num_categories=num_categories,
            cat_embed_dim=cat_embed_dim,
            embedding_dropout=embedding_dropout,
        )


