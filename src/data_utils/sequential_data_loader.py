"""
Sequential Data Loader Module - Memory Optimized

This module provides a memory-efficient custom PyTorch Dataset for creating sequential
time-series data suitable for RNNs, LSTMs, and Transformers.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple

class TimeSeriesDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for creating overlapping time-series sequences.
    
    Key optimizations:
    - Stores data as numpy arrays instead of PyTorch tensors
    - Converts to tensors on-demand during __getitem__
    - Uses float32 to reduce memory usage
    """
    def __init__(self, features: pd.DataFrame, targets: pd.Series, sequence_length: int = 30):
        """
        Args:
            features (pd.DataFrame): DataFrame of feature data.
            targets (pd.Series): Series of target data.
            sequence_length (int): The number of time steps to include in each sequence.
        """
        if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.Series):
            raise TypeError("Features must be a pandas DataFrame and targets a pandas Series.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        
        # 🔧 MEMORY OPTIMIZATION: Store as numpy arrays, convert to tensors on-demand
        self.features = features.values.astype(np.float32)  # More memory efficient
        self.targets = targets.values.astype(np.float32)
        self.sequence_length = sequence_length
        
        # Log memory usage for debugging
        features_mb = self.features.nbytes / (1024 * 1024)
        targets_mb = self.targets.nbytes / (1024 * 1024)
        print(f"📊 TimeSeriesDataset memory: Features={features_mb:.1f}MB, Targets={targets_mb:.1f}MB")

    def __len__(self) -> int:
        """
        Returns the total number of samples (sequences) in the dataset.
        """
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sequence and its corresponding target.
        
        🔧 OPTIMIZATION: Convert to tensors on-demand instead of storing them upfront
        
        Args:
            idx: The index of the sample to retrieve.
            
        Returns:
            A tuple containing:
            - The feature sequence (shape: [sequence_length, num_features]).
            - The target value.
        """
        sequence_start = idx
        sequence_end = idx + self.sequence_length
        target_idx = sequence_end

        if target_idx >= len(self.targets):
            raise IndexError("Target index out of range.")

        # 🔧 Convert to tensors only when needed (during training)
        X = torch.tensor(self.features[sequence_start:sequence_end], dtype=torch.float32)
        y = torch.tensor(self.targets[target_idx], dtype=torch.float32)
        
        return X, y 