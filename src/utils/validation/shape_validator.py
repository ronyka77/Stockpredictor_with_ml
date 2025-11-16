"""
ShapeValidator Utility for TFT-Diff Model

This module provides comprehensive tensor shape validation for all TFT-Diff components,
providing clear error messages and shape compatibility checks to eliminate scattered
shape checking logic throughout the codebase.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ShapeValidationResult:
    """Result of a shape validation operation."""

    is_valid: bool
    expected_shape: Union[Tuple[int, ...], str]
    actual_shape: Tuple[int, ...]
    error_message: Optional[str] = None
    context: Optional[str] = None


class ShapeValidator:
    """
    Comprehensive tensor shape validator for TFT-Diff components.

    This utility centralizes all tensor shape validation logic used across TFT-Diff
    components, providing consistent error messages and validation patterns.

    Supported tensor types:
    - Temporal features: (batch_size, seq_len, input_size)
    - Static features: (batch_size, static_size)
    - Correlation matrices: (n_stocks, n_stocks) or (batch_size, n_stocks, n_stocks)
    - Graph adjacency matrices: (batch_size, n_stocks, n_stocks)
    - Node features: (batch_size, n_stocks, embed_dim) or (batch_size * n_stocks, seq_len, input_size)
    - Prediction outputs: (batch_size, num_horizons)
    """

    # Standard shape patterns for TFT-Diff components
    SHAPE_PATTERNS = {
        "temporal_features": {
            "dimensions": 3,
            "description": "Temporal features with shape (batch_size, seq_len, input_size)",
        },
        "static_features": {
            "dimensions": 2,
            "description": "Static features with shape (batch_size, static_size)",
        },
        "correlation_matrix": {
            "dimensions": [2, 3],  # Can be 2D or 3D
            "description": "Correlation matrix with shape (n_stocks, n_stocks) or (batch_size, n_stocks, n_stocks)",
        },
        "adjacency_matrix": {
            "dimensions": 3,
            "description": "Graph adjacency matrix with shape (batch_size, n_stocks, n_stocks)",
        },
        "node_features": {
            "dimensions": [3, 4],  # Can be 3D or 4D (flattened)
            "description": "Node features with shape (batch_size, n_stocks, embed_dim) or (batch_size * n_stocks, seq_len, input_size)",
        },
        "prediction_output": {
            "dimensions": [2, 3],  # Can be 2D or 3D (multi-horizon)
            "description": "Prediction output with shape (batch_size, num_horizons) or (batch_size, seq_len, num_horizons)",
        },
    }

    def __init__(self, enable_logging: bool = True):
        """
        Initialize ShapeValidator.

        Args:
            enable_logging: Whether to log validation errors
        """
        self.enable_logging = enable_logging

    def validate_temporal_features(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        input_size: Optional[int] = None,
        context: str = "temporal_features",
    ) -> ShapeValidationResult:
        """
        Validate temporal features tensor shape.

        Args:
            tensor: Input tensor
            batch_size: Expected batch size (None for any)
            seq_len: Expected sequence length (None for any)
            input_size: Expected input feature size (None for any)
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        expected_dims = 3
        actual_shape = tensor.shape
        actual_dims = len(actual_shape)

        if actual_dims != expected_dims:
            error_msg = (
                f"{context}: Expected 3D tensor (batch_size, seq_len, input_size), "
                f"got {actual_dims}D tensor with shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(batch_size, seq_len, input_size)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        # Validate specific dimensions if provided
        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if seq_len is not None and actual_shape[1] != seq_len:
            error_msg = (
                f"{context}: Sequence length mismatch - expected {seq_len}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {seq_len}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if input_size is not None and actual_shape[2] != input_size:
            error_msg = (
                f"{context}: Input size mismatch - expected {input_size}, got {actual_shape[2]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {input_size})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})",
            actual_shape=actual_shape,
            context=context,
        )

    def validate_static_features(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        batch_size: Optional[int] = None,
        static_size: Optional[int] = None,
        context: str = "static_features",
    ) -> ShapeValidationResult:
        """
        Validate static features tensor shape.

        Args:
            tensor: Input tensor
            batch_size: Expected batch size (None for any)
            static_size: Expected static feature size (None for any)
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        expected_dims = 2
        actual_shape = tensor.shape
        actual_dims = len(actual_shape)

        if actual_dims != expected_dims:
            error_msg = (
                f"{context}: Expected 2D tensor (batch_size, static_size), "
                f"got {actual_dims}D tensor with shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(batch_size, static_size)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        # Validate specific dimensions if provided
        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if static_size is not None and actual_shape[1] != static_size:
            error_msg = (
                f"{context}: Static size mismatch - expected {static_size}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {static_size})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]})",
            actual_shape=actual_shape,
            context=context,
        )

    def _validate_correlation_matrix_2d(
        self, actual_shape: Tuple[int, ...], n_stocks: Optional[int], context: str
    ) -> ShapeValidationResult:
        """Validate 2D correlation matrix shape."""
        # 2D correlation matrix: (n_stocks, n_stocks)
        if actual_shape[0] != actual_shape[1]:
            error_msg = f"{context}: 2D correlation matrix must be square, got shape {actual_shape}"
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(n_stocks, n_stocks)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if n_stocks is not None and actual_shape[0] != n_stocks:
            error_msg = (
                f"{context}: Number of stocks mismatch - expected {n_stocks}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({n_stocks}, {n_stocks})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]})",
            actual_shape=actual_shape,
            context=context,
        )

    def _validate_correlation_matrix_3d(
        self,
        actual_shape: Tuple[int, ...],
        n_stocks: Optional[int],
        batch_size: Optional[int],
        context: str,
    ) -> ShapeValidationResult:
        """Validate 3D correlation matrix shape."""
        # 3D correlation matrix: (batch_size, n_stocks, n_stocks)
        if actual_shape[1] != actual_shape[2]:
            error_msg = (
                f"{context}: 3D correlation matrix must have square stock dimensions, "
                f"got shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(batch_size, n_stocks, n_stocks)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if n_stocks is not None and actual_shape[1] != n_stocks:
            error_msg = (
                f"{context}: Number of stocks mismatch - expected {n_stocks}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {n_stocks}, {n_stocks})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})",
            actual_shape=actual_shape,
            context=context,
        )

    def validate_correlation_matrix(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        n_stocks: Optional[int] = None,
        batch_size: Optional[int] = None,
        context: str = "correlation_matrix",
    ) -> ShapeValidationResult:
        """
        Validate correlation matrix tensor shape.

        Args:
            tensor: Input tensor (can be 2D or 3D)
            n_stocks: Expected number of stocks (None for any)
            batch_size: Expected batch size for 3D tensors (None for any)
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        actual_shape = tensor.shape
        actual_dims = len(actual_shape)

        # Correlation matrices can be 2D (n_stocks, n_stocks) or 3D (batch_size, n_stocks, n_stocks)
        if actual_dims not in [2, 3]:
            error_msg = (
                f"{context}: Expected 2D or 3D tensor, "
                f"got {actual_dims}D tensor with shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(n_stocks, n_stocks) or (batch_size, n_stocks, n_stocks)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if actual_dims == 2:
            return self._validate_correlation_matrix_2d(actual_shape, n_stocks, context)
        else:  # actual_dims == 3
            return self._validate_correlation_matrix_3d(actual_shape, n_stocks, batch_size, context)

    def validate_adjacency_matrix(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        batch_size: Optional[int] = None,
        n_stocks: Optional[int] = None,
        context: str = "adjacency_matrix",
    ) -> ShapeValidationResult:
        """
        Validate graph adjacency matrix tensor shape.

        Args:
            tensor: Input tensor
            batch_size: Expected batch size (None for any)
            n_stocks: Expected number of stocks (None for any)
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        expected_dims = 3
        actual_shape = tensor.shape
        actual_dims = len(actual_shape)

        if actual_dims != expected_dims:
            error_msg = (
                f"{context}: Expected 3D tensor (batch_size, n_stocks, n_stocks), "
                f"got {actual_dims}D tensor with shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(batch_size, n_stocks, n_stocks)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        # Validate square stock dimensions
        if actual_shape[1] != actual_shape[2]:
            error_msg = (
                f"{context}: Adjacency matrix must have square stock dimensions, "
                f"got shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, n_stocks, n_stocks)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        # Validate specific dimensions if provided
        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if n_stocks is not None and actual_shape[1] != n_stocks:
            error_msg = (
                f"{context}: Number of stocks mismatch - expected {n_stocks}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {n_stocks}, {n_stocks})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})",
            actual_shape=actual_shape,
            context=context,
        )

    def _validate_node_features_3d(
        self,
        actual_shape: Tuple[int, ...],
        batch_size: Optional[int],
        n_stocks: Optional[int],
        embed_dim: Optional[int],
        context: str,
    ) -> ShapeValidationResult:
        """Validate 3D node features shape."""
        # 3D node features: (batch_size, n_stocks, embed_dim)
        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if n_stocks is not None and actual_shape[1] != n_stocks:
            error_msg = (
                f"{context}: Number of stocks mismatch - expected {n_stocks}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {n_stocks}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if embed_dim is not None and actual_shape[2] != embed_dim:
            error_msg = f"{context}: Embedding dimension mismatch - expected {embed_dim}, got {actual_shape[2]}"
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {embed_dim})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})",
            actual_shape=actual_shape,
            context=context,
        )

    def _validate_node_features_4d(
        self,
        actual_shape: Tuple[int, ...],
        batch_size: Optional[int],
        n_stocks: Optional[int],
        seq_len: Optional[int],
        input_size: Optional[int],
        context: str,
    ) -> ShapeValidationResult:
        """Validate 4D flattened node features shape."""
        # 4D flattened node features: (batch_size * n_stocks, seq_len, input_size)
        if seq_len is not None and actual_shape[1] != seq_len:
            error_msg = (
                f"{context}: Sequence length mismatch - expected {seq_len}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {seq_len}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if input_size is not None and actual_shape[2] != input_size:
            error_msg = (
                f"{context}: Input size mismatch - expected {input_size}, got {actual_shape[2]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {input_size})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        # For 4D tensors, batch_size * n_stocks should equal the first dimension
        if batch_size is not None and n_stocks is not None:
            expected_first_dim = batch_size * n_stocks
            if actual_shape[0] != expected_first_dim:
                error_msg = (
                    f"{context}: Flattened batch dimension mismatch - "
                    f"expected batch_size * n_stocks = {expected_first_dim}, got {actual_shape[0]}"
                )
                return ShapeValidationResult(
                    is_valid=False,
                    expected_shape=f"({expected_first_dim}, {actual_shape[1]}, {actual_shape[2]})",
                    actual_shape=actual_shape,
                    error_message=error_msg,
                    context=context,
                )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})",
            actual_shape=actual_shape,
            context=context,
        )

    def validate_node_features(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        batch_size: Optional[int] = None,
        n_stocks: Optional[int] = None,
        embed_dim: Optional[int] = None,
        seq_len: Optional[int] = None,
        input_size: Optional[int] = None,
        context: str = "node_features",
    ) -> ShapeValidationResult:
        """
        Validate node features tensor shape.

        Node features can be either:
        - 3D: (batch_size, n_stocks, embed_dim) - for graph convolution
        - 4D: (batch_size * n_stocks, seq_len, input_size) - flattened for TFT processing

        Args:
            tensor: Input tensor
            batch_size: Expected batch size (None for any)
            n_stocks: Expected number of stocks (None for any)
            embed_dim: Expected embedding dimension for 3D tensors (None for any)
            seq_len: Expected sequence length for 4D tensors (None for any)
            input_size: Expected input size for 4D tensors (None for any)
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        actual_shape = tensor.shape
        actual_dims = len(actual_shape)

        if actual_dims not in [3, 4]:
            error_msg = (
                f"{context}: Expected 3D or 4D tensor, "
                f"got {actual_dims}D tensor with shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(batch_size, n_stocks, embed_dim) or (batch_size * n_stocks, seq_len, input_size)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if actual_dims == 3:
            return self._validate_node_features_3d(
                actual_shape, batch_size, n_stocks, embed_dim, context
            )
        else:  # actual_dims == 4
            return self._validate_node_features_4d(
                actual_shape, batch_size, n_stocks, seq_len, input_size, context
            )

    def _validate_prediction_output_2d(
        self,
        actual_shape: Tuple[int, ...],
        batch_size: Optional[int],
        num_horizons: Optional[int],
        context: str,
    ) -> ShapeValidationResult:
        """Validate 2D prediction output shape."""
        # 2D predictions: (batch_size, num_horizons)
        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if num_horizons is not None and actual_shape[1] != num_horizons:
            error_msg = f"{context}: Number of horizons mismatch - expected {num_horizons}, got {actual_shape[1]}"
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {num_horizons})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]})",
            actual_shape=actual_shape,
            context=context,
        )

    def _validate_prediction_output_3d(
        self,
        actual_shape: Tuple[int, ...],
        batch_size: Optional[int],
        seq_len: Optional[int],
        num_horizons: Optional[int],
        context: str,
    ) -> ShapeValidationResult:
        """Validate 3D prediction output shape."""
        # 3D predictions: (batch_size, seq_len, num_horizons)
        if batch_size is not None and actual_shape[0] != batch_size:
            error_msg = (
                f"{context}: Batch size mismatch - expected {batch_size}, got {actual_shape[0]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({batch_size}, {actual_shape[1]}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if seq_len is not None and actual_shape[1] != seq_len:
            error_msg = (
                f"{context}: Sequence length mismatch - expected {seq_len}, got {actual_shape[1]}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {seq_len}, {actual_shape[2]})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if num_horizons is not None and actual_shape[2] != num_horizons:
            error_msg = f"{context}: Number of horizons mismatch - expected {num_horizons}, got {actual_shape[2]}"
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {num_horizons})",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"({actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})",
            actual_shape=actual_shape,
            context=context,
        )

    def validate_prediction_output(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        batch_size: Optional[int] = None,
        num_horizons: Optional[int] = None,
        seq_len: Optional[int] = None,
        context: str = "prediction_output",
    ) -> ShapeValidationResult:
        """
        Validate prediction output tensor shape.

        Prediction outputs can be either:
        - 2D: (batch_size, num_horizons) - standard multi-horizon predictions
        - 3D: (batch_size, seq_len, num_horizons) - sequence predictions

        Args:
            tensor: Input tensor
            batch_size: Expected batch size (None for any)
            num_horizons: Expected number of horizons (None for any)
            seq_len: Expected sequence length for 3D tensors (None for any)
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        actual_shape = tensor.shape
        actual_dims = len(actual_shape)

        if actual_dims not in [2, 3]:
            error_msg = (
                f"{context}: Expected 2D or 3D tensor, "
                f"got {actual_dims}D tensor with shape {actual_shape}"
            )
            return ShapeValidationResult(
                is_valid=False,
                expected_shape="(batch_size, num_horizons) or (batch_size, seq_len, num_horizons)",
                actual_shape=actual_shape,
                error_message=error_msg,
                context=context,
            )

        if actual_dims == 2:
            return self._validate_prediction_output_2d(
                actual_shape, batch_size, num_horizons, context
            )
        else:  # actual_dims == 3
            return self._validate_prediction_output_3d(
                actual_shape, batch_size, seq_len, num_horizons, context
            )

    def validate_tensor_compatibility(
        self,
        tensor1: Union[torch.Tensor, np.ndarray],
        tensor2: Union[torch.Tensor, np.ndarray],
        dimensions_to_check: List[int],
        context: str = "tensor_compatibility",
    ) -> ShapeValidationResult:
        """
        Validate that two tensors are compatible along specified dimensions.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            dimensions_to_check: List of dimension indices to check for compatibility
            context: Context for error messages

        Returns:
            ShapeValidationResult with validation details
        """
        shape1 = tensor1.shape
        shape2 = tensor2.shape

        max_dims = max(len(shape1), len(shape2))

        # Pad shapes to same length for comparison
        if len(shape1) < max_dims:
            shape1 = (1,) * (max_dims - len(shape1)) + shape1
        if len(shape2) < max_dims:
            shape2 = (1,) * (max_dims - len(shape2)) + shape2

        mismatches = []
        for dim in dimensions_to_check:
            if (
                dim < max_dims
                and shape1[dim] != shape2[dim]
                and shape1[dim] != 1
                and shape2[dim] != 1
            ):
                mismatches.append(f"dimension {dim}: {shape1[dim]} vs {shape2[dim]}")

        if mismatches:
            error_msg = f"{context}: Tensor shape incompatibility - {'; '.join(mismatches)}"
            return ShapeValidationResult(
                is_valid=False,
                expected_shape=f"compatible with {tensor1.shape}",
                actual_shape=tensor2.shape,
                error_message=error_msg,
                context=context,
            )

        return ShapeValidationResult(
            is_valid=True,
            expected_shape=f"compatible with {tensor1.shape}",
            actual_shape=tensor2.shape,
            context=context,
        )

    def validate_and_log(self, result: ShapeValidationResult, raise_on_error: bool = True) -> bool:
        """
        Validate a shape validation result and optionally log/raise errors.

        Args:
            result: ShapeValidationResult to process
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            True if validation passed, False otherwise

        Raises:
            ValueError: If validation failed and raise_on_error is True
        """
        if result.is_valid:
            return True

        error_msg = result.error_message or f"Shape validation failed for {result.context}"

        if self.enable_logging:
            logger.error(f"Shape validation failed: {error_msg}")
            logger.error(f"Expected: {result.expected_shape}, Actual: {result.actual_shape}")

        if raise_on_error:
            raise ValueError(error_msg)

        return False

    def validate_tft_diff_shapes(
        self,
        temporal_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        static_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        correlation_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        input_size: Optional[int] = None,
        static_size: Optional[int] = None,
        n_stocks: Optional[int] = None,
        context: str = "TFT-Diff",
    ) -> Dict[str, ShapeValidationResult]:
        """
        Comprehensive validation of all TFT-Diff tensor shapes.

        Args:
            temporal_features: Temporal input features
            static_features: Static input features
            correlation_matrix: Correlation matrix
            batch_size: Expected batch size
            seq_len: Expected sequence length
            input_size: Expected input feature size
            static_size: Expected static feature size
            n_stocks: Expected number of stocks
            context: Context for error messages

        Returns:
            Dictionary of validation results for each tensor type
        """
        results = {}

        if temporal_features is not None:
            results["temporal_features"] = self.validate_temporal_features(
                temporal_features, batch_size, seq_len, input_size, f"{context}.temporal_features"
            )

        if static_features is not None:
            results["static_features"] = self.validate_static_features(
                static_features, batch_size, static_size, f"{context}.static_features"
            )

        if correlation_matrix is not None:
            results["correlation_matrix"] = self.validate_correlation_matrix(
                correlation_matrix, n_stocks, batch_size, f"{context}.correlation_matrix"
            )

        return results
