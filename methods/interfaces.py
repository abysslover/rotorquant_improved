"""
Common interfaces for vector quantization algorithms.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from enum import Enum


class Backend(Enum):
    """Available computation backends."""

    PYTHON = "python"  # Pure Python/PyTorch (default)
    CUDA = "cuda"  # CUDA kernels (if available)
    TRITON = "triton"  # Triton kernels (if available)
    CPU = "cpu"  # Force CPU execution
    METAL = "metal"  # Apple Metal (future)


class QuantizerBase(nn.Module):
    """
    Base class for MSE-only quantizers (Stage 1).

    Common interface:
        - quantize(x) -> (quantized_tensor, indices_dict)
        - dequantize(indices_dict) -> reconstructed_tensor
        - forward(x) -> (reconstructed_tensor, indices_dict)
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.seed = seed
        self.device = device

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize vectors.

        Args:
            x: Input tensor of shape (..., d)

        Returns:
            Tuple of (quantized_tensor, indices_dict)
        """
        raise NotImplementedError

    def dequantize(self, indices_dict: dict) -> torch.Tensor:
        """
        Reconstruct vectors from quantized indices.

        Args:
            indices_dict: Dictionary containing quantized indices
                         Must contain 'indices' key with int tensor
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Full quantize-dequantize cycle."""
        v_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class ProdQuantizerBase(nn.Module):
    """
    Base class for inner product quantizers (Stage 1 + Stage 2 with QJL).

    Common interface:
        - quantize(x) -> dict with mse_indices, qjl_signs, residual_norm
        - dequantize(compressed) -> reconstructed_tensor
        - inner_product(y, compressed) -> unbiased inner product estimate
        - forward(x) -> dict
    """

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.qjl_dim = qjl_dim or d
        self.seed = seed
        self.device = device

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Full quantization with QJL.

        Args:
            x: Input tensor of shape (..., d)

        Returns:
            Dictionary with:
                - 'mse_indices': int indices from Stage 1
                - 'qjl_signs': ±1 signs from Stage 2
                - 'residual_norm': L2 norm of residual
        """
        raise NotImplementedError

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """
        Reconstruct vectors from MSE indices.

        Args:
            compressed: Dictionary from quantize()
        """
        raise NotImplementedError

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Unbiased inner product estimate: <y, x>.

        Args:
            y: Query vectors (..., d)
            compressed: Compressed representation from quantize()

        Returns:
            Estimated inner products (...,)
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)


class KVCacheBase:
    """
    Base class for KV cache wrappers using compression.

    Common interface:
        - append(keys, values)
        - attention_scores(queries)
        - get_values()
        - __len__()
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        bits: int = 3,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.seed = seed
        self.device = device

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append new key-value pairs to cache.

        Args:
            keys: (..., seq_len, d_key) or (seq_len, d_key)
            values: (..., seq_len, d_value) or (seq_len, d_value)
        """
        raise NotImplementedError

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between queries and all cached keys.

        Args:
            queries: (..., d_key)

        Returns:
            Attention scores for each cached position
        """
        raise NotImplementedError

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        raise NotImplementedError

    def __len__(self):
        """Number of cached key-value pairs."""
        raise NotImplementedError
