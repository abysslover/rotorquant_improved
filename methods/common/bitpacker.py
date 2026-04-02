"""
Bit-packing utilities for compressed KV cache storage.

Provides PyTorch and NumPy implementations for packing/unpacking arbitrary
bit-width integer arrays into/from byte arrays for efficient KV cache storage.

This is the V3 feature ported from turboquant-pytorch.
"""

import torch
import numpy as np
from typing import Tuple, Optional


# ── PyTorch Implementation ───────────────────────────────────────────────────


def pack_indices_torch(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack integer indices into a compact byte tensor using PyTorch.

    Args:
        indices: 2D tensor of shape (N, D) with non-negative integers
        bits: Bit width per index (e.g., 2, 3, 4)

    Returns:
        Packed tensor of shape (N, num_bytes) with dtype uint8

    Example:
        >>> indices = torch.randint(0, 8, (100, 128), dtype=torch.uint8)
        >>> packed = pack_indices_torch(indices, bits=3)
        >>> packed.shape  # (100, 48) since 128*3/8 = 48
    """
    if bits < 1 or bits > 8:
        raise ValueError(
            f"bits must be between 1 and 8 for PyTorch implementation, got {bits}"
        )

    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D (N, D), got shape {indices.shape}")

    if not torch.is_floating_point(indices):
        indices = indices.to(torch.uint8)

    n_samples, d = indices.shape
    max_val = (1 << bits) - 1

    if indices.min() < 0 or indices.max() > max_val:
        raise ValueError(
            f"All indices must be in range [0, {max_val}], "
            f"got range [{indices.min()}, {indices.max()}]"
        )

    indices_per_byte = 8 // bits
    d_padded = ((d + indices_per_byte - 1) // indices_per_byte) * indices_per_byte
    idx_pad = d_padded - d

    # Pad indices if needed
    if idx_pad > 0:
        indices = torch.cat(
            [
                indices,
                torch.zeros(
                    n_samples, idx_pad, dtype=torch.uint8, device=indices.device
                ),
            ],
            dim=1,
        )

    # Reshape to (N, num_groups, indices_per_byte)
    n_groups = d_padded // indices_per_byte
    indices = indices.view(n_samples, n_groups, indices_per_byte)

    # Create power-of-2 weights for each position
    idx_powers = torch.tensor(
        [bits << i for i in range(indices_per_byte - 1, -1, -1)],
        dtype=torch.uint8,
        device=indices.device,
    )

    # Pack: multiply and sum across the last dimension
    packed = (indices * idx_powers).sum(dim=-1)

    return packed


def unpack_indices_torch(
    packed: torch.Tensor, bits: int, original_shape: int
) -> torch.Tensor:
    """
    Unpack byte tensor into integer indices using PyTorch.

    Args:
        packed: Packed tensor from pack_indices_torch
        bits: Bit width per index (must match packing)
        original_shape: Original D dimension before padding

    Returns:
        Unpacked tensor of shape (N, original_shape) with dtype uint8
    """
    if bits < 1 or bits > 8:
        raise ValueError(
            f"bits must be between 1 and 8 for PyTorch implementation, got {bits}"
        )

    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D (N, num_bytes), got shape {packed.shape}")

    n_samples, num_bytes = packed.shape
    indices_per_byte = 8 // bits

    # Calculate original number of indices
    total_bits = num_bytes * 8
    n_indices = total_bits // bits
    d_padded = n_indices  # This is the padded dimension

    # Create shift amounts for each position
    idx_shifts = torch.tensor(
        [bits * i for i in range(indices_per_byte - 1, -1, -1)],
        dtype=torch.uint8,
        device=packed.device,
    )

    # Unpack
    packed_expanded = packed.unsqueeze(-1)  # (N, num_bytes, 1)
    bits_tensor = (
        torch.arange(indices_per_byte, device=packed.device).unsqueeze(0).unsqueeze(-1)
    )  # (1, indices_per_byte, 1)

    # Extract each bits-wide field
    unpacked = (packed_expanded >> idx_shifts) & (
        (1 << bits) - 1
    )  # (N, num_bytes, indices_per_byte)
    indices = unpacked.view(n_samples, d_padded)  # (N, d_padded)

    # Trim padding if needed
    if d_padded > original_shape:
        indices = indices[:, :original_shape]

    return indices


# ── NumPy Implementation (Legacy) ───────────────────────────────────────────


def pack_indices(indices: np.ndarray, bits: int) -> bytes:
    """
    Pack integer indices into a compact byte array (NumPy version).

    Args:
        indices: 1D numpy array of non-negative integers
        bits: Bit width per index (e.g., 2, 3, 4)

    Returns:
        Bytes object containing packed indices

    Example:
        >>> indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        >>> packed = pack_indices(indices, bits=3)
        >>> len(packed)  # 8 indices * 3 bits = 24 bits = 3 bytes
        3
    """
    if bits < 1 or bits > 64:
        raise ValueError(f"bits must be between 1 and 64, got {bits}")

    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {indices.shape}")

    if not np.issubdtype(indices.dtype, np.integer):
        indices = indices.astype(np.int64)

    max_val = (1 << bits) - 1
    if np.any(indices < 0) or np.any(indices > max_val):
        raise ValueError(
            f"All indices must be in range [0, {max_val}], "
            f"got range [{indices.min()}, {indices.max()}]"
        )

    n_indices = len(indices)
    total_bits = n_indices * bits
    n_bytes = (total_bits + 7) // 8

    output = bytearray(n_bytes)

    for i, val in enumerate(indices):
        bit_offset = i * bits
        byte_offset = bit_offset // 8
        bit_in_byte = bit_offset % 8

        for b in range(bits):
            bit = (val >> b) & 1
            if bit:
                output[byte_offset + (b // 8)] |= 1 << (b % 8)

    return bytes(output)


def unpack_indices(packed: bytes, bits: int, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Unpack byte array into integer indices (NumPy version).

    Args:
        packed: Bytes object from pack_indices()
        bits: Bit width per index (must match packing)
        shape: Target shape for output array

    Returns:
        Numpy array of unpacked indices
    """
    if bits < 1 or bits > 64:
        raise ValueError(f"bits must be between 1 and 64, got {bits}")

    total_bits = len(packed) * 8
    n_indices = total_bits // bits

    if total_bits % bits != 0:
        raise ValueError(
            f"Packed data ({total_bits} bits) is not evenly divisible by bits ({bits})"
        )

    output = np.empty(n_indices, dtype=np.int64)

    for i in range(n_indices):
        bit_offset = i * bits
        val = 0

        for b in range(bits):
            byte_offset = (bit_offset + b) // 8
            bit_in_byte = (bit_offset + b) % 8
            if byte_offset < len(packed):
                bit = (packed[byte_offset] >> bit_in_byte) & 1
                val |= bit << b

        output[i] = val

    return output.reshape(shape)
