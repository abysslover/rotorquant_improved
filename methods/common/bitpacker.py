"""
Bit-packing utilities for compressed KV cache storage.

Provides functions to pack/unpack arbitrary bit-width integer arrays
into/from byte arrays for efficient KV cache storage.
"""

import numpy as np
from typing import Tuple


def pack_indices(indices: np.ndarray, bits: int) -> bytes:
    """
    Pack integer indices into a compact byte array.

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
    Unpack byte array into integer indices.

    Args:
        packed: Bytes object from pack_indices()
        bits: Bit width per index (must match packing)
        shape: Target shape for output array

    Returns:
        Numpy array of unpacked indices

    Example:
        >>> indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        >>> packed = pack_indices(indices, bits=3)
        >>> unpacked = unpack_indices(packed, bits=3, shape=(8,))
        >>> np.array_equal(indices, unpacked)
        True
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
