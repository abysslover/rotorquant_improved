"""
Common utilities for TurboQuant engines.
Provides shared matrix generation functions for CPU and Torch backends.
"""

import torch
import numpy as np
from scipy.linalg import qr
from typing import Optional


def generate_rotation_matrix_cpu(
    d: int, seed: Optional[int] = None, device: Optional[str] = None
) -> np.ndarray:
    """Generate orthogonal rotation matrix for CPU; device arg ignored for API compatibility."""
    np.random.seed(seed)
    G = np.random.randn(d, d).astype(np.float32)
    Q, R = qr(G)
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.reshape(1, -1)).astype(np.float32)


def generate_qjl_matrix_cpu(
    d: int,
    m: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Generate QJL projection matrix for CPU; device arg ignored for API compatibility."""
    m = m or d
    np.random.seed(seed)
    return np.random.randn(m, d).astype(np.float32)


def generate_rotation_matrix_torch(
    d: int, seed: Optional[int] = None, device: str = "cpu"
) -> torch.Tensor:
    """Generate orthogonal rotation matrix for Torch. args: d(int), seed(int), device(str) -> Tensor(d, d)"""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix_torch(
    d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu"
) -> torch.Tensor:
    """Generate QJL projection matrix for Torch. args: d(int), m(int), seed(int), device(str) -> Tensor(m, d)"""
    m = m or d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)
