"""
PlanarQuant CPU implementation - pure Python/PyTorch on CPU.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from ..common.lloyd_max import LloydMaxCodebook


def make_random_rotations(n_groups: int, device="cpu", seed=None) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    angles = torch.rand(n_groups, generator=gen) * (2 * math.pi)
    angles = angles.to(device)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def rot2_apply(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    c = cs[..., 0]
    s = cs[..., 1]
    v0 = v[..., 0]
    v1 = v[..., 1]
    return torch.stack([c * v0 - s * v1, s * v0 + c * v1], dim=-1)


def rot2_inverse(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    c = cs[..., 0]
    s = cs[..., 1]
    v0 = v[..., 0]
    v1 = v[..., 1]
    return torch.stack([c * v0 + s * v1, -s * v0 + c * v1], dim=-1)


class PlanarQuantMSE(nn.Module):
    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device
        self.n_groups = (d + 1) // 2
        self.d_padded = self.n_groups * 2

        cb = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", cb.centroids.to(device))

        rot = make_random_rotations(self.n_groups, device=device, seed=seed)
        self.register_buffer("rot2", rot)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 2)

    def _extract(self, v: torch.Tensor) -> torch.Tensor:
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., : self.d]

    def _quantize_scalar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diffs = x.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        x_q = self.centroids[indices]
        return x_q, indices

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        v = self._embed(x_unit)
        v_rot = rot2_apply(self.rot2, v)
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)
        q_flat, indices = self._quantize_scalar(flat)
        v_q = q_flat.reshape_as(v_rot)
        return v_q, {"indices": indices, "_norms": norms.squeeze(-1)}

    def dequantize(self, indices_dict: dict) -> torch.Tensor:
        idx = indices_dict["indices"]
        values = self.centroids[idx]
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 2)
        v_recon = rot2_inverse(self.rot2, v_q)
        x_hat = self._extract(v_recon)
        if "_norms" in indices_dict:
            norms = indices_dict["_norms"]
            if norms.dim() < x_hat.dim():
                norms = norms.unsqueeze(-1)
            x_hat = x_hat * norms
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        v_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class PlanarQuantProd(nn.Module):
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
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        self.mse = PlanarQuantMSE(d, self.mse_bits, seed=seed, device=device)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 1)
        S = torch.randn(self.qjl_dim, d, generator=gen)
        self.register_buffer("S", S.to(device))

    def quantize(self, x: torch.Tensor) -> dict:
        x_hat, mse_indices = self.mse(x)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0
        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)
        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip
        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)
