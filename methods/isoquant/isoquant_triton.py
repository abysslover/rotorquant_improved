import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from ..common.lloyd_max import LloydMaxCodebook


def quat_conjugate(q):
    signs = torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)
    return q * signs


def quat_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([rw, rx, ry, rz], dim=-1)


def make_random_unit_quaternion(shape, device="cpu", seed=None):
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    q = torch.randn(*shape, 4, generator=gen).to(device)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class IsoQuantMSE(nn.Module):
    def __init__(
        self, d: int, bits: int, seed: int = 42, mode: str = "full", device: str = "cpu"
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device
        self.mode = mode
        self.n_groups = (d + 3) // 4
        self.d_padded = self.n_groups * 4

        cb = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", cb.centroids.to(device))

        q_L = make_random_unit_quaternion((self.n_groups,), device=device, seed=seed)
        self.register_buffer("q_L", q_L)

        if mode == "full":
            q_R = make_random_unit_quaternion(
                (self.n_groups,), device=device, seed=seed + 10000
            )
            self.register_buffer("q_R", q_R)

    def _embed(self, x):
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 4)

    def _extract(self, v):
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., : self.d]

    def _rotate(self, v):
        if self.mode == "full":
            temp = quat_multiply(self.q_L, v)
            return quat_multiply(temp, quat_conjugate(self.q_R))
        else:
            return quat_multiply(self.q_L, v)

    def _unrotate(self, v):
        if self.mode == "full":
            temp = quat_multiply(quat_conjugate(self.q_L), v)
            return quat_multiply(temp, self.q_R)
        else:
            return quat_multiply(quat_conjugate(self.q_L), v)

    def _quantize_scalar(self, x):
        diffs = x.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        x_q = self.centroids[indices]
        return x_q, indices

    def quantize(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        v = self._embed(x_unit)
        v_rot = self._rotate(v)
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)
        q_flat, indices = self._quantize_scalar(flat)
        v_q = q_flat.reshape_as(v_rot)
        return v_q, {"indices": indices, "_norms": norms.squeeze(-1)}

    def dequantize(self, indices_dict):
        idx = indices_dict["indices"]
        values = self.centroids[idx]
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 4)
        v_recon = self._unrotate(v_q)
        x_hat = self._extract(v_recon)
        if "_norms" in indices_dict:
            norms = indices_dict["_norms"]
            if norms.dim() < x_hat.dim():
                norms = norms.unsqueeze(-1)
            x_hat = x_hat * norms
        return x_hat

    def forward(self, x):
        v_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class IsoQuantProd(nn.Module):
    def __init__(
        self,
        d: int,
        bits: int,
        mode: str = "full",
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

        self.mse = IsoQuantMSE(d, self.mse_bits, seed=seed, mode=mode, device=device)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 1)
        S = torch.randn(self.qjl_dim, d, generator=gen)
        self.register_buffer("S", S.to(device))

    def quantize(self, x):
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

    def dequantize(self, compressed):
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y, compressed):
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)
        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip
        return term1 + term2

    def forward(self, x):
        return self.quantize(x)
