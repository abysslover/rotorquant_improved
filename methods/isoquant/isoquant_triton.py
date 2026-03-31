import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

import triton
import triton.language as tl

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


@triton.jit
def _quat_mul(aw, ax, ay, az, bw, bx, by, bz):
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return rw, rx, ry, rz


@triton.jit
def _quantize_nearest(val, centroids_ptr, n_levels: tl.constexpr):
    best_val = tl.load(centroids_ptr)
    best_dist = tl.abs(val - best_val)
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        d = tl.abs(val - c)
        mask = d < best_dist
        best_dist = tl.where(mask, d, best_dist)
        best_val = tl.where(mask, c, best_val)
    return best_val


@triton.jit
def _iso_full_fused_kernel(
    input_ptr,
    output_ptr,
    ql_ptr,
    qr_ptr,
    centroids_ptr,
    batch_size,
    emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b,
    stride_in_d,
    stride_out_b,
    stride_out_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    ql_w = tl.load(ql_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    ql_x = tl.load(ql_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    ql_y = tl.load(ql_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    ql_z = tl.load(ql_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    qr_w = tl.load(qr_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    qr_x = tl.load(qr_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    qr_y = tl.load(qr_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    qr_z = tl.load(qr_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    d0 = g_offs * 4
    v0 = tl.load(
        input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
        mask=g_mask & (d0 < emb_dim),
        other=0.0,
    )
    v1 = tl.load(
        input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
        mask=g_mask & ((d0 + 1) < emb_dim),
        other=0.0,
    )
    v2 = tl.load(
        input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
        mask=g_mask & ((d0 + 2) < emb_dim),
        other=0.0,
    )
    v3 = tl.load(
        input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d,
        mask=g_mask & ((d0 + 3) < emb_dim),
        other=0.0,
    )

    tw, tx, ty, tz = _quat_mul(ql_w, ql_x, ql_y, ql_z, v0, v1, v2, v3)
    rw, rx, ry, rz = _quat_mul(tw, tx, ty, tz, qr_w, -qr_x, -qr_y, -qr_z)

    qw = _quantize_nearest(rw, centroids_ptr, n_levels)
    qx = _quantize_nearest(rx, centroids_ptr, n_levels)
    qy = _quantize_nearest(ry, centroids_ptr, n_levels)
    qz = _quantize_nearest(rz, centroids_ptr, n_levels)

    tw2, tx2, ty2, tz2 = _quat_mul(ql_w, -ql_x, -ql_y, -ql_z, qw, qx, qy, qz)
    fw, fx, fy, fz = _quat_mul(tw2, tx2, ty2, tz2, qr_w, qr_x, qr_y, qr_z)

    tl.store(
        output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
        fw,
        mask=g_mask & (d0 < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
        fx,
        mask=g_mask & ((d0 + 1) < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
        fy,
        mask=g_mask & ((d0 + 2) < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
        fz,
        mask=g_mask & ((d0 + 3) < emb_dim),
    )


@triton.jit
def _iso_fast_fused_kernel(
    input_ptr,
    output_ptr,
    ql_ptr,
    centroids_ptr,
    batch_size,
    emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b,
    stride_in_d,
    stride_out_b,
    stride_out_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    ql_w = tl.load(ql_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    ql_x = tl.load(ql_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    ql_y = tl.load(ql_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    ql_z = tl.load(ql_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    d0 = g_offs * 4
    v0 = tl.load(
        input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
        mask=g_mask & (d0 < emb_dim),
        other=0.0,
    )
    v1 = tl.load(
        input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
        mask=g_mask & ((d0 + 1) < emb_dim),
        other=0.0,
    )
    v2 = tl.load(
        input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
        mask=g_mask & ((d0 + 2) < emb_dim),
        other=0.0,
    )
    v3 = tl.load(
        input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d,
        mask=g_mask & ((d0 + 3) < emb_dim),
        other=0.0,
    )

    rw, rx, ry, rz = _quat_mul(ql_w, ql_x, ql_y, ql_z, v0, v1, v2, v3)

    qw = _quantize_nearest(rw, centroids_ptr, n_levels)
    qx = _quantize_nearest(rx, centroids_ptr, n_levels)
    qy = _quantize_nearest(ry, centroids_ptr, n_levels)
    qz = _quantize_nearest(rz, centroids_ptr, n_levels)

    fw, fx, fy, fz = _quat_mul(ql_w, -ql_x, -ql_y, -ql_z, qw, qx, qy, qz)

    tl.store(
        output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
        fw,
        mask=g_mask & (d0 < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
        fx,
        mask=g_mask & ((d0 + 1) < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
        fy,
        mask=g_mask & ((d0 + 2) < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
        fz,
        mask=g_mask & ((d0 + 3) < emb_dim),
    )


def triton_iso_full_fused(
    input: torch.Tensor,
    q_left: torch.Tensor,
    q_right: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    batch_size, emb_dim = input.shape
    n_groups = q_left.shape[0]
    n_levels = centroids.shape[0]

    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    ql_f32 = q_left.float().contiguous()
    qr_f32 = q_right.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _iso_full_fused_kernel[grid](
        input_f32,
        output,
        ql_f32,
        qr_f32,
        c_f32,
        batch_size,
        emb_dim,
        n_groups,
        n_levels,
        input_f32.stride(0),
        input_f32.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    output = output * norms
    return output.to(input.dtype)


def triton_iso_fast_fused(
    input: torch.Tensor,
    q_left: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    batch_size, emb_dim = input.shape
    n_groups = q_left.shape[0]
    n_levels = centroids.shape[0]

    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    ql_f32 = q_left.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _iso_fast_fused_kernel[grid](
        input_f32,
        output,
        ql_f32,
        c_f32,
        batch_size,
        emb_dim,
        n_groups,
        n_levels,
        input_f32.stride(0),
        input_f32.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    output = output * norms
    return output.to(input.dtype)
