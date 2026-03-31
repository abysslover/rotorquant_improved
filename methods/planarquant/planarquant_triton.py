import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple
import torch.nn as nn

from ..common.lloyd_max import LloydMaxCodebook


def make_random_rotations(n_groups: int, device="cpu", seed=None) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    angles = torch.rand(n_groups, generator=gen) * (2 * math.pi)
    angles = angles.to(device)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


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
def _planar2_fused_kernel(
    input_ptr,
    output_ptr,
    rot2_ptr,
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

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2
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

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    q0 = _quantize_nearest(r0, centroids_ptr, n_levels)
    q1 = _quantize_nearest(r1, centroids_ptr, n_levels)

    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    tl.store(
        output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
        f0,
        mask=g_mask & (d0 < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
        f1,
        mask=g_mask & ((d0 + 1) < emb_dim),
    )


@triton.jit
def _planar2_quantize_kernel(
    input_ptr,
    indices_ptr,
    rot2_ptr,
    centroids_ptr,
    batch_size,
    emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b,
    stride_in_d,
    stride_idx_b,
    stride_idx_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2
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

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    best_idx0 = tl.zeros_like(r0).to(tl.int32)
    best_dist0 = tl.abs(r0 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r0 - c)
        mask = dd < best_dist0
        best_dist0 = tl.where(mask, dd, best_dist0)
        best_idx0 = tl.where(mask, i, best_idx0)

    best_idx1 = tl.zeros_like(r1).to(tl.int32)
    best_dist1 = tl.abs(r1 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r1 - c)
        mask = dd < best_dist1
        best_dist1 = tl.where(mask, dd, best_dist1)
        best_idx1 = tl.where(mask, i, best_idx1)

    tl.store(
        indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
        best_idx0.to(tl.int8),
        mask=g_mask & (d0 < emb_dim),
    )
    tl.store(
        indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
        best_idx1.to(tl.int8),
        mask=g_mask & ((d0 + 1) < emb_dim),
    )


@triton.jit
def _planar2_dequantize_kernel(
    indices_ptr,
    output_ptr,
    rot2_ptr,
    centroids_ptr,
    batch_size,
    emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_idx_b,
    stride_idx_d,
    stride_out_b,
    stride_out_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2

    idx0 = tl.load(
        indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
        mask=g_mask & (d0 < emb_dim),
        other=0,
    ).to(tl.int32)
    idx1 = tl.load(
        indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
        mask=g_mask & ((d0 + 1) < emb_dim),
        other=0,
    ).to(tl.int32)

    q0 = tl.load(centroids_ptr + idx0, mask=g_mask, other=0.0)
    q1 = tl.load(centroids_ptr + idx1, mask=g_mask, other=0.0)

    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    tl.store(
        output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
        f0,
        mask=g_mask & (d0 < emb_dim),
    )
    tl.store(
        output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
        f1,
        mask=g_mask & ((d0 + 1) < emb_dim),
    )


def triton_planar2_fused(input, rot2, centroids):
    batch_size, emb_dim = input.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_fused_kernel[grid](
        input_f32,
        output,
        rot2_f32,
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


def triton_planar2_quantize(input, rot2, centroids):
    batch_size, emb_dim = input.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    indices = torch.empty(batch_size, emb_dim, dtype=torch.int8, device=input.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_quantize_kernel[grid](
        input_f32,
        indices,
        rot2_f32,
        c_f32,
        batch_size,
        emb_dim,
        n_groups,
        n_levels,
        input_f32.stride(0),
        input_f32.stride(1),
        indices.stride(0),
        indices.stride(1),
        BLOCK_G=BLOCK_G,
    )

    return indices, norms.squeeze(-1)


def triton_planar2_dequantize(indices, norms, rot2, centroids):
    batch_size, emb_dim = indices.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty(
        batch_size, emb_dim, dtype=torch.float32, device=indices.device
    )

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_dequantize_kernel[grid](
        indices,
        output,
        rot2_f32,
        c_f32,
        batch_size,
        emb_dim,
        n_groups,
        n_levels,
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    if norms.dim() == 1:
        norms = norms.unsqueeze(-1)
    return output * norms


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

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if (
            x.device.type == "cuda"
            and hasattr(self, "triton_available")
            and self.triton_available
        ):
            x_f32 = x.float()
            norms = x_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            x_unit = (x_f32 / norms).contiguous()

            indices, stored_norms = triton_planar2_quantize(
                x_unit, self.rot2.float(), self.centroids.float()
            )
            return indices, {"_norms": stored_norms}
        else:
            norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
            x_unit = x / norms
            v = self._embed(x_unit)
            c = self.rot2[..., 0]
            s = self.rot2[..., 1]
            v0, v1 = v[..., 0], v[..., 1]
            v_rot = torch.stack([c * v0 - s * v1, s * v0 + c * v1], dim=-1)
            flat = v_rot.reshape(*v_rot.shape[:-2], -1)
            diffs = flat.unsqueeze(-1) - self.centroids
            indices = diffs.abs().argmin(dim=-1)
            v_q = self.centroids[indices].reshape_as(v_rot)
            c_inv, s_inv = c, -s
            v_recon = torch.stack(
                [
                    c_inv * v_q[..., 0] + s_inv * v_q[..., 1],
                    -s_inv * v_q[..., 0] + c_inv * v_q[..., 1],
                ],
                dim=-1,
            )
            x_hat = self._extract(v_recon) * norms.squeeze(-1)
            return x_hat, {"indices": indices, "_norms": norms.squeeze(-1)}

    def dequantize(self, indices_dict: dict) -> torch.Tensor:
        if (
            "indices" in indices_dict
            and indices_dict["indices"].dim() == 2
            and indices_dict["indices"].dtype == torch.int8
        ):
            if (
                indices_dict["indices"].device.type == "cuda"
                and hasattr(self, "triton_available")
                and self.triton_available
            ):
                return triton_planar2_dequantize(
                    indices_dict["indices"],
                    indices_dict["_norms"],
                    self.rot2.float(),
                    self.centroids.float(),
                ).to(self.centroids.dtype)

        idx = indices_dict["indices"]
        values = self.centroids[idx]
        if values.dim() == 3:
            v_q = values.reshape(*values.shape[:-1], self.n_groups, 2)
        else:
            v_q = values.reshape(*values.shape[:-1], self.n_groups, 2)
        c, s = self.rot2[..., 0], self.rot2[..., 1]
        v_recon = torch.stack(
            [c * v_q[..., 0] + s * v_q[..., 1], -s * v_q[..., 0] + c * v_q[..., 1]],
            dim=-1,
        )
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

        try:
            import triton

            self.triton_available = True
        except ImportError:
            self.triton_available = False

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
