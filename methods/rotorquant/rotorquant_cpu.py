import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from .clifford import (
    MV_DIM,
    geometric_product,
    reverse,
    make_random_rotor,
    rotor_sandwich,
    embed_vectors_as_multivectors,
    extract_vectors_from_multivectors,
    multivector_norm_sq,
)
from ..common.lloyd_max import LloydMaxCodebook


class RotorQuantMSE(nn.Module):
    def __init__(
        self,
        d: int,
        bits: int,
        seed: int = 42,
        grade_bits: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        self.n_groups = (d + 2) // 3
        self.mv_dim = self.n_groups * MV_DIM

        if grade_bits is None:
            grade_bits = {"vector": bits}
        self.grade_bits = grade_bits

        d_eff_vector = d
        self.codebooks = nn.ModuleDict()
        for grade_name, gb in grade_bits.items():
            cb = LloydMaxCodebook(d_eff_vector, gb)
            self.register_buffer(f"centroids_{grade_name}", cb.centroids.to(device))

        self.grade_map = {"vector": [1, 2, 3]}

        rotors = []
        for i in range(self.n_groups):
            r = make_random_rotor((), device=device, seed=seed + i)
            rotors.append(r)
        self.register_buffer("rotors", torch.stack(rotors))

    def _apply_rotors(self, mv):
        return rotor_sandwich(self.rotors, mv)

    def _unapply_rotors(self, mv):
        rotor_rev = reverse(self.rotors)
        return rotor_sandwich(rotor_rev, mv)

    def _quantize_grade(self, x, grade_name):
        centroids = getattr(self, f"centroids_{grade_name}")
        diffs = x.unsqueeze(-1) - centroids
        indices = diffs.abs().argmin(dim=-1)
        x_q = centroids[indices]
        return x_q, indices

    def quantize(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        mv = embed_vectors_as_multivectors(x_unit)
        mv_rot = self._apply_rotors(mv)

        mv_q = torch.zeros_like(mv_rot)
        all_indices = {}

        for grade_name, component_indices in self.grade_map.items():
            grade_data = mv_rot[..., component_indices]
            flat = grade_data.reshape(*grade_data.shape[:-1], -1)
            q_flat, idx = self._quantize_grade(flat, grade_name)
            q_data = q_flat.reshape_as(grade_data)
            mv_q[..., component_indices] = q_data
            all_indices[grade_name] = idx

        all_indices["_norms"] = norms.squeeze(-1)
        return mv_q, all_indices

    def dequantize(self, indices):
        sample_centroids = getattr(self, "centroids_vector")
        vector_idx = indices["vector"]
        flat_batch = vector_idx.shape[0] if vector_idx.dim() >= 1 else 1

        mv_q = torch.zeros(
            flat_batch,
            self.n_groups,
            MV_DIM,
            dtype=sample_centroids.dtype,
            device=sample_centroids.device,
        )

        for grade_name, component_indices in self.grade_map.items():
            if grade_name.startswith("_"):
                continue
            centroids = getattr(self, f"centroids_{grade_name}")
            idx = indices[grade_name]
            values = centroids[idx]
            n_components = len(component_indices)
            values = values.reshape(flat_batch, self.n_groups, n_components)
            mv_q[..., component_indices] = values

        mv_recon = self._unapply_rotors(mv_q)
        x_hat = extract_vectors_from_multivectors(mv_recon, self.d)
        if "_norms" in indices:
            norms = indices["_norms"]
            if norms.dim() < x_hat.dim():
                norms = norms.unsqueeze(-1)
            x_hat = x_hat * norms
        return x_hat

    def forward(self, x):
        mv_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class RotorQuantProd(nn.Module):
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

        self.mse = RotorQuantMSE(d, self.mse_bits, seed=seed, device=device)

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


class RotorQuantKVCache:
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
        self.device = device

        self.key_quantizer = RotorQuantProd(d_key, bits, seed=seed, device=device)
        self.value_quantizer = RotorQuantMSE(
            d_value, bits, seed=seed + 100, device=device
        )

        self.key_cache = []
        self.value_cache = []

    def append(self, keys, values):
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        _, value_indices = self.value_quantizer(flat_values)

        self.key_cache.append(compressed_keys)
        self.value_cache.append(value_indices)

    def attention_scores(self, queries):
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self):
        values = []
        for indices in self.value_cache:
            v = self.value_quantizer.dequantize(indices)
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def __len__(self):
        return (
            sum(c["qjl_signs"].shape[0] for c in self.key_cache)
            if self.key_cache
            else 0
        )
