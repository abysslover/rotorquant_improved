import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from ..common.lloyd_max import LloydMaxCodebook
from ..interfaces import QuantizerBase, ProdQuantizerBase, KVCacheBase
from .common_utils import generate_rotation_matrix_torch as generate_rotation_matrix
from .common_utils import generate_qjl_matrix_torch as generate_qjl_matrix


class TurboQuantMSE(QuantizerBase):
    """PyTorch-based TurboQuant MSE quantizer (CPU/CUDA通用)"""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__(d, bits, seed, device)

        self.register_buffer(
            "Pi", generate_rotation_matrix(d, seed=seed, device=device)
        )

        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """회전 후 Lloyd-Max 양자화 수행. args: x(Tensor [..., d]) -> (x_hat Tensor, {"indices": Tensor int64})"""
        y = self.rotate(x)
        diffs = y.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.int64)

        x_hat = self.dequantize({"indices": indices})
        return x_hat, {"indices": indices}

    def dequantize(self, indices_dict: Dict) -> torch.Tensor:
        """양자화된 인덱스를 역양자화하여 원래 값으로 복원 (args: indices_dict -> Tensor)"""
        # 엔진 간 일관성을 위해 long 타입 캐스팅 적용
        indices = indices_dict["indices"].long()
        y_hat = self.centroids[indices]
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        x_hat, indices = self.quantize(x)
        return x_hat, indices


class TurboQuantProd(ProdQuantizerBase):
    """PyTorch-based TurboQuant Prod quantizer (Stage 1 + Stage 2)"""

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.mse_bits = max(bits - 1, 1)
        super().__init__(d, bits, qjl_dim, seed, device)

        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)

        self.register_buffer(
            "S", generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device)
        )

    def quantize(self, x: torch.Tensor) -> Dict:
        x_hat, mse_indices_dict = self.mse(x)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0
        return {
            "mse_indices": mse_indices_dict,
            "qjl_signs": qjl_signs.to(torch.int8),
            "residual_norm": residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: Dict) -> torch.Tensor:
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: Dict) -> torch.Tensor:
        x_mse = self.mse.dequantize(compressed["mse_indices"])

        if y.shape == x_mse.shape:
            term1 = (y * x_mse).sum(dim=-1)
        else:
            term1 = torch.matmul(y, x_mse.transpose(-2, -1))

        y_projected = y @ self.S.T
        signs = compressed["qjl_signs"].to(y.dtype)

        if y_projected.shape == signs.shape:
            qjl_ip = (y_projected * signs).sum(dim=-1)
        else:
            qjl_ip = torch.matmul(y_projected, signs.transpose(-2, -1))

        correction_scale = math.sqrt(math.pi / 2) / self.qjl_dim

        r_norm = compressed["residual_norm"]
        if qjl_ip.shape == r_norm.shape:
            term2 = r_norm * correction_scale * qjl_ip
        else:
            while r_norm.dim() < qjl_ip.dim():
                r_norm = r_norm.unsqueeze(-2)
            term2 = r_norm * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> Dict:
        return self.quantize(x)


class TurboQuantKVCache(KVCacheBase):
    """TurboQuant KV Cache with MSE-only compression (QJL removed from KV Cache)."""

    def __init__(
        self,
        d_key: int,
        d_value: int,
        bits: int = 3,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__(d_key, d_value, bits, key_bits, value_bits, seed, device)

        self.key_quantizer = TurboQuantMSE(
            d_key, self.key_bits, seed=seed, device=device
        )
        self.value_quantizer = TurboQuantMSE(
            d_value, self.value_bits, seed=seed + 100, device=device
        )

        self.key_cache = []
        self.value_cache = []

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        key_shape = keys.shape
        value_shape = values.shape

        _, key_indices_dict = self.key_quantizer.quantize(keys)
        _, value_indices_dict = self.value_quantizer.quantize(values)

        self.key_cache.append(
            {
                "indices": key_indices_dict,
                "shape": key_shape,
            }
        )
        self.value_cache.append(
            {
                "indices": value_indices_dict,
                "shape": value_shape,
            }
        )

    def _detect_gqa_and_expand(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """GQA 런타임 지원으로 Key 텐서를 Query 헤드 수에 맞춰 확장.

        Args:
            queries: Query tensor (batch, num_heads, seq_len, d_key) or (batch, seq_len, d_key)
            keys: Key tensor (batch, num_kv_heads, seq_len, d_key)

        Returns:
            Expanded keys tensor if GQA detected, otherwise original keys
        """
        if queries.dim() >= 3 and keys.dim() >= 3:
            h_q = queries.shape[-3]
            h_k = keys.shape[-3]
            if h_q != h_k and h_q % h_k == 0:
                group_size = h_q // h_k
                keys = keys.repeat_interleave(group_size, dim=-3)
        return keys

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """GQA 지원 attention score 계산.

        Args:
            queries: Query tensor (..., d_key)

        Returns:
            Attention scores (..., seq_len)
        """
        scores = []
        for cached in self.key_cache:
            keys = self.key_quantizer.dequantize(cached["indices"])
            keys = self._detect_gqa_and_expand(queries, keys)

            if queries.shape == keys.shape:
                s = (queries * keys).sum(dim=-1)
            else:
                s = torch.matmul(queries, keys.transpose(-2, -1))
            scores.append(s)
        return (
            torch.cat(scores, dim=-1) if scores else torch.empty(0, device=self.device)
        )

    def get_values(self, group_size: Optional[int] = None) -> torch.Tensor:
        """GQA 지원 Value 디양자화.

        Args:
            group_size: GQA group size for head expansion. If None, no expansion.

        Returns:
            Dequantized values tensor with optional GQA expansion.
        """
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["indices"])
            if group_size is not None and v.dim() >= 3:
                v = v.repeat_interleave(group_size, dim=-3)
            values.append(v)
        return (
            torch.cat(values, dim=-2)
            if values
            else torch.empty(0, self.d_value, device=self.device)
        )

    def __len__(self) -> int:
        """
        Return total logical sequence length (sum of S over cache entries).

        Note: For total token count including batch/head dims, use memory_usage_bits().
        """
        return sum(c["shape"][-2] for c in self.key_cache) if self.key_cache else 0

    def memory_usage_bits(self) -> dict:
        """Calculate memory usage statistics."""
        if not self.key_cache:
            return {
                "total_bits": 0,
                "fp16_bits": 0,
                "compression_ratio": 0.0,
            }

        # Calculate compressed size: key_indices × key_bits + value_indices × value_bits
        key_indices = sum(c["indices"]["indices"].numel() for c in self.key_cache)
        value_indices = sum(c["indices"]["indices"].numel() for c in self.value_cache)
        total_bits = key_indices * self.key_bits + value_indices * self.value_bits

        # Calculate FP16 size: 2 bytes (16 bits) per element
        fp16_bits = (key_indices + value_indices) * 16

        compression_ratio = fp16_bits / total_bits if total_bits > 0 else 0.0

        return {
            "total_bits": int(total_bits),
            "fp16_bits": int(fp16_bits),
            "compression_ratio": compression_ratio,
        }
