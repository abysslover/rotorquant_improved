"""
CUDA backend for TurboQuant.

Wraps the QJL CUDA kernels with a Python-friendly interface that integrates
into TurboQuant's two-stage quantization pipeline.

Falls back to pure PyTorch if kernels are not available.
"""

import math
import warnings
import os
import sys
import site
import ctypes
import ctypes.util
from typing import Optional, Tuple
from scipy.linalg import hadamard

from ..interfaces import QuantizerBase, ProdQuantizerBase, KVCacheBase
from .common_utils import generate_rotation_matrix_torch as generate_rotation_matrix
from .common_utils import generate_qjl_matrix_torch as generate_qjl_matrix


# CRITICAL: Preload CUDA libraries using ctypes before torch import
# This is necessary because LD_LIBRARY_PATH changes don't affect already-loaded libraries
def _preload_cuda_libraries():
    """Manually preload CUDA libraries using ctypes before torch import."""
    # Discover PyTorch lib directory - CUDA libs are in torch/lib, not torch/lib/cuda/lib
    torch_lib_candidates = []

    for site_dir in site.getsitepackages():
        torch_lib = os.path.join(site_dir, "torch", "lib")
        if os.path.isdir(torch_lib):
            torch_lib_candidates.append(torch_lib)

    torch_lib_candidates.extend(
        [
            "/data/anaconda3/lib/python3.10/site-packages/torch/lib",
            "/data/anaconda3/envs/py310/lib/python3.10/site-packages/torch/lib",
            "/data/anaconda3/lib/python3.9/site-packages/torch/lib",
        ]
    )

    for torch_lib in torch_lib_candidates:
        if os.path.isdir(torch_lib):
            # Preload key CUDA libraries directly from torch/lib
            # libc10.so is required by all other CUDA libs
            libc10_path = os.path.join(torch_lib, "libc10.so")
            if os.path.exists(libc10_path):
                try:
                    ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

            # Preload libtorch_python.so
            libtorch_python_path = os.path.join(torch_lib, "libtorch_python.so")
            if os.path.exists(libtorch_python_path):
                try:
                    ctypes.CDLL(libtorch_python_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

            # Set LD_LIBRARY_PATH for any remaining libraries
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            if torch_lib not in ld_library_path:
                os.environ["LD_LIBRARY_PATH"] = torch_lib + (
                    ":" + ld_library_path if ld_library_path else ""
                )
            break


# Preload CUDA libraries before torch import
_preload_cuda_libraries()

# Now import torch and other dependencies
import torch
import torch.nn as nn
import ctypes
import ctypes.util
import importlib.util

# Try importing pre-built CUDA kernels
_CUDA_AVAILABLE = False
_CUDA_LOAD_ERROR = None

try:
    # 1 차 시도: 표준 패키지 경로에서 직접 임포트 (setup.py 설치 환경)
    import turboquant.cuda_qjl_quant as cuda_qjl_quant
    import turboquant.cuda_qjl_score as cuda_qjl_score
    import turboquant.cuda_qjl_gqa_score as cuda_qjl_gqa_score
    import turboquant.quantization as quantization
    _CUDA_AVAILABLE = True
except ImportError:
    try:
        # 2 차 시도: 개발 환경을 위한 수동 .so 파일 로드
        _kernel_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda_kernels"
        )

        # Try multiple possible Python version formats
        possible_versions = [
            f"cpython-{sys.version_info.major}{sys.version_info.minor}",
            f"cpython-{sys.version_info.major}{sys.version_info.minor - 1}",
            "cpython-310",
            "cpython-39",
        ]

        modules_to_load = {}
        for mod_name in [
            "cuda_qjl_quant",
            "cuda_qjl_score",
            "cuda_qjl_gqa_score",
            "quantization",
        ]:
            # Try each version format until we find the so file
            so_path = None
            for py_version in possible_versions:
                candidate = os.path.join(
                    _kernel_dir, f"{mod_name}.{py_version}-x86_64-linux-gnu.so"
                )
                if os.path.exists(candidate):
                    so_path = candidate
                    break

            if so_path and os.path.exists(so_path):
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, so_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        # Register in sys.modules to ensure proper loading
                        sys.modules[mod_name] = mod
                        spec.loader.exec_module(mod)
                        modules_to_load[mod_name] = mod
                    else:
                        _CUDA_LOAD_ERROR = f"Could not create spec for {mod_name}"
                except OSError as e:
                    _CUDA_LOAD_ERROR = f"OSError loading {mod_name}: {e}"
            else:
                _CUDA_LOAD_ERROR = f"So file not found for {mod_name}"

        # Set globals only if all modules loaded successfully
        if len(modules_to_load) == 4:
            for mod_name, mod in modules_to_load.items():
                globals()[mod_name] = mod
            _CUDA_AVAILABLE = True
        else:
            _CUDA_AVAILABLE = False
            _CUDA_LOAD_ERROR = f"Only {len(modules_to_load)}/4 CUDA modules loaded"
    except Exception as inner_e:
        _CUDA_AVAILABLE = False
        _CUDA_LOAD_ERROR = str(inner_e)

except Exception as e:
    _CUDA_AVAILABLE = False
    _CUDA_LOAD_ERROR = str(e)


def is_cuda_available():
    return _CUDA_AVAILABLE


# ─── QJL CUDA kernel wrappers ───────────────────────────────────────


def qjl_quant(key_states, outlier_indices, rand_prj, outlier_sketch_dim):
    """Fused QJL quantization via CUDA kernel."""
    key_dtype = key_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_quant_half_half",
        (torch.half, torch.float): "qjl_quant_half_float",
        (torch.float, torch.float): "qjl_quant_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_quant_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_quant_bf16_float",
    }
    fn_name = dispatch.get((key_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: key={key_dtype}, proj={rand_dtype}")
    return getattr(cuda_qjl_quant, fn_name)(
        key_states, outlier_indices, rand_prj, outlier_sketch_dim
    )


def qjl_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    """Fused QJL score computation via CUDA kernel."""
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_score_cuda_half_half",
        (torch.half, torch.float): "qjl_score_cuda_half_float",
        (torch.float, torch.float): "qjl_score_cuda_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_score_cuda_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_score_cuda_bf16_float",
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")
    return getattr(cuda_qjl_score, fn_name)(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )


def qjl_gqa_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    """Fused QJL GQA score computation via CUDA kernel."""
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_gqa_score_cuda_half_half",
        (torch.half, torch.float): "qjl_gqa_score_cuda_half_float",
        (torch.float, torch.float): "qjl_gqa_score_cuda_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_gqa_score_cuda_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_gqa_score_cuda_bf16_float",
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")
    return getattr(cuda_qjl_gqa_score, fn_name)(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )


def quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    """Quantized batched matmul for value reconstruction."""
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape
    feat_per_int = 32 // bits
    fA = fA.view(-1, M, K).contiguous()
    N = qB.shape[-1] * feat_per_int
    qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_B = B * nh if not mqa else B

    scales = (
        scales.view(flatten_B, scales.shape[-2], scales.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    zeros = (
        zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )

    assert bits in [2, 4]

    dispatch = {
        torch.float16: "batchedQuantizedMultiplyAccumulate_half",
        torch.float32: "batchedQuantizedMultiplyAccumulate_float",
        torch.bfloat16: "batchedQuantizedMultiplyAccumulate_bf16",
    }
    fn_name = dispatch.get(fA.dtype)
    if fn_name is None:
        raise TypeError(f"Unsupported dtype: {fA.dtype}")
    result = getattr(quantization, fn_name)(
        fA, qB, scales, zeros, bits, group_size, nh, mqa
    )
    return result.view(B, nh, result.shape[-2], result.shape[-1])


# ─── QJL Sketch (adapted from QJL repo) ────────────────────────────


class QJLSketch(nn.Module):
    """
    QJL random projection sketch for 1-bit quantization.
    Adapted from amirzandieh/QJL with CUDA kernel support.
    """

    def __init__(
        self,
        dim: Tuple[int, int],
        dim_outlier: int,
        device=None,
        rng=None,
        rot=True,
        rht=False,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        assert len(dim) == 2, "dim should be (head_dim, sketch_dim)"
        self.dim = dim
        self.dim_outlier = dim_outlier

        self.proj_dir = self._init_proj_dir(rng).contiguous()
        self.proj_dir_score = (
            self._init_rot_dir().contiguous() if rot else self.proj_dir
        )
        if rht:
            self.proj_dir_score = self._compose_rht().contiguous()
        self.proj_dir_quant = self.proj_dir_score.transpose(0, 1).contiguous()

    def _init_proj_dir(self, rng):
        return torch.randn(
            self.dim, generator=rng, dtype=torch.float32, device=self.device
        )

    def _init_rot_dir(self):
        rot_matrices = []
        num_chunks = (self.dim[1] + self.dim[0] - 1) // self.dim[0]
        for i in range(num_chunks):
            start = i * self.dim[0]
            end = (i + 1) * self.dim[0]
            q, _ = torch.linalg.qr(self.proj_dir[:, start:end], mode="reduced")
            rot_matrices.append(q)
        return torch.cat(rot_matrices, dim=-1) * math.sqrt(self.dim[0])

    def _compose_rht(self):
        H = torch.from_numpy(
            hadamard(self.dim[0], dtype=float) / math.sqrt(self.dim[0])
        ).to(self.device)
        D = 2.0 * torch.randint(0, 2, (self.dim[0],), device=self.device) - 1.0
        HD = (H * D).to(self.proj_dir_score.dtype)
        return torch.einsum("dn,dm->mn", self.proj_dir_score, HD)

    def quantize_cuda(self, data, outlier_indices):
        """Quantize using fused CUDA kernel."""
        assert data.shape[-1] == self.dim[0]
        return qjl_quant(
            data.contiguous(),
            outlier_indices.contiguous(),
            self.proj_dir_quant,
            self.dim_outlier,
        )

    def quantize_pytorch(self, data, outlier_mask):
        """Pure PyTorch fallback for quantization."""
        s = self.proj_dir_quant.shape[0]
        key_outlier = data * outlier_mask.unsqueeze(-2)
        key_inlier = data * (1 - outlier_mask.unsqueeze(-2))

        proj_dtype = self.proj_dir_quant.dtype
        sketched_outlier = torch.einsum(
            "...nd,...sd->...ns", key_outlier.to(proj_dtype), self.proj_dir_quant
        )
        sketched_inlier = torch.einsum(
            "...nd,...sd->...ns", key_inlier.to(proj_dtype), self.proj_dir_quant
        )

        bit_pack_len = 8
        sketched_outlier = sketched_outlier.view(
            *sketched_outlier.shape[:-1], -1, bit_pack_len
        )
        sketched_inlier = sketched_inlier.view(
            *sketched_inlier.shape[:-1], -1, bit_pack_len
        )

        enc_vec = 2 ** torch.arange(
            bit_pack_len, dtype=torch.uint8, device=data.device
        ).view(1, 1, 1, -1)
        hash_outlier = ((sketched_outlier > 0) * enc_vec).sum(dim=-1, dtype=torch.uint8)
        hash_inlier = ((sketched_inlier > 0) * enc_vec).sum(dim=-1, dtype=torch.uint8)

        hash_outlier = hash_outlier[:, :, :, :, : s // 16]
        return hash_inlier, hash_outlier

    def quantize(self, data, outlier_indices):
        """Dispatch to CUDA or PyTorch."""
        if _CUDA_AVAILABLE and data.is_cuda:
            return self.quantize_cuda(data, outlier_indices)
        else:
            # Convert outlier_indices to mask for PyTorch path
            mask = torch.zeros(
                data.shape[:3] + (data.shape[-1],), device=data.device, dtype=data.dtype
            )
            for i in range(outlier_indices.shape[-1]):
                idx = outlier_indices[..., i].long()
                mask.scatter_(-1, idx.unsqueeze(-1), 1.0)
            return self.quantize_pytorch(data, mask)

    def calc_score_cuda(
        self, query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier
    ):
        """Compute attention scores using CUDA kernel."""
        sketched_q = torch.matmul(
            query.to(self.proj_dir_score.dtype), self.proj_dir_score
        )
        if data_quant.stride(-1) != 1:
            data_quant = data_quant.contiguous()
        return qjl_score(
            data_quant.contiguous(),
            outlier_quant.contiguous(),
            norm_data.contiguous(),
            norm_outlier.contiguous(),
            outlier_indices.contiguous(),
            sketched_q.contiguous(),
            query.contiguous(),
            self.proj_dir_score,
        )

    def calc_score_pytorch(
        self, query, data_quant, outlier_quant, norm_data, norm_outlier, sketch_dim
    ):
        """Pure PyTorch fallback for score computation."""
        # Unpack bit-packed quantized keys and compute inner products
        sketched_q = torch.matmul(
            query.to(self.proj_dir_score.dtype), self.proj_dir_score
        )

        bit_pack_len = 8
        B, H = data_quant.shape[:2]

        scores_list = []
        for n in range(data_quant.shape[2]):
            for g in range(data_quant.shape[3]):
                # Unpack bits
                k_packed = data_quant[:, :, n, g]  # (B, H, hash_dim)
                bits_unpacked = torch.zeros(B, H, sketch_dim, device=query.device)
                for byte_idx in range(k_packed.shape[-1]):
                    for bit in range(8):
                        dim_idx = byte_idx * 8 + bit
                        if dim_idx < sketch_dim:
                            bits_unpacked[:, :, dim_idx] = (
                                (k_packed[:, :, byte_idx] >> bit) & 1
                            ).float() * 2 - 1

                ip = (sketched_q.squeeze(-2) * bits_unpacked).sum(dim=-1)
                scl = math.sqrt(math.pi / 2) / sketch_dim
                nk = norm_data[:, :, n, g]
                score = scl * nk * ip
                scores_list.append(score)

        return torch.stack(scores_list, dim=-1).unsqueeze(-1)

    def calc_score(
        self, query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier
    ):
        """Dispatch to CUDA or PyTorch."""
        if _CUDA_AVAILABLE and query.is_cuda:
            return self.calc_score_cuda(
                query,
                data_quant,
                outlier_quant,
                outlier_indices,
                norm_data,
                norm_outlier,
            )
        raise RuntimeError(
            "CUDA kernels required for calc_score. Build with: cd turboquant/csrc && python setup.py build_ext --inplace"
        )


# ─── Key Quantizer with streaming support ──────────────────────────


class QJLKeyQuantizer:
    """
    Online key quantizer with streaming support.
    Buffers incoming keys and quantizes them in groups.
    """

    def __init__(
        self,
        qjl_sketch: QJLSketch,
        outliers_count: int,
        buffer_size: int,
        group_size: int,
        qjl_dim: int,
    ):
        self.qjl_sketch = qjl_sketch
        self.outliers_count = outliers_count
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.qjl_dim = qjl_dim
        self.seq_len = None
        self.outlier_indices = None
        self.key_states_quant = None
        self.key_outliers_quant = None
        self.key_outliers_norm = None
        self.key_states_norm = None
        self.key_residual = None

    def build_sketch(self, key_states: torch.Tensor):
        """Initial quantization of a batch of keys."""
        b, h, _, dim = key_states.shape
        self.seq_len = key_states.shape[-2]
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.key_residual = key_states[:, :, self.seq_len - residual_size :, :]
        if residual_size == self.seq_len:
            return None

        num_groups = (self.seq_len - residual_size) // self.group_size
        key_states = (
            key_states[:, :, : self.seq_len - residual_size, :]
            .reshape((b, h, num_groups, self.group_size, dim))
            .contiguous()
        )

        norms = key_states.norm(dim=-2)
        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        self.outlier_indices = outlier_indices.to(torch.uint8).contiguous()

        self.key_states_quant, self.key_outliers_quant, self.key_outliers_norm = (
            self.qjl_sketch.quantize(key_states, self.outlier_indices)
        )
        self.key_states_norm = torch.norm(key_states, dim=-1)

    def update_sketch(self, key_states: torch.Tensor):
        """Append a single new key token."""
        assert key_states.shape[-2] == 1
        self.seq_len += 1

        if self.key_residual is not None:
            self.key_residual = torch.cat([self.key_residual, key_states], dim=-2)
        else:
            self.key_residual = key_states

        if self.seq_len % self.buffer_size != 0:
            return None

        b, h, _, dim = self.key_residual.shape
        self.key_residual = self.key_residual.reshape((b, h, -1, self.group_size, dim))

        norms = self.key_residual.norm(dim=-2)
        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        outlier_indices = outlier_indices.to(torch.uint8)
        self.outlier_indices = torch.cat(
            [self.outlier_indices, outlier_indices], dim=2
        ).contiguous()

        kq, koq, kon = self.qjl_sketch.quantize(self.key_residual, outlier_indices)
        self.key_states_quant = torch.cat(
            [self.key_states_quant, kq], dim=2
        ).contiguous()
        self.key_outliers_quant = torch.cat(
            [self.key_outliers_quant, koq], dim=2
        ).contiguous()
        self.key_outliers_norm = torch.cat(
            [self.key_outliers_norm, kon], dim=2
        ).contiguous()

        residual_norm = torch.norm(self.key_residual, dim=-1)
        self.key_states_norm = torch.cat(
            [self.key_states_norm, residual_norm], dim=2
        ).contiguous()

        self.key_residual = None

    def attention_score(self, query_states: torch.Tensor) -> torch.Tensor:
        """Compute attention scores against all quantized keys."""
        residual = None
        if self.key_residual is not None:
            residual = torch.matmul(query_states, self.key_residual.transpose(-1, -2))

        scores = self.qjl_sketch.calc_score(
            query_states,
            self.key_states_quant,
            self.key_outliers_quant,
            self.outlier_indices,
            self.key_states_norm,
            self.key_outliers_norm,
        ).transpose(-1, -2)

        if residual is not None:
            return torch.cat([scores, residual], dim=-1)
        return scores


class TurboQuantMSE(QuantizerBase):
    """CUDA Kernel-based TurboQuant MSE quantizer."""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cuda"):
        super().__init__(d, bits, seed, device)

        if device.startswith("cuda") and not _CUDA_AVAILABLE:
            raise RuntimeError(
                "TurboQuantMSE(cuda_kernel) requires compiled CUDA extensions. "
                "Use engine='torch_cuda' for PyTorch CUDA fallback."
            )

        self.register_buffer(
            "Pi", generate_rotation_matrix(d, seed=seed, device=device)
        )

        from ..common.lloyd_max import LloydMaxCodebook

        codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", codebook.centroids.to(device))
        self.register_buffer("boundaries", codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi

    def quantize(self, x: torch.Tensor):
        y = self.rotate(x)
        diffs = y.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)

        x_hat = self.dequantize({"indices": indices})
        return x_hat, {"indices": indices}

    def dequantize(self, indices_dict):
        """양자화된 인덱스를 역양자화하여 원래 값으로 복원 (args: indices_dict -> Tensor)"""
        # 엔진 간 일관성을 위해 long 타입 캐스팅 적용
        y_hat = self.centroids[indices_dict["indices"].long()]
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Avoid double dequantization by returning quantize results directly."""
        return self.quantize(x)


class TurboQuantProd(ProdQuantizerBase):
    """CUDA Kernel-based TurboQuant Prod quantizer."""

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.mse_bits = max(bits - 1, 1)
        super().__init__(d, bits, qjl_dim, seed, device)

        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)

        self.register_buffer(
            "S",
            generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device),
        )

    def quantize(self, x: torch.Tensor):
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

    def dequantize(self, compressed):
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed):
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


class TurboQuantKVCache(KVCacheBase):
    """KV Cache wrapper for TurboQuant with MSE-only compression (QJL removed)."""

    def __init__(
        self,
        d_key: int,
        d_value: int,
        bits: int = 3,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        seed: int = 42,
        device: str = "cuda",
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
            return {"total_bits": 0, "fp16_bits": 0, "compression_ratio": 0.0}

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
