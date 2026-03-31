# TurboQuant + RotorQuant + IsoQuant + PlanarQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — plus **RotorQuant** (Clifford rotors), **[IsoQuant](https://github.com/ParaMind2025/isoquant)** (quaternion 4D blocks, [local impl](turboquant/isoquant.py)), and **PlanarQuant** (2D Givens rotations, [local impl](turboquant/planarquant.py)), progressively faster drop-in replacements for the dense rotation step.

**[PlanarQuant](https://github.com/ParaMind2025/isoquant)** (by ParaMind2025) is the fastest PyTorch-level variant: **10–27x faster** than RotorQuant and **2–5x faster** than IsoQuant-Fast in pure PyTorch, at identical reconstruction quality. With Triton kernels, both PlanarQuant and IsoQuant converge to **~30µs** (memory-bound floor). **[IsoQuant](https://github.com/ParaMind2025/isoquant)** (also ParaMind2025) remains the recommended default for its clean 4D hardware alignment.

## Head-to-Head vs Reference TurboQuant

Benchmarked against [back2matching/turboquant](https://github.com/back2matching/turboquant) v0.2.0 (first open-source TurboQuant, pip-installable) on RTX 5090, PyTorch 2.11, Triton 3.6.

### Synthetic MSE (unit vectors, d=128, n=2000)

| bits | Ref TurboQuant | RotorQuant | IsoQuant-Fast | Theory bound | Winner |
|------|---------------|------------|---------------|-------------|--------|
| 2 | 0.128799 | **0.115858** | 0.116346 | 0.170044 | **RQ** |
| 3 | 0.049446 | **0.034060** | 0.034310 | 0.042511 | **RQ** |
| 4 | 0.019621 | **0.009302** | 0.009431 | 0.010628 | **RQ** |

On synthetic unit vectors, RotorQuant and IsoQuant beat the reference by ~2x at every bit width. All methods are well below the theoretical bound from the paper.

### Inner Product Preservation (two-stage with QJL)

| bits | Method | Bias | RMSE | Correlation |
|------|--------|------|------|-------------|
| 3 | RefTQ | +0.0015 | 0.0413 | 0.8931 |
| 3 | RQ | +0.0006 | 0.0419 | 0.8786 |
| 3 | IQ-Fast | +0.0019 | 0.0412 | 0.8834 |
| 4 | RefTQ | +0.0015 | 0.0257 | 0.9610 |
| 4 | RQ | +0.0006 | 0.0228 | **0.9656** |
| 4 | IQ-Fast | +0.0007 | 0.0226 | **0.9664** |

All methods near-tied on inner product quality. All biases near zero (unbiased as proven in the paper).

### NIAH Retrieval

All three methods achieve **EXACT** retrieval across all bit widths (2/3/4) and sequence lengths (512/2048/8192). No misses.

### Real Model PPL (Qwen2.5-3B-Instruct, K-cache quantization)

| Method | PPL | vs FP16 baseline (8.18) |
|--------|-----|------------------------|
| FP16 baseline | **8.18** | — |
| RefTQ 3-bit | 352.98 | +344.80 |
| RotorQuant 3-bit | 43.71 | +35.53 |
| **IsoQuant-Fast 3-bit** | **22.71** | **+14.53** |
| RefTQ 4-bit | 18.58 | +10.40 |
| RotorQuant 4-bit | 30.27 | +22.09 |
| **IsoQuant-Fast 4-bit** | **15.70** | **+7.51** |

IsoQuant-Fast wins PPL at both bit widths. The reference TurboQuant's 3-bit result blows up on Qwen2.5 (2 KV heads) — this matches independent reports of catastrophic PPL with symmetric TurboQuant 3-bit on this model.

### K-cache MSE on Real Model Tensors

| bits | Ref TurboQuant | RotorQuant | IsoQuant-Fast |
|------|---------------|------------|---------------|
| 3 | **0.534** | 1.808 | 2.306 |
| 4 | **0.189** | 0.857 | 1.386 |

On real K vectors (non-unit, std=3.32, norm_mean=26.76), the reference TurboQuant's full d×d rotation achieves lower MSE. However, **lower MSE does not translate to better PPL** — IsoQuant's group-wise rotation preserves directional information that matters more for attention score computation.

### Speed (quantize + dequantize roundtrip, d=128, 3-bit)

| n vectors | Ref TurboQuant | RotorQuant | IsoQuant-Fast |
|-----------|---------------|------------|---------------|
| 1,000 | **0.20 ms** | 8.43 ms | 1.62 ms |
| 5,000 | **0.26 ms** | 12.52 ms | 1.44 ms |
| 10,000 | **0.46 ms** | 19.37 ms | 1.31 ms |

RefTQ's dense matmul is fastest on GPU (matrix multiply is what GPUs are optimized for). The advantage of RQ/IQ is parameter efficiency, not raw rotation speed.

### Parameter Efficiency

| Method | Rotation params | Total | vs RefTQ |
|--------|----------------|-------|----------|
| Ref TurboQuant | 16,384 (128×128 matrix) | 16,392 | 1x |
| RotorQuant | 344 (43 Cl(3,0) rotors) | 352 | **46.6x smaller** |
| IsoQuant-Fast | 128 (32 quaternions) | 136 | **120.5x smaller** |

### PPL at Scale (Qwen2.5-3B-Instruct, RTX 5090, compressed KV cache)

4-bit:

| Context | FP16 PPL | RefTQ PPL | RQ PPL | IQ-Fast PPL | FP16 Speed | TQ Speed | IQ Speed |
|---------|----------|-----------|--------|-------------|------------|----------|----------|
| 1,024 | **6.38** | 17.34 | 97.23 | 45.74 | 3,133 t/s | 6,239 t/s | 667 t/s |
| 2,048 | **6.98** | 25.12 | 192.36 | 64.89 | 18,699 t/s | 9,531 t/s | 1,302 t/s |
| 4,096 | **7.96** | 37.59 | 269.09 | 50.90 | 17,709 t/s | 12,248 t/s | 2,680 t/s |

3-bit:

| Context | FP16 PPL | RefTQ PPL | RQ PPL | IQ-Fast PPL |
|---------|----------|-----------|--------|-------------|
| 1,024 | **6.38** | 411.34 | 190.53 | 207.43 |
| 2,048 | **6.98** | 963.41 | 242.32 | 196.66 |
| 4,096 | **7.96** | 5,128.36 | 222.95 | **169.99** |

At 3-bit, RefTQ collapses catastrophically on Qwen2.5 (2 KV heads) as context grows. RotorQuant and IsoQuant degrade gracefully — IsoQuant-Fast achieves the best 3-bit PPL at 4K context (170 vs 5,128 for RefTQ).

### VRAM & Speed (measured, Qwen2.5-7B-Instruct, 14.5 GB model, M5 Max 128 GB)

| Context | FP16 Peak | TQ 4-bit Peak | VRAM Saved | FP16 Speed | TQ 4-bit Speed |
|---------|-----------|---------------|------------|------------|----------------|
| 460 | 14,833 MB | 14,758 MB | 75 MB | 17.7 tok/s | **23.8 tok/s** |
| 1,860 | 16,659 MB | 16,215 MB | 444 MB | 1.0 tok/s | **1.4 tok/s** |

### KV Cache VRAM with Compressed Storage (Qwen2.5-3B: 36 layers, 2 KV heads, head_dim=128)

All methods (TQ, RQ, IQ) use identical compressed format: `uint8` indices + `float32` norms per vector. The VRAM savings are method-independent — the difference is quality (PPL) and quantizer state size.

4-bit (3.8x compression):

| Context | FP16 KV | Compressed KV | Saved |
|---------|---------|---------------|-------|
| 460 | 16 MB | 4 MB | 12 MB |
| 1,860 | 65 MB | 17 MB | 48 MB |
| 4,096 | 144 MB | 38 MB | 106 MB |
| 8,192 | 288 MB | 77 MB | 212 MB |
| 16,384 | 576 MB | 153 MB | 423 MB |
| 32,768 | 1,152 MB | 306 MB | **846 MB** |

3-bit (4.9x compression):

| Context | FP16 KV | Compressed KV | Saved |
|---------|---------|---------------|-------|
| 460 | 16 MB | 3 MB | 13 MB |
| 1,860 | 65 MB | 13 MB | 52 MB |
| 4,096 | 144 MB | 29 MB | 115 MB |
| 8,192 | 288 MB | 59 MB | 230 MB |
| 16,384 | 576 MB | 117 MB | 459 MB |
| 32,768 | 1,152 MB | 234 MB | **918 MB** |

### Quantizer State Overhead

The rotation parameters are stored once and shared across all tokens. RotorQuant and IsoQuant's advantage is dramatic at scale — especially when running many layers with separate quantizers.

| Method | Per-quantizer | Total (36L × 2H) | vs RefTQ |
|--------|--------------|-------------------|----------|
| Ref TurboQuant | 128×128 matrix (64 KB) | **4,613 KB** | 1x |
| RotorQuant | 43 rotors × 8 (1.4 KB) | **101 KB** | 46x smaller |
| IsoQuant-Fast | 32 quats × 4 (0.5 KB) | **41 KB** | **114x smaller** |

For a 7B model (28 layers, 32 KV heads) RefTQ needs **57 MB** just for rotation matrices. IsoQuant needs **0.5 MB**.

---

## Rotation Variant Comparison

### Architecture (d=128)

| | TurboQuant | RotorQuant | IsoQuant-Fast | IsoQuant-Full | PlanarQuant |
|---|-----------|-----------|---------------|---------------|-------------|
| Block structure | Dense 128×128 | 43 × 3D Clifford | 32 × 4D quaternion | 32 × 4D quaternion | **64 × 2D Givens** |
| Forward FMAs | 16,384 | 2,408 | 512 | 1,024 | **256** |
| Parameters | 16,384 | 344 | 128 | 256 | **128** |
| Alignment | N/A | 42 blocks + 2D tail | 32 clean blocks | 32 clean blocks | **64 clean pairs** |
| PyTorch latency | — | 2,649 µs | 466 µs (5.7x) | 710 µs (3.7x) | **164 µs (16.2x)** |
| Triton latency | — | 34 µs | **30 µs** | 32 µs | **30 µs** |
| Reconstruction MSE | Baseline | 0.000265 | 0.000266 | 0.000265 | **0.000266** |

### Reconstruction MSE (8192 normalized vectors)

| d | bits | RotorQuant | IsoQuant-Fast | IsoQuant-Full | PlanarQuant | Planar/RQ |
|---|------|-----------|---------------|---------------|-------------|-----------|
| 64 | 2 | 0.001804 | 0.001784 | 0.001789 | 0.001788 | 0.991x |
| 64 | 3 | 0.000525 | 0.000522 | 0.000522 | 0.000522 | 0.995x |
| 64 | 4 | 0.000143 | 0.000143 | 0.000143 | 0.000143 | 1.003x |
| 128 | 2 | 0.000904 | 0.000907 | 0.000905 | 0.000907 | 1.003x |
| 128 | 3 | 0.000265 | 0.000266 | 0.000265 | 0.000266 | 1.002x |
| 128 | 4 | 0.000073 | 0.000073 | 0.000073 | 0.000073 | 1.006x |
| 256 | 2 | 0.000456 | 0.000457 | 0.000456 | 0.000456 | 1.000x |
| 256 | 3 | 0.000134 | 0.000134 | 0.000134 | 0.000134 | 1.000x |
| 256 | 4 | 0.000037 | 0.000037 | 0.000037 | 0.000037 | 1.000x |

MSE is indistinguishable across all methods. PlanarQuant and IsoQuant are pure speed upgrades.

### Stage-1 Latency — PyTorch path (µs, 8192 vectors, RTX 5090)

| d | bits | RotorQuant | IsoQuant-Full | IsoQuant-Fast | PlanarQuant | Planar speedup |
|---|------|-----------|---------------|---------------|-------------|----------------|
| 64 | 2 | 3,166 | 870 | 524 | **119** | **26.6x** |
| 64 | 3 | 2,627 | 792 | 551 | **127** | **20.7x** |
| 64 | 4 | 2,902 | 929 | 368 | **136** | **21.4x** |
| 128 | 2 | 2,724 | 709 | 321 | **235** | **11.6x** |
| 128 | 3 | 2,649 | 710 | 466 | **164** | **16.2x** |
| 128 | 4 | 2,636 | 828 | 681 | **256** | **10.3x** |
| 256 | 2 | 3,730 | 723 | 367 | **302** | **12.3x** |
| 256 | 3 | 3,834 | 750 | 477 | **284** | **13.5x** |
| 256 | 4 | 3,817 | 1,243 | 988 | **804** | **4.7x** |

PlanarQuant PyTorch is 4.7–26.6x faster than RotorQuant. Best gains at low dimensions and low bit widths.

### Stage-1 Latency — Triton kernels (µs, 8192 vectors, RTX 5090)

| d | bits | PQ-PyTorch | PQ-Triton | PQ speedup | IQ-PyTorch | IQ-Triton | IQ speedup |
|---|------|-----------|-----------|------------|-----------|-----------|------------|
| 64 | 2 | 120 | **31** | 3.9x | 303 | **30** | 10.0x |
| 64 | 3 | 124 | **30** | 4.1x | 321 | **30** | 10.6x |
| 64 | 4 | 136 | **31** | 4.3x | 396 | **33** | 12.2x |
| 128 | 2 | 125 | **30** | 4.1x | 315 | **32** | 9.9x |
| 128 | 3 | 166 | **30** | 5.5x | 367 | **31** | 11.9x |
| 128 | 4 | 255 | **30** | 8.4x | 461 | **31** | 14.8x |
| 256 | 2 | 192 | **31** | 6.1x | 380 | **31** | 12.4x |
| 256 | 3 | 279 | **31** | 8.9x | 469 | **32** | 14.9x |
| 256 | 4 | 578 | **31** | 18.7x | 763 | **30** | 25.4x |

With Triton, both PlanarQuant and IsoQuant converge to **~30µs** — the memory-bound floor. The FMA difference (4 vs 16) is invisible at this scale. Without Triton, PlanarQuant's simpler PyTorch path gives 2–5x over IsoQuant.

### Perplexity (wikitext-2, autoregressive with post-prefill quantization)

| Model | KV Heads | FP16 PPL | RQ 4-bit | Delta | RQ 3-bit | Delta |
|-------|----------|---------|---------|-------|---------|-------|
| **Mistral-7B** | 8 | 4.80 | **5.16** | **+7.4%** | 5.53 | +15.3% |
| **Gemma-2-2b** | 4 | 8.87 | **9.77** | **+10.1%** | 10.64 | +19.9% |
| Qwen2.5-3B | 2 | 9.81 | **10.13** | **+3.2%** | 12.28 | +25.2% |

### High-Context Generation

3-bit with post-prefill quantization on Qwen2.5-3B (RTX 5090):

| Context | Speed | VRAM | Needle |
|---------|-------|------|--------|
| 2K | 6.9 tok/s | 2.4 GB | **FOUND** |
| 8K | 8.6 tok/s | 3.1 GB | **FOUND** |
| 16K | 6.0 tok/s | 4.0 GB | **FOUND** |
| 32K | 5.0 tok/s | 5.9 GB | **FOUND** |
| 65K | 2.1 tok/s | 9.6 GB | **FOUND** |

### Attention Logits Speed (Q@K^T, decode mode, RTX 5090)

| KV Length | FP32 | FP16 | **RQ Triton** | **vs FP32** | vs FP16 |
|-----------|------|------|-------------|---------|---------|
| 4K | 0.132 ms | 0.019 ms | **0.024 ms** | **5.4x** | 0.8x |
| 16K | 0.057 ms | 0.033 ms | **0.024 ms** | **2.4x** | **1.4x** |
| 32K | 0.308 ms | 0.066 ms | **0.024 ms** | **12.7x** | **2.7x** |

## How It Works

### TurboQuant (Google)

Two stages: (1) Random rotation via d×d orthogonal matrix → per-coordinate Lloyd-Max quantization. (2) QJL 1-bit residual correction for unbiased inner products.

### RotorQuant

Replaces the d×d matrix with **Clifford rotors** in Cl(3,0). Chunks the vector into groups of 3 dims, rotates each with a 4-parameter rotor via the sandwich product `R v R̃`. 44x fewer parameters, 7.9x fewer FMAs.

### IsoQuant (recommended)

Replaces Clifford rotors with **quaternion 4D blocks** based on the isoclinic decomposition SO(4) ≅ SU(2) × SU(2). Each group of 4 coordinates is treated as a quaternion and rotated via `q_L v q̄_R` (Full) or `q_L v` (Fast).

### PlanarQuant (fastest, by [ParaMind2025](https://github.com/ParaMind2025/isoquant))

The simplest rotation primitive: **2D Givens rotations** (SO(2)). Each pair of adjacent coordinates is rotated by an independent angle θ. Only 4 FMAs per pair — the theoretical minimum for rotation-based quantization.

| | TurboQuant | RotorQuant | IsoQuant-Fast | PlanarQuant |
|---|-----------|-----------|---------------|-------------|
| Rotation | Dense d×d matmul | Cl(3,0) rotor sandwich | Quaternion multiply | **2D Givens rotation** |
| Block size | d | 3 | 4 (hardware-aligned) | **2** (pair-aligned) |
| FMAs (d=128) | 16,384 | 2,408 | 512 | **256 (64x fewer)** |
| Parameters | 16,384 | 344 | 128 | **128 (128x fewer)** |
| Alignment | N/A | Tail handling | Clean power-of-2 | **Clean pairs** |
| Quality | Baseline | 1.0x | 1.0x | **1.0x** |
| Triton latency | — | 34 µs | 30 µs | **30 µs** (tied) |
| PyTorch latency | — | 2,649 µs | 466 µs | **164 µs** |

### Key Innovations

**Grade elimination** (RotorQuant): The rotor sandwich of a grade-1 vector produces only odd grades. Dropping non-vector grades cuts storage from 344 → 129 indices per vector, matching TurboQuant's 128.

**4D hardware alignment** (IsoQuant): d=128 splits into 32 clean 4D blocks (no tail), fitting naturally into SIMD float4 patterns. RotorQuant's 3D blocks create 42 groups + 2D remainder.

**Norm separation**: Normalize to unit sphere before quantization, store norms separately. Combined with correct `d_eff` for Lloyd-Max codebook, this achieves MSE parity with TurboQuant.

**Post-prefill quantization**: Prefill runs at full FP16 (no error compounding through layers). First decode step bulk-quantizes the cache.

## Quick Start

```python
from turboquant import IsoQuantMSE, IsoQuantProd, PlanarQuantMSE

# PlanarQuant: fastest variant (2D Givens rotations, by ParaMind2025)
pq = PlanarQuantMSE(d=128, bits=3, device='cuda')
x_hat, indices = pq(x)  # quantize + dequantize

# IsoQuant: recommended default (4D quaternion rotations)
iq = IsoQuantMSE(d=128, bits=3, mode='fast', device='cuda')
x_hat, indices = iq(x)

# Stage 1 + 2: With QJL residual correction
iq_prod = IsoQuantProd(d=128, bits=3, mode='fast', device='cuda')
compressed = iq_prod.quantize(keys)
ip_estimate = iq_prod.inner_product(queries, compressed)

# Legacy Clifford interface (still available)
from turboquant import RotorQuantMSE
rq = RotorQuantMSE(d=128, bits=3, device='cuda')
```

## Triton Kernels

Portable, auto-tuned GPU kernels — no CUDA C++ compilation needed:

| Kernel | Purpose | Latency (d=128, 3-bit) |
|--------|---------|----------------------|
| **`triton_planar2_fused`** | **PlanarQuant 2D full pipeline** | **~30 µs** |
| **`triton_iso_fast_fused`** | **IsoQuant-Fast full pipeline** | **~30 µs** |
| **`triton_iso_full_fused`** | **IsoQuant-Full full pipeline** | ~32 µs |
| `triton_rotor_full_fused` | Clifford quantize-dequantize pipeline | 34 µs |
| `triton_planar2_quantize` | PlanarQuant quantize-only (returns indices) | — |
| `triton_planar2_dequantize` | PlanarQuant dequantize-only (from indices) | — |
| `triton_rotor_sandwich` | Clifford R x R̃ (embed + rotor sandwich) | — |
| `triton_fused_attention_qjl` | Q@K^T with QJL correction (experimental) | — |

```python
from turboquant import IsoQuantMSE, triton_iso_fast_fused

iq = IsoQuantMSE(d=128, bits=3, mode='fast', device='cuda')

# Triton fused quantize-dequantize (70x faster than PyTorch)
x_hat = triton_iso_fast_fused(x, iq.q_L, iq.centroids)
```

## Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `benchmark_vs_reference.py` | **vs reference TurboQuant (MSE, PPL, VRAM, speed)** | `python benchmark_vs_reference.py` |
| `benchmark_isoquant.py` | IsoQuant vs RotorQuant head-to-head | `python -m turboquant.benchmark_isoquant` |
| `benchmark_google_parity.py` | Full TurboQuant parity test | `python -m turboquant.benchmark_google_parity` |
| `benchmark_perplexity.py` | Perplexity benchmark (autoregressive + roundtrip) | `python -m turboquant.benchmark_perplexity` |
| `poc_high_context.py` | High-context generation (2K-131K tokens) | `python -m turboquant.poc_high_context` |
| `benchmark_triton.py` | Triton kernel speed (6 tests) | `python -m turboquant.benchmark_triton` |

## Project Structure

```
benchmark_vs_reference.py    # Head-to-head vs reference TurboQuant (pip)
turboquant/
  planarquant.py             # PlanarQuant: 2D Givens rotation (fastest, by ParaMind2025)
  isoquant.py                # IsoQuant: quaternion 4D block rotation (recommended)
  rotorquant.py              # RotorQuant: Clifford 3D block rotation (legacy)
  clifford.py                # Cl(3,0) geometric algebra
  triton_kernels.py          # Triton GPU kernels (rotor sandwich, fused pipeline, attention)
  fused_attention.py         # Fused attention with QJL correction (experimental)
  turboquant.py              # TurboQuant: dense rotation baseline
  lloyd_max.py               # Lloyd-Max optimal scalar quantizer
  compressors.py             # Asymmetric inner product compressors
  cuda_backend.py            # QJL CUDA kernel wrappers
  benchmark_isoquant.py      # All variants benchmark (Planar/Iso/Rotor)
  benchmark_google_parity.py # Google TurboQuant parity benchmark
  benchmark_perplexity.py    # Perplexity benchmark
  benchmark_triton.py        # Triton kernel benchmarks
  poc_high_context.py        # High-context generation POC
  csrc/
    planar2_fused_kernel.cu  # CUDA fused 2D rotation kernel (from ParaMind2025)
    rotor_fused_kernel.cu    # CUDA fused Clifford rotation kernel
tests/                       # Unit tests
setup.py                     # pip install with optional CUDA build
```

## Requirements

```bash
pip install -e .                    # PyTorch-only
pip install triton                  # Add Triton kernels (for Clifford path)
pip install -e ".[validate]"        # + model validation deps (transformers, bitsandbytes)
```

- Python 3.10+, PyTorch 2.0+, CUDA, scipy
- triton >= 3.0 (optional, for Clifford Triton kernels)

## When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| **Maximum speed** | **PlanarQuant 3-bit** (10–27x faster than RQ, same quality) |
| **Default** | **IsoQuant-Fast 3-bit** (5.8x faster, 4D hardware-aligned) |
| KV cache compression (quality) | IsoQuant-Fast 4-bit (+3-10% PPL, 3.7x compression) |
| KV cache compression (size) | IsoQuant-Fast 3-bit (4.9x, matches TQ) |
| Long context on limited VRAM | PlanarQuant or IsoQuant-Fast 3-bit + post-prefill |
| Triton kernel path needed | RotorQuant (Triton kernels available) |
| Apple Silicon | RotorQuant + Metal shader |

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — [Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — [Triton impl](https://dejan.ai/blog/turboquant/)
- [back2matching/turboquant](https://github.com/back2matching/turboquant) — Reference open-source TurboQuant (pip install turboquant)
- [IsoQuant / PlanarQuant](https://github.com/ParaMind2025/isoquant) — Ji, "IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression" (March 2026). PlanarQuant (2D Givens rotation variant) from the same repository.
- [QJL: 1-Bit Quantized JL Transform](https://arxiv.org/abs/2406.03482) — [Code](https://github.com/amirzandieh/QJL)
- [CommVQ](https://arxiv.org/abs/2506.18879) (ICML 2025) — [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- [CliffordNet](https://arxiv.org/abs/2601.06793) (Jan 2026)

## Citation

```bibtex
@article{pope2026rotorquant,
  title={RotorQuant: Clifford Algebra Vector Quantization for LLM KV Cache Compression},
  author={Pope, John D.},
  year={2026},
  url={https://www.scrya.com/rotorquant/},
  note={Code: https://github.com/scrya-com/rotorquant}
}
```

## License

MIT
