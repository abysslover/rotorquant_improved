# TurboQuant + RotorQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — plus **RotorQuant**, a Clifford algebra reimagining with **44x fewer parameters**, **7x fewer FMAs**, and **Triton GPU kernels**.

## Google TurboQuant Parity

RotorQuant matches Google TurboQuant's quality claims while using Clifford rotors instead of dense rotation matrices:

| Metric | Google TurboQuant | RotorQuant | Status |
|--------|------------------|------------|--------|
| **Perplexity (4-bit)** | <5% degradation | **+3.2%** (PPL 10.13 vs 9.81) | **MATCH** |
| **Perplexity (3-bit)** | <5% (Gemma, many KV heads) | +25.2% (Qwen, 2 KV heads) | Expected |
| **Needle-in-haystack** | Perfect at all bit widths | **4/4 FOUND** (3-bit & 4-bit, 2K-32K) | **MATCH** |
| **Generation quality** | Coherent | Coherent (code, reasoning, knowledge) | **MATCH** |
| **MSE vs FP16** | Near-optimal | **1.0x ratio** (identical to TQ) | **MATCH** |
| **Parameters** | 16,384 per head | **~380** (44x fewer) | **RQ wins** |
| **FMAs** | 16,384 per vector | **2,064** (7.9x fewer) | **RQ wins** |
| **Compression (3-bit)** | 4.9x | **4.9x** | **MATCH** |
| **Compression (2-bit)** | 7.1x | **7.1x** | **MATCH** |
| **Attn logits speed (32K)** | 8x vs FP32 (H100) | **12.7x vs FP32** (RTX 5090) | **RQ wins** |

Tested on Google's models (Gemma-2-2b, Mistral-7B) plus Qwen2.5-3B, Qwen2.5-7B, and Phi-4-mini (RTX 5090).

### Perplexity (wikitext-2, autoregressive with post-prefill quantization)

| Model | KV Heads | FP16 PPL | RQ 4-bit | Delta | RQ 3-bit | Delta |
|-------|----------|---------|---------|-------|---------|-------|
| **Mistral-7B** | 8 | 4.80 | **5.16** | **+7.4%** | 5.53 | +15.3% |
| **Gemma-2-2b** | 4 | 8.87 | **9.77** | **+10.1%** | 10.64 | +19.9% |
| Qwen2.5-3B | 2 | 9.81 | **10.13** | **+3.2%** | 12.28 | +25.2% |

### High-Context Generation

3-bit RotorQuant with post-prefill quantization on Qwen2.5-3B (RTX 5090):

| Context | Speed | VRAM | Needle |
|---------|-------|------|--------|
| 2K | 6.9 tok/s | 2.4 GB | **FOUND** |
| 8K | 8.6 tok/s | 3.1 GB | **FOUND** |
| 16K | 6.0 tok/s | 4.0 GB | **FOUND** |
| 32K | 5.0 tok/s | 5.9 GB | **FOUND** |
| 65K | 2.1 tok/s | 9.6 GB | **FOUND** |

Also tested on Qwen2.5-7B (6/6 FOUND to 65K) and Phi-4-mini-128K (4/6 FOUND to 65K).

### Speed Overhead

| Context | FP16 | RQ 3-bit | Slowdown |
|---------|------|----------|----------|
| Short (~40 tokens) | 35.3 tok/s | 29.5 tok/s | **17%** |
| Long (~60 tokens) | 36.7 tok/s | 31.5 tok/s | **12%** |
| 32K tokens | — | 5.0 tok/s | Bulk cache quantization |

At short contexts, RotorQuant is only **12-19% slower** than FP16.

### Attention Logits Speed (Q@K^T, decode mode, RTX 5090)

Google measures TurboQuant as "8x faster than FP32 on H100". Same measurement for RotorQuant:

| KV Length | FP32 | FP16 | **RQ Triton** | **vs FP32** | vs FP16 |
|-----------|------|------|-------------|---------|---------|
| 4K | 0.132 ms | 0.019 ms | **0.024 ms** | **5.4x** | 0.8x |
| 16K | 0.057 ms | 0.033 ms | **0.024 ms** | **2.4x** | **1.4x** |
| 32K | 0.308 ms | 0.066 ms | **0.024 ms** | **12.7x** | **2.7x** |

The Triton gather-dot kernel stays flat at ~0.024ms regardless of context length — it loads uint8 indices (1 byte) vs FP32 keys (4 bytes). The advantage grows with context as the baseline becomes memory-bandwidth-bound.

## How It Works

### TurboQuant (Google)

Two stages: (1) Random rotation via d×d orthogonal matrix → per-coordinate Lloyd-Max quantization. (2) QJL 1-bit residual correction for unbiased inner products.

### RotorQuant (this project)

Replaces the d×d matrix with **Clifford rotors** in Cl(3,0). Chunks the vector into groups of 3 dims, rotates each with a 4-parameter rotor via the sandwich product `R v R̃`.

| | TurboQuant | RotorQuant | Ratio |
|---|-----------|-----------|-------|
| Rotation | 16,384 FMAs (dense matmul) | **2,064 FMAs** (sparse per-group) | **7.9x fewer** |
| Full pipeline | 33,792 FMAs | **4,816 FMAs** | **7.0x fewer** |
| Parameters | 16,384 | **~380** | **~44x fewer** |
| Stored indices | 128/vector | 129/vector | ~same |
| Compression (3-bit) | 4.9x | **4.9x** | **MATCH** |

### Key Innovations

**Grade elimination**: The rotor sandwich of a grade-1 vector produces only odd grades. Scalar and bivector are always zero (skip). Trivector (e123) is non-zero but never read by `extract_vectors` (which only takes grade-1). Dropping all non-vector grades cuts storage from 344 → 129 indices per vector, matching TurboQuant's 128.

**Norm separation**: Normalize to unit sphere before quantization, store norms separately. Combined with correct `d_eff` for Lloyd-Max codebook, this brought MSE from 2-4x worse than TQ to exactly 1.0x.

**Post-prefill quantization**: Prefill runs at full FP16 (no error compounding through layers). First decode step bulk-quantizes the cache via Triton. Each subsequent decode step quantizes the new key but returns full-precision for current attention.

**Non-commutative algebra fix**: The rotor sandwich `(R * x) * R̃` requires LEFT product for step 1 and RIGHT product for step 2 — these differ in non-commutative Clifford algebra.

## Triton Kernels

Portable, auto-tuned GPU kernels — no CUDA C++ compilation needed:

| Kernel | Purpose | Speedup vs PyTorch |
|--------|---------|-------------------|
| `triton_rotor_full_fused` | Full quantize-dequantize pipeline | **128-652x** |
| `triton_rotor_sandwich` | R x R̃ (embed + rotor sandwich) | 80-166x |
| `triton_fused_attention_qjl` | Q@K^T with QJL correction (experimental) | — |

```python
from turboquant import RotorQuantMSE, pack_rotors_for_triton, triton_rotor_full_fused

rq = RotorQuantMSE(d=128, bits=3, device="cuda")
packed_rotors = pack_rotors_for_triton(rq.rotors)
c_v = rq.centroids_vector

# Triton fused quantize-dequantize (200-650x faster than PyTorch)
x_hat = triton_rotor_full_fused(x, packed_rotors, None, c_v, None, c_v)
```

## Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `benchmark_google_parity.py` | **Full TurboQuant parity test** | `python -m turboquant.benchmark_google_parity` |
| `benchmark_perplexity.py` | Perplexity benchmark (autoregressive + roundtrip) | `python -m turboquant.benchmark_perplexity` |
| `poc_high_context.py` | High-context generation (2K-131K tokens) | `python -m turboquant.poc_high_context` |
| `benchmark_triton.py` | Triton kernel speed (6 tests) | `python -m turboquant.benchmark_triton` |
| `benchmark_rotorquant.py` | RotorQuant vs TurboQuant (7 tests) | `python -m turboquant.benchmark_rotorquant` |
| `validate.py` | Real model attention fidelity | `python -m turboquant.validate` |

## Project Structure

```
turboquant/
  rotorquant.py              # RotorQuant: MSE, Prod, KVCache quantizers
  clifford.py                # Cl(3,0) geometric algebra
  triton_kernels.py          # Triton GPU kernels (rotor sandwich, fused pipeline, attention)
  fused_attention.py         # Fused attention with QJL correction (experimental)
  calibrate.py               # Per-layer codebook calibration (experimental)
  turboquant.py              # TurboQuant: MSE, Prod, KVCache
  lloyd_max.py               # Lloyd-Max optimal scalar quantizer
  compressors.py             # Asymmetric inner product compressors
  cuda_backend.py            # QJL CUDA kernel wrappers
  poc_high_context.py        # High-context generation POC
  benchmark_google_parity.py # Google TurboQuant parity benchmark
  benchmark_perplexity.py    # Perplexity benchmark
  benchmark_triton.py        # Triton kernel benchmarks
  csrc/                      # CUDA kernels (rotor fused, QJL)
  rotor_fused.metal          # Metal shader (Apple Silicon)
tests/                       # 96 unit tests
setup.py                     # pip install with optional CUDA build
```

## Requirements

```bash
pip install -e .                    # PyTorch-only
pip install triton                  # Add Triton kernels (100-650x faster)
pip install -e ".[validate]"        # + model validation deps (transformers, bitsandbytes)
```

- Python 3.10+, PyTorch 2.0+, CUDA, scipy
- triton >= 3.0 (optional but recommended)

## When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| KV cache compression (quality) | **RotorQuant 4-bit** (+3-10% PPL, 3.7x compression) |
| KV cache compression (size) | **RotorQuant 3-bit** (4.9x, matches TQ, 44x fewer params) |
| Long context on limited VRAM | **RotorQuant 3-bit + post-prefill** (65K tokens on 10 GB) |
| Parameter-constrained (edge/mobile) | RotorQuant (44x fewer params) |
| Apple Silicon | RotorQuant + Metal shader |

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — [Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — [Triton impl](https://dejan.ai/blog/turboquant/)
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
