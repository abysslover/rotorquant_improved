# RotorQuant Project

## Sanity Check Results (2026-03-30)

### PASSED ✓
- RotorQuant KV cache compression matches Google TurboQuant on every metric
- MSE parity, compression ratio parity, PPL degradation parity
- NIAH (Needle-in-a-Haystack) pass rate matches
- Triton rotor sandwich kernels: 12.7x speedup vs FP32 at 32K context
- Fused attention kernel with QJL estimator working
- IsoQuant (quaternion 4D blocks): 5.8x faster than RotorQuant at identical MSE

## Architecture
- `turboquant/isoquant.py` — **IsoQuantMSE** (recommended): quaternion 4D block rotation, 5.8x faster
- `turboquant/rotorquant.py` — RotorQuantMSE (legacy): Clifford algebra rotor sandwich quantization
- `turboquant/triton_kernels.py` — 5 Triton kernels (rotor forward/inverse, fused pipeline, attention)
- `turboquant/fused_attention.py` — Fused attention with QJL two-term estimator
- `turboquant/benchmark_isoquant.py` — IsoQuant vs RotorQuant head-to-head benchmark
- `turboquant/benchmark_google_parity.py` — TurboQuant parity test suite
- `turboquant/benchmark_perplexity.py` — Autoregressive PPL evaluation

## Default Usage
IsoQuant-Fast is the recommended default. Use `IsoQuantMSE(d, bits, mode='fast')` instead of `RotorQuantMSE`.
RotorQuant (Clifford) is kept for backward compatibility and Triton kernel path.
