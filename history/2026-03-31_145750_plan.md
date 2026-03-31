# Engine Implementation Verification & Benchmark Reorganization Plan

## Goal
Verify that each method-engine implementation (cpu/pytorch/triton) matches the original turboquant kernel implementations, and restructure benchmarks to output one TSV per function with method-engine comparisons.

## Execution Order (reverse of repo history)
1. TurboQuant → 2. RotorQuant → 3. IsoQuant → 4. PlanarQuant

---

## Step 1: Analyze Original Kernels

### TurboQuant (methods/turboquant/)
| File | Kernel | Purpose |
|------|--------|---------|
| turboquant.py | generate_rotation_matrix | Random orthogonal matrix via QR |
| turboquant.py | generate_qjl_matrix | QJL projection matrix |
| lloyd_max.py | LloydMaxCodebook | Per-coordinate scalar quantizer |
| triton_kernels.py | triton_rotor_* | Clifford rotor kernels |
| triton_isoquant.py | triton_iso_* | IsoQuant Triton kernels |
| triton_planarquant.py | triton_planar2_* | PlanarQuant Triton kernels |
| cuda_backend.py | QJLSketch | CUDA QJL kernels |

### PlanarQuant (methods/planarquant/)
| File | Engine | Implementation |
|------|--------|----------------|
| planarquant_cpu.py | CPU | Pure Python/PyTorch |
| planarquant_pytorch.py | PyTorch | CUDA-optimized PyTorch |
| planarquant_triton.py | Triton | triton_planar2_fused kernel |

### IsoQuant (methods/isoquant/)
| File | Engine | Implementation |
|------|--------|----------------|
| isoquant_cpu.py | CPU | Pure quaternion operations |
| isoquant_triton.py | Triton | triton_iso_fast_fused kernel |

### RotorQuant (methods/rotorquant/)
| File | Engine | Implementation |
|------|--------|----------------|
| rotorquant_cpu.py | CPU | Clifford algebra operations |
| clifford.py | - | Geometric algebra primitives |

---

## Step 2: Benchmark Reorganization

### Current benchmark_cuda.py has 4 functions:
1. `benchmark_qjl_quantize` → benchmark_qjl_quantization.tsv
2. `benchmark_attention_scores` → benchmark_attention_scores.tsv
3. `benchmark_e2e_pipeline` → benchmark_e2e_pipeline.tsv
4. `benchmark_accuracy_comparison` → benchmark_accuracy_comparison.tsv

### Current benchmark_rotorquant.py has functions:
1. `test_mse_distortion` → benchmark_mse_distortion.tsv
2. `test_inner_product` → benchmark_inner_product.tsv
3. `test_niah` → benchmark_niah.tsv
4. `test_equivariance` → benchmark_equivariance.tsv
5. `test_vram_usage` → benchmark_vram_usage.tsv
6. `test_speed` → benchmark_speed.tsv
7. `test_parameter_efficiency` → benchmark_parameter_efficiency.tsv

### Target: Method-Engine Comparison per Function
Each TSV should have columns:
- method (planarquant/isoquant/rotorquant/turboquant)
- engine (cpu/pytorch/triton)
- ... (function-specific metrics)

---

## Step 3: Verification Tests

For each method-engine pair, verify:
1. MSE reconstruction matches expected range
2. Inner product correlation > 0.95
3. Quantize/dequantize roundtrip works
4. Prod version inner_product works

---

## Current Status

- [x] Analyze original kernel implementations
- [x] Verify PlanarQuant engine implementations
- [x] Verify IsoQuant engine implementations
- [x] Verify RotorQuant engine implementations
- [x] Verify TurboQuant engine implementations
- [x] Reorganize benchmark_qjl_quantization.py → benchmark_qjl_quantization.tsv
- [x] Reorganize benchmark_rotorquant.py → benchmark_mse_distortion.tsv
- [x] Reorganize benchmark_rotorquant.py → benchmark_inner_product.tsv
- [x] Reorganize benchmark_rotorquant.py → benchmark_speed.tsv
- [x] Create benchmark_parameter_efficiency.py → benchmark_parameter_efficiency.tsv
- [x] Run all benchmarks and regenerate TSV files

## Generated Benchmark Files

| TSV | Description | Columns |
|-----|-------------|---------|
| benchmark_mse_distortion.tsv | MSE reconstruction error | method, engine, bits, mse, theory, ratio |
| benchmark_inner_product.tsv | Inner product preservation | method, engine, bits, ip_mse, bias, correlation |
| benchmark_qjl_quantization.tsv | QJL quantization speed | method, engine, bits, seq_len, time_ms |
| benchmark_speed.tsv | Quantization speed | method, engine, n, time_ms |
| benchmark_parameter_efficiency.tsv | Parameter count | method, engine, n_params, ratio |

## Sample Results Summary

### MSE Distortion (bits=3, d=128)
| Method | Engine | MSE | Ratio to Theory |
|--------|--------|-----|-----------------|
| planarquant | cpu | 0.0339 | 0.80x |
| isoquant | cpu | 0.0340 | 0.80x |
| rotorquant | cpu | 0.0339 | 0.80x |
| turboquant | pytorch | 0.0340 | 0.80x |

### Speed (n=1000, bits=3, d=128, CUDA)
| Method | Engine | Time |
|--------|--------|------|
| turboquant | pytorch | 76.1 us |
| planarquant | pytorch | 248.7 us |
| isoquant | pytorch | 888.3 us |
| rotorquant | pytorch | 2.92 ms |

### Parameter Efficiency (d=128, bits=3)
| Method | Engine | Params | Ratio to TQ |
|--------|--------|--------|------------|
| planarquant | pytorch | 136 | 120.5x smaller |
| isoquant | pytorch | 264 | 62.1x smaller |
| rotorquant | pytorch | 352 | 46.6x smaller |
| turboquant | pytorch | 16399 | baseline |
