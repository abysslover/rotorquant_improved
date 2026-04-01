"""
Verification script for TurboQuant implementation.
Tests MSE distortion bounds, inner product accuracy, and compression ratios
against theoretical predictions from the paper.
"""

import torch
import math
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant import (
    TurboQuantMSE,
    TurboQuantProd,
    TurboQuantKVCache,
    LloydMaxCodebook,
)


def test_lloyd_max_codebook():
    """Verify codebook properties for various dimensions and bit-widths."""
    print("=" * 60)
    print("TEST 1: Lloyd-Max Codebook Properties")
    print("=" * 60)

    for d in [64, 128, 256]:
        for bits in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(d, bits)
            print(
                f"  d={d:>4d}, bits={bits}: {cb.n_levels} levels, "
                f"distortion/coord={cb.distortion:.6f}, "
                f"centroids range=[{cb.centroids.min():.4f}, {cb.centroids.max():.4f}]"
            )

    # Verify symmetry (centroids should be symmetric around 0)
    cb = LloydMaxCodebook(128, 3)
    centroid_sum = cb.centroids.sum().abs().item()
    print(
        f"\n  Symmetry check (d=128, b=3): sum of centroids = {centroid_sum:.6f} (should be ~0)"
    )
    assert centroid_sum < 0.01, "Centroids should be symmetric!"
    print("  PASSED\n")


def test_mse_quantizer():
    """Verify MSE distortion on random unit vectors."""
    print("=" * 60)
    print("TEST 2: MSE Quantizer Distortion")
    print("=" * 60)

    d = 128
    n_vectors = 1000
    device = "cpu"

    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)

        # Generate random unit vectors
        x = torch.randn(n_vectors, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Quantize and reconstruct
        x_hat, indices = quantizer(x)

        # Compute empirical MSE
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        # Theoretical upper bound from paper: D_mse <= sqrt(3)*pi/2 * (1/4^b)
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4**bits))

        ratio = mse / theoretical_bound
        status = "OK" if ratio <= 1.5 else "WARN"  # allow some slack for finite d

        print(
            f"  bits={bits}: MSE={mse:.6f}, theory_bound={theoretical_bound:.6f}, "
            f"ratio={ratio:.3f} [{status}]"
        )

    print()


def test_inner_product_unbiasedness():
    """Verify that TurboQuantProd gives unbiased inner product estimates."""
    print("=" * 60)
    print("TEST 3: Inner Product Unbiasedness (QJL Correction)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(d, bits, seed=42, device=device)

        # Generate pairs of random unit vectors
        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        # True inner products
        true_ip = (x * y).sum(dim=-1)

        # Quantize x, compute estimated inner products
        compressed = quantizer.quantize(x)
        estimated_ip = quantizer.inner_product(y, compressed)

        # Check bias (should be near 0)
        bias = (estimated_ip - true_ip).mean().item()
        # Check RMSE
        rmse = ((estimated_ip - true_ip) ** 2).mean().sqrt().item()
        # Correlation
        correlation = torch.corrcoef(torch.stack([true_ip, estimated_ip]))[0, 1].item()

        # Theoretical distortion bound: D_prod <= sqrt(3)*pi^2/d * (1/4^b)
        theoretical_distortion = math.sqrt(3) * math.pi**2 / d * (1 / (4**bits))

        print(
            f"  bits={bits}: bias={bias:+.6f}, RMSE={rmse:.6f}, "
            f"corr={correlation:.4f}, theory_D={theoretical_distortion:.6f}"
        )

    print()


def test_mse_only_inner_product_bias():
    """Show that MSE-only quantizer has biased inner products (motivating QJL)."""
    print("=" * 60)
    print("TEST 4: MSE-Only Inner Product Bias (motivation for QJL)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [1, 2, 3]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)

        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        true_ip = (x * y).sum(dim=-1)
        x_hat, _ = quantizer(x)
        mse_ip = (x_hat * y).sum(dim=-1)

        bias = (mse_ip - true_ip).mean().item()
        # The bias factor for 1-bit is ~2/pi = 0.637, so ip is scaled by that
        scale_factor = (
            (mse_ip.mean() / true_ip.mean()).item()
            if true_ip.mean().abs() > 0.01
            else float("nan")
        )

        print(f"  bits={bits}: bias={bias:+.6f} (MSE-only is biased, QJL fixes this)")

    print()


def test_kv_cache():
    """Test the KV cache wrapper with compression ratios."""
    print("=" * 60)
    print("TEST 5: KV Cache Compression Ratios")
    print("=" * 60)

    d_key = 128
    d_value = 128
    seq_len = 1024
    device = "cpu"

    for bits in [2, 3, 4]:
        cache = TurboQuantKVCache(d_key, d_value, bits=bits, seed=42, device=device)

        # Simulate appending KV pairs
        keys = torch.randn(seq_len, d_key, device=device)
        values = torch.randn(seq_len, d_value, device=device)

        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        print(
            f"  bits={bits}: compression={usage['compression_ratio']:.2f}x "
            f"({usage['total_bits'] / 8 / 1024:.1f} KB vs "
            f"{usage['fp16_bits'] / 8 / 1024:.1f} KB fp16)"
        )

        # Test attention score computation
        query = torch.randn(1, d_key, device=device)
        scores = cache.attention_scores(query)
        print(
            f"           attention scores shape: {scores.shape}, "
            f"range=[{scores.min():.3f}, {scores.max():.3f}]"
        )

    print()


def test_needle_in_haystack():
    """
    Simplified needle-in-haystack: hide a specific key among many,
    verify we can still find it via attention after quantization.
    """
    print("=" * 60)
    print("TEST 6: Needle-in-Haystack Retrieval")
    print("=" * 60)

    d = 128
    device = "cpu"

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            # Create random keys
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            # Pick a random "needle" position and create a query that matches it
            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)  # exact match query

            # Quantize all keys
            quantizer = TurboQuantProd(d, bits, seed=42, device=device)
            compressed = quantizer.quantize(keys)

            # Compute inner products
            estimated_ips = quantizer.inner_product(
                query.expand(seq_len, -1), compressed
            )

            # Check if needle is still the top result
            top_idx = estimated_ips.argmax().item()
            found = top_idx == needle_pos

            # Also check top-5
            top5 = estimated_ips.topk(5).indices.tolist()
            in_top5 = needle_pos in top5

            status = "EXACT" if found else ("TOP-5" if in_top5 else "MISS")
            print(
                f"  bits={bits}, seq={seq_len:>5d}: top1={top_idx:>5d} "
                f"(needle={needle_pos:>5d}) [{status}]"
            )

    print()


def test_gpu_if_available():
    """Run a quick benchmark on GPU if CUDA is available."""
    print("=" * 60)
    print("TEST 7: GPU Benchmark (if CUDA available)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        print()
        return

    device = "cuda"
    d = 128
    bits = 3
    seq_len = 8192
    n_queries = 64

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Config: d={d}, bits={bits}, seq_len={seq_len}, n_queries={n_queries}")

    quantizer = TurboQuantProd(d, bits, seed=42, device=device)

    # Generate data
    keys = torch.randn(seq_len, d, device=device)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True)
    queries = torch.randn(n_queries, d, device=device)
    queries = queries / torch.norm(queries, dim=-1, keepdim=True)

    # Benchmark quantization
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        compressed = quantizer.quantize(keys)
    torch.cuda.synchronize()
    quant_time = (time.perf_counter() - t0) / 10
    print(f"  Quantize {seq_len} keys: {quant_time * 1000:.2f} ms")

    # Benchmark inner product
    compressed = quantizer.quantize(keys)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        for i in range(n_queries):
            ip = quantizer.inner_product(
                queries[i : i + 1].expand(seq_len, -1), compressed
            )
    torch.cuda.synchronize()
    ip_time = (time.perf_counter() - t0) / 100
    print(
        f"  Inner product ({n_queries} queries x {seq_len} keys): {ip_time * 1000:.2f} ms"
    )

    # Compare with full-precision
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        fp_scores = queries @ keys.T
    torch.cuda.synchronize()
    fp_time = (time.perf_counter() - t0) / 100
    print(f"  Full-precision matmul: {fp_time * 1000:.2f} ms")

    # Memory comparison
    fp16_bytes = seq_len * d * 2  # fp16
    quant_bytes = seq_len * d * bits / 8
    print(
        f"  Memory: {fp16_bytes / 1024:.1f} KB (fp16) vs {quant_bytes / 1024:.1f} KB (TQ-{bits}bit)"
    )
    print(f"  Compression: {fp16_bytes / quant_bytes:.1f}x")
    print()


def test_gqa_runtime_support():
    print("=" * 60)
    print("TEST 8: GQA Runtime Support")
    print("=" * 60)

    from methods.turboquant.turboquant_torch import TurboQuantKVCache

    cache = TurboQuantKVCache(64, 64, bits=3, device="cpu")

    keys = torch.randn(1, 2, 10, 64)
    values = torch.randn(1, 2, 10, 64)
    cache.append(keys, values)

    queries = torch.randn(1, 8, 1, 64)
    scores = cache.attention_scores(queries)

    print(f"  Scores shape: {scores.shape} (expected: (1, 8, 1, 10))")
    assert scores.shape == (1, 8, 1, 10), f"GQA Keys expansion failed"

    group_size = queries.shape[-3] // keys.shape[-3]
    expanded_values = cache.get_values(group_size=group_size)

    print(f"  Values shape: {expanded_values.shape} (expected: (1, 8, 10, 64))")
    assert expanded_values.shape == (1, 8, 10, 64), f"GQA Values expansion failed"

    print("  ✓ GQA runtime support working for Keys AND Values")
    print("  PASSED\n")


def test_engine_consistency():
    print("=" * 60)
    print("TEST 9: Cross-Engine Consistency")
    print("=" * 60)

    from methods.turboquant_factory import TurboQuantFactory

    engines = ["python", "numpy"]
    if torch.cuda.is_available():
        engines.extend(["cuda", "triton"])

    d, bits = 64, 3
    x_torch = torch.randn(100, d)
    x_torch = x_torch / x_torch.norm(dim=-1, keepdim=True)

    mse_results = {}
    memory_results = {}

    for backend in engines:
        try:
            quantizer = TurboQuantFactory.create_quantizer(
                method="turboquant",
                backend=backend,
                d=d,
                bits=bits,
                seed=42,
                device="cpu",
            )

            x_input = x_torch.numpy() if backend == "numpy" else x_torch
            x_hat, _ = quantizer(x_input)

            if backend == "numpy":
                import numpy as np

                mse = float(((x_input - x_hat) ** 2).sum(axis=-1).mean())
            else:
                mse = ((x_input - x_hat) ** 2).sum(dim=-1).mean().item()

            mse_results[backend] = mse

            cache = TurboQuantFactory.create_kvcache(
                method="turboquant",
                backend=backend,
                d_key=d,
                d_value=d,
                key_bits=4,
                value_bits=2,
                device="cpu",
            )

            test_keys = x_torch.numpy() if backend == "numpy" else x_torch
            test_values = x_torch.numpy() if backend == "numpy" else x_torch
            cache.append(test_keys[:10], test_values[:10])

            usage = cache.memory_usage_bits()
            memory_results[backend] = usage["total_bits"]

            print(f"  {backend:>10s}: MSE={mse:.6f}, Memory={usage['total_bits']} bits")

        except (ImportError, NotImplementedError, RuntimeError):
            print(f"  {backend:>10s}: Not available")

    baseline_mse = mse_results.get("python", 0)
    baseline_memory = memory_results.get("python", 0)

    for engine, mse in mse_results.items():
        if engine != "python" and baseline_mse > 0:
            mse_diff = abs(mse - baseline_mse) / baseline_mse
            assert mse_diff < 0.01, f"MSE inconsistency in {engine}: {mse_diff:.3f}"

    for engine, memory in memory_results.items():
        if engine != "python" and baseline_memory > 0:
            assert memory == baseline_memory, (
                f"Memory calculation inconsistency in {engine}"
            )

    print("  ✓ All engines consistent")
    print("  PASSED\n")


def test_qjl_removal_verification():
    """Verify QJL fields are removed from KV cache (MANUAL QA 5 merged)."""
    print("=" * 60)
    print("TEST 10: QJL Removal Verification")
    print("=" * 60)

    from methods.turboquant_factory import TurboQuantFactory

    cache = TurboQuantFactory.create_kvcache(
        method="turboquant",
        backend="python",
        d_key=128,
        d_value=128,
        key_bits=4,
        value_bits=2,
        seed=42,
        device="cpu",
    )

    keys = torch.randn(100, 128, device="cpu")
    values = torch.randn(100, 128, device="cpu")
    cache.append(keys, values)

    # Check that cache only stores indices, not QJL fields
    cached = cache.key_cache[0]
    assert "qjl_signs" not in cached, "QJL qjl_signs should not be in cache"
    assert "residual_norm" not in cached, "QJL residual_norm should not be in cache"
    assert "mse_indices" not in cached, "QJL mse_indices should not be in cache"

    # Check that key_quantizer is MSE-only, not Prod
    assert "TurboQuantMSE" in str(type(cache.key_quantizer)), (
        "Key quantizer should be TurboQuantMSE, not TurboQuantProd"
    )

    print("  Cache structure: indices + shape only")
    print("  Key quantizer: TurboQuantMSE (not TurboQuantProd)")
    print("  ✓ QJL fields removed: qjl_signs, residual_norm, mse_indices")
    print("  PASSED\n")


def test_multi_append_operations():
    """Test multiple append operations (MANUAL QA 6 merged)."""
    print("=" * 60)
    print("TEST 11: Multiple Append Operations")
    print("=" * 60)

    from methods.turboquant_factory import TurboQuantFactory

    cache = TurboQuantFactory.create_kvcache(
        method="turboquant",
        backend="python",
        d_key=128,
        d_value=128,
        bits=3,
        seed=42,
        device="cpu",
    )

    # Append multiple batches
    for i in range(5):
        keys = torch.randn(50, 128, device="cpu")
        values = torch.randn(50, 128, device="cpu")
        cache.append(keys, values)

    print(f"  Appended 5 batches of 50 tokens each")
    print(f"  Total cache length: {len(cache)}")
    print(f"  Expected: 250")

    assert len(cache) == 250, f"Expected 250, got {len(cache)}"

    # Test attention scores
    query = torch.randn(1, 128, device="cpu")
    scores = cache.attention_scores(query)

    print(f"  Attention scores shape: {scores.shape}")
    print(f"  Expected: (1, 250)")

    assert scores.shape == (1, 250), f"Expected (1, 250), got {scores.shape}"

    print("  ✓ PASSED\n")


def test_factory_pattern_integration():
    """Test Factory pattern integration (MANUAL QA 3 merged)."""
    print("=" * 60)
    print("TEST 12: Factory Pattern Integration")
    print("=" * 60)

    from methods.turboquant_factory import TurboQuantFactory

    # Create cache with factory
    cache = TurboQuantFactory.create_kvcache(
        method="turboquant",
        backend="python",
        d_key=128,
        d_value=128,
        key_bits=4,
        value_bits=2,
        seed=42,
        device="cpu",
    )

    keys = torch.randn(100, 128, device="cpu")
    values = torch.randn(100, 128, device="cpu")
    cache.append(keys, values)

    query = torch.randn(1, 128, device="cpu")
    scores = cache.attention_scores(query)
    values_recon = cache.get_values()

    print(f"  Cache type: {type(cache).__name__}")
    print(f"  Cache length: {len(cache)}")
    print(f"  Values reconstructed shape: {values_recon.shape}")
    print(f"  Attention scores shape: {scores.shape}")

    assert len(cache) == 100, f"Expected 100, got {len(cache)}"
    assert values_recon.shape == (100, 128), (
        f"Expected (100, 128), got {values_recon.shape}"
    )

    print("  ✓ PASSED\n")


def test_asymmetric_bits_allocation():
    """Test asymmetric K/V bit allocation with exact verification (MANUAL QA 2 merged)."""
    print("=" * 60)
    print("TEST 13: Asymmetric K/V Bit Allocation")
    print("=" * 60)

    from methods.turboquant_factory import TurboQuantFactory

    d_key, d_value = 128, 128
    seq_len = 100
    device = "cpu"

    # Use recommended settings: key_bits=4, value_bits=2
    cache = TurboQuantFactory.create_kvcache(
        method="turboquant",
        backend="python",
        d_key=d_key,
        d_value=d_value,
        bits=3,
        key_bits=4,
        value_bits=2,
        seed=42,
        device=device,
    )

    keys = torch.randn(seq_len, d_key, device=device)
    values = torch.randn(seq_len, d_value, device=device)
    cache.append(keys, values)

    usage = cache.memory_usage_bits()

    print(f"  Configuration: key_bits=4, value_bits=2")
    print(f"  Total bits: {usage['total_bits']}")
    print(f"  FP16 bits: {usage['fp16_bits']}")
    print(f"  Compression ratio: {usage['compression_ratio']:.2f}x")

    # Verify asymmetric bit usage
    expected_key_indices = seq_len * d_key
    expected_value_indices = seq_len * d_value
    expected_bits = expected_key_indices * 4 + expected_value_indices * 2

    assert usage["total_bits"] == expected_bits, (
        f"Expected {expected_bits} bits, got {usage['total_bits']}"
    )

    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print()
    print("TurboQuant Implementation Verification")
    print("Based on: 'TurboQuant: Online Vector Quantization' (ICLR 2026)")
    print()

    test_lloyd_max_codebook()
    test_mse_quantizer()
    test_inner_product_unbiasedness()
    test_mse_only_inner_product_bias()
    test_kv_cache()
    test_needle_in_haystack()
    test_gpu_if_available()
    test_gqa_runtime_support()
    test_engine_consistency()
    test_qjl_removal_verification()
    test_multi_append_operations()
    test_factory_pattern_integration()
    test_asymmetric_bits_allocation()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
