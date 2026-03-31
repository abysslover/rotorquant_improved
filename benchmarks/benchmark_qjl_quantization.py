"""
Benchmark: QJL Quantization Speed

Compares QJL quantization across all methods and engines.
Outputs: benchmarks/results/benchmark_qjl_quantization.tsv
"""

import torch
import time
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def format_time(ms):
    if ms < 1:
        return f"{ms * 1000:.1f} us"
    return f"{ms:.2f} ms"


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: QJL Quantization")
    print("=" * 70)

    n_warmup = 5
    n_iter = 50
    results = []

    for bits in [2, 3, 4]:
        for seq_len in [1024, 4096, 8192]:
            d = 128
            n = seq_len

            x = torch.randn(n, d, device=device)
            x = x / torch.norm(x, dim=-1, keepdim=True)

            for method_name in ["planarquant", "isoquant", "rotorquant", "turboquant"]:
                for engine in ["cpu", "torch_cuda"]:
                    if engine == "cpu":
                        dev = "cpu"
                    else:
                        dev = device if device == "cuda" else "cpu"

                    if dev == "cpu" and method_name in ["turboquant"]:
                        continue

                    try:
                        prod = TurboQuantProdFactory.create(
                            method=method_name,
                            engine=engine,
                            d=d,
                            bits=bits,
                            seed=42,
                            device=dev,
                        )

                        if dev.startswith("cuda"):
                            torch.cuda.synchronize()

                        for _ in range(n_warmup):
                            _ = prod(x)

                        if dev.startswith("cuda"):
                            torch.cuda.synchronize()

                        t0 = time.perf_counter()
                        for _ in range(n_iter):
                            _ = prod(x)

                        if dev.startswith("cuda"):
                            torch.cuda.synchronize()

                        ms = (time.perf_counter() - t0) / n_iter * 1000

                        results.append(
                            {
                                "method": method_name,
                                "engine": engine,
                                "bits": bits,
                                "seq_len": seq_len,
                                "time_ms": ms,
                            }
                        )

                        print(
                            f"  {method_name} {engine}: bits={bits} seq={seq_len}: {format_time(ms)}"
                        )
                    except Exception as e:
                        print(f"  {method_name} {engine}: SKIP ({e})")

    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "bits", "seq_len", "time_ms"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["bits"],
                    r["seq_len"],
                    f"{r['time_ms']:.4f}",
                ]
            )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_qjl_quantization.tsv")

    save_tsv(results, output_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
