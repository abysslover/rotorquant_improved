import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.planarquant import PlanarQuantProd
from methods.isoquant import IsoQuantProd
from methods.rotorquant import RotorQuantProd
from methods.turboquant import TurboQuantProd


def format_time(ms):
    if ms < 1:
        return f"{ms * 1000:.1f} us"
    return f"{ms:.2f} ms"


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: End-to-End Pipeline")
    print("=" * 70)

    d = 128
    n_warmup = 3
    n_iter = 20

    results = []

    for seq_len in [2048, 8192]:
        for bits in [2, 3, 4]:
            batch, heads = 1, 32

            keys = torch.randn(batch * heads, seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            for method_name, ProdClass in [
                ("planarquant", PlanarQuantProd),
                ("isoquant", IsoQuantProd),
                ("rotorquant", RotorQuantProd),
                ("turboquant", TurboQuantProd),
            ]:
                for engine in ["pytorch"]:
                    try:
                        dev = device if engine == "pytorch" else "cpu"
                        if dev == "cpu" and method_name == "turboquant":
                            continue

                        prod = ProdClass(d=d, bits=bits, seed=42, device=dev)

                        torch.cuda.synchronize()
                        for _ in range(n_warmup):
                            _ = prod(keys)
                        torch.cuda.synchronize()

                        t0 = time.perf_counter()
                        for _ in range(n_iter):
                            _ = prod(keys)
                        torch.cuda.synchronize()
                        ms = (time.perf_counter() - t0) / n_iter * 1000

                        fp_bytes = keys.numel() * 4
                        compressed = prod(keys)

                        if isinstance(compressed["mse_indices"], dict):
                            mse_indices = compressed["mse_indices"]["indices"]
                        else:
                            mse_indices = compressed["mse_indices"]
                        qjl_signs = compressed["qjl_signs"]

                        compressed_bytes = (
                            mse_indices.numel() * bits + qjl_signs.numel()
                        ) / 8
                        ratio = fp_bytes / compressed_bytes

                        results.append(
                            {
                                "method": method_name,
                                "engine": engine,
                                "seq_len": seq_len,
                                "bits": bits,
                                "time_ms": ms,
                                "compression_ratio": ratio,
                            }
                        )
                        print(
                            f"  {method_name} {engine}: seq={seq_len}, bits={bits}: {format_time(ms):>10s} ({ratio:.1f}x)"
                        )
                    except Exception as e:
                        print(f"  {method_name} {engine}: SKIP ({e})")

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "engine", "seq_len", "bits", "time_ms", "compression_ratio"]
        )

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["seq_len"],
                    r["bits"],
                    f"{r['time_ms']:.4f}",
                    f"{r['compression_ratio']:.2f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("CUDA not available. Exiting.")
        sys.exit(1)

    print()
    print("End-to-End Pipeline Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_e2e_pipeline.tsv")

    save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
