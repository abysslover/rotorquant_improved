import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def format_time(ms):
    if ms < 1:
        return f"{ms * 1000:.1f} us"
    return f"{ms:.2f} ms"


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Speed Comparison")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = 128
    bits = 3

    n_warmup = 5
    n_iter = 50

    print(f"  d={d}, bits={bits}, device={device}\n")
    print(f"  {'method':>12s}  {'engine':>10s}  {'n':>8s}  {'time':>12s}")
    print("  " + "-" * 54)

    results = []

    for n in [1000, 5000, 10000]:
        for method_name in ["planarquant", "isoquant", "rotorquant", "turboquant"]:
            for engine in ["cpu", "torch_cuda", "cuda_kernel"]:
                try:
                    dev = device if engine in ["torch_cuda", "cuda_kernel"] else "cpu"

                    if method_name == "turboquant" and engine == "cpu":
                        continue

                    mse = TurboQuantProdFactory.create_quantizer(
                        method=method_name,
                        engine=engine,
                        d=d,
                        bits=bits,
                        seed=42,
                        device=dev,
                    )

                    x = torch.randn(n, d, device=dev)

                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()

                    for _ in range(n_warmup):
                        _ = mse(x)

                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()

                    t0 = time.perf_counter()
                    for _ in range(n_iter):
                        _ = mse(x)

                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()

                    ms = (time.perf_counter() - t0) / n_iter * 1000

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "n": n,
                            "time_ms": ms,
                        }
                    )

                    print(
                        f"  {method_name:>12s}  {engine:>10s}  {n:>8d}  {format_time(ms):>12s}"
                    )
                except Exception as e:
                    pass

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "n", "time_ms"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["n"],
                    f"{r['time_ms']:.4f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    results = run_benchmark()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_speed.tsv")

    save_tsv(results, output_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
