import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.planarquant import PlanarQuantMSE
from methods.isoquant import IsoQuantMSE
from methods.rotorquant import RotorQuantMSE
from methods.turboquant import TurboQuantMSE


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Parameter Efficiency")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = 128

    print(f"  d={d}, device={device}\n")
    print(f"  {'method':>12s}  {'engine':>8s}  {'params':>10s}  {'ratio':>10s}")
    print("  " + "-" * 50)

    results = []

    for method_name, MSEClass in [
        ("planarquant", PlanarQuantMSE),
        ("isoquant", IsoQuantMSE),
        ("rotorquant", RotorQuantMSE),
        ("turboquant", TurboQuantMSE),
    ]:
        for engine in ["cpu", "pytorch"]:
            try:
                dev = device if engine == "pytorch" else "cpu"

                if dev == "cpu" and method_name == "turboquant":
                    continue

                q = MSEClass(d=d, bits=3, seed=42, device=dev)

                n_params = sum(p.numel() for p in q.parameters()) + sum(
                    b.numel() for b in q.buffers()
                )

                turbo_params = d * d + 8
                ratio = turbo_params / n_params if n_params > 0 else 0

                results.append(
                    {
                        "method": method_name,
                        "engine": engine,
                        "n_params": n_params,
                        "ratio": ratio,
                    }
                )

                print(
                    f"  {method_name:>12s}  {engine:>8s}  {n_params:>10d}  {ratio:>10.1f}x"
                )
            except Exception as e:
                pass

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "n_params", "ratio"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["n_params"],
                    f"{r['ratio']:.2f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    results = run_benchmark()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        output_dir, "results", "benchmark_parameter_efficiency.tsv"
    )

    save_tsv(results, output_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
