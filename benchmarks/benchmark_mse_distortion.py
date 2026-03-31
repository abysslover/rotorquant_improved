import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.planarquant import PlanarQuantMSE
from methods.isoquant import IsoQuantMSE
from methods.rotorquant import RotorQuantMSE
from methods.turboquant import TurboQuantMSE


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: MSE Distortion")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = 128
    n = 2000

    print(f"  d={d}, n_vectors={n}, device={device}\n")
    print(
        f"  {'method':>12s}  {'engine':>8s}  {'bits':>5s}  {'mse':>12s}  {'theory':>12s}  {'ratio':>10s}"
    )
    print("  " + "-" * 75)

    results = []

    for bits in [1, 2, 3, 4]:
        theory = math.sqrt(3) * math.pi / 2 * (1 / (4**bits))

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

                    q = MSEClass(d=d, bits=bits, seed=42, device=dev)

                    x = torch.randn(n, d, device=dev)
                    x = x / torch.norm(x, dim=-1, keepdim=True)

                    x_hat, _ = q(x)
                    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
                    ratio = mse / theory

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "bits": bits,
                            "mse": mse,
                            "theory": theory,
                            "ratio": ratio,
                        }
                    )

                    print(
                        f"  {method_name:>12s}  {engine:>8s}  {bits:>5d}  {mse:>12.6f}  {theory:>12.6f}  {ratio:>10.3f}"
                    )
                except Exception as e:
                    pass

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "bits", "mse", "theory", "ratio"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["bits"],
                    f"{r['mse']:.6f}",
                    f"{r['theory']:.6f}",
                    f"{r['ratio']:.4f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    results = run_benchmark()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_mse_distortion.tsv")

    save_tsv(results, output_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
