import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Inner Product Preservation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = 128
    n = 1000

    print(f"  d={d}, n={n}, device={device}\n")
    print(
        f"  {'method':>12s}  {'engine':>10s}  {'bits':>5s}  {'ip_mse':>10s}  {'bias':>10s}  {'corr':>10s}"
    )
    print("  " + "-" * 67)

    results = []

    for bits in [2, 3, 4]:
        for method_name in ["planarquant", "isoquant", "rotorquant", "turboquant"]:
            for engine in ["cpu", "torch_cuda"]:
                try:
                    dev = device if engine == "torch_cuda" else "cpu"

                    if method_name == "turboquant" and engine == "cpu":
                        continue

                    prod = TurboQuantProdFactory.create(
                        method=method_name,
                        engine=engine,
                        d=d,
                        bits=bits,
                        seed=42,
                        device=dev,
                    )

                    x = torch.randn(n, d, device=dev)
                    y = torch.randn(n, d, device=dev)

                    ip_true = (x * y).sum(dim=-1)

                    compressed = prod(x)
                    ip_est = prod.inner_product(y, compressed)

                    ip_mse = ((ip_true - ip_est) ** 2).mean().item()
                    bias = (ip_est - ip_true).mean().item()

                    ip_true_mean = ip_true.mean()
                    ip_est_mean = ip_est.mean()
                    cov = ((ip_true - ip_true_mean) * (ip_est - ip_est_mean)).mean()
                    std_x = ((ip_true - ip_true_mean) ** 2).mean().sqrt()
                    std_y = ((ip_est - ip_est_mean) ** 2).mean().sqrt()
                    corr = cov / (std_x * std_y + 1e-8)

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "bits": bits,
                            "ip_mse": ip_mse,
                            "bias": bias,
                            "correlation": corr,
                        }
                    )

                    print(
                        f"  {method_name:>12s}  {engine:>10s}  {bits:>5d}  {ip_mse:>10.6f}  {bias:>10.6f}  {corr:>10.6f}"
                    )
                except Exception as e:
                    pass

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "bits", "ip_mse", "bias", "correlation"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["bits"],
                    f"{r['ip_mse']:.6f}",
                    f"{r['bias']:.6f}",
                    f"{r['correlation']:.6f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    results = run_benchmark()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_inner_product.tsv")

    save_tsv(results, output_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
