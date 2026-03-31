import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: Structured Data (Low-rank + Directional)")
    print("=" * 70)

    d = 128
    n = 2000
    rank = 16

    basis = torch.randn(rank, d, device=device)
    basis = basis / torch.norm(basis, dim=-1, keepdim=True)
    coeffs = torch.randn(n, rank, device=device)
    x = coeffs @ basis
    x = x / torch.norm(x, dim=-1, keepdim=True)

    y = torch.randn(n, d, device=device)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    true_ip = (x * y).sum(dim=-1)

    results = []

    for bits in [2, 3, 4]:
        for method_name in ["planarquant", "isoquant", "rotorquant", "turboquant"]:
            for engine in ["torch_cuda"]:
                try:
                    dev = device if engine == "torch_cuda" else "cpu"
                    if dev == "cpu" and method_name == "turboquant":
                        continue

                    prod = TurboQuantProdFactory.create(
                        method=method_name,
                        engine=engine,
                        d=d,
                        bits=bits,
                        seed=42,
                        device=dev,
                    )

                    compressed = prod(x)
                    x_hat = prod.dequantize(compressed)

                    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
                    est_ip = prod.inner_product(y, compressed)
                    corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()
                    rmse = ((est_ip - true_ip) ** 2).mean().sqrt().item()

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "bits": bits,
                            "mse": mse,
                            "ip_corr": corr,
                            "ip_rmse": rmse,
                        }
                    )
                    print(
                        f"  {method_name} {engine} bits={bits}: mse={mse:.6f} corr={corr:.4f} rmse={rmse:.6f}"
                    )
                except Exception as e:
                    print(f"  {method_name} {engine}: SKIP ({e})")

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "bits", "mse", "ip_corr", "ip_rmse"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["bits"],
                    f"{r['mse']:.6f}",
                    f"{r['ip_corr']:.4f}",
                    f"{r['ip_rmse']:.6f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("Structured Data Benchmark")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_structured_data.tsv")

    save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
