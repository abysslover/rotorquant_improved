import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: Rotation Equivariance")
    print("=" * 70)

    d = 128
    n = 1000

    G = torch.randn(d, d, device=device)
    R, _ = torch.linalg.qr(G)

    x = torch.randn(n, d, device=device)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    x_rot = x @ R.T

    results = []

    for bits in [2, 3, 4]:
        for method_name in ["planarquant", "isoquant", "rotorquant", "turboquant"]:
            for engine in ["pytorch"]:
                try:
                    dev = device if engine == "pytorch" else "cpu"
                    if dev == "cpu" and method_name == "turboquant":
                        continue

                    mse = TurboQuantProdFactory.create_quantizer(
                        method=method_name,
                        engine=engine,
                        d=d,
                        bits=bits,
                        seed=42,
                        device=dev,
                    )

                    qx, _ = mse(x)
                    qrx, _ = mse(x_rot)
                    rqx = qx @ R.T

                    equiv_err = ((qrx - rqx) ** 2).sum(dim=-1).mean().sqrt().item()
                    cos = F.cosine_similarity(qrx, rqx, dim=-1).mean().item()

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "bits": bits,
                            "equiv_err": equiv_err,
                            "cosine": cos,
                        }
                    )
                    print(
                        f"  {method_name} {engine} bits={bits}: err={equiv_err:.6f} cos={cos:.6f}"
                    )
                except Exception as e:
                    print(f"  {method_name} {engine}: SKIP ({e})")

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "bits", "equiv_err", "cosine"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["bits"],
                    f"{r['equiv_err']:.6f}",
                    f"{r['cosine']:.6f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("Rotation Equivariance Benchmark")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_equivariance.tsv")

    save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
