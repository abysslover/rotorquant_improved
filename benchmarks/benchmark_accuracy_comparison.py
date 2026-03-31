import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.planarquant import PlanarQuantProd
from methods.isoquant import IsoQuantProd
from methods.rotorquant import RotorQuantProd
from methods.turboquant import TurboQuantProd


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: Accuracy Comparison (Inner Product Fidelity)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return []

    d = 128
    bits = 3
    seq_len = 2048
    batch, heads = 1, 8

    keys = torch.randn(batch, heads, seq_len, d, device=device, dtype=torch.float32)
    query = torch.randn(batch, heads, 1, d, device=device, dtype=torch.float32)

    scores_fp = torch.matmul(query, keys.transpose(-2, -1)).squeeze(-2)

    results = []

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

                x = keys.reshape(-1, d)
                y = (
                    query.squeeze(2)
                    .unsqueeze(2)
                    .expand(-1, -1, seq_len, -1)
                    .reshape(batch * heads * seq_len, d)
                )

                x_normalized = x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
                y_normalized = y / torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-8)

                compressed = prod(x_normalized)
                scores_est = prod.inner_product(y_normalized, compressed)
                scores_est = scores_est.reshape(batch, heads, seq_len)

                cos_sims = []
                top1_matches = 0
                n_checks = 0
                for b in range(batch):
                    for h in range(heads):
                        fp = scores_fp[b, h]
                        est = scores_est[b, h]
                        if est.shape[0] > 0:
                            cos = F.cosine_similarity(
                                fp.unsqueeze(0).float(), est.unsqueeze(0).float()
                            ).item()
                            cos_sims.append(cos)
                            if fp.argmax() == est.argmax():
                                top1_matches += 1
                            n_checks += 1

                avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0
                top1_pct = 100 * top1_matches / n_checks if n_checks > 0 else 0

                results.append(
                    {
                        "method": method_name,
                        "engine": engine,
                        "bits": bits,
                        "cosine_similarity": avg_cos,
                        "top1_match_pct": top1_pct,
                    }
                )
                print(
                    f"  {method_name} {engine}: cos={avg_cos:.6f}, top1={top1_pct:.1f}%"
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
            ["method", "engine", "bits", "cosine_similarity", "top1_match_pct"]
        )

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["bits"],
                    f"{r['cosine_similarity']:.6f}",
                    f"{r['top1_match_pct']:.2f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("Accuracy Comparison Benchmark")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    if results:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(
            output_dir, "results", "benchmark_accuracy_comparison.tsv"
        )

        save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
