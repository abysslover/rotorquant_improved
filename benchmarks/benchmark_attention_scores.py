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


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: Attention Score Computation")
    print("=" * 70)

    d = 128
    bits = 3
    n_warmup = 5
    n_iter = 50

    results = []

    for seq_len in [1024, 4096, 8192]:
        batch, heads = 1, 8

        keys = torch.randn(batch, heads, seq_len, d, device=device, dtype=torch.float16)
        query = torch.randn(batch, heads, 1, d, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        for _ in range(n_warmup):
            _ = torch.matmul(query, keys.transpose(-2, -1))
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = torch.matmul(query, keys.transpose(-2, -1))
        torch.cuda.synchronize()
        fp_ms = (time.perf_counter() - t0) / n_iter * 1000

        results.append(
            {
                "method": "fp16",
                "engine": "cuda",
                "seq_len": seq_len,
                "time_ms": fp_ms,
            }
        )
        print(f"  fp16: seq={seq_len:>6d}: {format_time(fp_ms):>10s}")

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

                    x = keys.reshape(-1, d).float()
                    y = (
                        query.squeeze(2)
                        .unsqueeze(2)
                        .expand(-1, -1, seq_len, -1)
                        .reshape(batch * heads * seq_len, d)
                        .float()
                    )

                    x_normalized = x / torch.norm(x, dim=-1, keepdim=True).clamp(
                        min=1e-8
                    )
                    y_normalized = y / torch.norm(y, dim=-1, keepdim=True).clamp(
                        min=1e-8
                    )

                    x_expanded = x_normalized
                    y_expanded = y_normalized

                    compressed = prod(x_expanded)

                    for _ in range(n_warmup):
                        _ = prod.inner_product(y_expanded, compressed)
                    torch.cuda.synchronize()

                    t0 = time.perf_counter()
                    for _ in range(n_iter):
                        _ = prod.inner_product(y_expanded, compressed)
                    torch.cuda.synchronize()

                    ms = (time.perf_counter() - t0) / n_iter * 1000

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "seq_len": seq_len,
                            "time_ms": ms,
                        }
                    )
                    print(
                        f"  {method_name} {engine}: seq={seq_len:>6d}: {format_time(ms):>10s}"
                    )
                except Exception as e:
                    print(f"  {method_name} {engine}: SKIP ({e})")

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "seq_len", "time_ms"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["seq_len"],
                    f"{r['time_ms']:.4f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("CUDA not available. Exiting.")
        sys.exit(1)

    print()
    print("Attention Score Computation Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_attention_scores.tsv")

    save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
