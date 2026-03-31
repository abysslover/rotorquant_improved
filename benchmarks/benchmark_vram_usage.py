import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: VRAM Usage")
    print("=" * 70)

    d = 128
    bits = 3
    n_kv_heads = 8

    results = []

    for seq_len in [1024, 4096, 16384, 32768]:
        fp16_bytes = n_kv_heads * seq_len * d * 2

        for method_name in ["planarquant", "isoquant", "rotorquant", "turboquant"]:
            for engine in ["torch_cuda"]:
                try:
                    dev = device if engine == "torch_cuda" else "cpu"
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

                    d_padded = getattr(mse, "d_padded", d)

                    idx_bytes = n_kv_heads * seq_len * d_padded * 1
                    norm_bytes = n_kv_heads * seq_len * 4
                    quant_bytes = idx_bytes + norm_bytes

                    ratio = fp16_bytes / quant_bytes

                    results.append(
                        {
                            "method": method_name,
                            "engine": engine,
                            "seq_len": seq_len,
                            "fp16_mb": fp16_bytes / 1024 / 1024,
                            "quant_mb": quant_bytes / 1024 / 1024,
                            "ratio": ratio,
                        }
                    )
                    print(
                        f"  {method_name} {engine}: seq={seq_len}: {fp16_bytes / 1024 / 1024:.2f}MB -> {quant_bytes / 1024 / 1024:.2f}MB ({ratio:.1f}x)"
                    )
                except Exception as e:
                    print(f"  {method_name} {engine}: SKIP ({e})")

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "seq_len", "fp16_mb", "quant_mb", "ratio"])

        for r in results:
            writer.writerow(
                [
                    r["method"],
                    r["engine"],
                    r["seq_len"],
                    f"{r['fp16_mb']:.2f}",
                    f"{r['quant_mb']:.2f}",
                    f"{r['ratio']:.2f}",
                ]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("VRAM Usage Benchmark")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_vram_usage.tsv")

    save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
