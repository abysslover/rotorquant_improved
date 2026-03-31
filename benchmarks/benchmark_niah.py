import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.turboquant_factory import TurboQuantProdFactory


def run_benchmark(device="cuda"):
    print("=" * 70)
    print("BENCHMARK: Needle-in-Haystack Retrieval")
    print("=" * 70)

    d = 128

    results = []

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)

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

                        compressed = prod(keys)
                        ips = prod.inner_product(query.expand(seq_len, -1), compressed)
                        found = ips.argmax().item() == needle_pos

                        results.append(
                            {
                                "method": method_name,
                                "engine": engine,
                                "bits": bits,
                                "seq_len": seq_len,
                                "found": "EXACT" if found else "MISS",
                            }
                        )
                        print(
                            f"  {method_name} {engine}: bits={bits} seq={seq_len}: {'FOUND' if found else 'MISS'}"
                        )
                    except Exception as e:
                        print(f"  {method_name} {engine}: SKIP ({e})")

    print()
    return results


def save_tsv(results, output_path):
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "engine", "bits", "seq_len", "found"])

        for r in results:
            writer.writerow(
                [r["method"], r["engine"], r["bits"], r["seq_len"], r["found"]]
            )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("Needle-in-Haystack Retrieval Benchmark")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = run_benchmark(device)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "results", "benchmark_niah.tsv")

    save_tsv(results, output_path)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
