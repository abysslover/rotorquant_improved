"""
IsoQuant High-Context POC

Patches Qwen2.5 models with IsoQuant quaternion 4D rotation KV cache compression
and tests needle-in-haystack retrieval + generation at increasing context.

Measures: VRAM, prefill tok/s, decode tok/s, attention fidelity, generation quality.

Usage:
    python -m methods.isoquant.isoquant_poc_high_context
    python -m methods.isoquant.isoquant_poc_high_context --bits 3 --max-ctx 65536

Based on: https://github.com/ParaMind2025/isoquant
         https://arxiv.org/abs/2504.19874 (TurboQuant, ICLR 2026)

Note: IsoQuant uses quaternion 4D block rotation for better hardware alignment.
"""

import torch
import time
import gc
import argparse
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from methods.isoquant.isoquant import IsoQuantMSE
from methods.common.high_context import (
    build_prompt,
    measure_attention_fidelity,
    test_generation,
    test_generation_fp16,
    patch_model_kv_cache,
    unpatch_model_kv_cache,
    PatchedCache,
)


# ── IsoQuant Key Compressor ────────────────────────────────────────────


class IsoQuantKeyCompressor:
    """Per-layer key compressor using IsoQuant-Fast quaternion rotation."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.iq = IsoQuantMSE(head_dim, bits, seed=seed, mode="fast", device=device)
        self.q_L = self.iq.q_L.to(device)
        self.centroids = self.iq.centroids.to(device)
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, keys: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize keys using IsoQuant-Fast."""
        B, H, S, D = keys.shape
        orig_dtype = keys.dtype
        flat = keys.reshape(-1, D)

        # Use quaternion rotation: q_L @ v
        flat_rotated = flat @ self.q_L.T
        flat_quantized = (
            flat_rotated / self.centroids.abs().max() * 2 ** (bits - 1)
        ).long()
        flat_quantized = torch.clamp(flat_quantized, 0, 2**bits - 1)

        # Dequantize
        flat_recon = (
            flat_quantized.float() / (2 ** (bits - 1))
        ) * self.centroids.abs().max()
        flat_recon = flat_recon @ self.q_L.T

        return flat_recon.to(orig_dtype).reshape(B, H, S, D)


class IsoQuantValueCompressor:
    """Per-layer value compressor using IsoQuant-Fast quaternion rotation."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.iq = IsoQuantMSE(head_dim, bits, seed=seed, mode="fast", device=device)
        self.q_L = self.iq.q_L.to(device)
        self.centroids = self.iq.centroids.to(device)
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, values: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize values using IsoQuant-Fast."""
        B, H, S, D = values.shape
        orig_dtype = values.dtype
        flat = values.reshape(-1, D)

        # Use quaternion rotation: q_L @ v
        flat_rotated = flat @ self.q_L.T
        flat_quantized = (
            flat_rotated / self.centroids.abs().max() * 2 ** (bits - 1)
        ).long()
        flat_quantized = torch.clamp(flat_quantized, 0, 2**bits - 1)

        # Dequantize
        flat_recon = (
            flat_quantized.float() / (2 ** (bits - 1))
        ) * self.centroids.abs().max()
        flat_recon = flat_recon @ self.q_L.T

        return flat_recon.to(orig_dtype).reshape(B, H, S, D)


# ── IsoQuant-specific Patched Cache ─────────────────────────────────────


class IsoQuantPatchedCache(PatchedCache):
    """IsoQuant-specific implementation of PatchedCache."""

    def __init__(self, bits: int, device: str, quantize_values: bool = True):
        super().__init__(bits, device, quantize_values, backend="isoquant")
        self._key_compressors = {}
        self._val_compressors = {}

    def get_key_compressor(self, layer_idx: int, head_dim: int):
        if layer_idx not in self._key_compressors:
            self._key_compressors[layer_idx] = IsoQuantKeyCompressor(
                head_dim, self.bits, seed=layer_idx * 1000, device=self.device
            )
        return self._key_compressors[layer_idx]

    def get_val_compressor(self, layer_idx: int, head_dim: int):
        if layer_idx not in self._val_compressors:
            self._val_compressors[layer_idx] = IsoQuantValueCompressor(
                head_dim, self.bits, seed=layer_idx * 1000 + 500, device=self.device
            )
        return self._val_compressors[layer_idx]


def patch_isoquant_kv_cache(
    model,
    bits: int = 4,
    quantize_values: bool = False,
    device: str = "cuda",
):
    """Patch model for IsoQuant KV cache compression.

    Args:
        model: HuggingFace model
        bits: Quantization bits per dimension
        quantize_values: Whether to also compress values
        device: Target device

    Returns:
        Tuple of (original_update, isoquant_cache)
    """
    return patch_model_kv_cache(
        model,
        bits=bits,
        quantize_values=quantize_values,
        backend="isoquant",
        device=device,
    )


# ── IsoQuant-specific Fidelity Measurement ──────────────────────────────


@torch.no_grad()
def measure_isoquant_fidelity(model, tokenizer, context_len: int, bits: int):
    """Measure IsoQuant attention fidelity vs FP16.

    Args:
        model: HuggingFace model (evaluated mode)
        tokenizer: HuggingFace tokenizer
        context_len: Context length in tokens
        bits: Quantization bits

    Returns:
        Dict with fidelity metrics
    """

    def get_compressor(layer_idx, head_dim):
        return IsoQuantKeyCompressor(
            head_dim, bits, seed=layer_idx * 1000, device="cuda"
        )

    return measure_attention_fidelity(
        model, tokenizer, context_len, bits, "isoquant", get_compressor
    )


# ── IsoQuant-specific Generation Test ───────────────────────────────────


@torch.no_grad()
def test_isoquant_generation(
    model,
    tokenizer,
    context_len: int,
    bits: int,
    max_new_tokens: int = 60,
    keys_only: bool = True,
):
    """Generate text with IsoQuant-compressed KV cache.

    Args:
        model: HuggingFace model (evaluated mode)
        tokenizer: HuggingFace tokenizer
        context_len: Context length in tokens
        bits: Quantization bits
        max_new_tokens: Number of tokens to generate
        keys_only: Whether to only compress keys

    Returns:
        Dict with generation metrics
    """

    def get_key_compressor(layer_idx, head_dim):
        return IsoQuantKeyCompressor(
            head_dim, bits, seed=layer_idx * 1000, device="cuda"
        )

    def get_val_compressor(layer_idx, head_dim):
        return IsoQuantValueCompressor(
            head_dim, bits, seed=layer_idx * 1000 + 500, device="cuda"
        )

    return test_generation(
        model,
        tokenizer,
        context_len,
        bits,
        max_new_tokens=max_new_tokens,
        keys_only=keys_only,
        backend="isoquant",
        get_key_compressor=get_key_compressor,
        get_val_compressor=get_val_compressor,
    )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="IsoQuant High-Context POC")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (2-4)")
    parser.add_argument(
        "--max-ctx", type=int, default=32768, help="Max context to test"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--skip-fidelity", action="store_true", help="Skip attention fidelity test"
    )
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--keys-only",
        action="store_true",
        default=True,
        help="Only compress keys, leave values in fp16 (recommended)",
    )
    parser.add_argument(
        "--compress-values",
        dest="keys_only",
        action="store_false",
        help="Also compress values (higher error)",
    )
    args = parser.parse_args()

    print()
    print("=" * 74)
    print("  IsoQuant High-Context POC")
    print(f"  Model: {args.model}")
    print(
        f"  Bits: {args.bits}  |  Max context: {args.max_ctx:,}  |  Keys only: {args.keys_only}"
    )
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(
        f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print("=" * 74)
    print()

    # Load model
    print("Loading model...", flush=True)
    import logging

    logging.disable(logging.WARNING)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    model_vram = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded. VRAM: {model_vram:.0f} MB")
    print()

    # Context lengths to test
    ctx_lengths = []
    ctx = 2048
    while ctx <= args.max_ctx:
        ctx_lengths.append(ctx)
        ctx *= 2

    # ── Phase 1: Attention fidelity ──
    if not args.skip_fidelity:
        print("=" * 74)
        print("PHASE 1: Attention Fidelity (IsoQuant vs FP16)")
        print("=" * 74)
        print()
        print(
            f"  {'Context':>8s}  {'Cosine Sim':>12s}  {'Top-1 Match':>12s}  {'Layers':>8s}"
        )
        print(f"  {'─' * 8}  {'─' * 12}  {'─' * 12}  {'─' * 8}")

        for ctx_len in ctx_lengths:
            if ctx_len > 8192:
                # Fidelity test requires 2x memory
                print(f"  {ctx_len:>8,}  {'(skipped — needs 2x VRAM)':>40s}")
                continue
            try:
                result = measure_isoquant_fidelity(model, tokenizer, ctx_len, args.bits)
                print(
                    f"  {result['seq_len']:>8,}  {result['cosine_sim']:>12.6f}  "
                    f"{result['top1_match']:>10.1f}%  {result['n_layers']:>8d}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  {ctx_len:>8,}  {'OOM':>12s}")
                torch.cuda.empty_cache()
                break

        print()

    # ── Phase 2: Generation with compressed KV cache ──
    print("=" * 74)
    print("PHASE 2: Generation with IsoQuant (4-bit KV cache)")
    print("=" * 74)
    print()

    # Baseline at smallest context
    print("  FP16 baseline (2K context):")
    try:
        baseline = test_generation_fp16(model, tokenizer, 2048, args.max_new_tokens)
        print(
            f"    Tokens: {baseline['input_tokens']:,} in + {baseline['gen_tokens']} gen"
        )
        print(f"    Speed:  {baseline['tok_per_sec']:.1f} tok/s")
        print(f"    VRAM:   {baseline['vram_peak_mb']:.0f} MB peak")
        print(f"    Needle: {'FOUND' if baseline['needle_found'] else 'NOT FOUND'}")
        print(f"    Output: {baseline['text'][:120]}...")
    except Exception as e:
        print(f"    Error: {e}")
        baseline = None
    print()

    # Compressed KV at each context length
    print("  IsoQuant 4-bit results:")
    print()
    print(
        f"  {'Context':>8s}  {'Prefill':>10s}  {'Decode':>10s}  {'VRAM':>8s}  {'Needle':>8s}  {'Output (first 60 chars)'}"
    )
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 40}")

    for ctx_len in ctx_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            result = test_isoquant_generation(
                model,
                tokenizer,
                ctx_len,
                args.bits,
                args.max_new_tokens,
                args.keys_only,
            )
            needle_str = "FOUND" if result["needle_found"] else "MISS"
            text_preview = result["text"][:60].replace("\n", " ")
            print(
                f"  {result['input_tokens']:>8,}  "
                f"{result['prefill_tok_s']:>8.1f}/s  "
                f"{result['decode_tok_s']:>8.1f}/s  "
                f"{result['vram_peak_mb']:>6.0f}MB  "
                f"{needle_str:>8s}  "
                f"{text_preview}"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {ctx_len:>8,}  {'OOM':>10s}  --- VRAM limit reached ---")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  {ctx_len:>8,}  Error: {e}")
            break

    print()

    # ── Phase 3: Memory projection ──
    print("=" * 74)
    print("PHASE 3: Memory Projection")
    print("=" * 74)
    print()

    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )

    fp16_per_token = n_layers * 2 * n_kv_heads * head_dim * 2  # bytes (K+V)
    # IsoQuant stores compressed indices, not full FP16
    tq_per_token = (n_layers * 2 * n_kv_heads * head_dim * args.bits) / 8  # compressed

    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    avail_for_kv = (gpu_total - model_vram / 1024) * 1024**3  # bytes

    print(f"  Model: {args.model}")
    print(f"  Layers: {n_layers}, KV heads: {n_kv_heads}, head_dim: {head_dim}")
    print(
        f"  FP16 KV per token: {fp16_per_token:,} bytes ({fp16_per_token / 1024:.1f} KB)"
    )
    print(
        f"  IsoQuant {args.bits}-bit per token: {tq_per_token:,} bytes ({tq_per_token / 1024:.1f} KB)"
    )
    print(f"  Compression: {fp16_per_token / tq_per_token:.1f}x")
    print(f"  GPU total: {gpu_total:.1f} GB, model: {model_vram / 1024:.1f} GB")
    print(f"  Available for KV cache: {avail_for_kv / 1024**3:.1f} GB")
    print()

    print(f"  {'Context':>10s}  {'FP16 KV':>10s}  {'Iso 4-bit':>10s}  {'Status'}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 20}")
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
        kv_bytes_fp16 = fp16_per_token * ctx
        kv_bytes_iq = tq_per_token * ctx
        kv_gb_fp16 = kv_bytes_fp16 / 1024**3
        kv_gb_iq = kv_bytes_iq / 1024**3
        fits = "fits" if kv_bytes_iq < avail_for_kv else "OOM"
        print(f"  {ctx:>10,}  {kv_gb_fp16:>8.2f}GB  {kv_gb_iq:>8.2f}GB  {fits}")

    print()
    print("=" * 74)
    print("POC COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
