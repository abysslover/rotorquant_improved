"""
High-Context Testing Framework for TurboQuant and Other Methods

This module provides shared utilities for high-context generation testing,
needle-in-haystack retrieval, and attention fidelity measurement.

Common across all rotation methods (IsoQuant, PlanarQuant, RotorQuant, TurboQuant).

Usage:
    from methods.common.high_context import (
        build_prompt,
        measure_attention_fidelity,
        test_generation,
        patch_model_kv_cache,
    )
"""

import torch
import torch.nn.functional as F
import time
import gc
import os
from typing import Dict, Any, Optional, Tuple, Callable


# ── Needle-in-haystack constants ───────────────────────────────────────────

NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name mentioned in the documents?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.33) -> str:
    """Build a needle-in-haystack prompt at the target token count.

    Args:
        tokenizer: HuggingFace tokenizer
        target_tokens: Target context length in tokens
        needle_pos: Position of needle as fraction (0.0-1.0)

    Returns:
        Formatted prompt string ready for model generation
    """
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Important Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)

    haystack = "".join(parts)
    messages = [{"role": "user", "content": f"{haystack}\n\n{QUESTION}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── KV Cache Patching Utilities ───────────────────────────────────────────


class PatchedCache:
    """Wraps HuggingFace DynamicCache to quantize keys/values on insertion.

    This class provides a unified interface for KV cache patching across
    different quantization backends. Subclasses should implement the compressor
    retrieval methods.
    """

    def __init__(
        self,
        bits: int,
        device: str,
        quantize_values: bool = True,
        backend: str = "auto",
    ):
        """Initialize patched cache.

        Args:
            bits: Quantization bits per dimension
            device: Target device (e.g., 'cuda')
            quantize_values: Whether to also compress values
            backend: Backend identifier (e.g., 'iso', 'planar', 'turboquant')
        """
        self.bits = bits
        self.device = device
        self.quantize_values = quantize_values
        self.backend = backend
        self._key_compressors: Dict[int, Any] = {}
        self._val_compressors: Dict[int, Any] = {}

    def get_key_compressor(self, layer_idx: int, head_dim: int) -> Any:
        """Get or create key compressor for layer.

        Subclasses must override this to return method-specific compressors.
        """
        raise NotImplementedError

    def get_val_compressor(self, layer_idx: int, head_dim: int) -> Any:
        """Get or create value compressor for layer.

        Subclasses must override this to return method-specific compressors.
        """
        raise NotImplementedError


def patch_model_kv_cache(
    model,
    bits: int,
    quantize_values: bool = True,
    backend: str = "auto",
    device: str = "cuda",
) -> Tuple[Callable, PatchedCache]:
    """Monkey-patch model's cache update for post-prefill quantization compression.

    Strategy:
      - Prefill: full precision (no quantization, no error compounding)
      - First decode step: quantize entire prefill cache in bulk
      - Subsequent decode steps: quantize each new key, return full-precision
        key for current attention to avoid compounding

    This gives perfect prefill quality + compressed cache for decode.
    Works with any HuggingFace model that uses DynamicCache.

    Args:
        model: HuggingFace AutoModelForCausalLM instance
        bits: Quantization bits per dimension
        quantize_values: Whether to also compress values
        backend: Backend identifier
        device: Target device

    Returns:
        Tuple of (original_update_function, patched_cache_instance)
    """
    from transformers import DynamicCache

    patched_cache = PatchedCache(bits, device, quantize_values, backend)
    prefill_done: Dict[int, bool] = {}

    _original_update = DynamicCache.update

    def _compress_keys_inplace(
        key_states: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Quantize keys, returning new tensor (same shape/device)."""
        D = key_states.shape[-1]
        kc = patched_cache.get_key_compressor(layer_idx, D)
        return kc.compress_dequantize(key_states)

    def _patched_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Patched update that quantizes keys for storage."""
        new_seq_len = key_states.shape[2]
        is_prefill = new_seq_len > 1

        if is_prefill:
            # Prefill: store at full precision — no quantization
            prefill_done[layer_idx] = True
            return _original_update(
                self, key_states, value_states, layer_idx, cache_kwargs
            )

        # Decode: quantize new key for storage, full-precision for current attention.
        key_quantized = _compress_keys_inplace(key_states, layer_idx)

        # Optionally quantize values
        if patched_cache.quantize_values:
            D = value_states.shape[-1]
            vc = patched_cache.get_val_compressor(layer_idx, D)
            value_states = vc.compress_dequantize(value_states)

        k_out, v_out = _original_update(
            self, key_quantized, value_states, layer_idx, cache_kwargs
        )

        # On first decode step: quantize all prefill keys in bulk (one-time cost)
        if prefill_done.get(layer_idx) is True:
            cached_keys = self.layers[layer_idx].keys
            B, H, S, D = cached_keys.shape
            if S > 1:
                prefill_keys = cached_keys[:, :, :-1, :]
                prefill_q = _compress_keys_inplace(prefill_keys, layer_idx)
                cached_keys[:, :, :-1, :] = prefill_q
            prefill_done[layer_idx] = "done"

        # Patch last position to full-precision for current-step attention
        k_out[:, :, -1:, :] = key_states

        return k_out, v_out

    DynamicCache.update = _patched_update
    return _original_update, patched_cache


def unpatch_model_kv_cache(original_update: Callable) -> None:
    """Restore original cache update function."""
    from transformers import DynamicCache

    DynamicCache.update = original_update


# ── Attention Fidelity Measurement ─────────────────────────────────────────


@torch.no_grad()
def measure_attention_fidelity(
    model,
    tokenizer,
    context_len: int,
    bits: int,
    backend: str,
    get_compressor: Callable[[int, int], Any],
) -> Dict[str, Any]:
    """Compare quantized attention scores vs FP16 on real KV cache.

    Args:
        model: HuggingFace model (evaluated mode)
        tokenizer: HuggingFace tokenizer
        context_len: Context length in tokens
        bits: Quantization bits
        backend: Backend identifier
        get_compressor: Function(layer_idx, head_dim) -> compressor

    Returns:
        Dict with cosine_sim, top1_match, needle_found metrics
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_len + 256
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Forward pass with FP16 KV cache
    outputs_fp16 = model(**inputs, use_cache=True, output_attentions=False)
    cache_fp16 = outputs_fp16.past_key_values

    n_layers = len(cache_fp16.layers)
    head_dim = cache_fp16.layers[0].keys.shape[-1]
    n_kv_heads = cache_fp16.layers[0].keys.shape[1]

    # Measure per-layer attention fidelity
    cosine_sims = []
    top1_matches = 0
    n_checks = 0

    for layer_idx in range(n_layers):
        keys = cache_fp16.layers[layer_idx].keys  # (1, H, S, D)
        B, H, S, D = keys.shape

        # Compress keys
        compressor = get_compressor(layer_idx, D)
        keys_quant = compressor.compress_dequantize(keys)

        # Query = last token attending to all keys
        query = keys[:, :, -1:, :]  # (1, H, 1, D)

        # Real scores
        real_scores = torch.matmul(
            query.float(), keys.float().transpose(-2, -1)
        ).squeeze(-2)

        # Quantized scores
        quant_scores = torch.matmul(
            query.float(), keys_quant.float().transpose(-2, -1)
        ).squeeze(-2)

        for h in range(H):
            cos = F.cosine_similarity(
                real_scores[0, h].unsqueeze(0), quant_scores[0, h].unsqueeze(0)
            ).item()
            cosine_sims.append(cos)

            if real_scores[0, h].argmax().item() == quant_scores[0, h].argmax().item():
                top1_matches += 1
            n_checks += 1

    # Clean up
    del cache_fp16, outputs_fp16
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "cosine_sim": sum(cosine_sims) / len(cosine_sims),
        "top1_match": top1_matches / n_checks * 100,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
    }


# ── Generation Testing ────────────────────────────────────────────────────


@torch.no_grad()
def test_generation(
    model,
    tokenizer,
    context_len: int,
    bits: int,
    max_new_tokens: int = 60,
    keys_only: bool = True,
    backend: str = "auto",
    get_key_compressor: Callable[[int, int], Any] = None,
    get_val_compressor: Callable[[int, int], Any] = None,
) -> Dict[str, Any]:
    """Generate text with compressed KV cache.

    Args:
        model: HuggingFace model (evaluated mode)
        tokenizer: HuggingFace tokenizer
        context_len: Context length in tokens
        bits: Quantization bits
        max_new_tokens: Number of tokens to generate
        keys_only: Whether to only compress keys (not values)
        backend: Backend identifier
        get_key_compressor: Optional compressor getter
        get_val_compressor: Optional value compressor getter

    Returns:
        Dict with speed, VRAM, needle_found metrics
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_len + 256
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    vram_before = torch.cuda.memory_allocated() / 1024**2

    # Patch KV cache
    original_update, patched_cache = patch_model_kv_cache(
        model, bits=bits, quantize_values=not keys_only, backend=backend, device="cuda"
    )

    try:
        # Prefill: single forward pass on full prompt
        t_prefill_start = time.perf_counter()
        prefill_out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t_prefill_start
        prefill_tok_s = input_len / t_prefill if t_prefill > 0 else 0

        # Decode: generate token by token
        t_decode_start = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_decode_start

        gen_tokens = outputs[0][input_len:]
        n_gen = len(gen_tokens)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Decode speed: subtract prefill time from total
        t_decode = t_total - t_prefill if t_total > t_prefill else t_total
        decode_tok_s = n_gen / t_decode if t_decode > 0 else 0

        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        vram_kv = vram_peak - vram_before

        needle_found = "AURORA-7749" in text or "aurora" in text.lower()

    finally:
        unpatch_model_kv_cache(original_update)
        torch.cuda.empty_cache()

    return {
        "input_tokens": input_len,
        "gen_tokens": n_gen,
        "text": text.strip(),
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "time_s": t_total,
        "vram_peak_mb": vram_peak,
        "vram_kv_est_mb": vram_kv,
        "needle_found": needle_found,
    }


@torch.no_grad()
def test_generation_fp16(
    model, tokenizer, context_len: int, max_new_tokens: int = 60
) -> Dict[str, Any]:
    """Baseline: generate with standard FP16 KV cache.

    Args:
        model: HuggingFace model (evaluated mode)
        tokenizer: HuggingFace tokenizer
        context_len: Context length in tokens
        max_new_tokens: Number of tokens to generate

    Returns:
        Dict with baseline speed and VRAM metrics
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_len + 256
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    vram_before = torch.cuda.memory_allocated() / 1024**2

    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    gen_tokens = outputs[0][input_len:]
    n_gen = len(gen_tokens)
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    vram_peak = torch.cuda.max_memory_allocated() / 1024**2
    vram_kv = vram_peak - vram_before

    torch.cuda.empty_cache()

    return {
        "input_tokens": input_len,
        "gen_tokens": n_gen,
        "text": text.strip(),
        "tok_per_sec": n_gen / t_gen if t_gen > 0 else 0,
        "time_s": t_gen,
        "vram_peak_mb": vram_peak,
        "vram_kv_est_mb": vram_kv,
        "needle_found": "AURORA-7749" in text or "aurora" in text.lower(),
    }
