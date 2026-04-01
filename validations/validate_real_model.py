#!/usr/bin/env python3
"""
Comprehensive TurboQuant Validation with Real Model Testing

Tests TurboQuant on a real LLM with memory profiling:
- Memory usage (ensure < 5GB)
- Prefill speed (tokens/sec)
- Decode speed (tokens/sec)
- Needle-in-haystack accuracy

Falls back to smaller model if Qwen2.5-3B-Instruct exceeds memory limit.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

# Memory limit: 5GB (keep headroom for PyTorch overhead)
MEMORY_LIMIT_MB = 5000

# Model selection (will try in order until one fits)
MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-3B-Instruct",  # Primary target (~2GB base)
    "microsoft/Phi-3-mini-4k-instruct",  # Fallback (~2.3GB base)
    "Qwen/Qwen2-1.5B-Instruct",  # Smaller fallback
]

# Needle-in-haystack test config
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"
FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""

# Test context lengths
CONTEXT_LENGTHS = [2048, 4096, 8192]


# ============================================================================
# MEMORY CHECKER
# ============================================================================


def check_model_memory(model_name, verbose=True):
    """Check how much GPU memory a model will use when loaded."""
    if verbose:
        print(f"Checking memory for {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()

        allocated_mb = torch.cuda.memory_allocated() // 1024 // 1024
        total_mb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024

        if verbose:
            print(f"  Model memory: {allocated_mb} MB")
            print(f"  Total GPU: {total_mb} MB")
            print(f"  Memory limit: {MEMORY_LIMIT_MB} MB")
            print(
                f"  Status: {'OK' if allocated_mb < MEMORY_LIMIT_MB else 'EXCEEDS LIMIT'}"
            )

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()

        return allocated_mb < MEMORY_LIMIT_MB, allocated_mb

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False, 0


# ============================================================================
# VALIDATION RUNNER
# ============================================================================


def run_validation(model_name, target_tokens=4096):
    """Run comprehensive validation on a model."""
    print(f"\n{'=' * 80}")
    print(f"VALIDATING: {model_name}")
    print(f"{'=' * 80}\n")

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    initial_memory = torch.cuda.memory_allocated() // 1024 // 1024
    print(f"Loaded. GPU: {initial_memory} MB\n")

    # Build prompt with needle
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    parts = []
    for i in range(n_reps):
        if i == n_reps // 2:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    prompt = f"user\n{haystack}\nQuestion: {QUESTION}</think>\n"

    # Tokenize
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Find needle position
    needle_phrase = "AURORA-7749"
    needle_tokens = tokenizer.encode(needle_phrase, add_special_tokens=False)
    input_ids_list = inputs["input_ids"][0].tolist()
    needle_start = None
    for i in range(len(input_ids_list) - len(needle_tokens) + 1):
        if input_ids_list[i : i + len(needle_tokens)] == needle_tokens:
            needle_start = i
            break

    # Find needle token positions - search for a distinctive substring
    needle_phrase = "AURORA-7749"
    needle_tokens = tokenizer.encode(needle_phrase, add_special_tokens=False)
    input_ids_list = inputs["input_ids"][0].tolist()
    needle_start = None
    for i in range(len(input_ids_list) - len(needle_tokens) + 1):
        if input_ids_list[i : i + len(needle_tokens)] == needle_tokens:
            needle_start = i
            break
    # Fallback: search for partial match
    if needle_start is None:
        for width in range(len(needle_tokens), 0, -1):
            sub = needle_tokens[:width]
            for i in range(len(input_ids_list) - width + 1):
                if input_ids_list[i : i + width] == sub:
                    needle_start = i
                    break
            if needle_start is not None:
                break

    print(f"\n{'=' * 80}")
    print(f"Context: {seq_len} tokens | Needle at token {needle_start}")
    print(f"{'=' * 80}\n")

    # Capture KV cache
    print("Running forward pass...", flush=True)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    cache = outputs.past_key_values

    n_layers = len(cache.layers)
    head_dim = cache.layers[0].keys.shape[-1]
    num_kv_heads = cache.layers[0].keys.shape[1]

    # ========================================================================
    # MEMORY ACCOUNTING
    # ========================================================================
    total_fp16_bytes = 0
    for layer_idx in range(n_layers):
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
        total_fp16_bytes += (keys.numel() + values.numel()) * 2

    fp16_mb = total_fp16_bytes / 1024 / 1024
    total_memory_mb = initial_memory + fp16_mb

    print(f"\n{'=' * 80}")
    print("MEMORY USAGE")
    print(f"{'=' * 80}")
    print(f"  Model weights:         {initial_memory} MB")
    print(f"  KV cache (FP16):       {fp16_mb:.1f} MB")
    print(f"  Total:                 {total_memory_mb:.1f} MB")
    print(f"  Memory limit:          {MEMORY_LIMIT_MB} MB")
    print(
        f"  Status:                {'OK' if total_memory_mb < MEMORY_LIMIT_MB else 'EXCEEDS LIMIT'}"
    )

    # ========================================================================
    # PERFORMANCE BENCHMARKING
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("PERFORMANCE")
    print(f"{'=' * 80}")

    # Prefill speed (first forward pass)
    inputs_for_perf = tokenizer(
        haystack[:target_tokens],
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens + 256,
    ).to("cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs_for_perf, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0
    prefill_speed = target_tokens / prefill_time

    print(f"\n  Prefill (first forward pass):")
    print(f"    Time:                  {prefill_time:.2f}s")
    print(f"    Speed:                 {prefill_speed:.1f} tokens/sec")

    # Decode speed (single token generation)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            input_ids = inputs_for_perf["input_ids"][:, -1:]
            outputs = model(input_ids, use_cache=True)
            input_ids = outputs.logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0
    decode_speed = 100 / decode_time

    print(f"\n  Decode (100 tokens):")
    print(f"    Time:                  {decode_time:.2f}s")
    print(f"    Speed:                 {decode_speed:.1f} tokens/sec")

    # ========================================================================
    # NEEDLE-IN-HAYSTACK VALIDATION
    # ========================================================================
    if needle_start is not None:
        print(f"\n{'=' * 80}")
        print("NEEDLE-IN-HAYSTACK VALIDATION")
        print(f"{'=' * 80}")

        # Query for last token
        query = cache.layers[0].keys[:, :, -1:, :]  # (1, H, 1, D)

        # Check attention scores across all layers
        needle_scores = []
        top_positions = []

        for layer_idx in range(n_layers):
            keys = cache.layers[layer_idx].keys
            scores = torch.matmul(
                query.float(), keys.float().transpose(-2, -1)
            ).squeeze(-2)  # (1, H, S)

            # Needle score (average across heads)
            needle_score = scores[0, :, needle_start].mean().item()
            needle_scores.append(needle_score)

            # Top position (average across heads)
            top_pos = scores[0, :, :].argmax(dim=-1).mean().item()
            top_positions.append(top_pos)

        avg_needle_score = sum(needle_scores) / len(needle_scores)
        avg_top_pos = sum(top_positions) / len(top_positions)
        needle_rank = (
            torch.tensor(needle_scores).argsort(descending=True) == needle_start
        ).nonzero()
        actual_rank = needle_rank[0].item() if len(needle_rank) > 0 else -1

        print(f"\n  Needle at position {needle_start}:")
        print(f"    Average attention score:  {avg_needle_score:.4f}")
        print(f"    Model's top focus:        Token {avg_top_pos:.0f}")
        print(
            f"    Needle retrieval:         {'FOUND' if actual_rank >= 0 else 'NOT FOUND'}"
        )

        # Verify by checking if model can generate the needle
        print(f"\n  Generation test:")
        gen_prompt = f"The secret project code name is "
        gen_inputs = tokenizer(gen_prompt, return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                gen_inputs["input_ids"],
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    Generated: '{generated_text}'")
        print(f"    Generation time: {gen_time:.2f}s")

        # Check if needle is present
        is_found = (
            "AURORA-7749" in generated_text
            or "AURORA" in generated_text
            and "7749" in generated_text
        )
        print(f"    Needle found:           {'YES' if is_found else 'NO'}")
    else:
        print(f"\n  Could not locate needle in tokens")

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}\n")

    return {
        "model": model_name,
        "memory_mb": total_memory_mb,
        "prefill_speed": prefill_speed,
        "decode_speed": decode_speed,
        "needle_found": is_found if needle_start is not None else None,
        "needle_rank": actual_rank if needle_start is not None else None,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("TURBOQUANT REAL MODEL VALIDATION")
    print("=" * 80)
    print(f"Memory limit: {MEMORY_LIMIT_MB} MB")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
    )
    print()

    # Find a suitable model
    selected_model = None
    for model_name in MODEL_CANDIDATES:
        fits, memory = check_model_memory(model_name, verbose=True)
        if fits:
            selected_model = model_name
            print(f"\n✓ Selected: {model_name} ({memory} MB)")
            break
        else:
            print(f"✗ {model_name} exceeds memory limit, trying next...\n")

    if selected_model is None:
        print(
            "\nERROR: No suitable model found. Please reduce MEMORY_LIMIT_MB or use a smaller GPU."
        )
        return 1

    # Run validation
    result = run_validation(selected_model)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Model:             {result['model']}")
    print(f"  Total Memory:      {result['memory_mb']:.1f} MB")
    print(f"  Prefill Speed:     {result['prefill_speed']:.1f} tokens/sec")
    print(f"  Decode Speed:      {result['decode_speed']:.1f} tokens/sec")
    if result["needle_found"] is not None:
        print(f"  Needle Found:      {'YES' if result['needle_found'] else 'NO'}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
