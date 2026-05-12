"""
Generate watermarked and non-watermarked completions using OPT-1.3B
on the C4 RealNewsLike dataset, matching the experimental setup of
Kirchenbauer et al. (ICML 2023).

Usage:
    python generate.py --num_samples 500 --output_file data/generations.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark import WatermarkLogitsProcessor, WatermarkDetector

# Paper defaults
MODEL_NAME = "facebook/opt-1.3b"
ORACLE_MODEL_NAME = "facebook/opt-2.7b"  # used only in evaluate.py for PPL
TARGET_LENGTH = 200
PROMPT_LENGTH = 50  # tokens to use as prompt (paper trims from end)
DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "realnewslike"
GAMMA = 0.25
DELTA = 2.0
HASH_KEY = 15485863


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--output_file", type=str, default="data/generations.jsonl")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--delta", type=float, default=DELTA)
    p.add_argument("--target_length", type=int, default=TARGET_LENGTH)
    p.add_argument("--prompt_length", type=int, default=PROMPT_LENGTH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model_and_tokenizer(model_name: str, device: str):
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def stream_c4_prompts(tokenizer, prompt_length: int, num_samples: int):
    """
    Yield (prompt_text, prompt_token_ids) from C4 RealNewsLike.
    Mirrors the paper: take a long string, trim the last `prompt_length` tokens
    as a prompt, use the rest as a human baseline.
    """
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    collected = 0
    for example in dataset:
        if collected >= num_samples:
            break
        text = example["text"]
        token_ids = tokenizer.encode(text)
        if len(token_ids) < prompt_length + TARGET_LENGTH:
            continue
        # Use first tokens as prompt; remaining as human baseline (not used in generation)
        prompt_ids = token_ids[:prompt_length]
        yield tokenizer.decode(prompt_ids, skip_special_tokens=True), prompt_ids
        collected += 1


@torch.inference_mode()
def generate_until_detected(
    model,
    prompt_ids: list[int],
    processor: WatermarkLogitsProcessor,
    detector: WatermarkDetector,
    max_tokens: int = 800,
    min_check: int = 20,
    eos_token_id: int = None,
    device: str = "cpu",
) -> list[int]:
    """
    Generate tokens until BOTH the z-score threshold is crossed AND an EOS token
    is produced (whichever comes last), or max_tokens is reached.
    Waiting for EOS avoids cutting off mid-sentence, which would inflate PPL.
    """
    all_ids = list(prompt_ids)
    generated = []
    past_key_values = None
    threshold_hit = False

    for _ in range(max_tokens):
        if past_key_values is None:
            input_tensor = torch.tensor([all_ids], dtype=torch.long, device=device)
        else:
            input_tensor = torch.tensor([[all_ids[-1]]], dtype=torch.long, device=device)

        out = model(input_tensor, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  # (1, vocab_size)

        context_tensor = torch.tensor([all_ids], dtype=torch.long, device=device)
        logits = processor(context_tensor, logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs[0], num_samples=1).item())

        all_ids.append(next_token)
        generated.append(next_token)

        if len(generated) >= min_check:
            threshold_hit = threshold_hit or detector.detect(generated)["is_watermarked"]

        eos_hit = (eos_token_id is not None and next_token == eos_token_id)
        if threshold_hit and (eos_hit or eos_token_id is None):
            break

    return generated


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt_ids: list[int],
    target_length: int,
    device: str,
    logits_processor=None,
) -> list[int]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    processors = [logits_processor] if logits_processor else []
    out = model.generate(
        input_ids,
        max_new_tokens=target_length,
        do_sample=True,
        logits_processor=processors,
    )
    # Return only the newly generated tokens (strip prompt)
    return out[0, len(prompt_ids):].tolist()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, args.device)
    vocab_size = len(tokenizer)

    fixed_processor = WatermarkLogitsProcessor(
        vocab_size=vocab_size,
        gamma=args.gamma,
        delta=args.delta,
        adaptive=False,
        hash_key=HASH_KEY,
    )
    adaptive_processor = WatermarkLogitsProcessor(
        vocab_size=vocab_size,
        gamma=args.gamma,
        delta=args.delta,
        adaptive=True,
        hash_key=HASH_KEY,
    )
    detector = WatermarkDetector(
        vocab_size=vocab_size,
        gamma=args.gamma,
        hash_key=HASH_KEY,
        z_threshold=4.0,
    )

    results = []
    prompt_stream = stream_c4_prompts(tokenizer, args.prompt_length, args.num_samples)

    for i, (prompt_text, prompt_ids) in enumerate(prompt_stream):
        print(f"[{i+1}/{args.num_samples}] generating …", end="\r")

        nw_tokens     = generate_one(model, tokenizer, prompt_ids, args.target_length, args.device)
        w_tokens      = generate_one(model, tokenizer, prompt_ids, args.target_length, args.device, fixed_processor)
        adp_tokens    = generate_one(model, tokenizer, prompt_ids, args.target_length, args.device, adaptive_processor)
        forced_tokens = generate_until_detected(model, prompt_ids, fixed_processor, detector,
                                                max_tokens=args.target_length * 4,
                                                eos_token_id=tokenizer.eos_token_id,
                                                device=args.device)

        results.append({
            "idx": i,
            "prompt": prompt_text,
            "prompt_ids": prompt_ids,
            "no_watermark_tokens": nw_tokens,
            "watermarked_tokens":  w_tokens,
            "adaptive_tokens":     adp_tokens,
            "forced_tokens":       forced_tokens,
            "no_watermark_text": tokenizer.decode(nw_tokens,     skip_special_tokens=True),
            "watermarked_text":  tokenizer.decode(w_tokens,      skip_special_tokens=True),
            "adaptive_text":     tokenizer.decode(adp_tokens,    skip_special_tokens=True),
            "forced_text":       tokenizer.decode(forced_tokens,  skip_special_tokens=True),
            "gamma": args.gamma,
            "delta": args.delta,
        })

    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
