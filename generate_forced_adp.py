"""
Add forced_adp_tokens to an existing generations.jsonl.

Reads prompt_ids from the existing file, runs generate_until_detected with
the adaptive processor, and writes the merged result back in place.

Usage:
    python generate_forced_adp.py --input_file data/generations.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark import WatermarkLogitsProcessor, WatermarkDetector
from generate import generate_until_detected

MODEL_NAME = "facebook/opt-1.3b"
HASH_KEY   = 15485863


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, default="data/generations.jsonl")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    records = []
    with open(args.input_file) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    if "forced_adp_tokens" in records[0]:
        print("forced_adp_tokens already present — nothing to do.")
        return

    gamma = records[0]["gamma"]
    delta = records[0]["delta"]

    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()

    vocab_size = len(tokenizer)
    target_length = len(records[0]["watermarked_tokens"])  # infer from existing data

    adaptive_processor = WatermarkLogitsProcessor(
        vocab_size=vocab_size,
        gamma=gamma,
        delta=delta,
        adaptive=True,
        alpha=0.5,  # sqrt scaling — more aggressive at low entropy than linear
        hash_key=HASH_KEY,
    )
    detector = WatermarkDetector(
        vocab_size=vocab_size,
        gamma=gamma,
        hash_key=HASH_KEY,
        z_threshold=4.0,
    )

    for i, rec in enumerate(records):
        print(f"[{i+1}/{len(records)}] generating forced_adp …", end="\r")
        forced_adp_tokens = generate_until_detected(
            model,
            rec["prompt_ids"],
            adaptive_processor,
            detector,
            max_tokens=target_length * 2,
            eos_token_id=tokenizer.eos_token_id,
            device=args.device,
        )
        rec["forced_adp_tokens"] = forced_adp_tokens
        rec["forced_adp_text"]   = tokenizer.decode(forced_adp_tokens, skip_special_tokens=True)

    with open(args.input_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone — wrote forced_adp_tokens for {len(records)} records to {args.input_file}")


if __name__ == "__main__":
    main()
