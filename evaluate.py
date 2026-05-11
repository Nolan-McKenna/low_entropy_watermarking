"""
Evaluate watermark detection and text quality.

Computes:
  - Z-scores for watermarked and non-watermarked sequences
  - Perplexity via OPT-2.7B oracle (matches Kirchenbauer et al. setup)
  - ROC curve and AUC
  - Detection accuracy at z-threshold = 4.0

Usage:
    python evaluate.py --input_file data/generations.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark import WatermarkDetector

ORACLE_MODEL_NAME = "facebook/opt-2.7b"
HASH_KEY = 15485863
Z_THRESHOLD = 4.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, default="data/generations.jsonl")
    p.add_argument("--compute_ppl", action="store_true",
                   help="Also compute oracle perplexity (requires OPT-2.7B, ~10GB RAM)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_perplexity(token_ids: list[int], model, device: str) -> float:
    if len(token_ids) < 2:
        return float("nan")
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    return float(torch.exp(out.loss).item())


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    records = []
    with open(args.input_file) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    gamma = records[0]["gamma"]
    detector = WatermarkDetector(
        vocab_size=max(max(r["no_watermark_tokens"]) for r in records) + 1,
        gamma=gamma,
        hash_key=HASH_KEY,
        z_threshold=Z_THRESHOLD,
    )

    # Re-derive vocab_size properly from a tokenizer
    # (The stored max token id is a lower bound; use a fixed large number for safety)
    # We rebuild the detector with the correct size after loading the tokenizer.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    vocab_size = len(tok)
    detector = WatermarkDetector(
        vocab_size=vocab_size,
        gamma=gamma,
        hash_key=HASH_KEY,
        z_threshold=Z_THRESHOLD,
    )

    oracle_model = None
    if args.compute_ppl:
        print(f"Loading oracle model {ORACLE_MODEL_NAME} …")
        oracle_model = AutoModelForCausalLM.from_pretrained(
            ORACLE_MODEL_NAME,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        ).to(args.device)
        oracle_model.eval()

    nw_zscores, w_zscores = [], []
    nw_ppls, w_ppls = [], []

    for i, rec in enumerate(records):
        nw_res = detector.detect(rec["no_watermark_tokens"])
        w_res = detector.detect(rec["watermarked_tokens"])
        nw_zscores.append(nw_res["z_score"])
        w_zscores.append(w_res["z_score"])

        if oracle_model is not None:
            nw_ppls.append(compute_perplexity(rec["no_watermark_tokens"], oracle_model, args.device))
            w_ppls.append(compute_perplexity(rec["watermarked_tokens"], oracle_model, args.device))

        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(records)}")

    nw_zscores = np.array(nw_zscores)
    w_zscores = np.array(w_zscores)

    # --- Detection accuracy at threshold ---
    tpr = np.mean(w_zscores > Z_THRESHOLD)
    fpr = np.mean(nw_zscores > Z_THRESHOLD)
    fnr = 1 - tpr
    print(f"\n=== Watermark detection (z > {Z_THRESHOLD}) ===")
    print(f"  TPR (watermarked detected):     {tpr:.3f}")
    print(f"  FPR (human text false-positive):{fpr:.3f}")
    print(f"  FNR (watermarked missed):       {fnr:.3f}")

    # --- Z-score summary ---
    print(f"\n=== Z-score statistics ===")
    print(f"  No-watermark  mean={nw_zscores.mean():.2f}  std={nw_zscores.std():.2f}")
    print(f"  Watermarked   mean={w_zscores.mean():.2f}  std={w_zscores.std():.2f}")

    # --- ROC / AUC ---
    labels = np.array([0] * len(nw_zscores) + [1] * len(w_zscores))
    scores = np.concatenate([nw_zscores, w_zscores])
    auc = roc_auc_score(labels, scores)
    print(f"\n=== ROC AUC ===")
    print(f"  AUC: {auc:.4f}")

    # --- Perplexity ---
    if oracle_model is not None:
        nw_ppls = np.array(nw_ppls)
        w_ppls = np.array(w_ppls)
        print(f"\n=== Oracle Perplexity (OPT-2.7B) ===")
        print(f"  No-watermark  mean={np.nanmean(nw_ppls):.2f}")
        print(f"  Watermarked   mean={np.nanmean(w_ppls):.2f}")

    # Save summary
    summary = {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auc": float(auc),
        "nw_z_mean": float(nw_zscores.mean()),
        "nw_z_std": float(nw_zscores.std()),
        "w_z_mean": float(w_zscores.mean()),
        "w_z_std": float(w_zscores.std()),
    }
    if oracle_model is not None:
        summary["nw_ppl_mean"] = float(np.nanmean(nw_ppls))
        summary["w_ppl_mean"] = float(np.nanmean(w_ppls))

    out_path = Path(args.input_file).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
