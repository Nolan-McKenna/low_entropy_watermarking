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
def compute_perplexity(
    prompt_ids: list[int], generated_ids: list[int], model, device: str
) -> float:
    """
    Compute oracle perplexity on generated_ids given prompt_ids as context.
    Loss is computed only over the generated tokens, matching Appendix A.2
    of Kirchenbauer et al.
    """
    if len(generated_ids) < 1:
        return float("nan")
    full_ids = prompt_ids + generated_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
    # -100 tells the model to ignore prompt positions when computing loss
    labels = torch.tensor(
        [[-100] * len(prompt_ids) + generated_ids], dtype=torch.long
    ).to(device)
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

    nw_zscores, w_zscores, adp_zscores, forced_zscores, forced_adp_zscores = [], [], [], [], []
    nw_ppls, w_ppls, adp_ppls, forced_ppls, forced_adp_ppls = [], [], [], [], []
    forced_lengths, forced_adp_lengths = [], []
    has_adaptive     = "adaptive_tokens"   in records[0]
    has_forced       = "forced_tokens"     in records[0]
    has_forced_adp   = "forced_adp_tokens" in records[0]

    for i, rec in enumerate(records):
        nw_res  = detector.detect(rec["no_watermark_tokens"])
        w_res   = detector.detect(rec["watermarked_tokens"])
        nw_zscores.append(nw_res["z_score"])
        w_zscores.append(w_res["z_score"])

        if has_adaptive:
            adp_res = detector.detect(rec["adaptive_tokens"])
            adp_zscores.append(adp_res["z_score"])

        if has_forced:
            forced_res = detector.detect(rec["forced_tokens"])
            forced_zscores.append(forced_res["z_score"])
            forced_lengths.append(len(rec["forced_tokens"]))

        if has_forced_adp:
            forced_adp_res = detector.detect(rec["forced_adp_tokens"])
            forced_adp_zscores.append(forced_adp_res["z_score"])
            forced_adp_lengths.append(len(rec["forced_adp_tokens"]))

        if oracle_model is not None:
            prompt_ids = rec["prompt_ids"]
            nw_ppls.append(compute_perplexity(prompt_ids, rec["no_watermark_tokens"], oracle_model, args.device))
            w_ppls.append(compute_perplexity(prompt_ids, rec["watermarked_tokens"], oracle_model, args.device))
            if has_adaptive:
                adp_ppls.append(compute_perplexity(prompt_ids, rec["adaptive_tokens"], oracle_model, args.device))
            if has_forced:
                forced_ppls.append(compute_perplexity(prompt_ids, rec["forced_tokens"], oracle_model, args.device))
            if has_forced_adp:
                forced_adp_ppls.append(compute_perplexity(prompt_ids, rec["forced_adp_tokens"], oracle_model, args.device))

        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(records)}")

    nw_zscores         = np.array(nw_zscores)
    w_zscores          = np.array(w_zscores)
    adp_zscores        = np.array(adp_zscores)        if adp_zscores        else None
    forced_zscores     = np.array(forced_zscores)     if forced_zscores     else None
    forced_adp_zscores = np.array(forced_adp_zscores) if forced_adp_zscores else None
    forced_lengths     = np.array(forced_lengths)     if forced_lengths     else None
    forced_adp_lengths = np.array(forced_adp_lengths) if forced_adp_lengths else None

    # --- Detection accuracy at threshold ---
    tpr            = np.mean(w_zscores > Z_THRESHOLD)
    fpr            = np.mean(nw_zscores > Z_THRESHOLD)
    fnr            = 1 - tpr
    adp_tpr        = float(np.mean(adp_zscores        > Z_THRESHOLD)) if adp_zscores        is not None else None
    forced_tpr     = float(np.mean(forced_zscores     > Z_THRESHOLD)) if forced_zscores     is not None else None
    forced_adp_tpr = float(np.mean(forced_adp_zscores > Z_THRESHOLD)) if forced_adp_zscores is not None else None
    print(f"\n=== Watermark detection (z > {Z_THRESHOLD}) ===")
    print(f"  TPR fixed:                           {tpr:.3f}")
    if adp_tpr        is not None: print(f"  TPR adaptive:                        {adp_tpr:.3f}")
    if forced_tpr     is not None: print(f"  TPR forced (fixed δ):                {forced_tpr:.3f}")
    if forced_adp_tpr is not None: print(f"  TPR forced (adaptive δ):             {forced_adp_tpr:.3f}")
    print(f"  FPR (no-watermark false-positive):   {fpr:.3f}")
    print(f"  FNR fixed:                           {fnr:.3f}")

    # --- Z-score summary ---
    print(f"\n=== Z-score statistics ===")
    print(f"  No-watermark       mean={nw_zscores.mean():.2f}  std={nw_zscores.std():.2f}")
    print(f"  Fixed δ            mean={w_zscores.mean():.2f}  std={w_zscores.std():.2f}")
    if adp_zscores        is not None: print(f"  Adaptive δ         mean={adp_zscores.mean():.2f}  std={adp_zscores.std():.2f}")
    if forced_zscores     is not None: print(f"  Forced (fixed δ)   mean={forced_zscores.mean():.2f}  std={forced_zscores.std():.2f}  len={forced_lengths.mean():.1f}")
    if forced_adp_zscores is not None: print(f"  Forced (adaptive)  mean={forced_adp_zscores.mean():.2f}  std={forced_adp_zscores.std():.2f}  len={forced_adp_lengths.mean():.1f}")

    # --- ROC / AUC ---
    labels = np.array([0] * len(nw_zscores) + [1] * len(w_zscores))
    scores = np.concatenate([nw_zscores, w_zscores])
    auc            = roc_auc_score(labels, scores)
    adp_auc        = None
    forced_auc     = None
    forced_adp_auc = None
    if adp_zscores is not None:
        adp_labels = np.array([0] * len(nw_zscores) + [1] * len(adp_zscores))
        adp_auc = roc_auc_score(adp_labels, np.concatenate([nw_zscores, adp_zscores]))
    if forced_zscores is not None:
        forced_labels = np.array([0] * len(nw_zscores) + [1] * len(forced_zscores))
        forced_auc = roc_auc_score(forced_labels, np.concatenate([nw_zscores, forced_zscores]))
    if forced_adp_zscores is not None:
        forced_adp_labels = np.array([0] * len(nw_zscores) + [1] * len(forced_adp_zscores))
        forced_adp_auc = roc_auc_score(forced_adp_labels, np.concatenate([nw_zscores, forced_adp_zscores]))
    print(f"\n=== ROC AUC ===")
    print(f"  Fixed δ:            {auc:.4f}")
    if adp_auc        is not None: print(f"  Adaptive δ:         {adp_auc:.4f}")
    if forced_auc     is not None: print(f"  Forced (fixed δ):   {forced_auc:.4f}")
    if forced_adp_auc is not None: print(f"  Forced (adaptive):  {forced_adp_auc:.4f}")

    # --- Perplexity ---
    if oracle_model is not None:
        nw_ppls         = np.array(nw_ppls)
        w_ppls          = np.array(w_ppls)
        adp_ppls        = np.array(adp_ppls)        if adp_ppls        else None
        forced_ppls     = np.array(forced_ppls)     if forced_ppls     else None
        forced_adp_ppls = np.array(forced_adp_ppls) if forced_adp_ppls else None
        print(f"\n=== Oracle Perplexity (OPT-2.7B) ===")
        print(f"  No-watermark      mean={np.nanmean(nw_ppls):.2f}")
        print(f"  Fixed δ           mean={np.nanmean(w_ppls):.2f}")
        if adp_ppls        is not None: print(f"  Adaptive δ        mean={np.nanmean(adp_ppls):.2f}")
        if forced_ppls     is not None: print(f"  Forced (fixed δ)  mean={np.nanmean(forced_ppls):.2f}")
        if forced_adp_ppls is not None: print(f"  Forced (adaptive) mean={np.nanmean(forced_adp_ppls):.2f}")

    # Save summary
    summary = {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auc": float(auc),
        "nw_z_mean": float(nw_zscores.mean()),
        "nw_z_std":  float(nw_zscores.std()),
        "w_z_mean":  float(w_zscores.mean()),
        "w_z_std":   float(w_zscores.std()),
    }
    if adp_zscores is not None:
        summary["adp_tpr"]    = float(adp_tpr)
        summary["adp_auc"]    = float(adp_auc)
        summary["adp_z_mean"] = float(adp_zscores.mean())
        summary["adp_z_std"]  = float(adp_zscores.std())
    if forced_zscores is not None:
        summary["forced_tpr"]      = float(forced_tpr)
        summary["forced_auc"]      = float(forced_auc)
        summary["forced_z_mean"]   = float(forced_zscores.mean())
        summary["forced_z_std"]    = float(forced_zscores.std())
        summary["forced_len_mean"] = float(forced_lengths.mean())
        summary["forced_len_std"]  = float(forced_lengths.std())
    if forced_adp_zscores is not None:
        summary["forced_adp_tpr"]      = float(forced_adp_tpr)
        summary["forced_adp_auc"]      = float(forced_adp_auc)
        summary["forced_adp_z_mean"]   = float(forced_adp_zscores.mean())
        summary["forced_adp_z_std"]    = float(forced_adp_zscores.std())
        summary["forced_adp_len_mean"] = float(forced_adp_lengths.mean())
        summary["forced_adp_len_std"]  = float(forced_adp_lengths.std())
    if oracle_model is not None:
        summary["nw_ppl_mean"] = float(np.nanmean(nw_ppls))
        summary["w_ppl_mean"]  = float(np.nanmean(w_ppls))
        if adp_ppls        is not None: summary["adp_ppl_mean"]        = float(np.nanmean(adp_ppls))
        if forced_ppls     is not None: summary["forced_ppl_mean"]     = float(np.nanmean(forced_ppls))
        if forced_adp_ppls is not None: summary["forced_adp_ppl_mean"] = float(np.nanmean(forced_adp_ppls))
        # Per-record arrays for entropy-split analysis in the notebook
        summary["nw_ppls"]     = [float(x) for x in nw_ppls]
        summary["w_ppls"]      = [float(x) for x in w_ppls]
        if adp_ppls        is not None: summary["adp_ppls"]        = [float(x) for x in adp_ppls]
        if forced_ppls     is not None: summary["forced_ppls"]     = [float(x) for x in forced_ppls]
        if forced_adp_ppls is not None: summary["forced_adp_ppls"] = [float(x) for x in forced_adp_ppls]

    out_path = Path(args.input_file).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
