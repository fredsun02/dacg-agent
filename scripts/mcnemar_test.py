#!/usr/bin/env python3
"""McNemar test: Conf+PRM vs each baseline (paired 2x2 table)."""

import json, argparse, numpy as np
from pathlib import Path
from scipy.stats import binom as binom_dist


def is_correct(pred, gt):
    if pred in ('NoEvidence', 'Uncertain', 'Unknown'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def find_oracle_step(history, gt):
    for i, s in enumerate(history):
        if is_correct(s.get('conclusion', ''), gt):
            return i + 1
    return None


def apply_stopping(history, config):
    if config.startswith('fixed_k_'):
        k = int(config.split('_')[-1])
        idx = min(k, len(history)) - 1
        return idx + 1, history[idx]
    if config == 'conf_only':
        for i, s in enumerate(history):
            if float(s.get('confidence', 0) or 0) > 0.8:
                return i + 1, s
        return len(history), history[-1]
    if config == 'kl_only':
        kls = [float(s.get('kl', 0) or 0) for s in history]
        for i in range(3, len(kls)):
            if all(kls[j] < 0.05 for j in range(i - 2, i + 1)):
                return i + 1, history[i]
        return len(history), history[-1]
    if config == 'conf_then_prm':
        for i, s in enumerate(history):
            if float(s.get('confidence', 0) or 0) > 0.8:
                return i + 1, s
        rewards = [float(s.get('prm_reward', 0) or 0) for s in history]
        if rewards:
            peak_idx = int(np.argmax(rewards))
            return peak_idx + 1, history[peak_idx]
        return len(history), history[-1]
    raise ValueError(f"Unknown config: {config}")


def mcnemar_exact(b, c):
    """Exact McNemar test (two-sided) using binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0, 1.0
    k = min(b, c)
    p_value = 2 * binom_dist.cdf(k, n, 0.5)
    p_value = min(p_value, 1.0)
    odds_ratio = b / c if c > 0 else float('inf')
    return p_value, odds_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", default="Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json")
    p.add_argument("--tagged", default="Stage5_Agent/evaluation/redesign/tagged_queries.json")
    p.add_argument("--e1", default="Stage5_Agent/evaluation/redesign/e1_zeroshot.json")
    p.add_argument("--e2", default="Stage5_Agent/evaluation/redesign/e2_rag.json")
    p.add_argument("--output", default="Stage5_Agent/evaluation/redesign/mcnemar_results.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.trajectories) as f:
        trajectories = json.load(f)
    with open(root / args.tagged) as f:
        tagged = json.load(f)
    clean_ids = {q["id"] for q in tagged if q.get("in_clean_103")}
    traj_map = {t["query_id"]: t for t in trajectories if t.get("history")}

    # Build Conf+PRM correctness vector
    ref_config = "conf_then_prm"
    ref_correct = {}
    for traj in trajectories:
        qid = traj["query_id"]
        if not traj.get("history") or qid not in clean_ids:
            continue
        _, snap = apply_stopping(traj["history"], ref_config)
        ref_correct[qid] = is_correct(snap.get("conclusion", ""), traj["ground_truth"])

    # Internal baselines
    internal_configs = ["fixed_k_3", "fixed_k_5", "fixed_k_10", "fixed_k_20",
                        "conf_only", "kl_only"]
    results = {}
    for cfg in internal_configs:
        b, c, n11, n00 = 0, 0, 0, 0
        for traj in trajectories:
            qid = traj["query_id"]
            if not traj.get("history") or qid not in clean_ids:
                continue
            _, snap = apply_stopping(traj["history"], cfg)
            base_ok = is_correct(snap.get("conclusion", ""), traj["ground_truth"])
            ref_ok = ref_correct[qid]
            if ref_ok and not base_ok: b += 1
            elif not ref_ok and base_ok: c += 1
            elif ref_ok and base_ok: n11 += 1
            else: n00 += 1
        pval, odds = mcnemar_exact(b, c)
        results[f"conf_prm_vs_{cfg}"] = {
            "table": {"both_correct": n11, "ref_only": b, "base_only": c, "both_wrong": n00},
            "p_value": round(pval, 4), "odds_ratio": round(odds, 3),
            "n": n11 + b + c + n00
        }
        sig = "*" if pval < 0.05 else ""
        print(f"Conf+PRM vs {cfg:16s}: b={b:3d} c={c:3d} p={pval:.4f}{sig} OR={odds:.3f}")

    # External baselines (E1, E2) if available
    for label, path in [("e1_zeroshot", args.e1), ("e2_rag", args.e2)]:
        fpath = root / path
        if not fpath.exists():
            print(f"  {label}: file not found, skipping")
            continue
        with open(fpath) as f:
            ext_data = json.load(f)
        ext_map = {r["id"]: r["correct"] for r in ext_data.get("results", [])}
        b, c, n11, n00 = 0, 0, 0, 0
        for qid in ref_correct:
            if qid not in ext_map:
                continue
            ref_ok = ref_correct[qid]
            base_ok = ext_map[qid]
            if ref_ok and not base_ok: b += 1
            elif not ref_ok and base_ok: c += 1
            elif ref_ok and base_ok: n11 += 1
            else: n00 += 1
        pval, odds = mcnemar_exact(b, c)
        results[f"conf_prm_vs_{label}"] = {
            "table": {"both_correct": n11, "ref_only": b, "base_only": c, "both_wrong": n00},
            "p_value": round(pval, 4), "odds_ratio": round(odds, 3),
            "n": n11 + b + c + n00
        }
        sig = "*" if pval < 0.05 else ""
        print(f"Conf+PRM vs {label:16s}: b={b:3d} c={c:3d} p={pval:.4f}{sig} OR={odds:.3f}")

    out_path = root / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
