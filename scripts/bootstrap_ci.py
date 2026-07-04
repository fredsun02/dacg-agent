#!/usr/bin/env python3
"""Bootstrap 95% CI for accuracy, regret, and drift across stopping configs.
Uses the same stopping logic as compute_trajectory_metrics.py."""

import json, argparse, numpy as np
from collections import defaultdict
from pathlib import Path


def is_correct(pred, gt):
    if pred in ('NoEvidence', 'Uncertain'):
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
        # Confidence first
        for i, s in enumerate(history):
            if float(s.get('confidence', 0) or 0) > 0.8:
                return i + 1, s
        # PRM peak fallback
        rewards = [float(s.get('prm_reward', 0) or 0) for s in history]
        if rewards:
            peak_idx = int(np.argmax(rewards))
            return peak_idx + 1, history[peak_idx]
        return len(history), history[-1]

    raise ValueError(f"Unknown config: {config}")


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(values)
    arr = np.array(values, dtype=float)
    boot_means = np.array([arr[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(arr)), float(lo), float(hi)


def compute_per_query(trajectories, clean_ids, config_name):
    per_query = {"clean": [], "full": []}
    for traj in trajectories:
        qid = traj["query_id"]
        gt = traj["ground_truth"]
        hist = traj["history"]
        if not hist:
            continue

        step, snap = apply_stopping(hist, config_name)
        pred = snap.get("conclusion", "")
        correct = int(is_correct(pred, gt))

        oracle_step = find_oracle_step(hist, gt)
        regret = abs(step - oracle_step) if oracle_step else step

        # Drift: had correct step at or before stop, but stopped wrong
        ever_correct_before_stop = (oracle_step is not None and oracle_step <= step)
        drifted = int(ever_correct_before_stop and not is_correct(pred, gt))

        entry = {"correct": correct, "regret": regret, "drifted": drifted,
                 "steps": step, "ever_correct_before_stop": int(ever_correct_before_stop)}
        per_query["full"].append(entry)
        if qid in clean_ids:
            per_query["clean"].append(entry)

    return per_query


def compute_cis(entries, n_boot=10000):
    if not entries:
        return {}
    acc_mean, acc_lo, acc_hi = bootstrap_ci([e["correct"] for e in entries], n_boot)
    reg_mean, reg_lo, reg_hi = bootstrap_ci([e["regret"] for e in entries], n_boot)
    eligible = [e for e in entries if e["ever_correct_before_stop"]]
    if eligible:
        drift_mean, drift_lo, drift_hi = bootstrap_ci([e["drifted"] for e in eligible], n_boot)
    else:
        drift_mean = drift_lo = drift_hi = 0.0
    steps_mean = np.mean([e["steps"] for e in entries])
    return {
        "n": len(entries),
        "acc": {"mean": round(acc_mean * 100, 1), "ci95": [round(acc_lo * 100, 1), round(acc_hi * 100, 1)]},
        "regret": {"mean": round(reg_mean, 2), "ci95": [round(reg_lo, 2), round(reg_hi, 2)]},
        "drift": {"mean": round(drift_mean * 100, 1), "ci95": [round(drift_lo * 100, 1), round(drift_hi * 100, 1)]},
        "steps": round(float(steps_mean), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", default="Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json")
    p.add_argument("--tagged", default="Stage5_Agent/evaluation/redesign/tagged_queries.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--output", default="Stage5_Agent/evaluation/redesign/bootstrap_cis.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.trajectories) as f:
        trajectories = json.load(f)
    with open(root / args.tagged) as f:
        tagged = json.load(f)

    clean_ids = {q["id"] for q in tagged if q.get("in_clean_103")}

    config_names = ["fixed_k_3", "fixed_k_5", "fixed_k_10", "fixed_k_20",
                    "conf_only", "kl_only", "conf_then_prm"]
    results = {}
    for cfg_name in config_names:
        per_query = compute_per_query(trajectories, clean_ids, cfg_name)
        results[cfg_name] = {
            "clean_103": compute_cis(per_query["clean"], args.n_boot),
            "full_173": compute_cis(per_query["full"], args.n_boot),
        }
        c = results[cfg_name]["clean_103"]
        print(f"{cfg_name:16s}: Clean acc={c['acc']['mean']:5.1f} [{c['acc']['ci95'][0]:5.1f}, {c['acc']['ci95'][1]:5.1f}]  "
              f"drift={c['drift']['mean']:5.1f} [{c['drift']['ci95'][0]:5.1f}, {c['drift']['ci95'][1]:5.1f}]")

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
