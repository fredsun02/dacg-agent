#!/usr/bin/env python3
"""
Build PRM preference pairs using GT-posterior continuous reward.

Reads trajectories from run_ablation.py (recording mode), computes
gt_reward = posterior_scores[gt_label_key] at each step, and generates
Bradley-Terry preference pairs based on reward differences.

Usage:
    python build_preferences_v2.py \
        --trajectories /path/to/trajectories_YYYYMMDD_HHMMSS.json \
        --output-dir /path/to/prm_data_v2/
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph_features import GRAPH_FEATURE_NAMES, GRAPH_FEATURE_NORMS

GT_KEY_MAP = {
    "beneficial": "beneficial",
    "harmful": "harmful",
    "noeffect": "neutral",
    "neutral": "neutral",
}

# Training-side filtering thresholds
MIN_PEAK_REWARD = 0.3       # discard trajectories with peak GT reward below this
STABLE_WINDOW_BONUS = 0.5   # extra margin multiplier for pairs from stable GT-correct windows


def _norm_label(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _gt_key(label: str) -> Optional[str]:
    return GT_KEY_MAP.get(_norm_label(label))


def _load_trajectories(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("trajectories"), list):
        return data["trajectories"]
    return []


def _build_records(history: list, gt_key: str) -> list:
    records = []
    for idx, snap in enumerate(history):
        posterior = snap.get("posterior_scores") or {}
        reward = posterior.get(gt_key)
        if reward is None and posterior:
            # Fallback: try canonical key mapping
            for k, v in posterior.items():
                if _gt_key(k) == gt_key or _norm_label(k) == gt_key:
                    reward = v
                    break
        if reward is None:
            continue
        conclusion = snap.get("conclusion", "")
        records.append({
            "step": snap.get("step", idx + 1),
            "features_dict": snap.get("graph_features") or {},
            "conclusion_norm": _norm_label(conclusion),
            "gt_reward": float(reward),
        })
    return records


def _find_peak_index(records: list) -> int:
    """Return index of peak GT reward."""
    best_i, best_r = 0, -1.0
    for i, r in enumerate(records):
        if r["gt_reward"] > best_r:
            best_r = r["gt_reward"]
            best_i = i
    return best_i


def _find_stable_window(records: list, gt_norm: str) -> tuple:
    """Find longest contiguous window where conclusion == GT. Returns (start, end) indices."""
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, r in enumerate(records):
        if r["conclusion_norm"] == gt_norm:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    return best_start, best_start + best_len


def _generate_preferences(traj: dict, prefix: str, delta: float, counters: dict,
                          truncate: bool = True) -> list:
    gt_key = _gt_key(traj.get("ground_truth", ""))
    if not gt_key:
        return []
    gt_norm = _norm_label(traj.get("ground_truth", ""))
    records = _build_records(traj.get("history", []), gt_key)
    records.sort(key=lambda r: r["step"])
    if len(records) < 2:
        return []

    # --- Filter 1: discard low-signal trajectories ---
    peak_reward = max(r["gt_reward"] for r in records)
    if peak_reward < MIN_PEAK_REWARD:
        counters["filtered"] += 1
        return []

    # --- Filter 2: truncate post-peak pollution ---
    # Keep records up to peak + small buffer (2 steps) to capture the decay onset
    if truncate:
        peak_idx = _find_peak_index(records)
        cut = min(peak_idx + 3, len(records))
        records_full = records          # keep full for stable_vs_unstable
        records = records[:cut]
    else:
        records_full = records

    # Find stable GT-correct window for bonus margin
    win_start, win_end = _find_stable_window(records_full, gt_norm)
    stable_steps = set(records_full[i]["step"] for i in range(win_start, win_end))

    query_id = traj.get("query_id", "q")
    gt_label = traj.get("ground_truth", "Unknown")
    counter = [0]
    prefs = []

    def add_pair(better, worse, ptype):
        margin = better["gt_reward"] - worse["gt_reward"]
        if margin <= 0:
            return
        # Bonus: pairs where better is inside a stable GT window get higher margin
        if better["step"] in stable_steps:
            margin *= (1.0 + STABLE_WINDOW_BONUS)
        counter[0] += 1
        prefs.append({
            "id": f"{prefix}_{query_id}_{counter[0]:04d}",
            "query_id": query_id,
            "state_better": better["features_dict"],
            "state_worse": worse["features_dict"],
            "preference_type": ptype,
            "margin": margin,
        })
        counters["type"][ptype] += 1
        counters["gt"][gt_label] += 1

    # 1. Peak vs others: best GT-reward step vs all others
    peak = max(records, key=lambda r: r["gt_reward"])
    for rec in records:
        if rec is peak:
            continue
        if (peak["gt_reward"] - rec["gt_reward"]) > delta:
            add_pair(peak, rec, "peak_vs_others")

    # 2. Monotonic pairs: nearby steps with reward difference
    window = 5
    for i in range(len(records)):
        for j in range(i + 1, min(len(records), i + window + 1)):
            ri, rj = records[i], records[j]
            if ri["gt_reward"] > rj["gt_reward"] + delta:
                add_pair(ri, rj, "monotonic_pairs")
            elif rj["gt_reward"] > ri["gt_reward"] + delta:
                add_pair(rj, ri, "monotonic_pairs")

    # 3. Correct vs incorrect (nearby steps, use truncated records)
    correct = [r for r in records if r["conclusion_norm"] == gt_norm]
    incorrect = [r for r in records if r["conclusion_norm"] != gt_norm]
    for c in correct[:5]:
        for ic in incorrect[:5]:
            if abs(c["step"] - ic["step"]) <= 5:
                add_pair(c, ic, "correct_vs_incorrect")

    # 4. Stable vs unstable: correct conclusion that flips (use full trajectory)
    for i in range(len(records_full) - 1):
        curr, nxt = records_full[i], records_full[i + 1]
        if nxt["step"] - curr["step"] != 1:
            continue
        if curr["conclusion_norm"] == gt_norm and nxt["conclusion_norm"] != curr["conclusion_norm"]:
            add_pair(curr, nxt, "stable_vs_unstable")

    return prefs


def main():
    parser = argparse.ArgumentParser(description="Build PRM preferences using GT posterior reward")
    parser.add_argument("--trajectories", required=True, help="Trajectories JSON from run_ablation.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for preference files")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--no-truncate", dest="truncate", action="store_false", default=True,
                        help="Disable post-peak truncation")
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = _load_trajectories(traj_path)
    if not trajectories:
        print(f"No trajectories found in {traj_path}")
        return

    # Split
    rng = random.Random(args.seed)
    rng.shuffle(trajectories)
    n = len(trajectories)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + (1 - args.train_ratio) / 2))

    splits = {
        "train": trajectories[:train_end],
        "val": trajectories[train_end:val_end],
        "test": trajectories[val_end:],
    }

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "feature_names": GRAPH_FEATURE_NAMES,
        "feature_norms": GRAPH_FEATURE_NORMS,
        "feature_dim": len(GRAPH_FEATURE_NAMES),
        "source": str(traj_path),
        "total_trajectories": n,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "delta": args.delta,
        "method": "gt_posterior_continuous",
        "truncate": args.truncate,
        "min_peak_reward": MIN_PEAK_REWARD,
        "stable_window_bonus": STABLE_WINDOW_BONUS,
    }

    counters = {"type": Counter(), "gt": Counter(), "filtered": 0}
    all_prefs = {}
    for split_name, split_trajs in splits.items():
        prefs = []
        for traj in split_trajs:
            prefs.extend(_generate_preferences(traj, split_name, args.delta, counters,
                                               truncate=args.truncate))
        all_prefs[split_name] = prefs

        out_path = output_dir / f"{split_name}_preferences.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "preferences": prefs}, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in all_prefs.values())
    print(f"\nPreferences saved to {output_dir}/")
    for split_name, prefs in all_prefs.items():
        print(f"  {split_name}: {len(prefs)}")
    print(f"  Total: {total}")

    print("\n  Type distribution:")
    for ptype, count in counters["type"].most_common():
        print(f"    {ptype}: {count}")

    print("\n  GT distribution:")
    for gt_label, count in counters["gt"].most_common():
        print(f"    {gt_label}: {count}")

    if counters["filtered"]:
        print(f"\n  Filtered (peak reward < {MIN_PEAK_REWARD}): {counters['filtered']} trajectories")


if __name__ == "__main__":
    main()
