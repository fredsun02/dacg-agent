#!/usr/bin/env python3
"""Offline alternative stopping strategies (Exp B) on pooled-201 trajectories.

Aligned with kgsa-v3 paper: is_correct() treats Uncertain as correct for NoEffect.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/home/thu/DRKG/KGSA")
TEST_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json"
VAL_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_val/trajectories_20260305_222147.json"
TEST_TAGGED = BASE / "Stage5_Agent/evaluation/redesign/tagged_queries.json"
VAL_TAGGED = BASE / "Stage5_Agent/evaluation/redesign_val/tagged_queries.json"
OUT_RESULTS = BASE / "Stage5_Agent/evaluation/redesign/results"


# ── Label helpers (matching compute_trajectory_metrics.py) ────────────────

def norm_pred(label):
    if not label:
        return 'Uncertain'
    l = str(label).strip().lower()
    if l in ('beneficial', 'treat', 'positive'):
        return 'Beneficial'
    if l in ('harmful', 'negative', 'cause'):
        return 'Harmful'
    if l in ('noeffect', 'no_effect', 'no effect', 'neutral'):
        return 'NoEffect'
    if l in ('noevidence', 'no_evidence', 'uncertain', 'unknown'):
        return 'Uncertain'
    return str(label)


def is_correct(pred, gt):
    pred = norm_pred(pred)
    if pred in ('NoEvidence', 'Uncertain'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def step_conclusion(step):
    return step.get('conclusion', 'Uncertain')


# ── Data loading ──────────────────────────────────────────────────────────

def _ids_201():
    with open(TEST_TAGGED) as f:
        t = json.load(f)
    with open(VAL_TAGGED) as f:
        v = json.load(f)
    ids = set()
    for q in t + v:
        if q['ground_truth'] in ('Beneficial', 'NoEffect') and q.get('gold_label') == q['ground_truth']:
            ids.add(q['id'])
    return ids


def _filtered_trajs():
    with open(TEST_TRAJ) as f:
        t = json.load(f)
    with open(VAL_TRAJ) as f:
        v = json.load(f)
    ids = _ids_201()
    return [tr for tr in t + v if tr['query_id'] in ids and tr.get('history')]


def _ever_correct_up_to(hist, stop_idx, gt):
    for j in range(stop_idx + 1):
        if is_correct(step_conclusion(hist[j]), gt):
            return True
    return False


def normalize_posterior(scores):
    b = max(0.0, float(scores.get("beneficial", 0) or 0))
    h = max(0.0, float(scores.get("harmful", 0) or 0))
    n = max(0.0, float(scores.get("neutral", 0) or 0))
    t = b + h + n
    if t <= 0:
        return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}
    return {"beneficial": b/t, "harmful": h/t, "neutral": n/t}


# ── Metrics ───────────────────────────────────────────────────────────────

def init_metrics():
    return {"total": 0, "correct": 0, "steps_sum": 0,
            "drift": 0, "drift_eligible": 0,
            "class_total": Counter(), "class_correct": Counter()}


def finalize_metrics(m):
    total = m["total"]
    noe_t, ben_t = m["class_total"]["NoEffect"], m["class_total"]["Beneficial"]
    return {
        "overall_acc": round((m["correct"] / total * 100) if total else 0, 2),
        "noe_acc": round((m["class_correct"]["NoEffect"] / noe_t * 100) if noe_t else 0, 2),
        "ben_acc": round((m["class_correct"]["Beneficial"] / ben_t * 100) if ben_t else 0, 2),
        "drift_rate": round((m["drift"] / m["drift_eligible"] * 100) if m["drift_eligible"] else 0, 2),
        "mean_steps": round(m["steps_sum"] / total if total else 0, 2),
        "n": total,
    }


# ── Stopping strategies ──────────────────────────────────────────────────

def _entropy(post):
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in post.values())


def _margin(post):
    vals = sorted(post.values(), reverse=True)
    return vals[0] - vals[1]


def _has_real_conclusion(s):
    return norm_pred(step_conclusion(s)) not in ("Uncertain", "")


def _find_stop_entropy(hist, tau_h):
    for i, s in enumerate(hist):
        if not _has_real_conclusion(s):
            continue
        post = normalize_posterior(s.get("posterior_scores") or {})
        if _entropy(post) < tau_h:
            return i
    return len(hist) - 1


def _find_stop_margin(hist, tau_m):
    for i, s in enumerate(hist):
        if not _has_real_conclusion(s):
            continue
        post = normalize_posterior(s.get("posterior_scores") or {})
        if _margin(post) > tau_m:
            return i
    return len(hist) - 1


def _find_stop_patience(hist, n):
    count, prev = 0, None
    for i, s in enumerate(hist):
        label = norm_pred(step_conclusion(s))
        if label in ("Uncertain", ""):
            count = 0
            prev = None
            continue
        if label == prev:
            count += 1
        else:
            count = 1
            prev = label
        if count >= n:
            return i
    return len(hist) - 1


STRATEGY_FN = {
    "entropy": _find_stop_entropy,
    "margin": _find_stop_margin,
    "patience": _find_stop_patience,
}


def evaluate(trajs, strategy, threshold):
    find_stop = STRATEGY_FN[strategy]
    m = init_metrics()
    for tr in trajs:
        hist = tr["history"]
        gt = tr["ground_truth"]
        stop_idx = max(0, min(find_stop(hist, threshold), len(hist) - 1))
        pred_raw = step_conclusion(hist[stop_idx])
        correct = is_correct(pred_raw, gt)
        ever_correct = _ever_correct_up_to(hist, stop_idx, gt)

        m["total"] += 1
        m["correct"] += int(correct)
        m["steps_sum"] += stop_idx + 1
        m["class_total"][gt] += 1
        m["class_correct"][gt] += int(correct)
        if ever_correct:
            m["drift_eligible"] += 1
            if not correct:
                m["drift"] += 1
    return finalize_metrics(m)


def pick_best(sweep):
    return max(sweep, key=lambda e: (e["metrics"]["overall_acc"], -e["metrics"]["mean_steps"]))


def pick_step_matched(sweep, target=2.7):
    return min(sweep, key=lambda e: (abs(e["metrics"]["mean_steps"] - target),
                                     -e["metrics"]["overall_acc"]))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    trajs = _filtered_trajs()
    print(f"Pooled: {len(trajs)} trajectories")

    sweeps = {
        "entropy":  [0.3, 0.5, 0.7, 0.9, 1.0],
        "margin":   [0.3, 0.4, 0.5, 0.6, 0.7],
        "patience": [2, 3, 4, 5],
    }

    results = {"meta": {"n": len(trajs)}, "strategies": {}}
    for strategy, thresholds in sweeps.items():
        sweep_rows = []
        for t in thresholds:
            metrics = evaluate(trajs, strategy, t)
            sweep_rows.append({"threshold": t, "metrics": metrics})
        best = pick_best(sweep_rows)
        step_matched = pick_step_matched(sweep_rows)
        results["strategies"][strategy] = {
            "sweep": sweep_rows,
            "best": best,
            "step_matched": step_matched,
        }

    with open(OUT_RESULTS / "alt_stopping.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Strategy':<10} {'Type':<12} {'Thr':>6} {'Acc':>7} {'NoE':>7} {'Ben':>7} "
          f"{'Drift':>7} {'Steps':>6}")
    print("-" * 68)
    for strategy, d in results["strategies"].items():
        for label, entry in [("best", d["best"]), ("step-match", d["step_matched"])]:
            t = entry["threshold"]
            m = entry["metrics"]
            print(f"{strategy:<10} {label:<12} {t:>6} {m['overall_acc']:>6.1f}% {m['noe_acc']:>6.1f}% "
                  f"{m['ben_acc']:>6.1f}% {m['drift_rate']:>6.1f}% {m['mean_steps']:>6.2f}")

    print(f"\nSaved to {OUT_RESULTS / 'alt_stopping.json'}")


if __name__ == "__main__":
    main()
