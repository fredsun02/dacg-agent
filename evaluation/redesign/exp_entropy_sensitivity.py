#!/usr/bin/env python3
"""Sweep entropy and margin thresholds to test sensitivity."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np

BASE = Path("/home/thu/DRKG/KGSA")
TEST_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json"
VAL_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_val/trajectories_20260305_222147.json"
TEST_TAGGED = BASE / "Stage5_Agent/evaluation/redesign/tagged_queries.json"
VAL_TAGGED = BASE / "Stage5_Agent/evaluation/redesign_val/tagged_queries.json"

def norm_pred(label):
    if not label: return 'Uncertain'
    l = str(label).strip().lower()
    if l in ('beneficial', 'treat', 'positive'): return 'Beneficial'
    if l in ('harmful', 'negative', 'cause'): return 'Harmful'
    if l in ('noeffect', 'no_effect', 'no effect', 'neutral'): return 'NoEffect'
    if l in ('noevidence', 'no_evidence', 'uncertain', 'unknown'): return 'Uncertain'
    return str(label)

def is_correct(pred, gt):
    pred = norm_pred(pred)
    if pred in ('NoEvidence', 'Uncertain'): pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'): return True
    return pred == gt

def step_conclusion(step): return step.get('conclusion', 'Uncertain')

def _ids_201():
    with open(TEST_TAGGED) as f: t = json.load(f)
    with open(VAL_TAGGED) as f: v = json.load(f)
    ids = set()
    for q in t + v:
        if q['ground_truth'] in ('Beneficial', 'NoEffect') and q.get('gold_label') == q['ground_truth']:
            ids.add(q['id'])
    return ids

def _filtered_trajs():
    with open(TEST_TRAJ) as f: t = json.load(f)
    with open(VAL_TRAJ) as f: v = json.load(f)
    ids = _ids_201()
    return [tr for tr in t + v if tr['query_id'] in ids and tr.get('history')]

def _normalize_posterior(scores):
    b = max(0.0, float(scores.get("beneficial", 0) or 0))
    h = max(0.0, float(scores.get("harmful", 0) or 0))
    n = max(0.0, float(scores.get("neutral", 0) or 0))
    t = b + h + n
    if t <= 0: return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}
    return {"beneficial": b/t, "harmful": h/t, "neutral": n/t}

def decide_entropy(hist, threshold):
    for i, s in enumerate(hist):
        ev = int(s.get("total_evidence", 0) or 0)
        if ev < 2: continue
        scores = s.get("posterior_scores", {})
        post = _normalize_posterior(scores)
        vals = [v for v in post.values() if v > 0]
        H = -sum(v * math.log(v + 1e-12) for v in vals)
        if H < threshold: return i, "entropy"
    return len(hist) - 1, "max_steps"

def decide_margin(hist, threshold):
    for i, s in enumerate(hist):
        ev = int(s.get("total_evidence", 0) or 0)
        if ev < 2: continue
        scores = s.get("posterior_scores", {})
        post = _normalize_posterior(scores)
        sorted_v = sorted(post.values(), reverse=True)
        margin = sorted_v[0] - sorted_v[1] if len(sorted_v) >= 2 else sorted_v[0]
        if margin > threshold: return i, "margin"
    return len(hist) - 1, "max_steps"

def evaluate(trajs, decide_fn):
    m = {"total": 0, "correct": 0, "steps_sum": 0, "drift": 0, "drift_elig": 0,
         "class_total": Counter(), "class_correct": Counter()}
    for tr in trajs:
        hist = tr["history"]; gt = tr["ground_truth"]
        stop_idx, _ = decide_fn(hist)
        stop_idx = max(0, min(stop_idx, len(hist) - 1))
        pred = step_conclusion(hist[stop_idx])
        correct = is_correct(pred, gt)
        ever = any(is_correct(step_conclusion(hist[j]), gt) for j in range(stop_idx + 1))
        m["total"] += 1; m["correct"] += int(correct); m["steps_sum"] += stop_idx + 1
        m["class_total"][gt] += 1; m["class_correct"][gt] += int(correct)
        if ever:
            m["drift_elig"] += 1
            if not correct: m["drift"] += 1
    t = m["total"]; noe_t = m["class_total"]["NoEffect"]; ben_t = m["class_total"]["Beneficial"]
    return {
        "acc": round(m["correct"]/t*100, 2),
        "noe": round(m["class_correct"]["NoEffect"]/noe_t*100, 2) if noe_t else 0,
        "ben": round(m["class_correct"]["Beneficial"]/ben_t*100, 2) if ben_t else 0,
        "drift": round(m["drift"]/m["drift_elig"]*100, 2) if m["drift_elig"] else 0,
        "steps": round(m["steps_sum"]/t, 2),
    }

def main():
    trajs = _filtered_trajs()
    print(f"n={len(trajs)}\n")

    # Entropy sweep
    print("=== Entropy threshold sweep ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}")
    print("-" * 55)
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05]:
        r = evaluate(trajs, lambda h, t=th: decide_entropy(h, t))
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.2f}")

    # Margin sweep
    print("\n=== Margin threshold sweep ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}")
    print("-" * 55)
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
        r = evaluate(trajs, lambda h, t=th: decide_margin(h, t))
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.2f}")

    # Reference: DACG (KL+PRM)
    print("\n=== Reference: DACG (KL+PRM) ===")
    print("             60.2%   73.5%   50.9%   14.8%  3.56")

if __name__ == "__main__":
    main()
