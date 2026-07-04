#!/usr/bin/env python3
"""Graph-structural stopping baselines:
1. Evidence saturation: stop when new_edges_ratio drops below threshold
2. Graph density: stop when graph_density stops changing
3. Marginal gain: stop when marginal_gain drops below threshold
4. Path discovery: stop when path_discovery_rate drops below threshold
5. Combined: saturation AND density
"""
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

# --- Stopping criteria ---

def decide_saturation(hist, threshold=0.1):
    """Stop when new_edges_ratio drops below threshold (no new edges being added)."""
    for i, s in enumerate(hist):
        if i < 1: continue  # need at least 2 steps
        gf = s.get('graph_features', {})
        ner = float(gf.get('new_edges_ratio', 1.0))
        if ner <= threshold: return i, "saturated"
    return len(hist) - 1, "max_steps"

def decide_path_discovery(hist, threshold=0.5):
    """Stop when path_discovery_rate drops below threshold."""
    for i, s in enumerate(hist):
        if i < 1: continue
        gf = s.get('graph_features', {})
        pdr = float(gf.get('path_discovery_rate', 1.0))
        if pdr < threshold: return i, "no_new_paths"
    return len(hist) - 1, "max_steps"

def decide_marginal_gain(hist, threshold=0.5):
    """Stop when marginal_gain drops below threshold."""
    for i, s in enumerate(hist):
        if i < 1: continue
        gf = s.get('graph_features', {})
        mg = float(gf.get('marginal_gain', 1.0))
        if mg < threshold: return i, "low_gain"
    return len(hist) - 1, "max_steps"

def decide_density_stable(hist, threshold=0.05):
    """Stop when graph_density change between steps is below threshold."""
    prev_density = None
    for i, s in enumerate(hist):
        gf = s.get('graph_features', {})
        density = float(gf.get('graph_density', 0.0))
        if prev_density is not None and i >= 1:
            change = abs(density - prev_density)
            if change < threshold: return i, "density_stable"
        prev_density = density
    return len(hist) - 1, "max_steps"

def decide_evidence_plateau(hist, window=3, threshold=0.1):
    """Stop when total_evidence growth rate drops (plateau)."""
    evidences = []
    for i, s in enumerate(hist):
        gf = s.get('graph_features', {})
        ev = float(gf.get('total_evidence', 0))
        evidences.append(ev)
        if len(evidences) >= int(window) + 1 and i >= 2:
            prev = evidences[-(int(window)+1)]
            curr = evidences[-1]
            if prev > 0:
                growth = (curr - prev) / prev
                if growth < threshold: return i, "evidence_plateau"
    return len(hist) - 1, "max_steps"

# --- Evaluation ---

def evaluate(trajs, decide_fn):
    m = {"total": 0, "correct": 0, "steps_sum": 0, "drift": 0, "drift_elig": 0,
         "class_total": Counter(), "class_correct": Counter(), "reasons": Counter()}
    for tr in trajs:
        hist = tr["history"]; gt = tr["ground_truth"]
        stop_idx, reason = decide_fn(hist)
        stop_idx = max(0, min(stop_idx, len(hist) - 1))
        pred = step_conclusion(hist[stop_idx])
        correct = is_correct(pred, gt)
        ever = any(is_correct(step_conclusion(hist[j]), gt) for j in range(stop_idx + 1))
        m["total"] += 1; m["correct"] += int(correct); m["steps_sum"] += stop_idx + 1
        m["class_total"][gt] += 1; m["class_correct"][gt] += int(correct)
        m["reasons"][reason] += 1
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
        "reasons": dict(m["reasons"]),
    }

def main():
    trajs = _filtered_trajs()
    print(f"n={len(trajs)}\n")

    # 1. Evidence saturation (new_edges_ratio)
    print("=== Evidence Saturation (new_edges_ratio <= threshold) ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}  {'stopped':>7} {'max':>4}")
    print("-" * 70)
    for th in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        r = evaluate(trajs, lambda h, t=th: decide_saturation(h, t))
        rs = r["reasons"]
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.2f}  {rs.get('saturated',0):>7} {rs.get('max_steps',0):>4}")

    # 2. Path discovery rate
    print("\n=== Path Discovery Rate (< threshold) ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}  {'stopped':>7} {'max':>4}")
    print("-" * 70)
    for th in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        r = evaluate(trajs, lambda h, t=th: decide_path_discovery(h, t))
        rs = r["reasons"]
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.02f}  {rs.get('no_new_paths',0):>7} {rs.get('max_steps',0):>4}")

    # 3. Marginal gain
    print("\n=== Marginal Gain (< threshold) ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}  {'stopped':>7} {'max':>4}")
    print("-" * 70)
    for th in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        r = evaluate(trajs, lambda h, t=th: decide_marginal_gain(h, t))
        rs = r["reasons"]
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.02f}  {rs.get('low_gain',0):>7} {rs.get('max_steps',0):>4}")

    # 4. Density stability
    print("\n=== Density Stability (|Δdensity| < threshold) ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}  {'stopped':>7} {'max':>4}")
    print("-" * 70)
    for th in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        r = evaluate(trajs, lambda h, t=th: decide_density_stable(h, t))
        rs = r["reasons"]
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.02f}  {rs.get('density_stable',0):>7} {rs.get('max_steps',0):>4}")

    # 5. Evidence plateau
    print("\n=== Evidence Plateau (growth < threshold over window=3) ===")
    print(f"{'threshold':<12} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}  {'stopped':>7} {'max':>4}")
    print("-" * 70)
    for th in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        r = evaluate(trajs, lambda h, t=th: decide_evidence_plateau(h, t))
        rs = r["reasons"]
        print(f"{th:<12.2f} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['ben']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.02f}  {rs.get('evidence_plateau',0):>7} {rs.get('max_steps',0):>4}")

    print("\n=== Reference ===")
    print("DACG (KL+PRM):  60.2%   73.5%   50.9%   14.8%  3.56")
    print("Entropy (0.5):   63.2%   75.9%   54.2%   13.0%  3.55")

if __name__ == "__main__":
    main()
