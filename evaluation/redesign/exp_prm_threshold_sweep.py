#!/usr/bin/env python3
"""Sweep PRM decline threshold α for KL+PRM(online) at ms=0, c=1."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np

BASE = Path("/home/thu/DRKG/KGSA")
TEST_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json"
VAL_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_val/trajectories_20260305_222147.json"
TEST_TAGGED = BASE / "Stage5_Agent/evaluation/redesign/tagged_queries.json"
VAL_TAGGED = BASE / "Stage5_Agent/evaluation/redesign_val/tagged_queries.json"
PRM_CONVERGENCE_THRESHOLD = 0.1

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

def _kl(p, q):
    eps = 1e-12
    return sum(max(eps, p.get(k, 0)) * math.log(max(eps, p.get(k, 0)) / max(eps, q.get(k, 0)))
               for k in ("beneficial", "harmful", "neutral"))

def compute_kl_stable(history, kl_consec=1):
    prev_post = None; below_count = 0; flags = []
    for step in history:
        ev = int(step.get("total_evidence", 0) or 0)
        scores = step.get("posterior_scores") or {}
        if ev < 2:
            prev_post = None; below_count = 0; flags.append(False); continue
        post = _normalize_posterior(scores)
        if prev_post is None:
            prev_post = post; below_count = 0; flags.append(False); continue
        kl_val = _kl(post, prev_post)
        prev_post = post
        if kl_val < 0.01: below_count += 1
        else: below_count = 0
        flags.append(below_count >= kl_consec)
    return flags

def decide_kl_prm_online(hist, kl_stable, alpha=0.3):
    reward_history = []; max_r = float('-inf')
    for i, s in enumerate(hist):
        r = float(s.get("prm_reward", 0) or 0)
        reward_history.append(r); max_r = max(max_r, r)
        if kl_stable[i]: return i, "kl_stable"
        ev = int(s.get("total_evidence", 0) or 0)
        if ev > 0 and r < max_r - alpha: return i, "prm_decline"
        if ev >= 3 and i >= 4 and len(reward_history) >= 4:
            rec = reward_history[-4:]
            if max(rec) - min(rec) < PRM_CONVERGENCE_THRESHOLD: return i, "prm_converged"
        if ev == 0 and i >= 6: return i, "no_evidence"
    return len(hist) - 1, "max_steps"

def evaluate(trajs, alpha):
    m = {"total": 0, "correct": 0, "steps_sum": 0, "drift": 0, "drift_elig": 0,
         "class_total": Counter(), "class_correct": Counter(), "stop_reasons": Counter()}
    for tr in trajs:
        hist = tr["history"]; gt = tr["ground_truth"]
        kl_stable = compute_kl_stable(hist)
        stop_idx, reason = decide_kl_prm_online(hist, kl_stable, alpha=alpha)
        stop_idx = max(0, min(stop_idx, len(hist) - 1))
        pred = step_conclusion(hist[stop_idx])
        correct = is_correct(pred, gt)
        ever = any(is_correct(step_conclusion(hist[j]), gt) for j in range(stop_idx + 1))
        m["total"] += 1; m["correct"] += int(correct); m["steps_sum"] += stop_idx + 1
        m["class_total"][gt] += 1; m["class_correct"][gt] += int(correct)
        m["stop_reasons"][reason] += 1
        if ever:
            m["drift_elig"] += 1
            if not correct: m["drift"] += 1
    t = m["total"]; noe_t = m["class_total"]["NoEffect"]
    return {
        "acc": round(m["correct"]/t*100, 2),
        "noe": round(m["class_correct"]["NoEffect"]/noe_t*100, 2) if noe_t else 0,
        "drift": round(m["drift"]/m["drift_elig"]*100, 2) if m["drift_elig"] else 0,
        "steps": round(m["steps_sum"]/t, 2),
        "reasons": dict(m["stop_reasons"]),
    }

def main():
    trajs = _filtered_trajs()
    print(f"n={len(trajs)}\n")

    # Sweep PRM decline threshold α
    print("=== PRM decline threshold α sweep (KL+PRM online, ms=0, c=1) ===")
    print(f"{'α':<8} {'Acc':>7} {'NoE':>7} {'Drift':>7} {'Steps':>6}  {'kl':>4} {'prm_d':>6} {'prm_c':>6} {'max':>4}")
    print("-" * 70)
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 999]:
        r = evaluate(trajs, alpha)
        rs = r["reasons"]
        label = f"{alpha:.2f}" if alpha < 100 else "∞(off)"
        print(f"{label:<8} {r['acc']:>6.1f}% {r['noe']:>6.1f}% {r['drift']:>6.1f}% {r['steps']:>5.2f}  "
              f"{rs.get('kl_stable',0):>4} {rs.get('prm_decline',0):>6} {rs.get('prm_converged',0):>6} {rs.get('max_steps',0):>4}")

    # Also sweep KL threshold δ at ms=0, c=1 for completeness
    print("\n=== KL threshold δ sweep (KL+PRM online, ms=0, c=1, α=0.3) ===")
    print(f"{'δ':<10} {'Acc':>7} {'NoE':>7} {'Drift':>7} {'Steps':>6}  {'kl':>4} {'prm_d':>6}")
    print("-" * 60)
    for delta in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        m = {"total": 0, "correct": 0, "steps_sum": 0, "drift": 0, "drift_elig": 0,
             "class_total": Counter(), "class_correct": Counter(), "stop_reasons": Counter()}
        for tr in trajs:
            hist = tr["history"]; gt = tr["ground_truth"]
            # Custom KL with different threshold
            prev_post = None; flags = []
            for step in hist:
                ev = int(step.get("total_evidence", 0) or 0)
                scores = step.get("posterior_scores") or {}
                if ev < 2:
                    prev_post = None; flags.append(False); continue
                post = _normalize_posterior(scores)
                if prev_post is None:
                    prev_post = post; flags.append(False); continue
                kl_val = _kl(post, prev_post)
                prev_post = post
                flags.append(kl_val < delta)
            stop_idx, reason = decide_kl_prm_online(hist, flags, alpha=0.3)
            stop_idx = max(0, min(stop_idx, len(hist) - 1))
            pred = step_conclusion(hist[stop_idx])
            correct = is_correct(pred, gt)
            ever = any(is_correct(step_conclusion(hist[j]), gt) for j in range(stop_idx + 1))
            m["total"] += 1; m["correct"] += int(correct); m["steps_sum"] += stop_idx + 1
            m["class_total"][gt] += 1; m["class_correct"][gt] += int(correct)
            m["stop_reasons"][reason] += 1
            if ever:
                m["drift_elig"] += 1
                if not correct: m["drift"] += 1
        t = m["total"]; noe_t = m["class_total"]["NoEffect"]
        acc = round(m["correct"]/t*100, 2)
        noe = round(m["class_correct"]["NoEffect"]/noe_t*100, 2) if noe_t else 0
        drift = round(m["drift"]/m["drift_elig"]*100, 2) if m["drift_elig"] else 0
        steps = round(m["steps_sum"]/t, 2)
        rs = dict(m["stop_reasons"])
        print(f"{delta:<10.3f} {acc:>6.1f}% {noe:>6.1f}% {drift:>6.1f}% {steps:>5.2f}  "
              f"{rs.get('kl_stable',0):>4} {rs.get('prm_decline',0):>6}")

if __name__ == "__main__":
    main()
