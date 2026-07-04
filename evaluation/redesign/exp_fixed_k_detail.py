#!/usr/bin/env python3
"""Detailed fixed-k vs KL+PRM comparison with per-class breakdown."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np

BASE = Path("/home/thu/DRKG/KGSA")
TEST_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json"
VAL_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_val/trajectories_20260305_222147.json"
TEST_TAGGED = BASE / "Stage5_Agent/evaluation/redesign/tagged_queries.json"
VAL_TAGGED = BASE / "Stage5_Agent/evaluation/redesign_val/tagged_queries.json"

PRM_DECLINE_THRESHOLD = 0.3; PRM_CONVERGENCE_THRESHOLD = 0.1

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

def compute_kl_stable(history, kl_consec=2):
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

def decide_fixed_k(hist, k):
    return min(k, len(hist)) - 1, "fixed_k"

def decide_kl_only(hist, kl_stable, min_steps=0):
    for i in range(len(hist)):
        if i < min_steps: continue
        if kl_stable[i]: return i, "kl_stable"
    return len(hist) - 1, "max_steps"

def decide_prm_online_only(hist, min_steps=0):
    reward_history = []; max_r = float('-inf')
    for i, s in enumerate(hist):
        r = float(s.get("prm_reward", 0) or 0)
        reward_history.append(r); max_r = max(max_r, r)
        if i < min_steps: continue
        ev = int(s.get("total_evidence", 0) or 0)
        if ev > 0 and r < max_r - PRM_DECLINE_THRESHOLD: return i, "prm_decline"
        if ev >= 3 and i >= 4 and len(reward_history) >= 4:
            rec = reward_history[-4:]
            if max(rec) - min(rec) < PRM_CONVERGENCE_THRESHOLD: return i, "prm_converged"
        if ev == 0 and i >= max(min_steps + 3, 6): return i, "no_evidence"
    return len(hist) - 1, "max_steps"

def decide_kl_prm_online(hist, kl_stable, min_steps=0):
    reward_history = []; max_r = float('-inf')
    for i, s in enumerate(hist):
        r = float(s.get("prm_reward", 0) or 0)
        reward_history.append(r); max_r = max(max_r, r)
        if i < min_steps: continue
        if kl_stable[i]: return i, "kl_stable"
        ev = int(s.get("total_evidence", 0) or 0)
        if ev > 0 and r < max_r - PRM_DECLINE_THRESHOLD: return i, "prm_decline"
        if ev >= 3 and i >= 4 and len(reward_history) >= 4:
            rec = reward_history[-4:]
            if max(rec) - min(rec) < PRM_CONVERGENCE_THRESHOLD: return i, "prm_converged"
        if ev == 0 and i >= max(min_steps + 3, 6): return i, "no_evidence"
    return len(hist) - 1, "max_steps"

def decide_oracle(hist, gt):
    for i, s in enumerate(hist):
        if is_correct(step_conclusion(s), gt): return i, "oracle"
    return len(hist) - 1, "never_correct"


def evaluate_detailed(trajs, decide_fn):
    """Per-class metrics + regret."""
    classes = ['Beneficial', 'NoEffect']
    m = {c: {"total": 0, "correct": 0, "steps": 0, "drift": 0, "drift_elig": 0, "regrets": []}
         for c in classes}
    m["all"] = {"total": 0, "correct": 0, "steps": 0, "drift": 0, "drift_elig": 0, "regrets": []}

    for tr in trajs:
        hist = tr["history"]; gt = tr["ground_truth"]
        kl_stable = compute_kl_stable(hist, kl_consec=2)
        stop_idx, reason = decide_fn(hist, kl_stable, gt)
        stop_idx = max(0, min(stop_idx, len(hist) - 1))
        pred = step_conclusion(hist[stop_idx])
        correct = is_correct(pred, gt)
        # ever correct
        ever = False
        first_correct = None
        for j, s in enumerate(hist):
            if is_correct(step_conclusion(s), gt):
                if first_correct is None: first_correct = j
                ever = True; break
        for bucket in [m[gt], m["all"]]:
            bucket["total"] += 1
            bucket["correct"] += int(correct)
            bucket["steps"] += stop_idx + 1
            if ever:
                bucket["drift_elig"] += 1
                if not correct: bucket["drift"] += 1
                bucket["regrets"].append(max(0, stop_idx - first_correct))

    return m


def fmt(m):
    t = m["total"]
    acc = m["correct"]/t*100 if t else 0
    steps = m["steps"]/t if t else 0
    drift = m["drift"]/m["drift_elig"]*100 if m["drift_elig"] else 0
    reg = np.mean(m["regrets"]) if m["regrets"] else float('nan')
    return acc, drift, steps, reg


def main():
    trajs = _filtered_trajs()
    print(f"Total: {len(trajs)} (NoE:{sum(1 for t in trajs if t['ground_truth']=='NoEffect')}, "
          f"Ben:{sum(1 for t in trajs if t['ground_truth']=='Beneficial')})\n")

    configs = [
        ("Fixed k=1", lambda h,kl,gt: decide_fixed_k(h, 1)),
        ("Fixed k=2", lambda h,kl,gt: decide_fixed_k(h, 2)),
        ("Fixed k=3", lambda h,kl,gt: decide_fixed_k(h, 3)),
        ("Fixed k=5", lambda h,kl,gt: decide_fixed_k(h, 5)),
        ("Fixed k=10", lambda h,kl,gt: decide_fixed_k(h, 10)),
        ("Fixed k=15", lambda h,kl,gt: decide_fixed_k(h, 15)),
        ("Fixed k=20", lambda h,kl,gt: decide_fixed_k(h, 20)),
        ("KL only (c=1)", lambda h,kl,gt: decide_kl_only(h, compute_kl_stable(h, kl_consec=1))),
        ("KL only (c=2)", lambda h,kl,gt: decide_kl_only(h, kl)),
        ("PRM online", lambda h,kl,gt: decide_prm_online_only(h)),
        ("KL+PRM (c=1)", lambda h,kl,gt: decide_kl_prm_online(h, compute_kl_stable(h, kl_consec=1))),
        ("KL+PRM (c=2)", lambda h,kl,gt: decide_kl_prm_online(h, kl)),
        ("Oracle", lambda h,kl,gt: decide_oracle(h, gt)),
    ]

    # Header
    print(f"{'Method':<18} │ {'Acc':>6} {'NoE':>6} {'Ben':>6} │ {'Drift':>6} {'NoE_dr':>7} {'Ben_dr':>7} │ "
          f"{'Steps':>5} │ {'Regret':>6} {'NoE_reg':>7} {'Ben_reg':>7}")
    print("─"*120)

    for name, fn in configs:
        m = evaluate_detailed(trajs, fn)
        a_acc, a_drift, a_steps, a_reg = fmt(m["all"])
        n_acc, n_drift, n_steps, n_reg = fmt(m["NoEffect"])
        b_acc, b_drift, b_steps, b_reg = fmt(m["Beneficial"])
        print(f"{name:<18} │ {a_acc:>5.1f}% {n_acc:>5.1f}% {b_acc:>5.1f}% │ "
              f"{a_drift:>5.1f}% {n_drift:>6.1f}% {b_drift:>6.1f}% │ "
              f"{a_steps:>5.2f} │ {a_reg:>6.2f} {n_reg:>7.2f} {b_reg:>7.2f}")

    # Also show step distribution for key methods
    print("\n\n=== STOP STEP DISTRIBUTION ===")
    key_configs = [
        ("Fixed k=3", lambda h,kl,gt: decide_fixed_k(h, 3)),
        ("KL+PRM (c=2)", lambda h,kl,gt: decide_kl_prm_online(h, kl)),
        ("Oracle", lambda h,kl,gt: decide_oracle(h, gt)),
    ]
    for name, fn in key_configs:
        steps_noe = []; steps_ben = []
        for tr in trajs:
            hist = tr["history"]; gt = tr["ground_truth"]
            kl_stable = compute_kl_stable(hist, kl_consec=2)
            stop_idx, _ = fn(hist, kl_stable, gt)
            stop_idx = max(0, min(stop_idx, len(hist) - 1))
            if gt == 'NoEffect': steps_noe.append(stop_idx + 1)
            else: steps_ben.append(stop_idx + 1)
        print(f"\n{name}:")
        print(f"  NoEffect steps: mean={np.mean(steps_noe):.2f}, median={np.median(steps_noe):.1f}, "
              f"dist={Counter(steps_noe).most_common(5)}")
        print(f"  Beneficial steps: mean={np.mean(steps_ben):.2f}, median={np.median(steps_ben):.1f}, "
              f"dist={Counter(steps_ben).most_common(5)}")


if __name__ == "__main__":
    main()
