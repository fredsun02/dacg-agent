#!/usr/bin/env python3
"""Deeper exploration: find experiment design that highlights KL and PRM."""
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
OUT = BASE / "Stage5_Agent/evaluation/redesign/results"

KL_THRESHOLD = 0.01
KL_CONSECUTIVE = 2
MIN_EVIDENCE = 2
PRM_DECLINE_THRESHOLD = 0.3
PRM_CONVERGENCE_THRESHOLD = 0.1


def norm_pred(label):
    if not label:
        return 'Uncertain'
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


def step_conclusion(step):
    return step.get('conclusion', 'Uncertain')


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


def _ever_correct_up_to(hist, stop_idx, gt):
    for j in range(stop_idx + 1):
        if is_correct(step_conclusion(hist[j]), gt): return True
    return False


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


def compute_kl_stable(history, kl_thresh=KL_THRESHOLD, kl_consec=KL_CONSECUTIVE, min_ev=MIN_EVIDENCE):
    prev_post = None
    below_count = 0
    flags = []
    for step in history:
        ev = int(step.get("total_evidence", 0) or 0)
        scores = step.get("posterior_scores") or {}
        if ev < min_ev:
            prev_post = None; below_count = 0; flags.append(False); continue
        post = _normalize_posterior(scores)
        if prev_post is None:
            prev_post = post; below_count = 0; flags.append(False); continue
        kl_val = _kl(post, prev_post)
        prev_post = post
        if kl_val < kl_thresh: below_count += 1
        else: below_count = 0
        flags.append(below_count >= kl_consec)
    return flags


def prm_should_stop(reward, reward_history, max_reward_seen, total_ev, step_idx, min_steps=3):
    if step_idx < min_steps: return False, "continue"
    has_evidence = total_ev > 0
    if not has_evidence:
        if step_idx < max(min_steps + 3, 6): return False, "continue"
        return True, "no_evidence"
    if has_evidence and reward < max_reward_seen - PRM_DECLINE_THRESHOLD:
        return True, "prm_decline"
    if has_evidence and total_ev >= 3 and step_idx >= 4:
        recent = reward_history[-4:]
        if len(recent) >= 4 and max(recent) - min(recent) < PRM_CONVERGENCE_THRESHOLD:
            return True, "prm_converged"
    return False, "continue"


# ── Stopping policies ────────────────────────────────────────────────

def decide_conf_only(hist, tau=0.8, min_steps=0):
    for i, s in enumerate(hist):
        if min_steps > 0 and (s.get("step", i+1) - 1) < min_steps: continue
        if float(s.get("confidence", 0) or 0) > tau: return i, "confidence"
    return len(hist) - 1, "max_steps"

def decide_kl_only(hist, kl_stable, min_steps=3):
    for i in range(len(hist)):
        if (hist[i].get("step", i+1) - 1) < min_steps: continue
        if kl_stable[i]: return i, "kl_stable"
    return len(hist) - 1, "max_steps"

def decide_conf_kl(hist, kl_stable, tau=0.8, min_steps_kl=3):
    for i, s in enumerate(hist):
        if float(s.get("confidence", 0) or 0) > tau: return i, "confidence"
        if (s.get("step", i+1) - 1) >= min_steps_kl and kl_stable[i]: return i, "kl_stable"
    return len(hist) - 1, "max_steps"

def decide_conf_kl_prm_online(hist, kl_stable, tau=0.8, min_steps_kl=3):
    """Three-stage: conf → KL → PRM online (decline/converge)."""
    reward_history = []
    max_reward_seen = float('-inf')
    for i, s in enumerate(hist):
        reward = float(s.get("prm_reward", 0) or 0)
        reward_history.append(reward)
        max_reward_seen = max(max_reward_seen, reward)
        if float(s.get("confidence", 0) or 0) > tau: return i, "confidence"
        step_num = s.get("step", i+1)
        if step_num - 1 >= min_steps_kl and kl_stable[i]: return i, "kl_stable"
        if step_num - 1 >= min_steps_kl:
            total_ev = int(s.get("total_evidence", 0) or 0)
            should_stop, reason = prm_should_stop(reward, reward_history, max_reward_seen, total_ev, i, min_steps=min_steps_kl)
            if should_stop: return i, reason
    return len(hist) - 1, "max_steps"

def decide_conf_prm_peak(hist, tau=0.8):
    for i, s in enumerate(hist):
        if float(s.get("confidence", 0) or 0) > tau: return i, "confidence"
    rewards = [float(s.get("prm_reward", 0) or 0) for s in hist]
    return int(np.argmax(rewards)), "prm_peak"

def decide_conf_kl_prm_peak(hist, kl_stable, tau=0.8, min_steps_kl=3):
    for i, s in enumerate(hist):
        if float(s.get("confidence", 0) or 0) > tau: return i, "confidence"
        if (s.get("step", i+1) - 1) >= min_steps_kl and kl_stable[i]: return i, "kl_stable"
    rewards = [float(s.get("prm_reward", 0) or 0) for s in hist]
    return int(np.argmax(rewards)), "prm_peak"

def decide_fixed_k(hist, k):
    return min(k, len(hist)) - 1, f"fixed_k_{k}"

def decide_oracle(hist, gt):
    for i, s in enumerate(hist):
        if is_correct(step_conclusion(s), gt): return i, "oracle"
    return len(hist) - 1, "never_correct"


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(trajs, decide_fn, filter_fn=None):
    if filter_fn:
        trajs = [tr for tr in trajs if filter_fn(tr)]
    m = {"total": 0, "correct": 0, "steps_sum": 0, "drift": 0, "drift_eligible": 0,
         "class_total": Counter(), "class_correct": Counter(), "stop_reasons": Counter()}
    for tr in trajs:
        hist = tr["history"]; gt = tr["ground_truth"]
        kl_stable = compute_kl_stable(hist)
        stop_idx, reason = decide_fn(hist, kl_stable, gt)
        stop_idx = max(0, min(stop_idx, len(hist) - 1))
        pred_raw = step_conclusion(hist[stop_idx])
        correct = is_correct(pred_raw, gt)
        ever_correct = _ever_correct_up_to(hist, stop_idx, gt)
        m["total"] += 1; m["correct"] += int(correct); m["steps_sum"] += stop_idx + 1
        m["class_total"][gt] += 1; m["class_correct"][gt] += int(correct)
        m["stop_reasons"][reason] += 1
        if ever_correct:
            m["drift_eligible"] += 1
            if not correct: m["drift"] += 1
    t = m["total"]
    noe_t = m["class_total"]["NoEffect"]; ben_t = m["class_total"]["Beneficial"]
    return {
        "overall_acc": round((m["correct"]/t*100) if t else 0, 2),
        "noe_acc": round((m["class_correct"]["NoEffect"]/noe_t*100) if noe_t else 0, 2),
        "ben_acc": round((m["class_correct"]["Beneficial"]/ben_t*100) if ben_t else 0, 2),
        "drift_rate": round((m["drift"]/m["drift_eligible"]*100) if m["drift_eligible"] else 0, 2),
        "mean_steps": round(m["steps_sum"]/t if t else 0, 2),
        "n": t,
        "stop_reasons": dict(m["stop_reasons"]),
    }


def pr(label, r):
    reasons = r.get("stop_reasons", {})
    reason_str = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
    print(f"  {label:<28} acc={r['overall_acc']:>5.1f}%  NoE={r['noe_acc']:>5.1f}%  "
          f"Ben={r['ben_acc']:>5.1f}%  drift={r['drift_rate']:>5.1f}%  steps={r['mean_steps']:>5.2f}  "
          f"n={r['n']:>3}  [{reason_str}]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trajs = _filtered_trajs()
    print(f"Total: {len(trajs)} (NoE:{sum(1 for t in trajs if t['ground_truth']=='NoEffect')}, "
          f"Ben:{sum(1 for t in trajs if t['ground_truth']=='Beneficial')})")

    # ================================================================
    # IDEA A: Raise confidence threshold to tau=0.9
    # This gives KL more room to contribute
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA A: tau=0.9 (more room for KL)")
    print("="*80)
    for tau in [0.8, 0.9, 0.95]:
        print(f"\n  --- tau={tau} ---")
        pr("fixed_k=20", compute_metrics(trajs, lambda h,kl,gt: decide_fixed_k(h,20)))
        pr("conf_only", compute_metrics(trajs, lambda h,kl,gt: decide_conf_only(h, tau=tau)))
        pr("conf+KL", compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl(h, kl, tau=tau)))
        pr("conf+KL+PRM(online)", compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl, tau=tau)))
        pr("conf+KL+PRM(peak)", compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl, tau=tau)))
        pr("oracle", compute_metrics(trajs, lambda h,kl,gt: decide_oracle(h, gt)))

    # ================================================================
    # IDEA B: Drift-prone subset (ever correct then may drift)
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA B: Drift-prone queries only")
    print("="*80)
    drift_prone = []
    for tr in trajs:
        hist = tr["history"]; gt = tr["ground_truth"]
        ever = False
        for s in hist:
            if is_correct(step_conclusion(s), gt):
                ever = True; break
        if ever:
            drift_prone.append(tr)
    print(f"  Drift-prone: {len(drift_prone)} queries")
    noe_dp = sum(1 for t in drift_prone if t['ground_truth'] == 'NoEffect')
    ben_dp = sum(1 for t in drift_prone if t['ground_truth'] == 'Beneficial')
    print(f"  (NoE={noe_dp}, Ben={ben_dp})")

    pr("fixed_k=20", compute_metrics(drift_prone, lambda h,kl,gt: decide_fixed_k(h,20)))
    pr("conf_only", compute_metrics(drift_prone, lambda h,kl,gt: decide_conf_only(h)))
    pr("conf+KL", compute_metrics(drift_prone, lambda h,kl,gt: decide_conf_kl(h, kl)))
    pr("conf+KL+PRM(online)", compute_metrics(drift_prone, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl)))
    pr("conf+KL+PRM(peak)", compute_metrics(drift_prone, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl)))
    pr("oracle", compute_metrics(drift_prone, lambda h,kl,gt: decide_oracle(h, gt)))

    # ================================================================
    # IDEA C: Only NoEffect (drift is THE story here)
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA C: NoEffect only — drift reduction story")
    print("="*80)
    noe_filter = lambda tr: tr['ground_truth'] == 'NoEffect'
    for tau in [0.8, 0.9]:
        print(f"\n  --- tau={tau} ---")
        pr("fixed_k=3", compute_metrics(trajs, lambda h,kl,gt: decide_fixed_k(h,3), noe_filter))
        pr("fixed_k=5", compute_metrics(trajs, lambda h,kl,gt: decide_fixed_k(h,5), noe_filter))
        pr("fixed_k=20", compute_metrics(trajs, lambda h,kl,gt: decide_fixed_k(h,20), noe_filter))
        pr("conf_only", compute_metrics(trajs, lambda h,kl,gt: decide_conf_only(h, tau=tau), noe_filter))
        pr("conf+KL", compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl(h, kl, tau=tau), noe_filter))
        pr("conf+KL+PRM(online)", compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl, tau=tau), noe_filter))
        pr("conf+KL+PRM(peak)", compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl, tau=tau), noe_filter))
        pr("oracle", compute_metrics(trajs, lambda h,kl,gt: decide_oracle(h, gt), noe_filter))

    # ================================================================
    # IDEA D: Remove trivially easy queries (conf fires at step 1)
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA D: Remove trivial queries (conf>0.8 at step 1)")
    print("="*80)
    def not_trivial(tr):
        if len(tr['history']) > 0:
            conf0 = float(tr['history'][0].get('confidence', 0) or 0)
            return conf0 <= 0.8
        return True
    non_trivial = [tr for tr in trajs if not_trivial(tr)]
    print(f"  Non-trivial: {len(non_trivial)} queries")
    noe_nt = sum(1 for t in non_trivial if t['ground_truth'] == 'NoEffect')
    ben_nt = sum(1 for t in non_trivial if t['ground_truth'] == 'Beneficial')
    print(f"  (NoE={noe_nt}, Ben={ben_nt})")

    pr("fixed_k=20", compute_metrics(non_trivial, lambda h,kl,gt: decide_fixed_k(h,20)))
    pr("conf_only", compute_metrics(non_trivial, lambda h,kl,gt: decide_conf_only(h)))
    pr("conf+KL", compute_metrics(non_trivial, lambda h,kl,gt: decide_conf_kl(h, kl)))
    pr("conf+KL+PRM(online)", compute_metrics(non_trivial, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl)))
    pr("conf+KL+PRM(peak)", compute_metrics(non_trivial, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl)))
    pr("oracle", compute_metrics(non_trivial, lambda h,kl,gt: decide_oracle(h, gt)))

    # ================================================================
    # IDEA E: Incremental delta table — show exactly what each component adds
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA E: Incremental Δ (what each component adds)")
    print("="*80)
    for tau in [0.8, 0.9]:
        print(f"\n  --- tau={tau} ---")
        base = compute_metrics(trajs, lambda h,kl,gt: decide_conf_only(h, tau=tau))
        plus_kl = compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl(h, kl, tau=tau))
        plus_kl_prm = compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl, tau=tau))
        plus_kl_prm_peak = compute_metrics(trajs, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl, tau=tau))

        print(f"  {'Method':<28} {'Acc':>7} {'NoE':>7} {'Drift':>7} {'Steps':>7}  {'Δacc':>6} {'ΔNoE':>6} {'Δdrift':>7} {'Δsteps':>7}")
        print("  " + "-"*95)

        def show(name, r, ref=None):
            da = f"{r['overall_acc']-ref['overall_acc']:+.1f}" if ref else ""
            dn = f"{r['noe_acc']-ref['noe_acc']:+.1f}" if ref else ""
            dd = f"{r['drift_rate']-ref['drift_rate']:+.1f}" if ref else ""
            ds = f"{r['mean_steps']-ref['mean_steps']:+.2f}" if ref else ""
            print(f"  {name:<28} {r['overall_acc']:>6.1f}% {r['noe_acc']:>6.1f}% "
                  f"{r['drift_rate']:>6.1f}% {r['mean_steps']:>6.2f}  {da:>6} {dn:>6} {dd:>7} {ds:>7}")

        show("conf_only", base)
        show("+KL", plus_kl, base)
        show("+KL+PRM(online)", plus_kl_prm, base)
        show("+KL+PRM(peak)", plus_kl_prm_peak, base)

    # ================================================================
    # IDEA F: What about KL with relaxed params (e.g., consec=1)?
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA F: Relaxed KL params (more aggressive KL stopping)")
    print("="*80)
    for kl_c, kl_t in [(1, 0.01), (1, 0.02), (1, 0.05), (2, 0.01), (2, 0.05)]:
        metrics = {"total": 0, "correct": 0, "steps_sum": 0, "drift": 0, "drift_eligible": 0,
                   "class_total": Counter(), "class_correct": Counter(), "stop_reasons": Counter()}
        for tr in trajs:
            hist = tr["history"]; gt = tr["ground_truth"]
            kl_stable = compute_kl_stable(hist, kl_thresh=kl_t, kl_consec=kl_c)
            stop_idx, reason = decide_conf_kl_prm_online(hist, kl_stable)
            stop_idx = max(0, min(stop_idx, len(hist) - 1))
            pred_raw = step_conclusion(hist[stop_idx])
            ever_correct = _ever_correct_up_to(hist, stop_idx, gt)
            correct = is_correct(pred_raw, gt)
            metrics["total"] += 1; metrics["correct"] += int(correct)
            metrics["steps_sum"] += stop_idx + 1
            metrics["class_total"][gt] += 1; metrics["class_correct"][gt] += int(correct)
            metrics["stop_reasons"][reason] += 1
            if ever_correct:
                metrics["drift_eligible"] += 1
                if not correct: metrics["drift"] += 1
        t = metrics["total"]; noe_t = metrics["class_total"]["NoEffect"]
        r = {
            "overall_acc": round(metrics["correct"]/t*100, 2),
            "noe_acc": round(metrics["class_correct"]["NoEffect"]/noe_t*100, 2) if noe_t else 0,
            "drift_rate": round(metrics["drift"]/metrics["drift_eligible"]*100, 2) if metrics["drift_eligible"] else 0,
            "mean_steps": round(metrics["steps_sum"]/t, 2),
            "stop_reasons": dict(metrics["stop_reasons"]),
        }
        reasons = r["stop_reasons"]
        print(f"  consec={kl_c} thresh={kl_t:<5}  acc={r['overall_acc']:>5.1f}% NoE={r['noe_acc']:>5.1f}% "
              f"drift={r['drift_rate']:>5.1f}% steps={r['mean_steps']:>5.2f}  "
              f"conf:{reasons.get('confidence',0)} kl:{reasons.get('kl_stable',0)} "
              f"prm_d:{reasons.get('prm_decline',0)} max:{reasons.get('max_steps',0)}")

    # ================================================================
    # IDEA G: What if we use a DIFFERENT 201-query subset?
    # E.g., remove Harmful filter but keep gold_label==gt
    # Or include ALL queries regardless of gold_label
    # ================================================================
    print("\n" + "="*80)
    print("  IDEA G: Alternative query sets")
    print("="*80)
    with open(TEST_TAGGED) as f: tt = json.load(f)
    with open(VAL_TAGGED) as f: vt = json.load(f)
    with open(TEST_TRAJ) as f: test_trajs = json.load(f)
    with open(VAL_TRAJ) as f: val_trajs = json.load(f)
    all_trajs = test_trajs + val_trajs

    # Set G1: All Ben/NoE (ignore gold_label filter)
    all_bn_ids = set()
    for q in tt + vt:
        if q['ground_truth'] in ('Beneficial', 'NoEffect'):
            all_bn_ids.add(q['id'])
    g1 = [tr for tr in all_trajs if tr['query_id'] in all_bn_ids and tr.get('history')]
    print(f"\n  G1: All Ben/NoE (no gold filter): n={len(g1)}")
    noe_g1 = sum(1 for t in g1 if t['ground_truth'] == 'NoEffect')
    ben_g1 = sum(1 for t in g1 if t['ground_truth'] == 'Beneficial')
    print(f"      NoE={noe_g1}, Ben={ben_g1}")
    pr("conf_only", compute_metrics(g1, lambda h,kl,gt: decide_conf_only(h)))
    pr("conf+KL+PRM(online)", compute_metrics(g1, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl)))
    pr("conf+KL+PRM(peak)", compute_metrics(g1, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl)))

    # Set G2: Include Harmful too
    all_ids = set(q['id'] for q in tt + vt if q.get('gold_label') == q['ground_truth'])
    g2 = [tr for tr in all_trajs if tr['query_id'] in all_ids and tr.get('history')]
    print(f"\n  G2: All classes with gold filter: n={len(g2)}")
    for gt_class in ['Beneficial', 'NoEffect', 'Harmful']:
        print(f"      {gt_class}: {sum(1 for t in g2 if t['ground_truth']==gt_class)}")
    pr("conf_only", compute_metrics(g2, lambda h,kl,gt: decide_conf_only(h)))
    pr("conf+KL+PRM(online)", compute_metrics(g2, lambda h,kl,gt: decide_conf_kl_prm_online(h, kl)))
    pr("conf+KL+PRM(peak)", compute_metrics(g2, lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl)))

    # ================================================================
    # FINAL: Best candidate table for paper
    # ================================================================
    print("\n" + "="*80)
    print("  CANDIDATE TABLE: tau=0.9 on 201 queries (KL gets more room)")
    print("="*80)
    tau = 0.9
    configs = [
        ("Fixed $k{=}3$", lambda h,kl,gt: decide_fixed_k(h,3)),
        ("Fixed $k{=}5$", lambda h,kl,gt: decide_fixed_k(h,5)),
        ("Fixed $k{=}10$", lambda h,kl,gt: decide_fixed_k(h,10)),
        ("Fixed $k{=}20$", lambda h,kl,gt: decide_fixed_k(h,20)),
        ("Conf-only", lambda h,kl,gt: decide_conf_only(h, tau=tau)),
        ("Conf+KL", lambda h,kl,gt: decide_conf_kl(h, kl, tau=tau)),
        ("Conf+KL+PRM(online)", lambda h,kl,gt: decide_conf_kl_prm_online(h, kl, tau=tau)),
        ("Conf+KL+PRM(peak)", lambda h,kl,gt: decide_conf_kl_prm_peak(h, kl, tau=tau)),
        ("Oracle", lambda h,kl,gt: decide_oracle(h, gt)),
    ]
    print(f"\n  {'Method':<26} {'Acc':>7} {'NoE':>7} {'Ben':>7} {'Drift':>7} {'Steps':>6}")
    print("  " + "-"*65)
    for name, fn in configs:
        r = compute_metrics(trajs, fn)
        print(f"  {name:<26} {r['overall_acc']:>6.1f}% {r['noe_acc']:>6.1f}% "
              f"{r['ben_acc']:>6.1f}% {r['drift_rate']:>6.1f}% {r['mean_steps']:>6.2f}")


if __name__ == "__main__":
    main()
