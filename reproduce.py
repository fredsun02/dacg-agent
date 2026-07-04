#!/usr/bin/env python3
"""
Offline reproduction of the core evidence-drift result.

NO network and NO LLM calls: this script replays the recorded retrieval
trajectories shipped in this repository and applies stopping rules offline.
It reproduces the paper's central qualitative finding on the bundled clean
Beneficial/NoEffect evaluation set:

  * under fixed-depth retrieval, NoEffect accuracy collapses and evidence
    drift rises monotonically as the budget k grows (1 -> 20);
  * a KL-convergence stopping rule recovers NoEffect accuracy, lowers drift,
    and uses far fewer retrieval steps.

Absolute numbers here are computed on the bundled test trajectories
(clean Beneficial/NoEffect subset); the headline 70/70 n=140 table in the
paper pools test+held-out and is produced by the full evaluation harness.
Run:  python reproduce.py
"""
import json, math, os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
TRAJ = ROOT / "trajectories/step2_v3_test/trajectories_20260305_053618.json"
TAGGED = ROOT / "evaluation/redesign/tagged_queries.json"


def norm_pred(label):
    if not label:
        return "Uncertain"
    l = str(label).strip().lower()
    if l in ("beneficial", "treat", "positive"):
        return "Beneficial"
    if l in ("harmful", "negative", "cause"):
        return "Harmful"
    if l in ("noeffect", "no_effect", "no effect", "neutral"):
        return "NoEffect"
    if l in ("noevidence", "no_evidence", "uncertain", "unknown"):
        return "Uncertain"
    return str(label)


def is_correct(pred, gt):
    p = norm_pred(pred)
    if p in ("NoEvidence", "Uncertain"):
        p = "Uncertain"
    if gt == "NoEffect" and p in ("NoEffect", "Uncertain"):
        return True
    return p == gt


def normalize_posterior(s):
    b = max(0.0, float(s.get("beneficial", 0) or 0))
    h = max(0.0, float(s.get("harmful", 0) or 0))
    n = max(0.0, float(s.get("neutral", 0) or 0))
    t = b + h + n
    if t <= 0:
        return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}
    return {"beneficial": b/t, "harmful": h/t, "neutral": n/t}


def load():
    traj = json.load(open(TRAJ))
    tagged = json.load(open(TAGGED))
    keep = {q["id"] for q in tagged
            if q["ground_truth"] in ("Beneficial", "NoEffect")
            and q.get("gold_label") == q["ground_truth"]}
    return [t for t in traj if t["query_id"] in keep and t.get("history")]


def _score(trajs, stop_of):
    m = Counter(); cc = Counter(); ct = Counter()
    drift = elig = steps = 0
    for t in trajs:
        h, gt = t["history"], t["ground_truth"]
        idx = max(0, min(stop_of(h), len(h) - 1))
        corr = is_correct(h[idx].get("conclusion", "Uncertain"), gt)
        ct[gt] += 1; cc[gt] += int(corr); m["tot"] += 1; m["cor"] += int(corr)
        steps += idx + 1
        if any(is_correct(h[j].get("conclusion", "Uncertain"), gt) for j in range(idx + 1)):
            elig += 1
            if not corr:
                drift += 1
    return dict(acc=round(m["cor"]/m["tot"]*100, 1),
                noe=round(cc["NoEffect"]/ct["NoEffect"]*100, 1),
                ben=round(cc["Beneficial"]/ct["Beneficial"]*100, 1),
                drift=round(drift/elig*100, 1) if elig else 0.0,
                steps=round(steps/m["tot"], 2))


def fixed_k(k):
    return lambda h: min(k, len(h)) - 1


def kl_stop(delta=0.05, consec=2):
    def stop_of(h):
        below = 0; prev = None; stop = len(h) - 1
        for i, s in enumerate(h):
            kl = s.get("kl")
            post = normalize_posterior(s.get("posterior_scores") or {})
            if kl is None and prev is not None:
                kl = sum(post[k] * math.log(max(post[k], 1e-12) / max(prev[k], 1e-12)) for k in post)
            prev = post
            if kl is not None and kl < delta:
                below += 1
                if below >= consec:
                    return i
            else:
                below = 0
        return stop
    return stop_of


def main():
    trajs = load()
    dist = Counter(t["ground_truth"] for t in trajs)
    print(f"Loaded {len(trajs)} recorded trajectories "
          f"(Beneficial={dist['Beneficial']}, NoEffect={dist['NoEffect']})\n")
    hdr = f"{'method':<14}{'acc':>7}{'NoEff':>7}{'Ben':>7}{'drift':>7}{'steps':>7}"
    print(hdr); print("-" * len(hdr))
    for k in (1, 3, 5, 10, 20):
        r = _score(trajs, fixed_k(k))
        print(f"{'fixed k='+str(k):<14}{r['acc']:>6.1f}{r['noe']:>7.1f}{r['ben']:>7.1f}{r['drift']:>7.1f}{r['steps']:>7.2f}")
    r = _score(trajs, kl_stop())
    print(f"{'KL-stopping':<14}{r['acc']:>6.1f}{r['noe']:>7.1f}{r['ben']:>7.1f}{r['drift']:>7.1f}{r['steps']:>7.2f}")
    print("\nDrift rises with fixed depth and concentrates in NoEffect; "
          "KL-stopping recovers NoEffect accuracy and cuts drift and steps.")


if __name__ == "__main__":
    main()
