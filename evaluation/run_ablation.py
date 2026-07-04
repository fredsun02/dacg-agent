"""
Offline ablation evaluation for GraphKGSAAgent.

Runs agent in recording_mode to capture full trajectories, then applies
multiple stopping criteria offline to produce an ablation table from ONE run.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph_agent import GraphKGSAAgent, KL_THRESHOLD, KL_CONSECUTIVE
from agent.path_inference import MIN_EVIDENCE
from agent.prm import PRM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_KS = [3, 5, 10, 20]
CONF_THRESHOLD = 0.8
DECISIVE_MIN_EV = 10
DECISIVE_MIN_CONF = 0.9
ENERGY_U_THRESHOLD = 0.5
ENERGY_V_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
_LABEL_MAP = {
    "beneficial": "Beneficial", "treat": "Beneficial", "positive": "Beneficial",
    "harmful": "Harmful", "negative": "Harmful", "cause": "Harmful",
    "noeffect": "NoEffect", "no_effect": "NoEffect", "no effect": "NoEffect",
    "neutral": "NoEffect",
    "uncertain": "Uncertain", "unknown": "Uncertain",
    "noevidence": "NoEvidence", "no_evidence": "NoEvidence",
}


def norm_label(label: str) -> str:
    if not label:
        return "Uncertain"
    return _LABEL_MAP.get(label.lower().strip(), label.title())


def norm_pred(label: str) -> str:
    n = norm_label(label)
    return "Uncertain" if n == "NoEvidence" else n


# ---------------------------------------------------------------------------
# Offline KL computation (mirrors GraphKGSAAgent._update_kl_stopping)
# ---------------------------------------------------------------------------
def _normalize_posterior(scores: Dict[str, float]) -> Dict[str, float]:
    b = max(0.0, float(scores.get("beneficial", 0.0)))
    h = max(0.0, float(scores.get("harmful", 0.0)))
    n = max(0.0, float(scores.get("neutral", 0.0)))
    t = b + h + n
    if t <= 0:
        return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}
    return {"beneficial": b/t, "harmful": h/t, "neutral": n/t}


def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    return sum(
        max(eps, p.get(k, 0.0)) * math.log(max(eps, p.get(k, 0.0)) / max(eps, q.get(k, 0.0)))
        for k in ("beneficial", "harmful", "neutral")
    )


def compute_kl_stable(history: List[Dict], min_evidence: int) -> List[bool]:
    """Return per-step kl_stable flags, matching agent's KL logic."""
    prev_post = None
    below_count = 0
    flags = []
    for step in history:
        ev = int(step.get("total_evidence", 0) or 0)
        scores = step.get("posterior_scores") or {}
        if ev < min_evidence:
            prev_post = None
            below_count = 0
            flags.append(False)
            continue
        post = _normalize_posterior(scores)
        if prev_post is None:
            prev_post = post
            below_count = 0
            flags.append(False)
            continue
        kl_val = _kl(post, prev_post)
        prev_post = post
        if kl_val < KL_THRESHOLD:
            below_count += 1
        else:
            below_count = 0
        flags.append(below_count >= KL_CONSECUTIVE)
    return flags


# ---------------------------------------------------------------------------
# Offline stopping criteria (each returns (stop_step_1based, reason, snapshot))
# ---------------------------------------------------------------------------
def _snap(history, idx, fallback):
    """Get snapshot at 0-based idx, or fallback."""
    if not history:
        return fallback
    return history[min(idx, len(history) - 1)]


def decide_fixed_k(history, fallback, k):
    if not history:
        return 0, f"fixed_k_{k}", fallback
    idx = min(k, len(history)) - 1
    return idx + 1, f"fixed_k_{k}", history[idx]


def decide_conf_only(history, fallback):
    for i, s in enumerate(history):
        if float(s.get("confidence", 0)) > CONF_THRESHOLD:
            return i + 1, "confidence_reached", s
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "confidence_only", fallback


def decide_kl_only(history, fallback, min_steps, kl_stable):
    for i, stable in enumerate(kl_stable):
        step_0 = history[i].get("step", i + 1) - 1  # 0-based step matching agent
        if step_0 < min_steps:
            continue
        if stable:
            return i + 1, "kl_stable", history[i]
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "kl_only", fallback


def decide_graph_only(history, fallback, min_steps, kl_stable):
    """KL + decisive fallback (current graph_only mode)."""
    for i, s in enumerate(history):
        step_0 = s.get("step", i + 1) - 1
        if step_0 < min_steps:
            continue
        if kl_stable[i]:
            return i + 1, "kl_stable", s
        ev = int(s.get("total_evidence", 0) or 0)
        conf = float(s.get("confidence", 0))
        decisive = s.get("is_decisive", False)
        if ev >= DECISIVE_MIN_EV and decisive and conf >= DECISIVE_MIN_CONF:
            return i + 1, "decisive_fallback", s
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "graph_only", fallback


def decide_full(history, fallback, min_steps, kl_stable, prm: PRM):
    """KL + PRM (current graph_prm mode). Feeds features sequentially to PRM."""
    prm.reset()
    for i, s in enumerate(history):
        step_0 = s.get("step", i + 1) - 1
        if step_0 < min_steps:
            continue
        if kl_stable[i]:
            return i + 1, "kl_stable", s
        feats = s.get("graph_features") or {}
        should_stop, reason, _ = prm.should_stop_from_dict(feats)
        if should_stop:
            return i + 1, reason, s
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "full", fallback


def decide_energy_u_only(history, fallback, min_steps, kl_stable):
    """KL + decisive with low edge uncertainty."""
    for i, s in enumerate(history):
        step_0 = s.get("step", i + 1) - 1
        if step_0 < min_steps:
            continue
        if kl_stable[i]:
            return i + 1, "kl_stable", s
        feats = s.get("graph_features") or {}
        eu = float(feats.get("edge_uncertainty", 1.0))
        ev = int(s.get("total_evidence", 0) or 0)
        conf = float(s.get("confidence", 0))
        decisive = s.get("is_decisive", False)
        if ev >= DECISIVE_MIN_EV and decisive and conf >= DECISIVE_MIN_CONF and eu < ENERGY_U_THRESHOLD:
            return i + 1, "decisive_low_u", s
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "energy_u_only", fallback


def decide_energy_v_only(history, fallback, min_steps, kl_stable):
    """KL + decisive with low constraint violation."""
    for i, s in enumerate(history):
        step_0 = s.get("step", i + 1) - 1
        if step_0 < min_steps:
            continue
        if kl_stable[i]:
            return i + 1, "kl_stable", s
        feats = s.get("graph_features") or {}
        cv = float(feats.get("constraint_violation", 1.0))
        ev = int(s.get("total_evidence", 0) or 0)
        conf = float(s.get("confidence", 0))
        decisive = s.get("is_decisive", False)
        if ev >= DECISIVE_MIN_EV and decisive and conf >= DECISIVE_MIN_CONF and cv < ENERGY_V_THRESHOLD:
            return i + 1, "decisive_low_v", s
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "energy_v_only", fallback


def decide_energy_full(history, fallback, min_steps, kl_stable):
    """KL + decisive with both energy criteria."""
    for i, s in enumerate(history):
        step_0 = s.get("step", i + 1) - 1
        if step_0 < min_steps:
            continue
        if kl_stable[i]:
            return i + 1, "kl_stable", s
        feats = s.get("graph_features") or {}
        eu = float(feats.get("edge_uncertainty", 1.0))
        cv = float(feats.get("constraint_violation", 1.0))
        ev = int(s.get("total_evidence", 0) or 0)
        conf = float(s.get("confidence", 0))
        decisive = s.get("is_decisive", False)
        if (ev >= DECISIVE_MIN_EV and decisive and conf >= DECISIVE_MIN_CONF
                and eu < ENERGY_U_THRESHOLD and cv < ENERGY_V_THRESHOLD):
            return i + 1, "decisive_energy", s
    if history:
        return len(history), "max_steps", history[-1]
    return 0, "energy_full", fallback


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _init_metrics():
    return {
        "correct": 0, "total": 0,
        "class_correct": Counter(), "class_total": Counter(),
        "steps_sum": 0, "stop_reasons": Counter(),
        "confusion": defaultdict(Counter),
    }


def _update(m, gt, snap, steps, reason):
    pred = norm_pred(snap.get("conclusion", "Uncertain"))
    gt = norm_pred(gt)
    m["total"] += 1
    m["steps_sum"] += int(steps)
    m["stop_reasons"][reason] += 1
    m["confusion"][gt][pred] += 1
    if pred == gt:
        m["correct"] += 1
        if gt in ("Beneficial", "NoEffect", "Harmful"):
            m["class_correct"][gt] += 1
    if gt in ("Beneficial", "NoEffect", "Harmful"):
        m["class_total"][gt] += 1


def _finalize(m):
    t = m["total"]
    acc = m["correct"] / t if t else 0
    avg_steps = m["steps_sum"] / t if t else 0
    by_class = {}
    for cls in ("Beneficial", "NoEffect", "Harmful"):
        ct = m["class_total"][cls]
        by_class[cls] = m["class_correct"][cls] / ct if ct else 0
    return {
        "accuracy": acc,
        "accuracy_by_class": by_class,
        "avg_steps": avg_steps,
        "total": t,
        "stop_reasons": dict(m["stop_reasons"]),
        "confusion": {gt: dict(preds) for gt, preds in m["confusion"].items()},
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------
def _fmt_table(configs_summary):
    hdr = f"{'Config':<20} {'Acc':>6} {'Ben':>6} {'NoEff':>6} {'Harm':>6} {'Steps':>6}"
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for name, s in configs_summary.items():
        a = s["accuracy"] * 100
        b = s["accuracy_by_class"]["Beneficial"] * 100
        n = s["accuracy_by_class"]["NoEffect"] * 100
        h = s["accuracy_by_class"]["Harmful"] * 100
        st = s["avg_steps"]
        lines.append(f"{name:<20} {a:>5.1f}% {b:>5.1f}% {n:>5.1f}% {h:>5.1f}% {st:>6.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline ablation evaluation (recording mode)")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N queries")
    parser.add_argument("--multihop", dest="multihop", action="store_true", default=True)
    parser.add_argument("--no-multihop", dest="multihop", action="store_false")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--prm-model", type=str, default=None,
                        help="PRM model path (default: models/graph_prm.pt)")
    parser.add_argument("--lambda-u", type=float, default=0.0,
                        help="Energy: edge uncertainty weight (online scoring)")
    parser.add_argument("--lambda-v", type=float, default=0.0,
                        help="Energy: constraint violation weight (online scoring)")
    parser.add_argument("--quality-floor", type=float, default=0.3,
                        help="Energy: minimum edge quality threshold")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    with open(args.benchmark) as f:
        data = json.load(f)
    benchmark = data["queries"] if isinstance(data, dict) and "queries" in data else (
        data["data"] if isinstance(data, dict) and "data" in data else data
    )
    if args.offset:
        benchmark = benchmark[args.offset:]
    if args.limit:
        benchmark = benchmark[:args.limit]
    print(f"Loaded {len(benchmark)} queries from {args.benchmark}")

    # PRM model path
    prm_path = args.prm_model or str(Path(__file__).parent.parent / "models" / "graph_prm.pt")
    if not Path(prm_path).exists():
        print(f"WARNING: PRM model not found at {prm_path}, Full config will be skipped")
        prm_path = None

    # Create agent in recording mode
    agent = GraphKGSAAgent(
        prm_model_path=prm_path,
        mode="graph_prm",
        enable_multihop=args.multihop,
        recording_mode=True,
        lambda_u=args.lambda_u,
        lambda_v=args.lambda_v,
        quality_floor=args.quality_floor,
    )

    # Separate PRM instance for offline Full simulation
    offline_prm = None
    if prm_path:
        offline_prm = PRM(model_path=prm_path, min_steps=agent.min_steps)

    # Init metrics for all configs
    config_names = [f"fixed_k_{k}" for k in FIXED_KS] + [
        "conf_only", "kl_only", "graph_only", "full",
        "energy_u_only", "energy_v_only", "energy_full",
    ]
    metrics = {name: _init_metrics() for name in config_names}

    # Save raw trajectories
    trajectories = []

    # Run evaluation
    for i, item in enumerate(benchmark):
        head = item.get("head_entity") or item.get("head") or item.get("drug") or ""
        tail = item.get("tail_entity") or item.get("tail") or item.get("disease") or ""
        gt = item.get("ground_truth") or item.get("relation_type") or item.get("label") or ""
        qid = item.get("id") or item.get("query_id") or f"q_{i}"

        if not head or not tail:
            print(f"[{i+1}/{len(benchmark)}] Skipping: missing entities")
            continue

        gt_norm = norm_pred(gt)
        print(f"[{i+1}/{len(benchmark)}] {head} -> {tail} (GT: {gt_norm})")

        try:
            result = agent.search(head_entity=head, tail_entity=tail, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        history = result.history or []
        fallback = {
            "step": 0, "conclusion": result.conclusion,
            "confidence": result.confidence, "total_evidence": result.total_evidence,
            "posterior_scores": result.conclusion_result.scores if result.conclusion_result else {},
            "graph_features": {},
        }

        # Compute offline KL stability flags
        kl_stable = compute_kl_stable(history, MIN_EVIDENCE)

        # Apply all offline stopping criteria
        for k in FIXED_KS:
            steps, reason, snap = decide_fixed_k(history, fallback, k)
            _update(metrics[f"fixed_k_{k}"], gt_norm, snap, steps, reason)

        steps, reason, snap = decide_conf_only(history, fallback)
        _update(metrics["conf_only"], gt_norm, snap, steps, reason)

        steps, reason, snap = decide_kl_only(history, fallback, agent.min_steps, kl_stable)
        _update(metrics["kl_only"], gt_norm, snap, steps, reason)

        steps, reason, snap = decide_graph_only(history, fallback, agent.min_steps, kl_stable)
        _update(metrics["graph_only"], gt_norm, snap, steps, reason)

        if offline_prm:
            steps, reason, snap = decide_full(history, fallback, agent.min_steps, kl_stable, offline_prm)
            _update(metrics["full"], gt_norm, snap, steps, reason)

        steps, reason, snap = decide_energy_u_only(history, fallback, agent.min_steps, kl_stable)
        _update(metrics["energy_u_only"], gt_norm, snap, steps, reason)

        steps, reason, snap = decide_energy_v_only(history, fallback, agent.min_steps, kl_stable)
        _update(metrics["energy_v_only"], gt_norm, snap, steps, reason)

        steps, reason, snap = decide_energy_full(history, fallback, agent.min_steps, kl_stable)
        _update(metrics["energy_full"], gt_norm, snap, steps, reason)

        # Record trajectory
        trajectories.append({
            "query_id": qid, "head": head, "tail": tail, "ground_truth": gt,
            "total_steps": len(history), "history": history,
        })

        # Progress summary
        pred = norm_pred(history[-1]["conclusion"]) if history else "Uncertain"
        status = "OK" if pred == gt_norm else "MISS"
        print(f"  -> {pred} (steps={len(history)}) [{status}]")

    # Finalize
    summary = {name: _finalize(m) for name, m in metrics.items()}
    if not offline_prm:
        summary.pop("full", None)

    # Print table
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    print(_fmt_table(summary))

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = out_dir / f"ablation_{ts}.json"
    with open(result_path, "w") as f:
        json.dump({
            "meta": {
                "benchmark": str(args.benchmark), "total": len(trajectories),
                "multihop": args.multihop, "limit": args.limit,
                "timestamp": ts, "prm_model": prm_path,
                "lambda_u": args.lambda_u, "lambda_v": args.lambda_v,
                "quality_floor": args.quality_floor,
            },
            "configs": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_path}")

    # Save trajectories separately (large file)
    traj_path = out_dir / f"trajectories_{ts}.json"
    with open(traj_path, "w") as f:
        json.dump(trajectories, f, ensure_ascii=False)
    print(f"Trajectories saved to {traj_path}")


if __name__ == "__main__":
    main()
