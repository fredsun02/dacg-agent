#!/usr/bin/env python3
"""Offline threshold sweep (Exp A) + calibration (Exp C) on pooled-201 trajectories.

Aligned with kgsa-v3 paper: is_correct() treats Uncertain as correct for NoEffect.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/home/thu/DRKG/KGSA")
TEST_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json"
VAL_TRAJ = BASE / "Stage5_Agent/trajectories/step2_v3_val/trajectories_20260305_222147.json"
TEST_TAGGED = BASE / "Stage5_Agent/evaluation/redesign/tagged_queries.json"
VAL_TAGGED = BASE / "Stage5_Agent/evaluation/redesign_val/tagged_queries.json"
OUT_RESULTS = BASE / "Stage5_Agent/evaluation/redesign/results"
OUT_FIG = BASE / "paper/kgsa-v3/figures"

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{mathptmx}',
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.3, 'grid.alpha': 0.15,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
})

C_ADA = '#4575b4'
C_PRM = '#d62728'
C_FIX = '#878787'
C_ORA = '#2ca02c'


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
    """Paper's correctness: Uncertain counts as correct for NoEffect."""
    pred = norm_pred(pred)
    if pred in ('NoEvidence', 'Uncertain'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def step_conclusion(step):
    return step.get('conclusion', 'Uncertain')


# ── Data loading (matching kgsa-v3 generate_all.py) ──────────────────────

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


def _update_metrics(m, gt, pred_raw, stop_step, ever_correct):
    correct = is_correct(pred_raw, gt)
    m["total"] += 1
    m["correct"] += int(correct)
    m["steps_sum"] += stop_step
    m["class_total"][gt] += 1
    m["class_correct"][gt] += int(correct)
    if ever_correct:
        m["drift_eligible"] += 1
        if not correct:
            m["drift"] += 1


# ── Experiment A: Threshold Sweep ─────────────────────────────────────────

def compute_threshold_sweep(trajs, taus):
    rows = []
    for tau in taus:
        for mode in ("conf_only", "conf_prm"):
            m = init_metrics()
            for tr in trajs:
                hist = tr["history"]
                gt = tr["ground_truth"]

                stop_idx = None
                for i, s in enumerate(hist):
                    if float(s.get("confidence", 0) or 0) > tau:
                        stop_idx = i
                        break

                if stop_idx is None:
                    if mode == "conf_only":
                        stop_idx = len(hist) - 1
                    else:
                        rewards = [float(s.get("prm_reward", 0) or 0) for s in hist]
                        stop_idx = int(np.argmax(rewards))

                stop_idx = max(0, min(stop_idx, len(hist) - 1))
                pred_raw = step_conclusion(hist[stop_idx])
                ever_correct = _ever_correct_up_to(hist, stop_idx, gt)
                _update_metrics(m, gt, pred_raw, stop_idx + 1, ever_correct)

            rows.append({"tau": tau, "mode": mode, **finalize_metrics(m)})
    return rows


def plot_threshold_sweep(rows, taus, out_path):
    by_key = {(r["tau"], r["mode"]): r for r in rows}

    fig, ax1 = plt.subplots(figsize=(3.5, 2.4))
    ax2 = ax1.twinx()

    for mode, color, lbl in [("conf_only", C_ADA, "Conf-only"),
                              ("conf_prm", C_PRM, "Conf+PRM")]:
        noe = [by_key[(t, mode)]["noe_acc"] for t in taus]
        steps = [by_key[(t, mode)]["mean_steps"] for t in taus]
        ax1.plot(taus, noe, 'o-', color=color, ms=3.5, lw=1.3,
                 label=r'%s NoE Acc' % lbl, markeredgecolor='white', markeredgewidth=0.3)
        ax2.plot(taus, steps, 's--', color=color, ms=2.5, lw=0.9, alpha=0.55,
                 label=r'%s Steps' % lbl)

    ax1.set_xlabel(r'Confidence threshold $\tau$')
    ax1.set_ylabel(r'NoEffect accuracy (\%)')
    ax2.set_ylabel(r'Mean steps')
    ax1.set_xlim(0.45, 1.0)
    ax1.grid(True, alpha=0.15)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc='center left')

    plt.tight_layout()
    plt.savefig(out_path, facecolor='white')
    plt.close()
    print(f"fig_threshold_sweep: saved to {out_path}")


# ── Experiment C: Calibration ─────────────────────────────────────────────

def compute_calibration(trajs, n_bins=10):
    pairs = []
    for tr in trajs:
        gt = tr["ground_truth"]
        for s in tr["history"]:
            conf = max(0.0, min(1.0, float(s.get("confidence", 0) or 0)))
            pred_raw = step_conclusion(s)
            if norm_pred(pred_raw) == "Uncertain":
                continue
            pairs.append((conf, int(is_correct(pred_raw, gt))))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = [{"count": 0, "conf_sum": 0.0, "acc_sum": 0} for _ in range(n_bins)]
    for conf, correct in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins_data[idx]["count"] += 1
        bins_data[idx]["conf_sum"] += conf
        bins_data[idx]["acc_sum"] += correct

    total = len(pairs)
    ece = 0.0
    bin_results = []
    for i, b in enumerate(bins_data):
        cnt = b["count"]
        mean_conf = b["conf_sum"] / cnt if cnt else (edges[i] + edges[i+1]) / 2
        acc = b["acc_sum"] / cnt if cnt else 0.0
        ece += abs(acc - mean_conf) * cnt / total if total else 0.0
        bin_results.append({
            "bin_lo": round(float(edges[i]), 3),
            "bin_hi": round(float(edges[i+1]), 3),
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(acc, 4),
            "count": cnt,
        })
    return {"n_pairs": total, "ece": round(ece, 4), "bins": bin_results}


def plot_calibration(calib, out_path):
    bins = calib["bins"]
    centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    accs = [b["accuracy"] for b in bins]
    counts = [b["count"] for b in bins]
    width = 0.08

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(centers, accs, width=width, color=C_ADA, alpha=0.75, edgecolor='white', lw=0.3,
           label='Observed accuracy', zorder=3)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7, alpha=0.4, label='Perfectly calibrated')

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=width, color=C_FIX, alpha=0.12, zorder=1)
    ax2.set_ylabel(r'Sample count', fontsize=7, color='#999')
    ax2.tick_params(axis='y', colors='#999')

    ax.set_xlabel(r'Confidence')
    ax.set_ylabel(r'Accuracy')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.15, zorder=0)

    ece_val = calib["ece"]
    ax.text(0.05, 0.92, r'ECE $= %.3f$' % ece_val, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#ccc'))
    ax.legend(fontsize=6.5, loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, facecolor='white')
    plt.close()
    print(f"fig_calibration: saved to {out_path} (ECE={ece_val:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    trajs = _filtered_trajs()
    print(f"Pooled: {len(trajs)} trajectories")

    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sweep = compute_threshold_sweep(trajs, taus)
    with open(OUT_RESULTS / "threshold_sweep.json", "w") as f:
        json.dump({"meta": {"n": len(trajs), "taus": taus}, "rows": sweep}, f, indent=2)
    plot_threshold_sweep(sweep, taus, OUT_FIG / "fig_threshold_sweep.pdf")

    print(f"\n{'tau':>5} {'mode':<10} {'Acc':>6} {'NoE':>6} {'Ben':>6} {'Drift':>6} {'Steps':>6}")
    print("-" * 50)
    for r in sweep:
        print(f"{r['tau']:>5.2f} {r['mode']:<10} {r['overall_acc']:>5.1f}% {r['noe_acc']:>5.1f}% "
              f"{r['ben_acc']:>5.1f}% {r['drift_rate']:>5.1f}% {r['mean_steps']:>6.2f}")

    calib = compute_calibration(trajs)
    with open(OUT_RESULTS / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    plot_calibration(calib, OUT_FIG / "fig_calibration.pdf")


if __name__ == "__main__":
    main()
