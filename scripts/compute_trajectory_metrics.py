#!/usr/bin/env python3
"""Compute trajectory-level metrics for evaluation redesign."""

import json
import argparse
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path


def is_correct(pred, gt):
    if pred in ('NoEvidence', 'Uncertain'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def find_oracle_step(history, gt):
    """Return 1-indexed oracle step (first correct), or None."""
    for i, s in enumerate(history):
        if is_correct(s.get('conclusion', ''), gt):
            return i + 1
    return None


def apply_stopping(history, config):
    """Return (stop_step_1indexed, snapshot) for a stopping config."""
    if config.startswith('fixed_k_'):
        k = int(config.split('_')[-1])
        idx = min(k, len(history)) - 1
        return idx + 1, history[idx]

    if config == 'conf_only':
        for i, s in enumerate(history):
            if float(s.get('confidence', 0) or 0) > 0.8:
                return i + 1, s
        return len(history), history[-1]

    if config == 'kl_only':
        kls = [float(s.get('kl', 0) or 0) for s in history]
        for i in range(3, len(kls)):
            if all(kls[j] < 0.05 for j in range(i - 2, i + 1)):
                return i + 1, history[i]
        return len(history), history[-1]

    if config == 'full':
        kls = [float(s.get('kl', 0) or 0) for s in history]
        rewards = [float(s.get('prm_reward', 0) or 0) for s in history]
        # KL stable
        for i in range(3, len(kls)):
            if all(kls[j] < 0.05 for j in range(i - 2, i + 1)):
                return i + 1, history[i]
        # PRM decline (reward drops for 2 consecutive steps after peak)
        if len(rewards) >= 3:
            peak_idx = int(np.argmax(rewards))
            if peak_idx >= 2:
                decline = 0
                for j in range(peak_idx + 1, len(rewards)):
                    if rewards[j] < rewards[j - 1]:
                        decline += 1
                        if decline >= 2:
                            return j + 1, history[j]
                    else:
                        decline = 0
        return len(history), history[-1]

    raise ValueError(f"Unknown config: {config}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    tagged = json.load(open(args.tagged))
    traj = json.load(open(args.trajectories))
    traj_map = {t['query_id']: t for t in traj}

    clean_ids = set(q['id'] for q in tagged if q['in_clean_103'])
    clean_traj = [t for t in traj if t['query_id'] in clean_ids and t.get('history')]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ['fixed_k_3', 'fixed_k_5', 'fixed_k_10', 'fixed_k_20',
               'conf_only', 'kl_only', 'full']

    # === 1. Ablation table ===
    ablation = {}
    for cfg in configs:
        metrics = {
            'correct': 0, 'total': 0, 'steps': [],
            'regrets': [], 'drifts': 0, 'drift_eligible': 0,
            'by_class': defaultdict(lambda: {'correct': 0, 'total': 0}),
        }
        for t in clean_traj:
            gt = t['ground_truth']
            hist = t['history']
            step, snap = apply_stopping(hist, cfg)
            pred = snap.get('conclusion', '')
            correct = is_correct(pred, gt)

            oracle_step = find_oracle_step(hist, gt)
            regret = abs(step - oracle_step) if oracle_step else step

            metrics['total'] += 1
            metrics['steps'].append(step)
            metrics['regrets'].append(regret)
            if correct:
                metrics['correct'] += 1
            metrics['by_class'][gt]['total'] += 1
            if correct:
                metrics['by_class'][gt]['correct'] += 1

            # Drift: had correct step before stop, but stopped at wrong step
            if oracle_step and oracle_step <= step and not correct:
                metrics['drifts'] += 1
            if oracle_step and oracle_step <= step:
                metrics['drift_eligible'] += 1

        acc = metrics['correct'] / metrics['total'] * 100
        avg_steps = np.mean(metrics['steps'])
        avg_regret = np.mean(metrics['regrets'])
        med_regret = np.median(metrics['regrets'])
        drift_rate = (metrics['drifts'] / metrics['drift_eligible'] * 100
                      if metrics['drift_eligible'] > 0 else 0)

        ablation[cfg] = {
            'accuracy': round(acc, 1),
            'avg_steps': round(avg_steps, 2),
            'mean_regret': round(avg_regret, 2),
            'median_regret': round(med_regret, 1),
            'drift_rate': round(drift_rate, 1),
            'n': metrics['total'],
        }
        for cls in ['Beneficial', 'NoEffect']:
            d = metrics['by_class'][cls]
            if d['total'] > 0:
                ablation[cfg][f'acc_{cls[:3].lower()}'] = round(
                    d['correct'] / d['total'] * 100, 1)

    # Oracle row
    oracle_correct = sum(1 for t in clean_traj
                         if find_oracle_step(t['history'], t['ground_truth']) is not None)
    oracle_steps = [find_oracle_step(t['history'], t['ground_truth']) or len(t['history'])
                    for t in clean_traj]
    ablation['oracle'] = {
        'accuracy': round(oracle_correct / len(clean_traj) * 100, 1),
        'avg_steps': round(np.mean(oracle_steps), 2),
        'mean_regret': 0.0,
        'median_regret': 0.0,
        'drift_rate': 0.0,
        'n': len(clean_traj),
    }

    # === 2. Correctness-over-steps curve ===
    max_step = 20
    correctness_curve = {}
    for step in range(1, max_step + 1):
        eligible = [t for t in clean_traj if len(t['history']) >= step]
        if not eligible:
            break
        correct = sum(1 for t in eligible
                      if is_correct(t['history'][step - 1].get('conclusion', ''),
                                    t['ground_truth']))
        correctness_curve[step] = {
            'accuracy': round(correct / len(eligible) * 100, 1),
            'n': len(eligible),
            'correct': correct,
        }

    # === 3. Drift analysis ===
    drift_analysis = {
        'total_queries': len(clean_traj),
        'has_correct_step': sum(1 for t in clean_traj
                                if find_oracle_step(t['history'], t['ground_truth'])),
        'drifted': sum(1 for t in clean_traj
                       if find_oracle_step(t['history'], t['ground_truth'])
                       and not is_correct(t['history'][-1].get('conclusion', ''),
                                          t['ground_truth'])),
        'time_to_correct': {
            'mean': round(np.mean([find_oracle_step(t['history'], t['ground_truth'])
                                   for t in clean_traj
                                   if find_oracle_step(t['history'], t['ground_truth'])]), 2),
            'median': int(np.median([find_oracle_step(t['history'], t['ground_truth'])
                                     for t in clean_traj
                                     if find_oracle_step(t['history'], t['ground_truth'])])),
        },
        'by_class': {},
    }
    for cls in ['Beneficial', 'NoEffect']:
        sub = [t for t in clean_traj if t['ground_truth'] == cls]
        has = sum(1 for t in sub if find_oracle_step(t['history'], cls))
        drifted = sum(1 for t in sub
                      if find_oracle_step(t['history'], cls)
                      and not is_correct(t['history'][-1].get('conclusion', ''), cls))
        drift_analysis['by_class'][cls] = {
            'total': len(sub),
            'has_correct': has,
            'drifted': drifted,
            'drift_rate': round(drifted / has * 100, 1) if has else 0,
        }

    # === 4. PRM analysis ===
    prm_analysis = {
        'correct_peak_reward': {},
        'wrong_peak_reward': {},
        'prm_decline_prevents_drift': 0,
        'prm_decline_total': 0,
    }
    correct_peaks, wrong_peaks = [], []
    for t in clean_traj:
        gt = t['ground_truth']
        rewards = [float(s.get('prm_reward', 0) or 0) for s in t['history']]
        if not rewards:
            continue
        peak = max(rewards)
        if is_correct(t['history'][-1].get('conclusion', ''), gt):
            correct_peaks.append(peak)
        else:
            wrong_peaks.append(peak)

    if correct_peaks:
        prm_analysis['correct_peak_reward'] = {
            'mean': round(np.mean(correct_peaks), 3),
            'std': round(np.std(correct_peaks), 3),
            'n': len(correct_peaks),
        }
    if wrong_peaks:
        prm_analysis['wrong_peak_reward'] = {
            'mean': round(np.mean(wrong_peaks), 3),
            'std': round(np.std(wrong_peaks), 3),
            'n': len(wrong_peaks),
        }

    # PRM peak alignment with oracle
    peak_at_oracle = 0
    peak_total = 0
    for t in clean_traj:
        gt = t['ground_truth']
        rewards = [float(s.get('prm_reward', 0) or 0) for s in t['history']]
        oracle = find_oracle_step(t['history'], gt)
        if not rewards or not oracle:
            continue
        peak_idx = int(np.argmax(rewards))
        peak_total += 1
        if is_correct(t['history'][peak_idx].get('conclusion', ''), gt):
            peak_at_oracle += 1
    prm_analysis['peak_at_correct_step'] = {
        'count': peak_at_oracle,
        'total': peak_total,
        'rate': round(peak_at_oracle / peak_total * 100, 1) if peak_total else 0,
    }

    # === Save all ===
    results = {
        'subset': 'clean_103 (Beneficial + NoEffect, gold==gt)',
        'n_queries': len(clean_traj),
        'gt_distribution': dict(Counter(t['ground_truth'] for t in clean_traj)),
        'ablation': ablation,
        'correctness_curve': correctness_curve,
        'drift_analysis': drift_analysis,
        'prm_analysis': prm_analysis,
    }

    out_path = out_dir / "trajectory_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # === Print summary ===
    print(f"=== Ablation Table (Clean-103, n={len(clean_traj)}) ===\n")
    print(f"{'Config':<15} {'Acc':>5} {'Ben':>5} {'NoE':>5} {'Steps':>6} {'Regret':>7} {'Drift%':>7}")
    print("-" * 58)
    for cfg in configs + ['oracle']:
        a = ablation[cfg]
        ben = a.get('acc_ben', '-')
        noe = a.get('acc_noe', '-')
        print(f"{cfg:<15} {a['accuracy']:>5.1f} {ben:>5} {noe:>5} "
              f"{a['avg_steps']:>6.2f} {a['mean_regret']:>7.2f} {a['drift_rate']:>6.1f}%")

    print(f"\n=== Drift Analysis ===")
    print(f"Has correct step: {drift_analysis['has_correct_step']}/{drift_analysis['total_queries']}")
    print(f"Drifted: {drift_analysis['drifted']} ({drift_analysis['drifted']/drift_analysis['has_correct_step']*100:.1f}%)")
    print(f"Time-to-correct: mean={drift_analysis['time_to_correct']['mean']}, median={drift_analysis['time_to_correct']['median']}")
    for cls, d in drift_analysis['by_class'].items():
        print(f"  {cls}: drift_rate={d['drift_rate']}%")

    print(f"\n=== PRM Analysis ===")
    print(f"Correct queries peak reward: {prm_analysis['correct_peak_reward']}")
    print(f"Wrong queries peak reward: {prm_analysis['wrong_peak_reward']}")
    print(f"PRM peak at correct step: {prm_analysis['peak_at_correct_step']}")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
