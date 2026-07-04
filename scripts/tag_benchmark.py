#!/usr/bin/env python3
"""Tag benchmark queries with evidence alignment and publication bias risk."""

import json
import argparse
from collections import Counter
from pathlib import Path


def is_correct(pred, gt):
    if pred in ('NoEvidence', 'Uncertain'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    test = json.load(open(args.benchmark))
    traj = json.load(open(args.trajectories))
    test_map = {q['id']: q for q in test}
    traj_map = {t['query_id']: t for t in traj}

    # --- Define subsets ---
    clean_ids = set()
    for q in test:
        gl = q.get('gold_label')
        if gl in (None, 'None', ''):
            continue
        if gl != q['ground_truth']:
            continue
        if q['ground_truth'] in ('Beneficial', 'NoEffect'):
            clean_ids.add(q['id'])

    # --- Tag each query ---
    tagged = []
    for q in test:
        qid = q['id']
        t = traj_map.get(qid)
        if not t or not t.get('history'):
            continue

        gt = q['ground_truth']
        hist = t['history']

        # Count extraction polarity
        n_treat = 0
        n_noeffect = 0
        n_other = 0
        for s in hist:
            n_treat += s.get('new_triples', 0)  # approximate

        # Use posterior to estimate bias
        last_ps = hist[-1].get('posterior_scores', {})
        ben_posterior = last_ps.get('beneficial', 0)
        neu_posterior = last_ps.get('neutral', 0)

        # Oracle step
        oracle_step = None
        for i, s in enumerate(hist):
            if is_correct(s.get('conclusion', ''), gt):
                oracle_step = i + 1
                break

        # Drift
        has_correct = oracle_step is not None
        final_correct = is_correct(hist[-1].get('conclusion', ''), gt)
        drifted = has_correct and not final_correct

        # Publication bias risk
        if gt == 'NoEffect' and ben_posterior > 0.5:
            pub_bias_risk = 'high'
        elif gt == 'NoEffect' and ben_posterior > 0.3:
            pub_bias_risk = 'medium'
        else:
            pub_bias_risk = 'low'

        tagged.append({
            'id': qid,
            'head': q['head_entity'],
            'tail': q['tail_entity'],
            'ground_truth': gt,
            'gold_label': q.get('gold_label'),
            'grade_certainty': q.get('grade_certainty'),
            'in_clean_103': qid in clean_ids,
            'total_steps': t['total_steps'],
            'oracle_step': oracle_step,
            'has_correct': has_correct,
            'final_correct': final_correct,
            'drifted': drifted,
            'ben_posterior': round(ben_posterior, 4),
            'neu_posterior': round(neu_posterior, 4),
            'pub_bias_risk': pub_bias_risk,
        })

    # --- Output ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "tagged_queries.json", "w") as f:
        json.dump(tagged, f, indent=2)

    # Clean-103 subset
    clean = [q for q in test if q['id'] in clean_ids]
    with open(out_dir / "clean_103.json", "w") as f:
        json.dump(clean, f, indent=2)

    # Report
    clean_tagged = [t for t in tagged if t['in_clean_103']]
    print(f"Total queries: {len(tagged)}")
    print(f"Clean-103: {len(clean_tagged)}")
    print(f"  GT: {dict(Counter(t['ground_truth'] for t in clean_tagged))}")
    print(f"  Has correct step: {sum(t['has_correct'] for t in clean_tagged)}/{len(clean_tagged)}")
    print(f"  Drifted: {sum(t['drifted'] for t in clean_tagged)}/{len(clean_tagged)}")
    print(f"  Pub bias risk: {dict(Counter(t['pub_bias_risk'] for t in clean_tagged))}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
