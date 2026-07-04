#!/usr/bin/env python3
"""E3 Single-Round KG baseline: extract history[0].conclusion from trajectories."""

import json, argparse
from pathlib import Path


def is_correct(pred, gt):
    if pred in ('NoEvidence', 'Uncertain'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def load_clean103_ids(tagged_path):
    with open(tagged_path) as f:
        tagged = json.load(f)
    return {q["id"] for q in tagged if q.get("in_clean_103")}


def find_oracle_step(history, gt):
    for i, s in enumerate(history):
        if is_correct(s.get('conclusion', ''), gt):
            return i + 1
    return None


def compute_metrics(trajectories, query_ids=None):
    results = []
    for traj in trajectories:
        qid = traj["query_id"]
        if query_ids is not None and qid not in query_ids:
            continue
        gt = traj["ground_truth"]
        hist = traj["history"]
        if not hist:
            continue
        pred = hist[0]["conclusion"]
        correct = is_correct(pred, gt)
        oracle_step = find_oracle_step(hist, gt)
        regret = abs(1 - oracle_step) if oracle_step else len(hist)
        results.append({
            "query_id": qid, "ground_truth": gt, "prediction": pred,
            "correct": correct, "regret": regret,
        })
    return results


def summarize(results, label="All"):
    n = len(results)
    if n == 0:
        return {}
    acc = sum(r["correct"] for r in results) / n * 100
    regret = sum(r["regret"] for r in results) / n
    by_class = {}
    for cls in ["Beneficial", "NoEffect", "Harmful"]:
        subset = [r for r in results if r["ground_truth"] == cls]
        if subset:
            by_class[cls] = {
                "n": len(subset),
                "acc": round(sum(r["correct"] for r in subset) / len(subset) * 100, 1)
            }
    return {"label": label, "n": n, "acc": round(acc, 1), "regret": round(regret, 2),
            "steps": 1.0, "drift": 0.0, "by_class": by_class}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", default="Stage5_Agent/trajectories/step2_v3_test/trajectories_20260305_053618.json")
    p.add_argument("--tagged", default="Stage5_Agent/evaluation/redesign/tagged_queries.json")
    p.add_argument("--output", default="Stage5_Agent/evaluation/redesign/e3_baseline.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.trajectories) as f:
        trajectories = json.load(f)
    clean_ids = load_clean103_ids(root / args.tagged)

    clean_results = compute_metrics(trajectories, clean_ids)
    full_results = compute_metrics(trajectories)

    output = {
        "clean_103": summarize(clean_results, "Clean-103"),
        "full_173": summarize(full_results, "Full-173"),
    }
    print(json.dumps(output, indent=2))
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
