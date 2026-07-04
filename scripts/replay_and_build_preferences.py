#!/usr/bin/env python3
"""
Replay trajectories through GraphStore to extract 18-dim features and build PRM preference pairs.

Usage:
    python replay_and_build_preferences.py \
        --trajectory-dir /data/DRKG/KGSA/Stage4/Task9_online_data/combined_trajectories_20260121_061308/short/ \
        --output-dir /data/DRKG/KGSA/Stage5_Agent/prm_data/
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.entity_resolver import EntityResolver
from agent.graph_store import GraphStore
from agent.path_inference import PathInference, ConclusionResult
from agent.graph_features import (
    GraphFeatureExtractor,
    GraphFeatures,
    GRAPH_FEATURE_NAMES,
    GRAPH_FEATURE_NORMS,
)


def replay_trajectory(trajectory: dict) -> list:
    """Replay a single trajectory through GraphStore, returning per-step records."""
    steps = trajectory.get("steps", [])
    if not steps:
        return []

    head_name = trajectory.get("query", {}).get("head", "")
    tail_name = trajectory.get("query", {}).get("tail", "")
    if not head_name or not tail_name:
        return []

    resolver = EntityResolver()
    graph = GraphStore(resolver=resolver)
    inference = PathInference()
    extractor = GraphFeatureExtractor()

    head_id = resolver.resolve(head_name, create=True).id
    tail_id = resolver.resolve(tail_name, create=True).id

    prev_features: GraphFeatures | None = None
    records = []

    for step in steps:
        # Add triples from all extractions in this step
        for extraction in step.get("extractions", []):
            pmid = extraction.get("pmid", "")
            for triple in extraction.get("triples", []):
                t = dict(triple)
                if pmid and not t.get("pmid"):
                    t["pmid"] = pmid
                graph.add_triple(t)

        # Infer conclusion via path inference
        conclusion_result = inference.infer_conclusion(graph, head_id, tail_id)

        # Extract 18-dim features
        features = extractor.extract(graph, head_id, tail_id, prev_features)
        prev_features = features

        records.append({
            "step": step.get("step", len(records) + 1),
            "features_dict": features.to_dict(),
            "conclusion": conclusion_result.conclusion,
            "confidence": conclusion_result.confidence,
            "is_correct": step.get("is_correct", False),
            "conclusion_confidence": step.get("conclusion_confidence", conclusion_result.confidence),
        })

    return records


def _generate_preferences(trajectory: dict, records: list, prefix: str = "pref") -> list:
    """Generate preference pairs from replayed trajectory records."""
    if not records:
        return []

    prefs = []
    counter = [0]
    query_id = trajectory.get("query_id", "q")

    def make_pair(better_rec, worse_rec, ptype, margin):
        counter[0] += 1
        return {
            "id": f"{prefix}_{query_id}_{counter[0]:04d}",
            "query_id": query_id,
            "state_better": better_rec["features_dict"],
            "state_worse": worse_rec["features_dict"],
            "preference_type": ptype,
            "margin": margin,
        }

    optimal_step = trajectory.get("optimal_stop_step")
    if not optimal_step:
        correct_recs = [r for r in records if r.get("is_correct")]
        if correct_recs:
            best = max(correct_recs, key=lambda r: r.get("conclusion_confidence", 0.0))
            optimal_step = best["step"]
        else:
            return _fallback_preferences(records, make_pair)

    optimal_idx = optimal_step - 1
    if optimal_idx < 0 or optimal_idx >= len(records):
        return _fallback_preferences(records, make_pair)

    optimal_rec = records[optimal_idx]
    optimal_conf = optimal_rec.get("conclusion_confidence", 0.0)

    # optimal vs early
    for rec in records[:optimal_idx]:
        rec_conf = rec.get("conclusion_confidence", 0.0)
        if not rec.get("is_correct"):
            prefs.append(make_pair(optimal_rec, rec, "optimal_vs_early_incorrect", max(0.1, optimal_conf - rec_conf)))
        elif rec_conf + 0.1 < optimal_conf:
            prefs.append(make_pair(optimal_rec, rec, "optimal_vs_early_lowconf", max(0.05, optimal_conf - rec_conf)))

    # optimal vs late
    for rec in records[optimal_idx + 1:]:
        efficiency_penalty = 0.05 * (rec["step"] - optimal_step)
        if rec.get("is_correct"):
            prefs.append(make_pair(optimal_rec, rec, "optimal_vs_late_correct", efficiency_penalty))
        else:
            prefs.append(make_pair(optimal_rec, rec, "optimal_vs_late_incorrect", efficiency_penalty + 0.2))

    # correct vs incorrect (nearby steps)
    correct_recs = [r for r in records if r.get("is_correct")]
    incorrect_recs = [r for r in records if not r.get("is_correct")]
    for c in correct_recs[:5]:
        for ic in incorrect_recs[:5]:
            if abs(c["step"] - ic["step"]) <= 5:
                prefs.append(make_pair(c, ic, "correct_vs_incorrect", max(0.1, c.get("conclusion_confidence", 0.0))))

    # high conf vs low conf among correct
    if len(correct_recs) >= 2:
        sorted_correct = sorted(correct_recs, key=lambda r: r.get("conclusion_confidence", 0.0), reverse=True)
        best = sorted_correct[0]
        for other in sorted_correct[1:min(4, len(sorted_correct))]:
            diff = best.get("conclusion_confidence", 0.0) - other.get("conclusion_confidence", 0.0)
            if diff > 0.05:
                prefs.append(make_pair(best, other, "high_conf_vs_low_conf", diff))

    # stable vs unstable
    for i in range(len(records) - 1):
        curr, nxt = records[i], records[i + 1]
        if curr.get("is_correct") and nxt.get("conclusion") != curr.get("conclusion"):
            prefs.append(make_pair(curr, nxt, "stable_vs_unstable", 0.15))

    return prefs


def _fallback_preferences(records: list, make_pair) -> list:
    prefs = []
    if len(records) < 2:
        return prefs

    # early vs late (both incorrect)
    third = max(1, len(records) // 3)
    for early in records[:third][:3]:
        for late in records[-third:][:3]:
            if early["step"] < late["step"]:
                prefs.append(make_pair(early, late, "early_vs_late_both_incorrect",
                                       0.03 * (late["step"] - early["step"])))

    # low marginal gain
    for i in range(len(records) - 1):
        curr, nxt = records[i], records[i + 1]
        if nxt["features_dict"].get("marginal_gain", 1.0) < 0.1:
            prefs.append(make_pair(curr, nxt, "low_marginal_gain", 0.1))

    return prefs


def process_trajectories(trajectory_dir: Path) -> tuple:
    """Load and replay all trajectory files, returning (trajectories, all_records)."""
    files = sorted(trajectory_dir.glob("trajectory_*.json"))
    if not files:
        print(f"No trajectory files found in {trajectory_dir}")
        return [], []

    all_pairs = []  # (trajectory_dict, records)
    skipped = 0

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            traj = json.load(f)
        records = replay_trajectory(traj)
        if not records:
            skipped += 1
            continue
        all_pairs.append((traj, records))

    print(f"Replayed {len(all_pairs)} trajectories ({skipped} skipped, {len(files)} total files)")
    return all_pairs


def build_preferences(all_pairs: list, prefix: str = "pref") -> list:
    """Generate preference pairs from all replayed trajectories."""
    prefs = []
    for traj, records in all_pairs:
        prefs.extend(_generate_preferences(traj, records, prefix))
    return prefs


def save_preferences(prefs: list, output_path: Path, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "preferences": prefs}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Replay trajectories and build 18-dim PRM preference pairs")
    parser.add_argument("--trajectory-dir", required=True, help="Directory with trajectory JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for preference files")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-steps", type=int, default=1, help="Min trajectory steps to include")
    parser.add_argument("--min-triples", type=int, default=1, help="Min total triples to include")
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)

    # Also check subdirectories (short/, long/)
    subdirs = [d for d in traj_dir.iterdir() if d.is_dir() and d.name in ("short", "long")]
    if subdirs:
        all_pairs = []
        for sd in sorted(subdirs):
            print(f"\nProcessing {sd.name}/...")
            pairs = process_trajectories(sd)
            all_pairs.extend(pairs)
    else:
        all_pairs = process_trajectories(traj_dir)

    # Filter by min-steps and min-triples
    if args.min_steps > 1 or args.min_triples > 1:
        before = len(all_pairs)
        all_pairs = [
            (traj, records) for traj, records in all_pairs
            if len(records) >= args.min_steps
            and sum(r["features_dict"].get("total_evidence", 0) for r in records[-1:]) >= args.min_triples
        ]
        print(f"\nFiltered: {before} -> {len(all_pairs)} (min_steps={args.min_steps}, min_triples={args.min_triples})")

    if not all_pairs:
        print("No valid trajectories found. Exiting.")
        return

    # Split
    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    n = len(all_pairs)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + (1 - args.train_ratio) / 2))

    train_pairs = all_pairs[:train_end]
    val_pairs = all_pairs[train_end:val_end]
    test_pairs = all_pairs[val_end:]

    train_prefs = build_preferences(train_pairs, "train")
    val_prefs = build_preferences(val_pairs, "val")
    test_prefs = build_preferences(test_pairs, "test")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "feature_names": GRAPH_FEATURE_NAMES,
        "feature_norms": GRAPH_FEATURE_NORMS,
        "feature_dim": len(GRAPH_FEATURE_NAMES),
        "source": str(traj_dir),
        "total_trajectories": len(all_pairs),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }

    save_preferences(train_prefs, output_dir / "train_preferences.json", metadata)
    save_preferences(val_prefs, output_dir / "val_preferences.json", metadata)
    save_preferences(test_prefs, output_dir / "test_preferences.json", metadata)

    print(f"\nPreferences saved to {output_dir}/")
    print(f"  Train: {len(train_prefs)}")
    print(f"  Val:   {len(val_prefs)}")
    print(f"  Test:  {len(test_prefs)}")

    all_types = Counter(p["preference_type"] for p in train_prefs + val_prefs + test_prefs)
    print("  Type distribution:")
    for ptype, count in all_types.most_common():
        print(f"    {ptype}: {count}")


if __name__ == "__main__":
    main()
