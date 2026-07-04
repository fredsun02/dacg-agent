"""
Offline Graph Inference Evaluation

Replays trajectory data through the new graph inference pipeline
to compare graph-based vs counter-based conclusions.
No API calls required - uses pre-extracted triples from trajectory files.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.entity_resolver import EntityResolver
from agent.graph_store import GraphStore
from agent.path_inference import PathInference


@dataclass
class ComparisonResult:
    """Result comparing graph vs counter inference."""
    query_id: str
    head_entity: str
    tail_entity: str
    ground_truth: str

    # Counter-based (baseline)
    counter_conclusion: str
    counter_correct: bool
    counter_scores: Dict[str, int]

    # Graph-based (new)
    graph_conclusion: str
    graph_correct: bool
    graph_confidence: float
    graph_scores: Dict[str, float]
    direct_paths: int
    two_hop_paths: int
    total_evidence: int

    # Analysis
    both_correct: bool
    graph_wins: bool  # graph correct, counter wrong
    counter_wins: bool  # counter correct, graph wrong


def load_trajectory(path: str) -> Dict:
    """Load a single trajectory file."""
    with open(path) as f:
        return json.load(f)


def load_all_trajectories(trajectory_dir: str) -> List[Dict]:
    """Load all trajectory files from a directory."""
    trajectories = []
    traj_path = Path(trajectory_dir)

    for f in sorted(traj_path.glob("trajectory_*.json")):
        try:
            traj = load_trajectory(str(f))
            trajectories.append(traj)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return trajectories


def extract_all_triples(trajectory: Dict) -> List[Dict]:
    """Extract all triples from all steps in a trajectory."""
    triples = []

    for step in trajectory.get("steps", []):
        for extraction in step.get("extractions", []):
            pmid = extraction.get("pmid", "")
            for triple in extraction.get("triples", []):
                # Add pmid to triple for evidence tracking
                triple_with_pmid = dict(triple)
                triple_with_pmid["pmid"] = pmid
                triples.append(triple_with_pmid)

    return triples


def counter_inference(triples: List[Dict]) -> Tuple[str, Dict[str, int]]:
    """
    Original counter-based inference: argmax(beneficial, harmful, neutral).

    Returns:
        (conclusion, counts_dict)
    """
    counts = {"beneficial": 0, "harmful": 0, "neutral": 0}

    for triple in triples:
        direction = triple.get("eval_direction") or triple.get("direction", "")
        direction = direction.lower()

        if direction == "beneficial":
            counts["beneficial"] += 1
        elif direction == "harmful":
            counts["harmful"] += 1
        elif direction in ("neutral", "noeffect", "no_effect"):
            counts["neutral"] += 1

    # Argmax logic
    if counts["beneficial"] > counts["harmful"] and counts["beneficial"] > counts["neutral"]:
        return "Beneficial", counts
    elif counts["harmful"] > counts["beneficial"] and counts["harmful"] > counts["neutral"]:
        return "Harmful", counts
    elif counts["neutral"] > counts["beneficial"] and counts["neutral"] > counts["harmful"]:
        return "NoEffect", counts
    elif counts["beneficial"] == 0 and counts["harmful"] == 0 and counts["neutral"] == 0:
        return "NoEvidence", counts
    else:
        # Tie-breaking: prefer Beneficial > Harmful > NoEffect
        if counts["beneficial"] >= counts["harmful"]:
            return "Beneficial", counts
        else:
            return "Harmful", counts


def graph_inference(triples: List[Dict], head: str, tail: str) -> Tuple[str, float, Dict]:
    """
    New graph-based inference using path scoring logic.

    Key improvement over counter: distinguishes NoEvidence vs NoEffect
    based on MIN_EVIDENCE threshold.

    Since offline trajectories have varying entity names that don't match
    query entities exactly, we use a simplified approach:
    - Treat all triples as direct evidence for the query pair
    - Apply path scoring formula: score = confidence * log(1 + evidence_count)
    - Use MIN_EVIDENCE threshold to distinguish NoEvidence vs NoEffect

    Returns:
        (conclusion, confidence, result_dict)
    """
    # Lower threshold for offline evaluation since trajectories are short
    # The key insight is the scoring formula, not the hard threshold
    MIN_EVIDENCE = 1
    MARGIN = 0.05

    # Aggregate scores by direction using path scoring formula
    import math

    direction_evidence = {"beneficial": [], "harmful": [], "neutral": []}

    for triple in triples:
        direction = triple.get("eval_direction") or triple.get("direction", "")
        direction = direction.lower()
        confidence = float(triple.get("confidence", 0.5))
        is_causal = triple.get("is_causal", True)

        # Weight causal evidence higher
        weight = 1.0 if is_causal else 0.5

        if direction == "beneficial":
            direction_evidence["beneficial"].append(confidence * weight)
        elif direction == "harmful":
            direction_evidence["harmful"].append(confidence * weight)
        elif direction in ("neutral", "noeffect", "no_effect"):
            direction_evidence["neutral"].append(confidence * weight)

    # Compute scores using path scoring formula
    def compute_score(evidences: List[float]) -> float:
        if not evidences:
            return 0.0
        mean_conf = sum(evidences) / len(evidences)
        evidence_bonus = math.log(1 + len(evidences))
        return mean_conf * evidence_bonus

    s_pos = compute_score(direction_evidence["beneficial"])
    s_neg = compute_score(direction_evidence["harmful"])
    s_neu = compute_score(direction_evidence["neutral"])

    total_evidence = (len(direction_evidence["beneficial"]) +
                      len(direction_evidence["harmful"]) +
                      len(direction_evidence["neutral"]))

    # Inference logic with NoEvidence vs NoEffect distinction
    if total_evidence < MIN_EVIDENCE:
        conclusion = "NoEvidence"
        confidence = 0.0
        is_decisive = False
    elif s_neu > s_pos + MARGIN and s_neu > s_neg + MARGIN:
        conclusion = "NoEffect"
        total_score = s_pos + s_neg + s_neu
        confidence = s_neu / total_score if total_score > 0 else 0.5
        is_decisive = True
    elif s_pos > s_neg + MARGIN:
        conclusion = "Beneficial"
        total_score = s_pos + s_neg + s_neu
        confidence = s_pos / total_score if total_score > 0 else 0.5
        is_decisive = True
    elif s_neg > s_pos + MARGIN:
        conclusion = "Harmful"
        total_score = s_pos + s_neg + s_neu
        confidence = s_neg / total_score if total_score > 0 else 0.5
        is_decisive = True
    else:
        conclusion = "Uncertain"
        confidence = 0.5
        is_decisive = False

    return conclusion, confidence, {
        "scores": {"beneficial": s_pos, "harmful": s_neg, "neutral": s_neu},
        "direct_paths": total_evidence,
        "two_hop_paths": 0,
        "total_evidence": total_evidence,
        "is_decisive": is_decisive,
    }


def normalize_label(label: str) -> str:
    """Normalize labels for comparison."""
    label = label.lower().strip()

    mappings = {
        "beneficial": "Beneficial",
        "treat": "Beneficial",
        "positive": "Beneficial",
        "harmful": "Harmful",
        "negative": "Harmful",
        "cause": "Harmful",
        "noeffect": "NoEffect",
        "no_effect": "NoEffect",
        "no effect": "NoEffect",
        "neutral": "NoEffect",
        "uncertain": "Uncertain",
        "unknown": "Uncertain",
        "noevidence": "NoEvidence",
        "no_evidence": "NoEvidence",
    }

    return mappings.get(label, label.title())


def evaluate_trajectory(trajectory: Dict) -> Optional[ComparisonResult]:
    """Evaluate a single trajectory with both methods."""
    query_id = trajectory.get("query_id", "unknown")
    query = trajectory.get("query", {})
    head = query.get("head", "")
    tail = query.get("tail", "")
    ground_truth = trajectory.get("ground_truth", "")

    if not head or not tail or not ground_truth:
        return None

    gt_normalized = normalize_label(ground_truth)

    # Extract all triples
    triples = extract_all_triples(trajectory)

    if not triples:
        # No evidence case
        counter_conclusion = "NoEvidence"
        counter_scores = {"beneficial": 0, "harmful": 0, "neutral": 0}
        graph_conclusion = "NoEvidence"
        graph_confidence = 0.0
        graph_result = {"scores": {}, "direct_paths": 0, "two_hop_paths": 0, "total_evidence": 0}
    else:
        # Counter-based inference
        counter_conclusion, counter_scores = counter_inference(triples)

        # Graph-based inference
        graph_conclusion, graph_confidence, graph_result = graph_inference(triples, head, tail)

    # Normalize conclusions
    counter_normalized = normalize_label(counter_conclusion)
    graph_normalized = normalize_label(graph_conclusion)

    # Check correctness
    counter_correct = counter_normalized == gt_normalized
    graph_correct = graph_normalized == gt_normalized

    return ComparisonResult(
        query_id=query_id,
        head_entity=head,
        tail_entity=tail,
        ground_truth=gt_normalized,
        counter_conclusion=counter_normalized,
        counter_correct=counter_correct,
        counter_scores=counter_scores,
        graph_conclusion=graph_normalized,
        graph_correct=graph_correct,
        graph_confidence=graph_confidence,
        graph_scores=graph_result.get("scores", {}),
        direct_paths=graph_result.get("direct_paths", 0),
        two_hop_paths=graph_result.get("two_hop_paths", 0),
        total_evidence=graph_result.get("total_evidence", 0),
        both_correct=counter_correct and graph_correct,
        graph_wins=graph_correct and not counter_correct,
        counter_wins=counter_correct and not graph_correct,
    )


def run_offline_evaluation(
    trajectory_dir: str,
    output_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run offline evaluation comparing graph vs counter inference.

    Args:
        trajectory_dir: Directory containing trajectory JSON files
        output_path: Optional path to save results
        max_trajectories: Limit number of trajectories (for testing)
        verbose: Print progress

    Returns:
        Evaluation results dictionary
    """
    print(f"Loading trajectories from {trajectory_dir}...")
    trajectories = load_all_trajectories(trajectory_dir)

    if max_trajectories:
        trajectories = trajectories[:max_trajectories]

    print(f"Evaluating {len(trajectories)} trajectories...")

    results = []

    # Per-class tracking
    class_stats = defaultdict(lambda: {
        "counter_correct": 0, "graph_correct": 0, "total": 0,
        "graph_wins": 0, "counter_wins": 0
    })

    for i, traj in enumerate(trajectories):
        result = evaluate_trajectory(traj)

        if result is None:
            continue

        results.append(result)

        # Update class stats
        gt = result.ground_truth
        class_stats[gt]["total"] += 1
        if result.counter_correct:
            class_stats[gt]["counter_correct"] += 1
        if result.graph_correct:
            class_stats[gt]["graph_correct"] += 1
        if result.graph_wins:
            class_stats[gt]["graph_wins"] += 1
        if result.counter_wins:
            class_stats[gt]["counter_wins"] += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(trajectories)}")

    # Compute overall metrics
    total = len(results)
    counter_correct_total = sum(1 for r in results if r.counter_correct)
    graph_correct_total = sum(1 for r in results if r.graph_correct)
    both_correct_total = sum(1 for r in results if r.both_correct)
    graph_wins_total = sum(1 for r in results if r.graph_wins)
    counter_wins_total = sum(1 for r in results if r.counter_wins)

    counter_accuracy = counter_correct_total / total if total > 0 else 0.0
    graph_accuracy = graph_correct_total / total if total > 0 else 0.0

    # Per-class accuracy
    class_accuracy = {}
    for cls, stats in class_stats.items():
        if stats["total"] > 0:
            class_accuracy[cls] = {
                "counter_accuracy": stats["counter_correct"] / stats["total"],
                "graph_accuracy": stats["graph_correct"] / stats["total"],
                "total": stats["total"],
                "graph_wins": stats["graph_wins"],
                "counter_wins": stats["counter_wins"],
            }

    summary = {
        "total": total,
        "counter_accuracy": counter_accuracy,
        "graph_accuracy": graph_accuracy,
        "improvement": graph_accuracy - counter_accuracy,
        "both_correct": both_correct_total,
        "graph_wins": graph_wins_total,
        "counter_wins": counter_wins_total,
        "class_accuracy": class_accuracy,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("OFFLINE EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal trajectories: {total}")
    print(f"\n{'Method':<20} {'Accuracy':<15} {'Correct':<10}")
    print("-" * 45)
    print(f"{'Counter (baseline)':<20} {counter_accuracy*100:>6.2f}%        {counter_correct_total}")
    print(f"{'Graph (new)':<20} {graph_accuracy*100:>6.2f}%        {graph_correct_total}")
    print(f"\nImprovement: {(graph_accuracy - counter_accuracy)*100:+.2f}%")
    print(f"\nGraph wins (graph correct, counter wrong): {graph_wins_total}")
    print(f"Counter wins (counter correct, graph wrong): {counter_wins_total}")

    print(f"\n{'Class':<15} {'Counter':<12} {'Graph':<12} {'Total':<8} {'Graph Wins':<12}")
    print("-" * 60)
    for cls in sorted(class_accuracy.keys()):
        stats = class_accuracy[cls]
        print(f"{cls:<15} {stats['counter_accuracy']*100:>6.2f}%     {stats['graph_accuracy']*100:>6.2f}%     {stats['total']:<8} {stats['graph_wins']}")

    # Detailed analysis for NoEffect (the key problem)
    if "NoEffect" in class_accuracy:
        ne_stats = class_accuracy["NoEffect"]
        print(f"\n{'='*70}")
        print("KEY METRIC: NoEffect Class (the main problem)")
        print(f"{'='*70}")
        print(f"Counter accuracy: {ne_stats['counter_accuracy']*100:.2f}%")
        print(f"Graph accuracy:   {ne_stats['graph_accuracy']*100:.2f}%")
        print(f"Improvement:      {(ne_stats['graph_accuracy'] - ne_stats['counter_accuracy'])*100:+.2f}%")

    # Save results
    if output_path:
        output = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "trajectory_dir": trajectory_dir,
            "results": [
                {
                    "query_id": r.query_id,
                    "ground_truth": r.ground_truth,
                    "counter_conclusion": r.counter_conclusion,
                    "counter_correct": r.counter_correct,
                    "graph_conclusion": r.graph_conclusion,
                    "graph_correct": r.graph_correct,
                    "graph_confidence": r.graph_confidence,
                    "graph_wins": r.graph_wins,
                    "counter_wins": r.counter_wins,
                }
                for r in results
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return summary


def analyze_disagreements(
    trajectory_dir: str,
    max_examples: int = 10,
) -> None:
    """Analyze cases where graph and counter disagree."""
    trajectories = load_all_trajectories(trajectory_dir)

    graph_wins = []
    counter_wins = []

    for traj in trajectories:
        result = evaluate_trajectory(traj)
        if result is None:
            continue

        if result.graph_wins:
            graph_wins.append((traj, result))
        elif result.counter_wins:
            counter_wins.append((traj, result))

    print(f"\n{'='*70}")
    print(f"DISAGREEMENT ANALYSIS")
    print(f"{'='*70}")

    print(f"\n--- Graph Wins ({len(graph_wins)} cases) ---")
    for traj, result in graph_wins[:max_examples]:
        print(f"\n[{result.query_id}] GT: {result.ground_truth}")
        print(f"  Counter: {result.counter_conclusion} (scores: {result.counter_scores})")
        print(f"  Graph:   {result.graph_conclusion} (conf: {result.graph_confidence:.2f})")
        print(f"  Evidence: {result.total_evidence} (direct: {result.direct_paths}, 2-hop: {result.two_hop_paths})")

    print(f"\n--- Counter Wins ({len(counter_wins)} cases) ---")
    for traj, result in counter_wins[:max_examples]:
        print(f"\n[{result.query_id}] GT: {result.ground_truth}")
        print(f"  Counter: {result.counter_conclusion} (scores: {result.counter_scores})")
        print(f"  Graph:   {result.graph_conclusion} (conf: {result.graph_confidence:.2f})")
        print(f"  Evidence: {result.total_evidence} (direct: {result.direct_paths}, 2-hop: {result.two_hop_paths})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline Graph Inference Evaluation")
    parser.add_argument("--trajectory-dir", "-t", type=str,
                        default="/data/DRKG/KGSA/Stage4/Task9_online_data/combined_trajectories_20260121_061308/short",
                        help="Directory containing trajectory JSON files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--max", "-m", type=int, default=None,
                        help="Max trajectories to evaluate")
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Analyze disagreement cases")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.analyze:
        analyze_disagreements(args.trajectory_dir)
    else:
        run_offline_evaluation(
            args.trajectory_dir,
            output_path=args.output,
            max_trajectories=args.max,
            verbose=args.verbose,
        )
