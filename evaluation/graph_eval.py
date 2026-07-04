"""
Graph-based Agent Evaluation

Compares graph-based inference against baseline counter-based inference.
Supports ablation study configurations from IMPROVEMENT_PLAN.md.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph_agent import GraphKGSAAgent, create_graph_agent


@dataclass
class GraphEvalResult:
    """Single evaluation result for graph agent."""
    query_id: str
    head_entity: str
    tail_entity: str
    ground_truth: str
    prediction: str
    confidence: float
    is_correct: bool
    steps: int
    papers: int
    stop_reason: str
    direct_paths: int
    two_hop_paths: int
    total_evidence: int
    multihop_expansions: int
    reward_trajectory: List[float] = None

    def __post_init__(self):
        if self.reward_trajectory is None:
            self.reward_trajectory = []


def load_benchmark(path: str) -> List[Dict]:
    """Load benchmark dataset."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "queries" in data:
            return data["queries"]
        elif "data" in data:
            return data["data"]
    return data


def normalize_label(label: str) -> str:
    """Normalize conclusion labels for comparison."""
    label = label.lower().strip()

    # Map variations to standard labels
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


def evaluate_graph_agent(
    agent: GraphKGSAAgent,
    benchmark: List[Dict],
    verbose: bool = False,
    max_queries: Optional[int] = None,
) -> Dict:
    """
    Evaluate graph agent on benchmark.

    Returns:
        {
            "accuracy": float,
            "accuracy_by_class": dict,
            "avg_steps": float,
            "avg_papers": float,
            "avg_confidence": float,
            "avg_evidence": float,
            "results": List[GraphEvalResult]
        }
    """
    results = []
    correct = 0
    total = 0

    # Per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    queries = benchmark[:max_queries] if max_queries else benchmark

    for i, item in enumerate(queries):
        head = item.get("head_entity") or item.get("head") or item.get("drug") or ""
        tail = item.get("tail_entity") or item.get("tail") or item.get("disease") or ""
        gt = item.get("ground_truth") or item.get("relation_type") or item.get("label") or ""
        query_id = item.get("id") or item.get("query_id") or f"q_{i}"

        if not head or not tail:
            print(f"[{i+1}/{len(queries)}] Skipping: missing entities")
            continue

        gt_normalized = normalize_label(gt)

        print(f"\n[{i+1}/{len(queries)}] {head} -> {tail}")
        print(f"  Ground truth: {gt_normalized}")

        try:
            search_result = agent.search(
                head_entity=head,
                tail_entity=tail,
                verbose=verbose
            )

            prediction = search_result.conclusion
            # Handle NoEvidence as prediction
            if prediction == "NoEvidence":
                prediction = "Uncertain"

            is_correct = normalize_label(prediction) == gt_normalized

            result = GraphEvalResult(
                query_id=query_id,
                head_entity=head,
                tail_entity=tail,
                ground_truth=gt_normalized,
                prediction=prediction,
                confidence=search_result.confidence,
                is_correct=is_correct,
                steps=search_result.total_steps,
                papers=search_result.papers_searched,
                stop_reason=search_result.stop_reason,
                direct_paths=search_result.direct_paths,
                two_hop_paths=search_result.two_hop_paths,
                total_evidence=search_result.total_evidence,
                multihop_expansions=search_result.multihop_expansions,
                reward_trajectory=search_result.reward_trajectory,
            )
            results.append(result)

            if is_correct:
                correct += 1
                class_correct[gt_normalized] += 1
            total += 1
            class_total[gt_normalized] += 1

            status = "✓" if is_correct else "✗"
            print(f"  Prediction: {prediction} (conf: {search_result.confidence:.2f}) {status}")
            print(f"  Evidence: {search_result.total_evidence} | Steps: {search_result.total_steps}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    accuracy_by_class = {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in class_total
    }

    avg_steps = sum(r.steps for r in results) / len(results) if results else 0.0
    avg_papers = sum(r.papers for r in results) / len(results) if results else 0.0
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
    avg_evidence = sum(r.total_evidence for r in results) / len(results) if results else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_by_class": accuracy_by_class,
        "correct": correct,
        "total": total,
        "avg_steps": avg_steps,
        "avg_papers": avg_papers,
        "avg_confidence": avg_confidence,
        "avg_evidence": avg_evidence,
        "results": results,
    }


def run_ablation_study(
    benchmark_path: str,
    output_dir: str,
    max_queries: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """
    Run ablation study comparing different configurations.

    Configurations:
    - graph_only: Graph + path inference (no PRM)
    - graph_multihop: Graph + path inference + multihop expansion
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(benchmark_path)
    print(f"Loaded {len(benchmark)} queries from {benchmark_path}")

    results = {}

    # Config 1: Graph only (no multihop)
    print("\n" + "="*60)
    print("ABLATION 1: Graph + Path Inference (no multihop)")
    print("="*60)

    agent1 = create_graph_agent(mode="graph_only", enable_multihop=False)
    eval1 = evaluate_graph_agent(agent1, benchmark, verbose=verbose, max_queries=max_queries)
    results["graph_only"] = {
        "accuracy": eval1["accuracy"],
        "accuracy_by_class": eval1["accuracy_by_class"],
        "avg_steps": eval1["avg_steps"],
        "avg_evidence": eval1["avg_evidence"],
    }

    # Config 2: Graph + multihop
    print("\n" + "="*60)
    print("ABLATION 2: Graph + Path Inference + Multihop")
    print("="*60)

    agent2 = create_graph_agent(mode="graph_only", enable_multihop=True)
    eval2 = evaluate_graph_agent(agent2, benchmark, verbose=verbose, max_queries=max_queries)
    results["graph_multihop"] = {
        "accuracy": eval2["accuracy"],
        "accuracy_by_class": eval2["accuracy_by_class"],
        "avg_steps": eval2["avg_steps"],
        "avg_evidence": eval2["avg_evidence"],
    }

    # Summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)

    for config, metrics in results.items():
        print(f"\n{config}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  By class: {metrics['accuracy_by_class']}")
        print(f"  Avg steps: {metrics['avg_steps']:.1f}")
        print(f"  Avg evidence: {metrics['avg_evidence']:.1f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"ablation_results_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {result_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Graph KGSA Agent")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    parser.add_argument("--output", default="./eval_results", help="Output directory")
    parser.add_argument("--max-queries", type=int, help="Max queries to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")

    args = parser.parse_args()

    if args.ablation:
        run_ablation_study(
            args.benchmark,
            args.output,
            max_queries=args.max_queries,
            verbose=args.verbose
        )
    else:
        agent = create_graph_agent(mode="graph_only", enable_multihop=True)
        benchmark = load_benchmark(args.benchmark)
        results = evaluate_graph_agent(
            agent, benchmark,
            verbose=args.verbose,
            max_queries=args.max_queries
        )
        print(f"\nFinal Accuracy: {results['accuracy']*100:.1f}%")
