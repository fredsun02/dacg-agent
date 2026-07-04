"""
Cochrane External Validation

Evaluates the KGSA agent against Cochrane systematic review ground truth.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.search_agent import KGSAAgent, create_agent


@dataclass
class EvalResult:
    """Single evaluation result"""
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
    reward_trajectory: List[float]


def load_benchmark(path: str) -> List[Dict]:
    """Load benchmark dataset"""
    with open(path) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict):
        if "queries" in data:
            return data["queries"]
        elif "data" in data:
            return data["data"]
    return data


def evaluate_agent(
    agent: KGSAAgent,
    benchmark: List[Dict],
    verbose: bool = False,
    save_details: bool = True
) -> Dict:
    """
    Evaluate agent on benchmark.

    Args:
        agent: KGSAAgent instance
        benchmark: List of query dicts with ground_truth
        verbose: Print detailed search process
        save_details: Save per-query details

    Returns:
        {
            "accuracy": float,
            "avg_steps": float,
            "avg_papers": float,
            "avg_confidence": float,
            "results": List[EvalResult]
        }
    """
    results = []

    for i, item in enumerate(benchmark):
        # Extract entities - handle different field names
        head = item.get("head_entity") or item.get("head") or item.get("drug") or ""
        tail = item.get("tail_entity") or item.get("tail") or item.get("disease") or ""
        gt = item.get("ground_truth") or item.get("relation_type") or item.get("label") or ""
        query_id = item.get("id") or item.get("query_id") or f"q_{i}"

        if not head or not tail:
            print(f"[{i+1}/{len(benchmark)}] Skipping: missing entities")
            continue

        print(f"\n[{i+1}/{len(benchmark)}] {head} -> {tail}")
        print(f"  Ground truth: {gt}")

        # Run agent
        search_result = agent.search(
            head_entity=head,
            tail_entity=tail,
            verbose=verbose
        )

        # Map prediction to standard labels if needed
        prediction = search_result.conclusion

        # Check correctness
        is_correct = (prediction == gt)

        # Also consider partial matches
        if not is_correct:
            # Beneficial matches Treat/Inhibit
            if gt in ["Beneficial", "Treat", "Inhibit"] and prediction in ["Beneficial", "Treat", "Inhibit"]:
                is_correct = True
            # Harmful matches Cause/Stimulate
            elif gt in ["Harmful", "Cause", "Stimulate"] and prediction in ["Harmful", "Cause", "Stimulate"]:
                is_correct = True

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Prediction: {prediction} (confidence: {search_result.confidence:.2f}) - {status}")

        eval_result = EvalResult(
            query_id=query_id,
            head_entity=head,
            tail_entity=tail,
            ground_truth=gt,
            prediction=prediction,
            confidence=search_result.confidence,
            is_correct=is_correct,
            steps=search_result.total_steps,
            papers=search_result.papers_searched,
            stop_reason=search_result.stop_reason,
            reward_trajectory=search_result.evidence_summary.get("reward_trajectory", [])
        )
        results.append(eval_result)

    # Calculate statistics
    if not results:
        return {"accuracy": 0, "avg_steps": 0, "avg_papers": 0, "results": []}

    accuracy = sum(r.is_correct for r in results) / len(results)
    avg_steps = sum(r.steps for r in results) / len(results)
    avg_papers = sum(r.papers for r in results) / len(results)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    # Stop reason distribution
    stop_reasons = {}
    for r in results:
        stop_reasons[r.stop_reason] = stop_reasons.get(r.stop_reason, 0) + 1

    return {
        "accuracy": accuracy,
        "correct": sum(r.is_correct for r in results),
        "total": len(results),
        "avg_steps": avg_steps,
        "avg_papers": avg_papers,
        "avg_confidence": avg_confidence,
        "stop_reasons": stop_reasons,
        "results": [asdict(r) for r in results] if save_details else []
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KGSA Agent on Cochrane benchmark")
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Path to benchmark JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for results (default: auto-generated)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Agent config file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of queries (for testing)")
    args = parser.parse_args()

    # Create agent
    print("Creating agent...")
    agent = create_agent(args.config)

    # Load benchmark
    print(f"Loading benchmark from {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)

    if args.limit:
        benchmark = benchmark[:args.limit]
        print(f"Limited to {args.limit} queries")

    print(f"Evaluating on {len(benchmark)} queries...")

    # Run evaluation
    results = evaluate_agent(agent, benchmark, verbose=args.verbose)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"Avg Steps: {results['avg_steps']:.2f}")
    print(f"Avg Papers: {results['avg_papers']:.2f}")
    print(f"Avg Confidence: {results['avg_confidence']:.2f}")
    print(f"\nStop reasons:")
    for reason, count in results.get("stop_reasons", {}).items():
        print(f"  {reason}: {count}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"/data/DRKG/KGSA/Stage5_Agent/results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
