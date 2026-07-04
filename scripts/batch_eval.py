#!/usr/bin/env python3
"""
Batch Evaluation Script

Run evaluations on multiple benchmarks or with different configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.search_agent import create_agent
from evaluation.cochrane_eval import evaluate_agent, load_benchmark
from evaluation.case_study import run_all_case_studies, CASE_STUDIES


def run_benchmark_evaluation(benchmark_path: str, config_path: str = None, limit: int = None, verbose: bool = False):
    """Run evaluation on a specific benchmark"""
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_path}")
    print(f"{'='*60}")

    agent = create_agent(config_path)
    benchmark = load_benchmark(benchmark_path)

    if limit:
        benchmark = benchmark[:limit]

    results = evaluate_agent(agent, benchmark, verbose=verbose)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Avg Steps: {results['avg_steps']:.2f}")
    print(f"  Avg Papers: {results['avg_papers']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for KGSA Agent")
    parser.add_argument("--mode", choices=["benchmark", "case_study", "all"], default="all",
                        help="Evaluation mode")
    parser.add_argument("--benchmarks", nargs="+", type=str,
                        help="Paths to benchmark files")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Agent config file")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="/data/DRKG/KGSA/Stage5_Agent/results",
                        help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit queries per benchmark")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "config": args.config,
        "benchmarks": {},
        "case_studies": None
    }

    # Run benchmark evaluations
    if args.mode in ["benchmark", "all"] and args.benchmarks:
        for benchmark_path in args.benchmarks:
            if not Path(benchmark_path).exists():
                print(f"Warning: Benchmark not found: {benchmark_path}")
                continue

            name = Path(benchmark_path).stem
            results = run_benchmark_evaluation(
                benchmark_path,
                args.config,
                args.limit,
                args.verbose
            )
            all_results["benchmarks"][name] = results

    # Run case studies
    if args.mode in ["case_study", "all"]:
        print(f"\n{'='*60}")
        print("Running Case Studies")
        print(f"{'='*60}")

        agent = create_agent(args.config)
        case_results = run_all_case_studies(agent, verbose=args.verbose)
        all_results["case_studies"] = case_results

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if all_results["benchmarks"]:
        print("\nBenchmark Results:")
        for name, results in all_results["benchmarks"].items():
            print(f"  {name}: {results['accuracy']*100:.1f}% ({results.get('correct', 0)}/{results.get('total', 0)})")

    if all_results["case_studies"]:
        cs = all_results["case_studies"]
        print(f"\nCase Studies: {cs['accuracy']*100:.1f}% ({cs['correct']}/{cs['total']})")

    # Save results
    output_path = output_dir / f"batch_eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
