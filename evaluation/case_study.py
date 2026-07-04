"""
Case Study Analysis

Run specific case studies to demonstrate agent capabilities.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.search_agent import KGSAAgent, create_agent


# Predefined case studies
CASE_STUDIES = [
    {
        "name": "Hydroxychloroquine COVID-19",
        "head_entity": "hydroxychloroquine",
        "tail_entity": "COVID-19",
        "expected": "NoEffect",
        "description": "Controversial treatment that was later shown to be ineffective"
    },
    {
        "name": "Aspirin Cardiovascular Disease",
        "head_entity": "aspirin",
        "tail_entity": "cardiovascular disease",
        "expected": "Beneficial",
        "description": "Well-established preventive treatment"
    },
    {
        "name": "Metformin Type 2 Diabetes",
        "head_entity": "metformin",
        "tail_entity": "type 2 diabetes",
        "expected": "Beneficial",
        "description": "First-line treatment for type 2 diabetes"
    },
    {
        "name": "Statins Heart Disease",
        "head_entity": "statins",
        "tail_entity": "heart disease",
        "expected": "Beneficial",
        "description": "Widely used for cholesterol reduction and heart disease prevention"
    },
    {
        "name": "Smoking Lung Cancer",
        "head_entity": "smoking",
        "tail_entity": "lung cancer",
        "expected": "Harmful",
        "description": "Well-established causal harmful relationship"
    },
    {
        "name": "Ivermectin COVID-19",
        "head_entity": "ivermectin",
        "tail_entity": "COVID-19",
        "expected": "NoEffect",
        "description": "Another controversial treatment shown to be ineffective"
    }
]


def run_case_study(
    agent: KGSAAgent,
    case: Dict,
    verbose: bool = True
) -> Dict:
    """
    Run a single case study.

    Args:
        agent: KGSAAgent instance
        case: Case study dict with head_entity, tail_entity, expected
        verbose: Print detailed output

    Returns:
        Case study result dict
    """
    print(f"\n{'='*70}")
    print(f"CASE STUDY: {case.get('name', 'Unknown')}")
    print(f"Description: {case.get('description', 'N/A')}")
    print(f"Expected: {case.get('expected', 'Unknown')}")
    print(f"{'='*70}")

    result = agent.search(
        head_entity=case["head_entity"],
        tail_entity=case["tail_entity"],
        verbose=verbose
    )

    expected = case.get("expected", "")
    is_correct = result.conclusion == expected

    # Also accept related labels
    if not is_correct:
        if expected in ["Beneficial", "Treat", "Inhibit"]:
            is_correct = result.conclusion in ["Beneficial", "Treat", "Inhibit"]
        elif expected in ["Harmful", "Cause", "Stimulate"]:
            is_correct = result.conclusion in ["Harmful", "Cause", "Stimulate"]

    status = "PASS" if is_correct else "FAIL"

    print(f"\n{'-'*70}")
    print(f"RESULT: {status}")
    print(f"  Expected: {expected}")
    print(f"  Got: {result.conclusion} (confidence: {result.confidence:.2f})")
    print(f"  Steps: {result.total_steps}, Papers: {result.papers_searched}")
    print(f"{'-'*70}")

    return {
        "name": case.get("name"),
        "head_entity": case["head_entity"],
        "tail_entity": case["tail_entity"],
        "expected": expected,
        "prediction": result.conclusion,
        "confidence": result.confidence,
        "is_correct": is_correct,
        "steps": result.total_steps,
        "papers": result.papers_searched,
        "stop_reason": result.stop_reason,
        "evidence_summary": result.evidence_summary
    }


def run_all_case_studies(
    agent: KGSAAgent,
    cases: List[Dict] = None,
    verbose: bool = False
) -> Dict:
    """
    Run all case studies.

    Args:
        agent: KGSAAgent instance
        cases: List of case studies (uses default if None)
        verbose: Print detailed output

    Returns:
        Summary results
    """
    if cases is None:
        cases = CASE_STUDIES

    results = []
    for case in cases:
        result = run_case_study(agent, case, verbose=verbose)
        results.append(result)

    # Summary
    correct = sum(r["is_correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("CASE STUDY SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    print()
    for r in results:
        status = "PASS" if r["is_correct"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['prediction']} (expected: {r['expected']})")
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run KGSA Agent case studies")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Agent config file path")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--case", type=str, default=None,
                        help="Run specific case by name")
    args = parser.parse_args()

    # Create agent
    print("Creating agent...")
    agent = create_agent(args.config)

    # Run case studies
    if args.case:
        # Find specific case
        case = next((c for c in CASE_STUDIES if c["name"].lower() == args.case.lower()), None)
        if case is None:
            print(f"Case '{args.case}' not found. Available cases:")
            for c in CASE_STUDIES:
                print(f"  - {c['name']}")
            return

        result = run_case_study(agent, case, verbose=True)
        results = {"results": [result]}
    else:
        results = run_all_case_studies(agent, verbose=args.verbose)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"/data/DRKG/KGSA/Stage5_Agent/results/case_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
