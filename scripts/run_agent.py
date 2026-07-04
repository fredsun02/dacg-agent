#!/usr/bin/env python3
"""
Command-line interface for KGSA Searching Agent

Usage:
    python run_agent.py "aspirin" "cardiovascular disease"
    python run_agent.py "hydroxychloroquine" "COVID-19" --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.search_agent import create_agent


def main():
    parser = argparse.ArgumentParser(
        description="KGSA Searching Agent - Find causal relationships between entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_agent.py aspirin "cardiovascular disease"
    python run_agent.py hydroxychloroquine COVID-19 --verbose
    python run_agent.py metformin "type 2 diabetes" --output result.json
        """
    )

    parser.add_argument("head", type=str,
                        help="Head entity (treatment/intervention)")
    parser.add_argument("tail", type=str,
                        help="Tail entity (condition/outcome)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed search process")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save result to JSON file")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max search steps")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Maximum search time in seconds (default: 300)")

    args = parser.parse_args()

    # Create agent
    print("Initializing KGSA Agent...")
    agent = create_agent(args.config)

    if args.max_steps:
        agent.max_steps = args.max_steps

    # Run search
    print(f"\nSearching: {args.head} -> {args.tail}")
    print("-" * 50)

    result = agent.search(
        head_entity=args.head,
        tail_entity=args.tail,
        verbose=args.verbose or True,
        max_search_time=args.timeout
    )

    # Output result
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(f"\nQuery: Does {args.head} affect {args.tail}?")
    print(f"\nConclusion: {result.conclusion}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nEvidence Summary:")
    print(f"  - Papers searched: {result.papers_searched}")
    print(f"  - Search steps: {result.total_steps}")
    print(f"  - Supporting evidence: {result.evidence_summary.get('supporting', 0)}")
    print(f"  - Opposing evidence: {result.evidence_summary.get('opposing', 0)}")
    print(f"  - Neutral evidence: {result.evidence_summary.get('neutral', 0)}")
    print(f"  - Stop reason: {result.stop_reason}")

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to {output_path}")


if __name__ == "__main__":
    main()
