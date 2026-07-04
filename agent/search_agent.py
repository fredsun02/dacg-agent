"""
KGSA Searching Agent

Main agent class that integrates:
- PubMed search
- LLM-based causal extraction
- PRM-guided stopping decisions
- Conclusion inference
"""

import time
import yaml
from typing import Optional, Dict
from pathlib import Path

from .pubmed_client import PubMedClient
from .extractor import CausalExtractor
from .state import SearchState, SearchResult
from .prm import PRM


class KGSAAgent:
    """Knowledge Graph Search Agent"""

    def __init__(
        self,
        prm_model_path: str,
        pubmed_api_key: Optional[str] = None,
        extractor_api_key: Optional[str] = None,
        extractor_api_base: Optional[str] = None,
        extractor_model: str = "gpt-5-chat",
        batch_size: int = 2,
        max_steps: int = 20,
        min_steps: int = 3,
        decline_threshold: float = 0.3,
        convergence_threshold: float = 0.1,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize the agent.

        Args:
            prm_model_path: Path to trained PRM model
            pubmed_api_key: NCBI API key (optional)
            extractor_api_key: LLM API key
            extractor_api_base: LLM API base URL
            extractor_model: LLM model name
            batch_size: Papers per search step (should match PRM training)
            max_steps: Maximum search steps
            min_steps: Minimum steps before PRM can stop
            decline_threshold: PRM stops if reward drops by this much from peak
            convergence_threshold: PRM stops if reward converges
            confidence_threshold: Confidence stop threshold
        """
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.pubmed = PubMedClient(api_key=pubmed_api_key)

        self.extractor = CausalExtractor(
            api_key=extractor_api_key,
            api_base=extractor_api_base,
            model=extractor_model
        )

        self.prm = PRM(
            model_path=prm_model_path,
            min_steps=min_steps,
            decline_threshold=decline_threshold,
            convergence_threshold=convergence_threshold
        )

    def search(
        self,
        head_entity: str,
        tail_entity: str,
        verbose: bool = True,
        max_search_time: int = 300
    ) -> SearchResult:
        """
        Execute search for causal relationship.

        Args:
            head_entity: Treatment/intervention entity
            tail_entity: Condition/outcome entity
            verbose: Print progress
            max_search_time: Maximum search time in seconds

        Returns:
            SearchResult with conclusion and evidence
        """
        start_time = time.time()

        # Reset PRM history
        self.prm.reset()

        # Initialize state
        state = SearchState(
            head_entity=head_entity,
            tail_entity=tail_entity
        )

        # Build search query
        query = f"{head_entity} {tail_entity}"

        if verbose:
            print(f"\n{'='*60}")
            print(f"KGSA Search: {head_entity} -> {tail_entity}")
            print(f"{'='*60}")

        stop_reason = "max_steps"

        # Search loop
        for step, papers in self.pubmed.search_and_fetch(
            query,
            batch_size=self.batch_size,
            max_batches=self.max_steps
        ):
            # Check timeout
            if time.time() - start_time > max_search_time:
                stop_reason = "timeout"
                if verbose:
                    print(f"\nStep {step + 1}: Timeout reached")
                break

            if not papers:
                if verbose:
                    print(f"\nStep {step + 1}: No papers found")
                continue

            if verbose:
                print(f"\nStep {step + 1}: Processing {len(papers)} papers")
                for p in papers:
                    print(f"  - PMID {p.pmid}: {p.title[:50]}...")

            # Extract causal relations
            extraction_results = self.extractor.extract_batch(
                papers, head_entity, tail_entity
            )

            # Update state
            state.update(extraction_results, self.batch_size)

            # Get current features and conclusion
            features = state.get_current_features()
            conclusion = state.get_conclusion()
            confidence = state.get_confidence()

            if verbose:
                snapshot = state.history[-1]
                print(f"  Evidence: +{snapshot['supporting_count'] - (state.history[-2]['supporting_count'] if len(state.history) > 1 else 0)} supporting, "
                      f"+{snapshot['opposing_count'] - (state.history[-2]['opposing_count'] if len(state.history) > 1 else 0)} opposing")
                print(f"  Total: {snapshot['total_evidences']} evidences from {snapshot['papers_seen']} papers")
                print(f"  Conclusion: {conclusion} (confidence: {confidence:.2f})")

            # PRM decision
            should_stop, reason, reward = self.prm.should_stop(features)

            if verbose:
                print(f"  PRM reward: {reward:.3f}")

            if should_stop:
                stop_reason = reason
                if verbose:
                    print(f"  -> Stopping: {reason}")
                break

            # Confidence threshold check
            if confidence >= self.confidence_threshold:
                stop_reason = "confidence_reached"
                if verbose:
                    print(f"  -> Stopping: high confidence ({confidence:.2f})")
                break

        # Build final result
        result = SearchResult(
            query=query,
            head_entity=head_entity,
            tail_entity=tail_entity,
            conclusion=state.get_conclusion(),
            confidence=state.get_confidence(),
            total_steps=len(state.history),
            papers_searched=state.papers_seen,
            stop_reason=stop_reason,
            history=state.history,
            evidence_summary={
                "supporting": state.supporting_count,
                "opposing": state.opposing_count,
                "neutral": state.neutral_count,
                "total_triples": len(state.triples),
                "reward_trajectory": self.prm.get_reward_trajectory()
            }
        )

        if verbose:
            print(f"\n{'='*60}")
            print("FINAL RESULT")
            print(f"{'='*60}")
            print(f"Conclusion: {result.conclusion}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Steps: {result.total_steps}")
            print(f"Papers: {result.papers_searched}")
            print(f"Stop reason: {result.stop_reason}")
            print(f"Evidence: {result.evidence_summary['supporting']} supporting, "
                  f"{result.evidence_summary['opposing']} opposing, "
                  f"{result.evidence_summary['neutral']} neutral")
            print(f"{'='*60}")

        return result


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_agent(config_path: Optional[str] = None) -> KGSAAgent:
    """
    Create agent from configuration file.

    Args:
        config_path: Path to config.yaml (uses default if None)

    Returns:
        Configured KGSAAgent
    """
    # Default config
    default_config = {
        "prm": {
            "model_path": "/data/DRKG/KGSA/Stage4/Task8_data_selection/models/prm_mlp.pt",
            "min_steps": 3,
            "decline_threshold": 0.3,
            "convergence_threshold": 0.1,
            "max_steps": 20
        },
        "extractor": {
            "api_base": "https://www.packyapi.com/v1",
            "model": "gpt-5-chat"
        },
        "agent": {
            "batch_size": 2,
            "confidence_threshold": 0.8
        }
    }

    # Load from file if provided
    if config_path:
        file_config = load_config(config_path)
        # Merge configs
        for key in file_config:
            if key in default_config:
                default_config[key].update(file_config[key])
            else:
                default_config[key] = file_config[key]

    return KGSAAgent(
        prm_model_path=default_config["prm"]["model_path"],
        extractor_api_key=default_config["extractor"].get("api_key"),
        extractor_api_base=default_config["extractor"].get("api_base"),
        extractor_model=default_config["extractor"].get("model", "gpt-5-chat"),
        batch_size=default_config["agent"].get("batch_size", 2),
        max_steps=default_config["prm"].get("max_steps", 20),
        min_steps=default_config["prm"].get("min_steps", 3),
        decline_threshold=default_config["prm"].get("decline_threshold", 0.3),
        convergence_threshold=default_config["prm"].get("convergence_threshold", 0.1),
        confidence_threshold=default_config["agent"].get("confidence_threshold", 0.8)
    )


# Test function
if __name__ == "__main__":
    # Quick test
    print("Creating agent...")
    agent = create_agent()

    print("\nRunning test search...")
    result = agent.search(
        head_entity="aspirin",
        tail_entity="cardiovascular disease",
        verbose=True
    )

    print(f"\nResult: {result.conclusion} (confidence: {result.confidence:.2f})")
