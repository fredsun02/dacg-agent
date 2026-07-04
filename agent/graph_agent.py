"""
Graph-based KGSA Agent

Enhanced agent that uses graph-based inference instead of counter-based.
Supports different modes for ablation study.
"""

import math
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from .pubmed_client import PubMedClient
from .extractor import CausalExtractor, CausalTriple
from .entity_resolver import EntityResolver
from .graph_store import GraphStore
from .path_inference import PathInference, ConclusionResult
from .graph_features import GraphFeatures, GraphFeatureExtractor
from .multihop_search import expand_search, should_expand
from .prm import PRM
from .state import SearchState, SearchResult

# KL-based stopping (Layer 1)
KL_THRESHOLD = 0.01
KL_CONSECUTIVE = 1



@dataclass
class GraphSearchResult:
    """Result from graph-based search."""
    query: str
    head_entity: str
    tail_entity: str
    conclusion: str
    confidence: float
    total_steps: int
    papers_searched: int
    stop_reason: str

    # Graph-specific metrics
    direct_paths: int = 0
    two_hop_paths: int = 0
    total_evidence: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    multihop_expansions: int = 0

    # History and trajectories
    history: List[Dict] = field(default_factory=list)
    conclusion_result: Optional[ConclusionResult] = None
    reward_trajectory: List[float] = field(default_factory=list)

    # For interface compatibility with cochrane_eval
    @property
    def evidence_summary(self) -> Dict:
        return {
            "reward_trajectory": self.reward_trajectory,
            "direct_paths": self.direct_paths,
            "two_hop_paths": self.two_hop_paths,
            "total_evidence": self.total_evidence,
        }

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "head_entity": self.head_entity,
            "tail_entity": self.tail_entity,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "total_steps": self.total_steps,
            "papers_searched": self.papers_searched,
            "stop_reason": self.stop_reason,
            "direct_paths": self.direct_paths,
            "two_hop_paths": self.two_hop_paths,
            "total_evidence": self.total_evidence,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "multihop_expansions": self.multihop_expansions,
        }


class GraphKGSAAgent:
    """
    Graph-based KGSA Agent.

    Modes:
    - 'graph_only': Use path inference only, no PRM
    - 'graph_prm': Use graph features with existing PRM
    - 'baseline': Use original counter-based logic (for comparison)
    """

    def __init__(
        self,
        prm_model_path: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        extractor_api_key: Optional[str] = None,
        extractor_api_base: Optional[str] = None,
        extractor_model: str = "claude-sonnet-4-6",
        batch_size: int = 2,
        max_steps: int = 20,
        min_steps: int = 0,
        mode: str = "graph_only",
        enable_multihop: bool = True,
        multihop_max_intermediates: int = 3,
        confidence_threshold: float = 0.8,
        recording_mode: bool = False,
        lambda_u: float = 0.0,
        lambda_v: float = 0.0,
        quality_floor: float = 0.3,
    ):
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.mode = mode
        self.enable_multihop = enable_multihop
        self.multihop_max_intermediates = multihop_max_intermediates
        self.confidence_threshold = confidence_threshold
        self.recording_mode = recording_mode

        # Initialize components
        self.pubmed = PubMedClient(api_key=pubmed_api_key)
        self.extractor = CausalExtractor(
            api_key=extractor_api_key,
            api_base=extractor_api_base,
            model=extractor_model
        )

        # PRM: load in graph_prm mode OR recording mode (for offline ablation)
        self.prm = None
        if prm_model_path and (mode == "graph_prm" or recording_mode):
            self.prm = PRM(model_path=prm_model_path, min_steps=min_steps)

        # Graph inference components
        self.inference = PathInference(
            lambda_u=lambda_u, lambda_v=lambda_v, quality_floor=quality_floor
        )
        self.feature_extractor = GraphFeatureExtractor()

        # Per-search state
        self._reward_trajectory: List[float] = []
        self._prev_posterior: Optional[Dict[str, float]] = None
        self._kl_below_count: int = 0
        self._last_kl: Optional[float] = None
        print(f"[GraphAgent] Initialized: mode={mode}, multihop={enable_multihop}, recording={recording_mode}")

    def search(
        self,
        head_entity: str,
        tail_entity: str,
        verbose: bool = True,
        max_search_time: int = 300
    ) -> GraphSearchResult:
        """Execute graph-based search."""
        start_time = time.time()

        # Initialize graph for this search
        resolver = EntityResolver()
        graph = GraphStore(resolver=resolver)

        # Resolve query entities
        head_resolved = resolver.resolve(head_entity, create=True)
        tail_resolved = resolver.resolve(tail_entity, create=True)
        head_id = head_resolved.id
        tail_id = tail_resolved.id

        base_query = f"{head_entity} {tail_entity}"

        if verbose:
            print(f"\n{'='*60}")
            print(f"GraphKGSA Search: {head_entity} -> {tail_entity}")
            print(f"Mode: {self.mode}")
            print(f"{'='*60}")

        stop_reason = "max_steps"
        papers_searched = 0
        history = []
        prev_features = None
        multihop_expansions = 0

        # Reset per-search state
        self._reward_trajectory = []
        self._prev_posterior = None
        self._kl_below_count = 0
        self._last_kl = None
        if self.prm:
            self.prm.reset()

        # Single-stream search
        base_gen = self.pubmed.search_and_fetch(
            base_query, batch_size=self.batch_size, max_batches=self.max_steps
        )
        base_done = False

        # Main search loop
        for step in range(self.max_steps):
            # Check timeout
            if time.time() - start_time > max_search_time:
                stop_reason = "timeout"
                if verbose:
                    print(f"\nStep {step + 1}: Timeout reached")
                break

            papers = []
            if not base_done:
                try:
                    _, papers = next(base_gen)
                except StopIteration:
                    base_done = True

            if base_done and not papers:
                stop_reason = "sources_exhausted"
                break

            if not papers:
                if verbose:
                    print(f"\nStep {step + 1}: No papers found")
                continue

            papers_searched += len(papers)

            if verbose:
                print(f"\nStep {step + 1}: Processing {len(papers)} papers")

            extraction_results = self.extractor.extract_batch(
                papers, head_entity, tail_entity
            )
            new_triples = self._add_results_to_graph(graph, extraction_results)

            if verbose:
                print(f"  Added {new_triples} triples to graph")
                print(f"  Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

            # Get conclusion via path inference
            conclusion_result = self.inference.infer_conclusion(graph, head_id, tail_id)

            if verbose:
                print(f"  Conclusion: {conclusion_result.conclusion} (conf: {conclusion_result.confidence:.2f})")
                print(f"  Evidence: {conclusion_result.total_evidence} (direct: {conclusion_result.direct_paths}, 2-hop: {conclusion_result.two_hop_paths})")

            # Extract graph features
            features = self.feature_extractor.extract(graph, head_id, tail_id, prev_features)
            prev_features = features

            # Stopping decisions (compute signals before snapshot so KL/PRM values are available)
            should_stop, reason = self._should_stop(
                step, conclusion_result, features, papers_searched
            )

            # Record history (includes signals computed by _should_stop)
            snapshot = {
                "step": step + 1,
                "papers_this_step": len(papers),
                "new_triples": new_triples,
                "conclusion": conclusion_result.conclusion,
                "confidence": conclusion_result.confidence,
                "total_evidence": conclusion_result.total_evidence,
                "direct_paths": conclusion_result.direct_paths,
                "two_hop_paths": conclusion_result.two_hop_paths,
                "is_decisive": conclusion_result.is_decisive,
                "posterior_scores": conclusion_result.scores,
                "kl": self._last_kl,
                "prm_reward": self._reward_trajectory[-1] if self._reward_trajectory else None,
                "graph_features": features.to_dict(),
            }
            history.append(snapshot)

            if should_stop:
                stop_reason = reason
                if verbose:
                    print(f"  -> Stopping: {reason}")
                break

            # Multi-hop expansion if needed
            if self.enable_multihop and conclusion_result.should_continue and step >= self.min_steps:
                if verbose:
                    print(f"  Attempting multihop expansion...")

                expanded = expand_search(
                    self, graph, head_entity, tail_entity, head_id, tail_id,
                    max_intermediates=self.multihop_max_intermediates,
                    verbose=verbose,
                    inference=self.inference,
                )
                if expanded > 0:
                    multihop_expansions += 1
                    # Re-evaluate after expansion
                    conclusion_result = self.inference.infer_conclusion(graph, head_id, tail_id)
                    if verbose:
                        print(f"  After expansion: {conclusion_result.conclusion} (conf: {conclusion_result.confidence:.2f})")

        # Final conclusion
        final_result = self.inference.infer_conclusion(graph, head_id, tail_id)

        return GraphSearchResult(
            query=base_query,
            head_entity=head_entity,
            tail_entity=tail_entity,
            conclusion=final_result.conclusion,
            confidence=final_result.confidence,
            total_steps=len(history),
            papers_searched=papers_searched,
            stop_reason=stop_reason,
            direct_paths=final_result.direct_paths,
            two_hop_paths=final_result.two_hop_paths,
            total_evidence=final_result.total_evidence,
            graph_nodes=graph.node_count(),
            graph_edges=graph.edge_count(),
            multihop_expansions=multihop_expansions,
            history=history,
            conclusion_result=final_result,
            reward_trajectory=self._reward_trajectory.copy(),
        )

    def _add_results_to_graph(
        self,
        graph: GraphStore,
        extraction_results: List[Dict]
    ) -> int:
        """Add extraction results to graph."""
        added = 0
        for result in extraction_results:
            if not isinstance(result, dict):
                continue
            for triple in result.get("triples", []):
                edge = graph.add_triple(triple)
                if edge is not None:
                    added += 1
        return added

    def _normalize_posterior(self, conclusion: ConclusionResult) -> Dict[str, float]:
        """Extract and normalize posterior from ConclusionResult.scores."""
        ben = max(0.0, float(conclusion.scores.get("beneficial", 0.0)))
        harm = max(0.0, float(conclusion.scores.get("harmful", 0.0)))
        neu = max(0.0, float(conclusion.scores.get("neutral", 0.0)))
        total = ben + harm + neu
        if total <= 0:
            return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}
        return {"beneficial": ben / total, "harmful": harm / total, "neutral": neu / total}

    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """KL(p || q) over the 3-class label distribution."""
        eps = 1e-12
        kl = 0.0
        for key in ("beneficial", "harmful", "neutral"):
            p_i = max(eps, p.get(key, 0.0))
            q_i = max(eps, q.get(key, 0.0))
            kl += p_i * math.log(p_i / q_i)
        return kl

    def _update_kl_stopping(self, conclusion: ConclusionResult) -> bool:
        """Layer 1: track Δ_t = KL(q_t || q_{t-1}), return True when stable."""
        if conclusion.total_evidence < self.inference.min_evidence:
            self._prev_posterior = None
            self._kl_below_count = 0
            self._last_kl = None
            return False

        posterior = self._normalize_posterior(conclusion)
        if self._prev_posterior is None:
            self._prev_posterior = posterior
            self._kl_below_count = 0
            self._last_kl = None
            return False

        kl = self._kl_divergence(posterior, self._prev_posterior)
        self._prev_posterior = posterior
        self._last_kl = kl

        if kl < KL_THRESHOLD:
            self._kl_below_count += 1
        else:
            self._kl_below_count = 0

        return self._kl_below_count >= KL_CONSECUTIVE

    def _should_stop(
        self,
        step: int,
        conclusion: ConclusionResult,
        features: GraphFeatures,
        papers_searched: int = 0,
    ) -> tuple:
        """Two-layer stopping: Layer 1 (KL stability) -> Layer 2 (mode-specific)."""
        kl_stable = self._update_kl_stopping(conclusion)

        # In recording mode, always compute PRM reward for trajectory recording
        if self.recording_mode and self.prm:
            features_dict = features.to_dict()
            _, _, reward = self.prm.should_stop_from_dict(features_dict)
            self._reward_trajectory.append(reward)
            return False, "continue"

        # --- Normal (non-recording) stopping logic ---

        # Minimum steps
        if step < self.min_steps:
            return False, "continue"

        # Layer 1: KL belief stability
        if kl_stable:
            return True, "kl_stable"

        # Mode-specific stopping (Layer 2)
        if self.mode == "graph_only":
            if conclusion.total_evidence >= 10 and conclusion.is_decisive and conclusion.confidence >= 0.9:
                return True, "decisive_fallback"
            return False, "continue"

        elif self.mode == "graph_prm" and self.prm:
            features_dict = features.to_dict()
            should_stop, reason, reward = self.prm.should_stop_from_dict(features_dict)
            self._reward_trajectory.append(reward)
            if should_stop:
                return True, reason
            return False, "continue"

        else:
            if conclusion.is_decisive:
                return True, "decisive"
            return False, "continue"


def create_graph_agent(
    mode: str = "graph_only",
    prm_model_path: Optional[str] = None,
    enable_multihop: bool = True,
    recording_mode: bool = False,
) -> GraphKGSAAgent:
    """Create a GraphKGSAAgent with default settings."""
    return GraphKGSAAgent(
        prm_model_path=prm_model_path,
        mode=mode,
        enable_multihop=enable_multihop,
        recording_mode=recording_mode,
    )
