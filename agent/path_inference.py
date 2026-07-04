"""
Path-based Conclusion Inference for KGSA.

Replaces counter-based inference with graph path queries.
Key improvement: Distinguishes NoEvidence (continue searching) from NoEffect (sufficient evidence shows no effect).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .graph_store import GraphStore, GraphEdge


# Thresholds / constants
MIN_EVIDENCE = 2
MARGIN = 0.1
RHO = 1.0
ETA = 1.5
POLARITY_SHARPNESS = 5.0
DEFAULT_LENGTH_PENALTY = RHO
NON_CAUSAL_WEIGHT = 0.25
TWO_HOP_CAP = 5


@dataclass(frozen=True)
class ConclusionResult:
    """Result of path-based conclusion inference."""
    conclusion: str          # Beneficial, Harmful, NoEffect, NoEvidence, Uncertain
    confidence: float        # 0.0 - 1.0
    scores: Dict[str, float] # beneficial, harmful, neutral scores
    total_evidence: int      # Unique evidence count
    direct_paths: int        # Number of direct edges
    two_hop_paths: int       # Number of two-hop paths

    @property
    def should_continue(self) -> bool:
        """Whether search should continue based on conclusion."""
        return self.conclusion in ("NoEvidence", "Uncertain")

    @property
    def is_decisive(self) -> bool:
        """Whether conclusion is decisive (Beneficial, Harmful, or NoEffect)."""
        return self.conclusion in ("Beneficial", "Harmful", "NoEffect")


class PathInference:
    """Path-based conclusion inference with configurable thresholds."""

    def __init__(
        self,
        min_evidence: int = MIN_EVIDENCE,
        margin: float = MARGIN,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        eta: float = ETA,
        polarity_sharpness: float = POLARITY_SHARPNESS,
        lambda_u: float = 0.0,
        lambda_v: float = 0.0,
        quality_floor: float = 0.3,
    ) -> None:
        self.min_evidence = min_evidence
        self.margin = margin
        self.length_penalty = length_penalty
        self.rho = length_penalty
        self.eta = eta
        self.polarity_sharpness = polarity_sharpness
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.quality_floor = quality_floor

    def infer_conclusion(
        self,
        graph: GraphStore,
        head_id: str,
        tail_id: str,
    ) -> ConclusionResult:
        """
        Infer conclusion from graph paths.

        Decision logic:
        1. If total_evidence < MIN_EVIDENCE: NoEvidence (continue searching)
        2. If top class beats second-best by margin: return that class
        3. Otherwise: Uncertain (evidence conflict)
        """
        scores = self.aggregate_scores(graph, head_id, tail_id)

        total_evidence = int(scores.get("total_evidence", 0))
        s_pos = float(scores.get("beneficial", 0.0))
        s_neg = float(scores.get("harmful", 0.0))
        s_neu = float(scores.get("neutral", 0.0))

        # Key distinction: NoEvidence vs NoEffect
        if total_evidence < self.min_evidence:
            conclusion = "NoEvidence"
            confidence = 0.0
        else:
            # Symmetric margin: top class must beat second-best by margin
            label_scores = [
                ("Beneficial", s_pos), ("Harmful", s_neg), ("NoEffect", s_neu),
            ]
            label_scores.sort(key=lambda x: x[1], reverse=True)
            top_label, top_score = label_scores[0]
            second_score = label_scores[1][1]
            if top_score > second_score + self.margin:
                conclusion = top_label
            else:
                conclusion = "Uncertain"
            confidence = self._compute_confidence(s_pos, s_neg, s_neu)

        return ConclusionResult(
            conclusion=conclusion,
            confidence=confidence,
            scores=scores,
            total_evidence=total_evidence,
            direct_paths=int(scores.get("direct_paths", 0)),
            two_hop_paths=int(scores.get("two_hop_paths", 0)),
        )

    def aggregate_scores(
        self,
        graph: GraphStore,
        head_id: str,
        tail_id: str,
    ) -> Dict[str, float]:
        """Compute answer posterior q_t(L|Q) via route portfolio."""
        routes, direct_count, two_hop_count = self._collect_routes(graph, head_id, tail_id)
        posterior = self._compute_route_posterior(routes)
        total_evidence = graph.count_unique_evidence(head_id, tail_id)

        return {
            "beneficial": posterior["beneficial"],
            "harmful": posterior["harmful"],
            "neutral": posterior["neutral"],
            "total_evidence": float(total_evidence),
            "direct_paths": float(direct_count),
            "two_hop_paths": float(two_hop_count),
        }

    def _collect_routes(
        self,
        graph: GraphStore,
        head_id: str,
        tail_id: str,
    ) -> Tuple[List[Tuple[float, int]], int, int]:
        """Collect (strength, polarity) pairs from all feasible routes."""
        routes: List[Tuple[float, int]] = []
        direct_edges = graph.get_direct_edges(head_id, tail_id)
        apply_energy = self.lambda_u > 0 or self.lambda_v > 0
        h_cache = {} if self.lambda_u > 0 else None

        for edge in direct_edges:
            strength = self._score_edge(edge)
            if strength > 0:
                if not edge.is_causal:
                    strength *= NON_CAUSAL_WEIGHT
                if apply_energy:
                    penalty = 0.0
                    if self.lambda_u > 0:
                        penalty += self.lambda_u * graph.compute_pair_entropy(
                            edge.head_id, edge.tail_id, h_cache
                        )
                    if self.lambda_v > 0:
                        gap = self.quality_floor - edge.confidence_agg
                        if gap > 0:
                            penalty += self.lambda_v * gap
                    strength *= math.exp(-penalty)
                routes.append((strength, edge.polarity))

        two_hop_paths = graph.get_two_hop_paths(head_id, tail_id)

        # Score all two-hop paths, sort by strength, cap at TWO_HOP_CAP
        scored_two_hops = []
        for path in two_hop_paths:
            strength = graph.compute_path_score(path, length_penalty=self.rho)
            if strength > 0:
                has_non_causal = any(
                    not graph.edges[eid].is_causal
                    for eid in path.edge_ids if eid in graph.edges
                )
                if has_non_causal:
                    strength *= NON_CAUSAL_WEIGHT
                if apply_energy:
                    penalty = (
                        graph.compute_path_edge_uncertainty(
                            path, self.lambda_u, h_cache
                        )
                        + graph.compute_path_constraint_violation(
                            path, self.quality_floor, self.lambda_v
                        )
                    )
                    strength *= math.exp(-penalty)
                polarity = graph.compute_path_polarity(path)
                scored_two_hops.append((strength, polarity))

        scored_two_hops.sort(key=lambda x: x[0], reverse=True)
        routes.extend(scored_two_hops[:TWO_HOP_CAP])

        return routes, len(direct_edges), len(two_hop_paths)

    def _compute_route_posterior(
        self,
        routes: List[Tuple[float, int]],
    ) -> Dict[str, float]:
        """q_t(L|Q) = Σ_π q(L|π,Q) · q(π|Q) where q(π|Q) ∝ Str(π,t)^η."""
        if not routes:
            return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}

        weights: List[float] = []
        label_probs: List[Tuple[float, float, float]] = []
        for strength, polarity in routes:
            if strength <= 0:
                continue
            weights.append(math.pow(strength, self.eta))
            label_probs.append(self._softmax(self._polarity_logits(polarity)))

        total_w = sum(weights)
        if total_w <= 0:
            return {"beneficial": 1/3, "harmful": 1/3, "neutral": 1/3}

        ben = sum(w * lp[0] for w, lp in zip(weights, label_probs)) / total_w
        harm = sum(w * lp[1] for w, lp in zip(weights, label_probs)) / total_w
        neu = sum(w * lp[2] for w, lp in zip(weights, label_probs)) / total_w
        return {"beneficial": ben, "harmful": harm, "neutral": neu}

    def _polarity_logits(self, polarity: int) -> Tuple[float, float, float]:
        """φ(π): map route polarity to 3-dim label logit vector."""
        s = self.polarity_sharpness
        h = -s / 2
        if polarity > 0:
            return (s, h, h)
        if polarity < 0:
            return (h, s, h)
        return (h, h, s)

    def _softmax(self, logits: Tuple[float, float, float]) -> Tuple[float, float, float]:
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        total = sum(exps)
        return (exps[0] / total, exps[1] / total, exps[2] / total)

    def _score_edge(self, edge: GraphEdge) -> float:
        """Route strength for a direct edge: w_e · exp(-ρ)."""
        return edge.confidence_agg * math.exp(-self.rho)

    def _compute_confidence(
        self,
        s_pos: float,
        s_neg: float,
        s_neu: float,
    ) -> float:
        """Confidence = max posterior probability."""
        total = s_pos + s_neg + s_neu
        if total <= 0:
            return 0.0
        return max(s_pos, s_neg, s_neu) / total


# Convenience functions for simple usage

def infer_conclusion(
    graph: GraphStore,
    head_id: str,
    tail_id: str,
    min_evidence: int = MIN_EVIDENCE,
    margin: float = MARGIN,
) -> ConclusionResult:
    """Infer conclusion using default PathInference settings."""
    inference = PathInference(min_evidence=min_evidence, margin=margin)
    return inference.infer_conclusion(graph, head_id, tail_id)


def aggregate_scores(
    graph: GraphStore,
    head_id: str,
    tail_id: str,
) -> Dict[str, float]:
    """Aggregate scores using default PathInference settings."""
    return PathInference().aggregate_scores(graph, head_id, tail_id)


def compute_confidence(scores: Dict[str, float]) -> float:
    """Compute confidence from score dict."""
    s_pos = max(0.0, float(scores.get("beneficial", 0.0)))
    s_neg = max(0.0, float(scores.get("harmful", 0.0)))
    s_neu = max(0.0, float(scores.get("neutral", 0.0)))
    total = s_pos + s_neg + s_neu
    if total <= 0:
        return 0.0
    return max(s_pos, s_neg, s_neu) / total
