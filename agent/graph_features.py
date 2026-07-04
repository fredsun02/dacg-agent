"""
Graph Topology Features for PRM.

Extracts 18-dimensional graph-based features for the Process Reward Model,
replacing/augmenting the counter-based features in StateFeatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .graph_store import GraphStore, GraphEdge
from .path_inference import PathInference, RHO


# Feature normalization constants
GRAPH_FEATURE_NORMS = {
    "direct_path_count": 20.0,
    "direct_score_max": 2.0,
    "two_hop_count": 100.0,
    "two_hop_score_max": 2.0,
    "best_path_score": 2.0,
    "path_score_gap": 5.0,
    "contradiction_ratio": 1.0,
    "polarity_entropy": 1.1,  # max entropy for 3 classes
    "head_degree": 50.0,
    "tail_degree": 50.0,
    "common_neighbors": 20.0,
    "new_edges_ratio": 1.0,
    "path_discovery_rate": 1.0,
    "marginal_gain": 1.0,
    "total_evidence": 100.0,
    "avg_confidence": 1.0,
    "graph_density": 1.0,
    "recency_ratio": 1.0,
    "edge_uncertainty": 1.1,
    "constraint_violation": 2.0,
}

GRAPH_FEATURE_NAMES = [
    "direct_path_count",
    "direct_score_max",
    "two_hop_count",
    "two_hop_score_max",
    "best_path_score",
    "path_score_gap",
    "contradiction_ratio",
    "polarity_entropy",
    "head_degree",
    "tail_degree",
    "common_neighbors",
    "new_edges_ratio",
    "path_discovery_rate",
    "marginal_gain",
    "total_evidence",
    "avg_confidence",
    "graph_density",
    "recency_ratio",
    "edge_uncertainty",
    "constraint_violation",
]


@dataclass
class GraphFeatures:
    """20-dimensional graph topology features for PRM."""

    # Path features
    direct_path_count: int = 0
    direct_score_max: float = 0.0
    two_hop_count: int = 0
    two_hop_score_max: float = 0.0
    best_path_score: float = 0.0
    path_score_gap: float = 0.0

    # Conflict features
    contradiction_ratio: float = 0.0
    polarity_entropy: float = 1.0

    # Coverage features
    head_degree: int = 0
    tail_degree: int = 0
    common_neighbors: int = 0
    new_edges_ratio: float = 1.0

    # Saturation features
    path_discovery_rate: float = 1.0
    marginal_gain: float = 1.0

    # Global features
    total_evidence: int = 0
    avg_confidence: float = 0.0
    graph_density: float = 0.0
    recency_ratio: float = 0.0

    # Energy features
    edge_uncertainty: float = 0.0
    constraint_violation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in GRAPH_FEATURE_NAMES}

    def to_normalized_array(self) -> List[float]:
        """Convert to normalized feature array for PRM input."""
        features = []
        for name in GRAPH_FEATURE_NAMES:
            value = float(getattr(self, name))
            norm = GRAPH_FEATURE_NORMS.get(name, 1.0)
            features.append(value / norm)
        return features


class GraphFeatureExtractor:
    """Extracts graph topology features from GraphStore."""

    def __init__(self, length_penalty: float = RHO):
        self.length_penalty = length_penalty
        self.inference = PathInference(length_penalty=length_penalty)

    def extract(
        self,
        graph: GraphStore,
        head_id: str,
        tail_id: str,
        prev_features: Optional[GraphFeatures] = None,
        recent_year_threshold: int = 2020,
    ) -> GraphFeatures:
        """
        Extract graph features for a query pair.

        Args:
            graph: GraphStore instance
            head_id: Head node ID
            tail_id: Tail node ID
            prev_features: Previous step features (for marginal gain)
            recent_year_threshold: Year threshold for recency

        Returns:
            GraphFeatures instance
        """
        # Path features
        direct_edges = graph.get_direct_edges(head_id, tail_id)
        two_hop_paths = graph.get_two_hop_paths(head_id, tail_id)

        direct_path_count = len(direct_edges)
        two_hop_count = len(two_hop_paths)

        # Direct edge scores
        direct_scores = [self._score_edge(e) for e in direct_edges]
        direct_score_max = max(direct_scores) if direct_scores else 0.0

        # Two-hop path scores
        two_hop_scores = [graph.compute_path_score(p, self.length_penalty) for p in two_hop_paths]
        two_hop_score_max = max(two_hop_scores) if two_hop_scores else 0.0

        best_path_score = max(direct_score_max, two_hop_score_max)

        # Aggregate scores by polarity
        scores = self.inference.aggregate_scores(graph, head_id, tail_id)
        s_pos = scores.get("beneficial", 0.0)
        s_neg = scores.get("harmful", 0.0)
        s_neu = scores.get("neutral", 0.0)
        path_score_gap = abs(s_pos - s_neg)

        # Conflict features
        contradiction_ratio, polarity_entropy = self._compute_conflict_features(
            direct_edges, two_hop_paths, graph
        )

        # Coverage features
        head_degree = graph.node_degree(head_id) if head_id in graph.nodes else 0
        tail_degree = graph.node_degree(tail_id) if tail_id in graph.nodes else 0
        common_neighbors = self._count_common_neighbors(graph, head_id, tail_id)

        # New edges ratio (compared to previous step)
        prev_edge_count = prev_features.direct_path_count if prev_features else 0
        new_edges = max(0, direct_path_count - prev_edge_count)
        new_edges_ratio = new_edges / max(1, direct_path_count) if direct_path_count > 0 else 1.0

        # Path discovery rate
        prev_paths = prev_features.two_hop_count if prev_features else 0
        new_paths = max(0, two_hop_count - prev_paths)
        path_discovery_rate = new_paths / max(1, two_hop_count) if two_hop_count > 0 else 1.0

        # Marginal gain
        if prev_features and prev_features.path_score_gap > 0:
            marginal_gain = abs(path_score_gap - prev_features.path_score_gap) / prev_features.path_score_gap
        else:
            marginal_gain = 1.0

        # Global features
        total_evidence = graph.count_unique_evidence(head_id, tail_id)

        # Average confidence
        all_confidences = [e.confidence_agg for e in direct_edges]
        for path in two_hop_paths:
            for eid in path.edge_ids:
                edge = graph.edges.get(eid)
                if edge:
                    all_confidences.append(edge.confidence_agg)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # Graph density (edges / possible edges in local subgraph)
        local_nodes = {head_id, tail_id}
        for e in direct_edges:
            local_nodes.add(e.head_id)
            local_nodes.add(e.tail_id)
        for path in two_hop_paths:
            for nid in path.node_ids:
                local_nodes.add(nid)
        n_nodes = len(local_nodes)
        # Count actual edges in local subgraph (not paths)
        n_edges = direct_path_count
        max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        graph_density = min(1.0, n_edges / max_edges) if max_edges > 0 else 0.0

        # Recency ratio
        recent_count = 0
        total_with_year = 0
        for edge in direct_edges:
            for ev in edge.evidences:
                if ev.pub_year > 0:
                    total_with_year += 1
                    if ev.pub_year >= recent_year_threshold:
                        recent_count += 1
        recency_ratio = recent_count / total_with_year if total_with_year > 0 else 0.0

        # Energy features: edge uncertainty H(θ) and constraint violation
        pairs_seen = set()
        h_sum = 0.0
        h_count = 0
        viol_sum = 0.0
        viol_edges_seen = set()
        _quality_floor = 0.3
        for edge in direct_edges:
            pair = (edge.head_id, edge.tail_id)
            if pair not in pairs_seen:
                pairs_seen.add(pair)
                h_sum += graph.compute_pair_entropy(edge.head_id, edge.tail_id)
                h_count += 1
            if edge.id not in viol_edges_seen:
                viol_edges_seen.add(edge.id)
                gap = _quality_floor - edge.confidence_agg
                if gap > 0:
                    viol_sum += gap
        for path in two_hop_paths:
            for eid in path.edge_ids:
                edge_obj = graph.edges.get(eid)
                if edge_obj:
                    pair = (edge_obj.head_id, edge_obj.tail_id)
                    if pair not in pairs_seen:
                        pairs_seen.add(pair)
                        h_sum += graph.compute_pair_entropy(edge_obj.head_id, edge_obj.tail_id)
                        h_count += 1
                    if eid not in viol_edges_seen:
                        viol_edges_seen.add(eid)
                        gap = _quality_floor - edge_obj.confidence_agg
                        if gap > 0:
                            viol_sum += gap
        edge_uncertainty = h_sum / h_count if h_count > 0 else 0.0
        constraint_violation = viol_sum

        return GraphFeatures(
            direct_path_count=direct_path_count,
            direct_score_max=direct_score_max,
            two_hop_count=two_hop_count,
            two_hop_score_max=two_hop_score_max,
            best_path_score=best_path_score,
            path_score_gap=path_score_gap,
            contradiction_ratio=contradiction_ratio,
            polarity_entropy=polarity_entropy,
            head_degree=head_degree,
            tail_degree=tail_degree,
            common_neighbors=common_neighbors,
            new_edges_ratio=new_edges_ratio,
            path_discovery_rate=path_discovery_rate,
            marginal_gain=marginal_gain,
            total_evidence=total_evidence,
            avg_confidence=avg_confidence,
            graph_density=graph_density,
            recency_ratio=recency_ratio,
            edge_uncertainty=edge_uncertainty,
            constraint_violation=constraint_violation,
        )

    def _score_edge(self, edge: GraphEdge) -> float:
        """Route strength for a direct edge: w_e · exp(-ρ)."""
        return edge.confidence_agg * math.exp(-self.length_penalty)

    def _compute_conflict_features(
        self,
        direct_edges: List[GraphEdge],
        two_hop_paths: List,
        graph: GraphStore,
    ) -> Tuple[float, float]:
        """Compute contradiction ratio and polarity entropy."""
        polarities = {"positive": 0, "negative": 0, "neutral": 0}

        for edge in direct_edges:
            if edge.polarity > 0:
                polarities["positive"] += 1
            elif edge.polarity < 0:
                polarities["negative"] += 1
            else:
                polarities["neutral"] += 1

        for path in two_hop_paths:
            polarity = graph.compute_path_polarity(path)
            if polarity > 0:
                polarities["positive"] += 1
            elif polarity < 0:
                polarities["negative"] += 1
            else:
                polarities["neutral"] += 1

        total = sum(polarities.values())
        if total == 0:
            return 0.0, 1.0

        # Contradiction ratio: min(pos, neg) / max(pos, neg)
        pos, neg = polarities["positive"], polarities["negative"]
        if pos == 0 and neg == 0:
            contradiction_ratio = 0.0
        elif max(pos, neg) == 0:
            contradiction_ratio = 0.0
        else:
            contradiction_ratio = min(pos, neg) / max(pos, neg)

        # Polarity entropy
        entropy = 0.0
        for count in polarities.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        return contradiction_ratio, max(0.0, entropy)

    def _count_common_neighbors(
        self,
        graph: GraphStore,
        head_id: str,
        tail_id: str,
    ) -> int:
        """Count common neighbors of head and tail."""
        head_neighbors = set()
        for eid in graph.out_index.get(head_id, set()):
            edge = graph.edges.get(eid)
            if edge:
                head_neighbors.add(edge.tail_id)
        for eid in graph.in_index.get(head_id, set()):
            edge = graph.edges.get(eid)
            if edge:
                head_neighbors.add(edge.head_id)

        tail_neighbors = set()
        for eid in graph.out_index.get(tail_id, set()):
            edge = graph.edges.get(eid)
            if edge:
                tail_neighbors.add(edge.tail_id)
        for eid in graph.in_index.get(tail_id, set()):
            edge = graph.edges.get(eid)
            if edge:
                tail_neighbors.add(edge.head_id)

        # Remove head and tail themselves
        head_neighbors.discard(head_id)
        head_neighbors.discard(tail_id)
        tail_neighbors.discard(head_id)
        tail_neighbors.discard(tail_id)

        return len(head_neighbors & tail_neighbors)


def extract_graph_features(
    graph: GraphStore,
    head_id: str,
    tail_id: str,
    prev_features: Optional[GraphFeatures] = None,
) -> GraphFeatures:
    """Convenience function to extract graph features."""
    extractor = GraphFeatureExtractor()
    return extractor.extract(graph, head_id, tail_id, prev_features)
