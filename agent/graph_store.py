"""
Graph Store for KGSA Dynamic Knowledge Graph.

Provides in-memory graph storage with efficient path queries.
Replaces counter-based evidence tracking with a queryable graph structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Any

from .extractor import CausalTriple

try:
    from .entity_resolver import EntityResolver, ResolvedEntity
except ImportError:
    EntityResolver = None  # type: ignore
    ResolvedEntity = None  # type: ignore


# Direction to polarity mapping
DIRECTION_POLARITY = {
    "beneficial": 1,
    "harmful": -1,
    "neutral": 0,
    "unclear": 0,
    "indirect": 0,
}

# Relation type to polarity mapping (fallback)
RELATION_POLARITY = {
    "Treat": 1,
    "Prevent": 1,
    "Inhibit": 1,
    "Cause": -1,
    "Worsen": -1,
    "Stimulate": -1,
    "NoEffect": 0,
    "Associated": 0,
}

# Length decay for multiplicative route strength
RHO = 1.0


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    name: str
    type: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

    def add_alias(self, alias: str) -> None:
        if alias and alias not in self.aliases:
            self.aliases.append(alias)


@dataclass
class Evidence:
    """Single piece of evidence supporting an edge."""
    pmid: str
    confidence: float
    snippet: str
    pub_year: int = 0


@dataclass
class GraphEdge:
    """Edge in the knowledge graph with aggregated evidence."""
    id: str
    head_id: str
    tail_id: str
    relation_type: str
    polarity: int  # +1 (beneficial), -1 (harmful), 0 (neutral)
    is_causal: bool
    evidences: List[Evidence] = field(default_factory=list)
    confidence_agg: float = 0.0

    # Track seen evidence to avoid duplicates
    _seen_keys: set = field(default_factory=set, repr=False)

    def add_evidence(self, ev: Evidence) -> bool:
        """Add evidence, aggregating confidence. Returns True if new."""
        key = (ev.pmid, ev.snippet[:100] if ev.snippet else "")
        if key in self._seen_keys:
            return False

        self._seen_keys.add(key)
        self.evidences.append(ev)

        # Aggregate confidence: s_new = 1 - (1 - s_old) * (1 - s_in)
        self.confidence_agg = 1.0 - (1.0 - self.confidence_agg) * (1.0 - ev.confidence)
        return True

    @property
    def evidence_count(self) -> int:
        return len(self.evidences)


@dataclass(frozen=True)
class GraphPath:
    """Path through the graph (direct or multi-hop)."""
    node_ids: Tuple[str, ...]
    edge_ids: Tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.edge_ids)

    @property
    def is_direct(self) -> bool:
        return len(self.edge_ids) == 1


@dataclass
class PathScores:
    """Aggregated path scores for conclusion inference."""
    beneficial: float = 0.0
    harmful: float = 0.0
    neutral: float = 0.0
    total_evidence: int = 0
    direct_paths: int = 0
    two_hop_paths: int = 0


class GraphStore:
    """In-memory graph store with path query support."""

    def __init__(self, resolver: Optional[EntityResolver] = None) -> None:
        self.resolver = resolver
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}

        # Indexes for efficient path queries
        self.out_index: Dict[str, set] = {}  # head_id -> {edge_id}
        self.in_index: Dict[str, set] = {}   # tail_id -> {edge_id}

        # Edge deduplication index
        self._edge_key_to_id: Dict[Tuple[str, str, str, int], str] = {}
        self._edge_counter: int = 0

    def add_triple(
        self,
        triple: CausalTriple | Dict[str, Any],
        head_type: Optional[str] = None,
        tail_type: Optional[str] = None
    ) -> Optional[GraphEdge]:
        """Add a causal triple to the graph, aggregating on existing edges."""
        # Extract fields from triple with safe type conversion
        if isinstance(triple, dict):
            head_name = triple.get("head_entity") or triple.get("head", "")
            tail_name = triple.get("tail_entity") or triple.get("tail", "")
            relation_type = triple.get("relation_type", "Associated")
            raw_conf = triple.get("confidence", 0.5)
            confidence = float(raw_conf) if raw_conf not in (None, "") else 0.5
            snippet = triple.get("evidence_text", "") or ""
            pmid = str(triple.get("pmid", "") or "")
            is_causal = bool(triple.get("is_causal", True))
            direction = triple.get("eval_direction") or triple.get("direction", "unclear")
            raw_year = triple.get("pub_year", 0)
            pub_year = int(raw_year) if raw_year not in (None, "") else 0
        else:
            head_name = triple.head_entity or ""
            tail_name = triple.tail_entity or ""
            relation_type = triple.relation_type or "Associated"
            raw_conf = triple.confidence
            confidence = float(raw_conf) if raw_conf is not None else 0.5
            snippet = triple.evidence_text or ""
            pmid = str(triple.pmid) if triple.pmid else ""
            is_causal = bool(getattr(triple, "is_causal", True))
            direction = getattr(triple, "direction", "unclear") or "unclear"
            pub_year = 0

        # Skip triples with empty entities
        if not head_name.strip() or not tail_name.strip():
            return None

        # Resolve entities to node IDs
        head_id = self._ensure_node(head_name, head_type)
        tail_id = self._ensure_node(tail_name, tail_type)

        # Determine polarity (non-causal / Associated → forced neutral)
        if not is_causal or relation_type == "Associated":
            polarity = 0
        else:
            polarity = DIRECTION_POLARITY.get(direction)
            if polarity is None or direction == "unclear":
                polarity = RELATION_POLARITY.get(relation_type, 0)

        # Find or create edge
        edge_key = (head_id, tail_id, relation_type, polarity)
        edge_id = self._edge_key_to_id.get(edge_key)

        if edge_id is None:
            edge_id = self._next_edge_id()
            edge = GraphEdge(
                id=edge_id,
                head_id=head_id,
                tail_id=tail_id,
                relation_type=relation_type,
                polarity=polarity,
                is_causal=is_causal,
            )
            self.edges[edge_id] = edge
            self._edge_key_to_id[edge_key] = edge_id
            self._index_edge(edge)
        else:
            edge = self.edges[edge_id]

        # Add evidence
        edge.add_evidence(Evidence(
            pmid=pmid,
            confidence=confidence,
            snippet=snippet,
            pub_year=pub_year,
        ))

        return edge

    def get_direct_edges(
        self,
        head_id: str,
        tail_id: Optional[str] = None,
        polarity: Optional[int] = None,
        is_causal: Optional[bool] = None,
    ) -> List[GraphEdge]:
        """Get direct edges from head_id with optional filtering."""
        results = []
        for edge_id in self.out_index.get(head_id, set()):
            edge = self.edges[edge_id]
            if tail_id is not None and edge.tail_id != tail_id:
                continue
            if polarity is not None and edge.polarity != polarity:
                continue
            if is_causal is not None and edge.is_causal != is_causal:
                continue
            results.append(edge)
        return results

    def get_two_hop_paths(
        self,
        head_id: str,
        tail_id: str,
        max_paths: int = 100,
    ) -> List[GraphPath]:
        """Find all two-hop paths: head_id -> mid -> tail_id."""
        # First hop: head -> mid
        first_hops: Dict[str, List[str]] = {}
        for edge_id in self.out_index.get(head_id, set()):
            edge = self.edges[edge_id]
            mid_id = edge.tail_id
            if mid_id != tail_id:  # Avoid trivial paths
                first_hops.setdefault(mid_id, []).append(edge_id)

        # Second hop: mid -> tail
        paths = []
        for edge_id in self.in_index.get(tail_id, set()):
            edge = self.edges[edge_id]
            mid_id = edge.head_id
            if mid_id in first_hops:
                for first_edge_id in first_hops[mid_id]:
                    paths.append(GraphPath(
                        node_ids=(head_id, mid_id, tail_id),
                        edge_ids=(first_edge_id, edge_id),
                    ))
                    if len(paths) >= max_paths:
                        return paths
        return paths

    def compute_path_polarity(self, path: GraphPath) -> int:
        """Compute path polarity: neutral-transparent + harmful-chain override."""
        if not path.edge_ids:
            return 0

        non_zero = []
        for edge_id in path.edge_ids:
            edge = self.edges.get(edge_id)
            if edge is None:
                return 0
            if edge.polarity != 0:
                non_zero.append(edge.polarity)

        if not non_zero:
            return 0
        if len(non_zero) == 1:
            return non_zero[0]
        # All harmful edges → harmful chain (override -1×-1=+1)
        if all(p == -1 for p in non_zero):
            return -1
        result = 1
        for p in non_zero:
            result *= p
        return result

    def compute_path_score(
        self,
        path: GraphPath,
        length_penalty: float = RHO,
    ) -> float:
        """Multiplicative route strength: Str(π,t) = ∏w_e · exp(-ρk)."""
        if not path.edge_ids:
            return 0.0

        strength = 1.0
        for eid in path.edge_ids:
            edge = self.edges.get(eid)
            if edge is None:
                return 0.0
            strength *= edge.confidence_agg

        return strength * math.exp(-length_penalty * path.length)

    def compute_path_scores(
        self,
        head_id: str,
        tail_id: str,
    ) -> PathScores:
        """Compute aggregated path scores for conclusion inference."""
        scores = PathScores()

        # Direct edges
        direct_edges = self.get_direct_edges(head_id, tail_id)
        for edge in direct_edges:
            score = edge.confidence_agg * math.exp(-RHO)
            if edge.polarity > 0:
                scores.beneficial += score
            elif edge.polarity < 0:
                scores.harmful += score
            else:
                scores.neutral += score
            scores.total_evidence += edge.evidence_count
        scores.direct_paths = len(direct_edges)

        # Two-hop paths
        two_hop = self.get_two_hop_paths(head_id, tail_id)
        for path in two_hop:
            polarity = self.compute_path_polarity(path)
            path_score = self.compute_path_score(path, length_penalty=RHO)

            if polarity > 0:
                scores.beneficial += path_score
            elif polarity < 0:
                scores.harmful += path_score
            else:
                scores.neutral += path_score

            for eid in path.edge_ids:
                edge = self.edges.get(eid)
                if edge:
                    scores.total_evidence += edge.evidence_count
        scores.two_hop_paths = len(two_hop)

        return scores

    def count_unique_evidence(self, head_id: str, tail_id: str) -> int:
        """Count unique PMIDs across all paths between head and tail."""
        pmids = set()

        for edge in self.get_direct_edges(head_id, tail_id):
            for ev in edge.evidences:
                pmids.add(ev.pmid)

        for path in self.get_two_hop_paths(head_id, tail_id):
            for eid in path.edge_ids:
                edge = self.edges.get(eid)
                if edge:
                    for ev in edge.evidences:
                        pmids.add(ev.pmid)

        return len(pmids)

    def get_node_id(self, name: str) -> Optional[str]:
        """Get node ID for a name, or None if not found."""
        if self.resolver:
            resolved = self.resolver.match(name)
            return resolved
        # Fallback: check all nodes
        normalized = (name or "").strip().lower().replace(" ", "_")
        for node_id, node in self.nodes.items():
            if node.name.lower().replace(" ", "_") == normalized:
                return node_id
        return None

    def _ensure_node(self, name: str, node_type: Optional[str]) -> str:
        """Ensure a node exists for the given name, creating if needed."""
        if self.resolver:
            resolved = self.resolver.resolve(name, create=True, node_type=node_type)
            if resolved.id not in self.nodes:
                self.nodes[resolved.id] = GraphNode(
                    id=resolved.id,
                    name=resolved.name,
                    type=node_type,
                )
            return resolved.id

        # Fallback: simple normalization
        node_id = self._fallback_node_id(name)
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(id=node_id, name=name, type=node_type)
        return node_id

    def _fallback_node_id(self, name: str) -> str:
        clean = (name or "").strip().lower().replace(" ", "_")[:40]
        return f"n_{clean}" if clean else f"n_{len(self.nodes) + 1}"

    def _next_edge_id(self) -> str:
        self._edge_counter += 1
        return f"e{self._edge_counter:06d}"

    def _index_edge(self, edge: GraphEdge) -> None:
        self.out_index.setdefault(edge.head_id, set()).add(edge.id)
        self.in_index.setdefault(edge.tail_id, set()).add(edge.id)

    def __len__(self) -> int:
        return len(self.edges)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def node_degree(self, node_id: str) -> int:
        """Return total degree (in + out) for a node."""
        out_deg = len(self.out_index.get(node_id, set()))
        in_deg = len(self.in_index.get(node_id, set()))
        return out_deg + in_deg

    def get_node_name(self, node_id: str) -> str:
        """Get node name by ID."""
        node = self.nodes.get(node_id)
        return node.name if node else ""

    # --- Energy foraging methods ---

    def get_pair_edges(self, head_id: str, tail_id: str) -> List[GraphEdge]:
        """Get all edges between (head_id, tail_id) regardless of polarity."""
        results = []
        for edge_id in self.out_index.get(head_id, set()):
            edge = self.edges[edge_id]
            if edge.tail_id == tail_id:
                results.append(edge)
        return results

    def compute_pair_polarity_distribution(
        self, head_id: str, tail_id: str
    ) -> Tuple[float, float, float]:
        """θ = (p+, p-, p0) for entity pair, weighted by evidence_count."""
        edges = self.get_pair_edges(head_id, tail_id)
        w_pos = w_neg = w_neu = 0.0
        for edge in edges:
            w = edge.evidence_count
            if edge.polarity > 0:
                w_pos += w
            elif edge.polarity < 0:
                w_neg += w
            else:
                w_neu += w
        total = w_pos + w_neg + w_neu
        if total <= 0:
            return (1/3, 1/3, 1/3)
        return (w_pos / total, w_neg / total, w_neu / total)

    def compute_pair_entropy(
        self,
        head_id: str,
        tail_id: str,
        _cache: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> float:
        """H(θ) = -Σ p·log(p) for the pair polarity distribution."""
        if _cache is not None:
            key = (head_id, tail_id)
            cached = _cache.get(key)
            if cached is not None:
                return cached
        dist = self.compute_pair_polarity_distribution(head_id, tail_id)
        entropy = 0.0
        for p in dist:
            if p > 0:
                entropy -= p * math.log(p)
        if _cache is not None:
            _cache[key] = entropy
        return entropy

    def compute_path_edge_uncertainty(
        self,
        path: GraphPath,
        lambda_u: float,
        _cache: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> float:
        """λ_U · Σ_{hop} H(θ_{u,v}) along path edges."""
        if lambda_u <= 0 or not path.edge_ids:
            return 0.0
        total_h = 0.0
        for eid in path.edge_ids:
            edge = self.edges.get(eid)
            if edge:
                total_h += self.compute_pair_entropy(
                    edge.head_id, edge.tail_id, _cache
                )
        return lambda_u * total_h

    def compute_path_constraint_violation(
        self, path: GraphPath, quality_floor: float, lambda_v: float
    ) -> float:
        """λ_V · Σ max(0, τ_q - w_e) along path edges."""
        if lambda_v <= 0 or not path.edge_ids:
            return 0.0
        total_viol = 0.0
        for eid in path.edge_ids:
            edge = self.edges.get(eid)
            if edge:
                gap = quality_floor - edge.confidence_agg
                if gap > 0:
                    total_viol += gap
        return lambda_v * total_viol
