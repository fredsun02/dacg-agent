"""
Search State Management

Tracks evidence accumulation and conclusion inference during search.
Compatible with the 17-feature PRM model from Stage 4.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import Counter

from .extractor import CausalTriple


# Legacy state-feature normalization (used only by to_normalized_array)
FEATURE_NORMS = {
    "papers_seen": 100.0,
    "total_evidences": 100.0,
    "supporting_count": 50.0,
    "opposing_count": 50.0,
    "neutral_count": 20.0,
    "causal_beneficial": 50.0,
    "causal_harmful": 50.0,
    "causal_neutral": 20.0,
    "associated_beneficial": 50.0,
    "associated_harmful": 50.0,
    "associated_neutral": 20.0,
    "indirect_count": 50.0,
    "avg_confidence": 1.0,
    "max_confidence": 1.0,
    "confidence_std": 0.5,
    "conclusion_entropy": 1.1,
    "conclusion_stability": 30.0,
    "marginal_gain": 1.0,
    "new_relation_ratio": 1.0,
    "latest_paper_year": 2024.0,
    "recent_paper_ratio": 1.0,
    "direct_path_exists": 1.0,
    "shortest_path_length": 5.0,
    "path_count": 1000.0
}

# Legacy state-feature order (used only by to_normalized_array)
FEATURE_NAMES = [
    "papers_seen",
    "total_evidences",
    "supporting_count",
    "opposing_count",
    "neutral_count",
    "avg_confidence",
    "max_confidence",
    "confidence_std",
    "conclusion_entropy",
    "conclusion_stability",
    "marginal_gain",
    "new_relation_ratio",
    "latest_paper_year",
    "recent_paper_ratio",
    "direct_path_exists",
    "shortest_path_length",
    "path_count",
    "causal_beneficial",
    "causal_harmful",
    "causal_neutral",
    "associated_beneficial",
    "associated_harmful",
    "associated_neutral",
    "indirect_count"
]


@dataclass
class StateFeatures:
    """24-field state feature vector. Graph PRM uses GraphFeatures (20-dim) instead."""

    # Basic counts
    papers_seen: int = 0
    total_evidences: int = 0

    # Relation counts
    supporting_count: int = 0
    opposing_count: int = 0
    neutral_count: int = 0
    causal_beneficial: int = 0
    causal_harmful: int = 0
    causal_neutral: int = 0
    associated_beneficial: int = 0
    associated_harmful: int = 0
    associated_neutral: int = 0
    indirect_count: int = 0

    # Confidence metrics
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_std: float = 0.0

    # Conclusion stability
    conclusion_entropy: float = 1.0
    conclusion_stability: int = 0

    # Saturation metrics
    marginal_gain: float = 1.0
    new_relation_ratio: float = 1.0

    # Temporal features
    latest_paper_year: int = 0
    recent_paper_ratio: float = 0.0

    # Path features (simplified for online search)
    direct_path_exists: bool = False
    shortest_path_length: int = -1
    path_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "papers_seen": self.papers_seen,
            "total_evidences": self.total_evidences,
            "supporting_count": self.supporting_count,
            "opposing_count": self.opposing_count,
            "neutral_count": self.neutral_count,
            "causal_beneficial": self.causal_beneficial,
            "causal_harmful": self.causal_harmful,
            "causal_neutral": self.causal_neutral,
            "associated_beneficial": self.associated_beneficial,
            "associated_harmful": self.associated_harmful,
            "associated_neutral": self.associated_neutral,
            "indirect_count": self.indirect_count,
            "avg_confidence": self.avg_confidence,
            "max_confidence": self.max_confidence,
            "confidence_std": self.confidence_std,
            "conclusion_entropy": self.conclusion_entropy,
            "conclusion_stability": self.conclusion_stability,
            "marginal_gain": self.marginal_gain,
            "new_relation_ratio": self.new_relation_ratio,
            "latest_paper_year": self.latest_paper_year,
            "recent_paper_ratio": self.recent_paper_ratio,
            "direct_path_exists": self.direct_path_exists,
            "shortest_path_length": self.shortest_path_length,
            "path_count": self.path_count
        }

    def to_normalized_array(self) -> np.ndarray:
        """[Legacy] Normalized array for state-based PRM checkpoints."""
        features = []

        for name in FEATURE_NAMES:
            value = getattr(self, name)

            # Handle boolean
            if isinstance(value, bool):
                value = 1.0 if value else 0.0

            # Special handling for year
            if name == "latest_paper_year":
                value = (value - 2020) / 10.0 if value > 0 else 0.0
            else:
                # Normalize by max value
                norm = FEATURE_NORMS.get(name, 1.0)
                value = float(value) / norm

            features.append(value)

        return np.array(features, dtype=np.float32)


@dataclass
class SearchState:
    """Full search state with evidence tracking"""

    head_entity: str
    tail_entity: str

    # Accumulated triples
    triples: List[CausalTriple] = field(default_factory=list)

    # Paper tracking
    papers_seen: int = 0
    paper_pmids: List[str] = field(default_factory=list)

    # Evidence counts (by direction)
    supporting_count: int = 0   # beneficial direction
    opposing_count: int = 0     # harmful direction
    neutral_count: int = 0      # no effect / unclear

    # Causal vs Associated tracking (overall)
    causal_count: int = 0       # strict causal relations
    associated_count: int = 0   # correlation/association relations

    # Causal vs Associated tracking (by direction)
    causal_beneficial: int = 0
    causal_harmful: int = 0
    causal_neutral: int = 0
    associated_beneficial: int = 0
    associated_harmful: int = 0
    associated_neutral: int = 0
    indirect_count: int = 0

    # Paper relevance tracking
    relevant_papers: int = 0    # papers discussing query entities
    off_topic_papers: int = 0   # papers not related to query

    # Confidence tracking
    confidences: List[float] = field(default_factory=list)

    # Conclusion tracking
    conclusion_history: List[str] = field(default_factory=list)

    # Year tracking
    paper_years: List[int] = field(default_factory=list)

    # Previous state for marginal gain calculation
    prev_supporting_ratio: float = 0.0

    # Step history
    history: List[Dict] = field(default_factory=list)

    def update(
        self,
        extraction_results: List[Dict],
        batch_size: int = 2
    ):
        """
        Update state with new extraction results.

        Args:
            extraction_results: List of extraction result dicts
            batch_size: Batch size for new_relation_ratio calculation
        """
        new_triples_count = 0
        new_papers = 0

        for result in extraction_results:
            pmid = result.get("pmid", "")
            if pmid and pmid not in self.paper_pmids:
                self.paper_pmids.append(pmid)
                new_papers += 1

            # Track paper year
            pub_year = result.get("pub_year", 0)
            if pub_year:
                self.paper_years.append(pub_year)

            # Track paper relevance
            relevance_score = result.get("relevance_score", 0.0)
            if relevance_score > 0.3 or result.get("is_relevant", False):
                self.relevant_papers += 1
            else:
                self.off_topic_papers += 1

            # Process triples (handle both CausalTriple objects and dict triples)
            for triple in result.get("triples", []):
                self.triples.append(triple)

                # Handle both object and dict access patterns
                if isinstance(triple, dict):
                    confidence = triple.get('confidence', 0.5)
                    is_causal = triple.get('is_causal', True)
                    # Check eval_direction first (from evaluator), then direction
                    direction = triple.get('eval_direction') or triple.get('direction')
                    rel_type = triple.get('relation_type', 'Associated')
                else:
                    confidence = triple.confidence
                    is_causal = getattr(triple, 'is_causal', True)
                    direction = getattr(triple, 'direction', None)
                    rel_type = triple.relation_type

                self.confidences.append(confidence)
                new_triples_count += 1

                # Track causal vs associated
                if is_causal:
                    self.causal_count += 1
                else:
                    self.associated_count += 1

                if direction == "beneficial":
                    if is_causal:
                        self.causal_beneficial += 1
                    else:
                        self.associated_beneficial += 1
                    self.supporting_count += 1
                elif direction == "harmful":
                    if is_causal:
                        self.causal_harmful += 1
                    else:
                        self.associated_harmful += 1
                    self.opposing_count += 1
                elif direction == "neutral":
                    if is_causal:
                        self.causal_neutral += 1
                    else:
                        self.associated_neutral += 1
                    self.neutral_count += 1
                elif direction == "indirect":
                    self.indirect_count += 1
                else:
                    # Fallback to relation type classification
                    if rel_type in {"Treat", "Inhibit", "Prevent"}:
                        if is_causal:
                            self.causal_beneficial += 1
                        else:
                            self.associated_beneficial += 1
                        self.supporting_count += 1
                    elif rel_type in {"Cause", "Stimulate", "Worsen"}:
                        if is_causal:
                            self.causal_harmful += 1
                        else:
                            self.associated_harmful += 1
                        self.opposing_count += 1
                    else:
                        if is_causal:
                            self.causal_neutral += 1
                        else:
                            self.associated_neutral += 1
                        self.neutral_count += 1

        self.papers_seen += new_papers

        # Update conclusion
        conclusion = self._infer_conclusion()
        self.conclusion_history.append(conclusion)

        # Compute marginal gain
        total_evidence = self.supporting_count + self.opposing_count + self.neutral_count
        if total_evidence > 0:
            new_ratio = self.supporting_count / total_evidence
            if self.prev_supporting_ratio > 0 and new_triples_count > 0:
                marginal_gain = abs(new_ratio - self.prev_supporting_ratio) / (new_triples_count / batch_size)
            else:
                marginal_gain = 1.0
            self.prev_supporting_ratio = new_ratio
        else:
            marginal_gain = 1.0

        # Build features and record snapshot
        features = self._build_features(new_triples_count, batch_size, marginal_gain)
        snapshot = {
            "step_id": len(self.history),
            "papers_seen": self.papers_seen,
            "total_evidences": len(self.triples),
            "supporting_count": self.supporting_count,
            "opposing_count": self.opposing_count,
            "neutral_count": self.neutral_count,
            "causal_count": self.causal_count,
            "associated_count": self.associated_count,
            "causal_beneficial": self.causal_beneficial,
            "causal_harmful": self.causal_harmful,
            "causal_neutral": self.causal_neutral,
            "associated_beneficial": self.associated_beneficial,
            "associated_harmful": self.associated_harmful,
            "associated_neutral": self.associated_neutral,
            "indirect_count": self.indirect_count,
            "relevant_papers": self.relevant_papers,
            "off_topic_papers": self.off_topic_papers,
            "conclusion": conclusion,
            "confidence": self._compute_confidence(),
            "features": features.to_dict()
        }
        self.history.append(snapshot)

    def _infer_conclusion(self) -> str:
        """Infer conclusion from current evidence"""
        scores = self._weighted_direction_scores()
        if scores["beneficial"] > scores["harmful"] and scores["beneficial"] > scores["neutral"]:
            return "Beneficial"
        elif scores["harmful"] > scores["beneficial"] and scores["harmful"] > scores["neutral"]:
            return "Harmful"
        elif scores["neutral"] > scores["beneficial"] and scores["neutral"] > scores["harmful"]:
            return "NoEffect"
        else:
            return "Uncertain"

    def _weighted_direction_scores(self) -> Dict[str, float]:
        """Compute weighted scores for conclusion inference."""
        causal_weight = 1.0
        associated_weight = 0.3
        return {
            "beneficial": self.causal_beneficial * causal_weight + self.associated_beneficial * associated_weight,
            "harmful": self.causal_harmful * causal_weight + self.associated_harmful * associated_weight,
            "neutral": self.causal_neutral * causal_weight + self.associated_neutral * associated_weight
        }

    def _compute_confidence(self) -> float:
        """Compute confidence in current conclusion"""
        scores = self._weighted_direction_scores()
        total = scores["beneficial"] + scores["harmful"] + scores["neutral"]
        if total == 0:
            return 0.0

        conclusion = self._infer_conclusion()
        if conclusion == "Beneficial":
            return scores["beneficial"] / total
        elif conclusion == "Harmful":
            return scores["harmful"] / total
        elif conclusion == "NoEffect":
            return scores["neutral"] / total
        else:
            return 0.5

    def _build_features(
        self,
        new_triples: int,
        batch_size: int,
        marginal_gain: float
    ) -> StateFeatures:
        """Build state feature vector."""

        # Basic counts
        papers_seen = self.papers_seen
        total_evidences = len(self.triples)

        # Confidence metrics
        if self.confidences:
            avg_confidence = float(np.mean(self.confidences))
            max_confidence = float(np.max(self.confidences))
            confidence_std = float(np.std(self.confidences)) if len(self.confidences) > 1 else 0.0
        else:
            avg_confidence = 0.0
            max_confidence = 0.0
            confidence_std = 0.0

        # Conclusion entropy
        total = self.supporting_count + self.opposing_count + self.neutral_count
        if total > 0:
            probs = []
            for count in [self.supporting_count, self.opposing_count, self.neutral_count]:
                if count > 0:
                    probs.append(count / total)
            if probs:
                entropy = float(-sum(p * np.log(p + 1e-10) for p in probs))
            else:
                entropy = 1.0
        else:
            entropy = 1.0

        # Conclusion stability
        stability = 0
        if len(self.conclusion_history) > 1:
            current = self.conclusion_history[-1]
            for prev in reversed(self.conclusion_history[:-1]):
                if prev == current:
                    stability += 1
                else:
                    break

        # New relation ratio
        new_relation_ratio = new_triples / max(1, batch_size)

        # Temporal features
        if self.paper_years:
            latest_year = max(self.paper_years)
            recent_count = sum(1 for y in self.paper_years if y >= 2020)
            recent_ratio = recent_count / len(self.paper_years)
        else:
            latest_year = 0
            recent_ratio = 0.0

        # Path features (simplified - we have evidence if triples exist)
        direct_path_exists = total_evidences > 0
        shortest_path_length = 1 if total_evidences > 0 else -1
        path_count = total_evidences

        return StateFeatures(
            papers_seen=papers_seen,
            total_evidences=total_evidences,
            supporting_count=self.supporting_count,
            opposing_count=self.opposing_count,
            neutral_count=self.neutral_count,
            causal_beneficial=self.causal_beneficial,
            causal_harmful=self.causal_harmful,
            causal_neutral=self.causal_neutral,
            associated_beneficial=self.associated_beneficial,
            associated_harmful=self.associated_harmful,
            associated_neutral=self.associated_neutral,
            indirect_count=self.indirect_count,
            avg_confidence=avg_confidence,
            max_confidence=max_confidence,
            confidence_std=confidence_std,
            conclusion_entropy=entropy,
            conclusion_stability=stability,
            marginal_gain=marginal_gain,
            new_relation_ratio=new_relation_ratio,
            latest_paper_year=latest_year,
            recent_paper_ratio=recent_ratio,
            direct_path_exists=direct_path_exists,
            shortest_path_length=shortest_path_length,
            path_count=path_count
        )

    def get_current_features(self) -> StateFeatures:
        """Get current state features"""
        if not self.history:
            return StateFeatures()
        return StateFeatures(**self.history[-1]["features"])

    def get_conclusion(self) -> str:
        """Get current conclusion"""
        return self._infer_conclusion()

    def get_confidence(self) -> float:
        """Get current confidence"""
        return self._compute_confidence()


@dataclass
class SearchResult:
    """Final search result"""

    query: str
    head_entity: str
    tail_entity: str

    conclusion: str
    confidence: float

    total_steps: int
    papers_searched: int
    stop_reason: str

    history: List[Dict] = field(default_factory=list)
    evidence_summary: Dict = field(default_factory=dict)

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
            "history": self.history,
            "evidence_summary": self.evidence_summary
        }
