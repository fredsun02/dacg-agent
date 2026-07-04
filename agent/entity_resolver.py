"""
Entity Resolver for KGSA Graph Store.

Normalizes entity names and resolves them to stable node IDs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable


@dataclass(frozen=True)
class ResolvedEntity:
    """Result of entity resolution."""
    id: str
    name: str
    normalized: str
    created: bool


class EntityResolver:
    """Normalize and resolve entity names to stable node IDs."""

    _STOPWORDS = frozenset({"the", "a", "an", "of", "and", "in", "on", "for", "with"})

    def __init__(self, alias_map: Optional[Dict[str, str]] = None) -> None:
        self.alias_to_id: Dict[str, str] = {}
        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}
        self.id_to_aliases: Dict[str, List[str]] = {}
        self.id_to_type: Dict[str, Optional[str]] = {}
        self._counter: int = 0

        if alias_map:
            for alias, node_id in alias_map.items():
                self.alias_to_id[self.normalize(alias)] = node_id

    def normalize(self, name: str) -> str:
        """Normalize entity name for matching."""
        if not name:
            return ""
        text = name.strip().lower()
        text = text.replace("-", " ").replace("_", " ")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def canonicalize(self, name: str) -> str:
        """Canonicalize name for display (preserve case, clean whitespace)."""
        if not name:
            return ""
        return re.sub(r"\s+", " ", name.strip())

    def expand_variants(self, normalized: str) -> List[str]:
        """Generate lightweight variants for improved matching."""
        if not normalized:
            return []

        variants = {normalized}

        # Remove stopwords
        tokens = [t for t in normalized.split() if t not in self._STOPWORDS]
        if tokens and len(tokens) < len(normalized.split()):
            variants.add(" ".join(tokens))

        # Simple singular/plural handling
        if normalized.endswith("s") and len(normalized) > 3:
            variants.add(normalized[:-1])
        if normalized.endswith("ies") and len(normalized) > 4:
            variants.add(normalized[:-3] + "y")

        # Space-less variant
        spaceless = normalized.replace(" ", "")
        if spaceless != normalized:
            variants.add(spaceless)

        return [v for v in variants if v]

    def add_alias(self, node_id: str, alias: str, overwrite: bool = False) -> bool:
        """Add an alias for a node."""
        normalized = self.normalize(alias)
        if not normalized:
            return False

        existing = self.alias_to_id.get(normalized)
        if existing and existing != node_id and not overwrite:
            return False

        self.alias_to_id[normalized] = node_id
        self.id_to_aliases.setdefault(node_id, [])
        if alias not in self.id_to_aliases[node_id]:
            self.id_to_aliases[node_id].append(alias)
        return True

    def register(
        self,
        node_id: str,
        canonical_name: str,
        node_type: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None
    ) -> None:
        """Register a node with its canonical name and optional aliases."""
        self.id_to_name[node_id] = canonical_name
        self.name_to_id[self.normalize(canonical_name)] = node_id
        self.id_to_type[node_id] = node_type
        if aliases:
            for alias in aliases:
                self.add_alias(node_id, alias)

    def match(self, name: str) -> Optional[str]:
        """Find matching node ID for a name."""
        normalized = self.normalize(name)
        if not normalized:
            return None

        # Exact name match
        if normalized in self.name_to_id:
            return self.name_to_id[normalized]

        # Alias match
        if normalized in self.alias_to_id:
            return self.alias_to_id[normalized]

        # Variant match
        for variant in self.expand_variants(normalized):
            node_id = self.name_to_id.get(variant) or self.alias_to_id.get(variant)
            if node_id:
                return node_id

        return None

    def resolve(
        self,
        name: str,
        create: bool = False,
        node_type: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None
    ) -> ResolvedEntity:
        """Resolve a name to a node ID, optionally creating a new node."""
        normalized = self.normalize(name)

        # Reject empty names
        if not normalized:
            return ResolvedEntity(id="", name="", normalized="", created=False)

        node_id = self.match(name)

        if node_id:
            return ResolvedEntity(
                id=node_id,
                name=self.id_to_name.get(node_id, name),
                normalized=normalized,
                created=False
            )

        if not create:
            return ResolvedEntity(id="", name="", normalized=normalized, created=False)

        node_id = self._next_id()
        canonical = self.canonicalize(name)
        self.register(node_id, canonical, node_type=node_type, aliases=aliases)

        return ResolvedEntity(
            id=node_id,
            name=canonical,
            normalized=normalized,
            created=True
        )

    def _next_id(self) -> str:
        self._counter += 1
        return f"E{self._counter:06d}"

    def __len__(self) -> int:
        return len(self.id_to_name)
