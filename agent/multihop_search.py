"""
Graph-guided Multi-hop Search for KGSA.

Expands search when direct evidence is insufficient by finding
intermediate nodes and searching for indirect causal paths.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .graph_store import GraphStore
from .path_inference import PathInference

if TYPE_CHECKING:
    from .search_agent import KGSAAgent


def select_intermediate_nodes(
    graph: GraphStore,
    head_id: str,
    tail_id: str,
    top_k: int = 5,
) -> List[str]:
    """
    Select intermediate node IDs for multi-hop expansion.

    Strategy:
    1. Prefer nodes that are neighbors of both head and tail (bridging nodes)
    2. Otherwise rank candidates by node degree (hub nodes)

    Args:
        graph: GraphStore instance
        head_id: Head node ID
        tail_id: Tail node ID
        top_k: Maximum number of intermediate nodes to return

    Returns:
        List of intermediate node IDs
    """
    if top_k <= 0:
        return []

    # Get head's outgoing neighbors
    head_neighbors = set()
    for edge_id in graph.out_index.get(head_id, set()):
        edge = graph.edges.get(edge_id)
        if edge:
            head_neighbors.add(edge.tail_id)

    # Get tail's incoming neighbors
    tail_neighbors = set()
    for edge_id in graph.in_index.get(tail_id, set()):
        edge = graph.edges.get(edge_id)
        if edge:
            tail_neighbors.add(edge.head_id)

    # Remove trivial nodes
    head_neighbors.discard(head_id)
    head_neighbors.discard(tail_id)
    tail_neighbors.discard(head_id)
    tail_neighbors.discard(tail_id)

    # Prefer common neighbors (bridging nodes)
    common = head_neighbors & tail_neighbors
    if common:
        # Sort for reproducibility
        return sorted(common, key=lambda nid: graph.node_degree(nid), reverse=True)[:top_k]

    # Otherwise rank by degree
    candidates = head_neighbors | tail_neighbors
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda nid: graph.node_degree(nid),
        reverse=True
    )
    return ranked[:top_k]


def expand_search(
    agent: "KGSAAgent",
    graph: GraphStore,
    head_name: str,
    tail_name: str,
    head_id: str,
    tail_id: str,
    max_intermediates: int = 3,
    verbose: bool = False,
    inference: Optional[PathInference] = None,
) -> int:
    """
    Perform graph-guided multi-hop search expansion.

    Triggered when conclusion is NoEvidence or Uncertain.
    Searches for head-intermediate and intermediate-tail pairs.

    Returns:
        Number of new triples added to graph
    """
    # Check if expansion is needed
    inf = inference or PathInference()
    result = inf.infer_conclusion(graph, head_id, tail_id)

    if not result.should_continue:
        if verbose:
            print(f"  [Multihop] No expansion needed: {result.conclusion}")
        return 0

    # Select intermediate nodes
    intermediates = select_intermediate_nodes(graph, head_id, tail_id, top_k=max_intermediates)

    if not intermediates:
        if verbose:
            print("  [Multihop] No intermediate nodes found")
        return 0

    if verbose:
        print(f"  [Multihop] Expanding with {len(intermediates)} intermediate nodes")

    new_triples = 0

    for mid_id in intermediates:
        mid_name = graph.get_node_name(mid_id)
        if not mid_name:
            continue

        if verbose:
            print(f"    -> Searching via: {mid_name}")

        # Search head -> mid
        added_hm = _search_pair(agent, graph, head_name, mid_name, verbose)
        new_triples += added_hm

        # Search mid -> tail
        added_mt = _search_pair(agent, graph, mid_name, tail_name, verbose)
        new_triples += added_mt

    if verbose:
        print(f"  [Multihop] Added {new_triples} new triples")

    return new_triples


def should_expand(
    graph: GraphStore,
    head_id: str,
    tail_id: str,
) -> bool:
    """Check if multi-hop expansion should be triggered."""
    result = PathInference().infer_conclusion(graph, head_id, tail_id)
    return result.should_continue


def _search_pair(
    agent: "KGSAAgent",
    graph: GraphStore,
    left: str,
    right: str,
    verbose: bool = False,
) -> int:
    """Search for papers about a specific entity pair and add to graph."""
    if not left or not right:
        return 0

    # Build query with exact phrase matching
    query = f'"{left}" AND "{right}"'
    added = 0

    try:
        # Search only 1 batch for expansion (to limit API calls)
        for _, papers in agent.pubmed.search_and_fetch(
            query,
            batch_size=agent.batch_size,
            max_batches=1
        ):
            if not papers:
                continue

            if verbose:
                print(f"      Found {len(papers)} papers for '{left}' <-> '{right}'")

            extraction_results = agent.extractor.extract_batch(papers, left, right)
            added += _add_results_to_graph(graph, extraction_results)

    except Exception as e:
        if verbose:
            print(f"      Error searching {left} <-> {right}: {e}")

    return added


def _add_results_to_graph(
    graph: GraphStore,
    extraction_results: List[Dict[str, Any]],
) -> int:
    """Add extraction results to graph, returning count of new triples."""
    added = 0

    for result in extraction_results:
        if not isinstance(result, dict):
            continue

        triples = result.get("triples", [])
        for triple in triples:
            edge = graph.add_triple(triple)
            if edge is not None:
                added += 1

    return added
