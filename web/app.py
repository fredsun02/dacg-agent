#!/usr/bin/env python3
"""
KGSA Web Interface

Real-time knowledge graph visualization during search.
Uses Server-Sent Events (SSE) for streaming updates.
"""

import json
import sys
import time
import queue
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Generator, Optional

from flask import Flask, render_template, request, Response, jsonify

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.pubmed_client import PubMedClient
from agent.extractor import CausalExtractor
from agent.state import SearchState
from agent.prm import PRM

# Local KG (Neo4j) demo support
sys.path.insert(0, "/data/DRKG/KGSA/Stage4/Task3/scripts")
from neo4j import GraphDatabase
from simulate_search import get_relevant_evidences


app = Flask(__name__)

# Global configuration
CONFIG = {
    "prm_model_path": "/data/DRKG/KGSA/Stage4/Task8_data_selection/models/prm_mlp.pt",
    "batch_size": 2,
    "max_steps": 15,
    "min_steps": 3,
}

# Local KG settings (Neo4j-backed)
KG_CONFIG = {
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "kgsa2024",
    "batch_size": 3,
    "max_steps": 10,
    "max_evidences": 300,
    "path_top_k": 3,
    "path_max_hops": 4,
    "path_min_hops": 2,
    "path_allow_undirected": True,
    "edge_delay": 0.35,
    "step_warmup_delay": 0.8,
    "step_delay": 1.2
}

# LLM API settings (can be updated via frontend)
LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", ""),
    "api_base": "https://www.packyapi.com/v1",
    "model": "gpt-5-chat"
}


@dataclass
class GraphEdge:
    """Edge in the knowledge graph representing evidence from a paper"""
    source: str
    target: str
    relation_type: str
    confidence: float
    pmid: str
    title: str
    abstract: str
    authors: list
    journal: str
    pub_year: int
    evidence_text: str

    def to_dict(self):
        # Color based on relation type
        colors = {
            "Treat": "#28a745",      # Green - Beneficial
            "Inhibit": "#28a745",    # Green - Beneficial
            "Cause": "#dc3545",      # Red - Harmful
            "Stimulate": "#dc3545",  # Red - Harmful
            "NoEffect": "#6c757d",   # Gray - Neutral
        }
        color = colors.get(self.relation_type, "#848484")

        return {
            "from": self.source,
            "to": self.target,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "color": {"color": color, "highlight": color, "hover": color},
            "width": 1.0 + self.confidence * 1.5,  # Thinner edges, still confidence-scaled
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            # Paper metadata for popup
            "paper": {
                "pmid": self.pmid,
                "title": self.title,
                "abstract": self.abstract,
                "authors": self.authors,
                "journal": self.journal,
                "pub_year": self.pub_year,
                "evidence_text": self.evidence_text,
                "relation_type": self.relation_type,
                "confidence": self.confidence
            }
        }


class StreamingSearchAgent:
    """Agent that streams search progress via SSE"""

    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        self.pubmed = PubMedClient()

        # Use provided settings or fall back to global config
        self.extractor = CausalExtractor(
            api_key=api_key or LLM_CONFIG.get("api_key") or None,
            api_base=api_base or LLM_CONFIG.get("api_base") or "https://www.packyapi.com/v1",
            model=model or LLM_CONFIG.get("model") or "gpt-4o"
        )

        self.prm = PRM(
            model_path=CONFIG["prm_model_path"],
            min_steps=CONFIG["min_steps"],
            decline_threshold=0.3,
            convergence_threshold=0.1
        )

    def search_stream(
        self,
        head_entity: str,
        tail_entity: str
    ) -> Generator[dict, None, None]:
        """
        Stream search progress as events.

        Yields:
            Event dictionaries with type and data
        """
        # Reset PRM
        self.prm.reset()

        # Initialize state
        state = SearchState(
            head_entity=head_entity,
            tail_entity=tail_entity
        )

        # Create initial graph - only head and tail nodes
        nodes = [
            {"id": "head", "label": head_entity, "color": "#ff6b6b", "size": 30, "font": {"size": 16, "color": "#fff"}},
            {"id": "tail", "label": tail_entity, "color": "#4ecdc4", "size": 30, "font": {"size": 16, "color": "#fff"}},
        ]
        edges = []
        edge_count = 0

        # Send initial state
        yield {
            "type": "init",
            "data": {
                "nodes": nodes,
                "edges": [],
                "head": head_entity,
                "tail": tail_entity,
                "message": f"Searching: {head_entity} → {tail_entity}",
                "status": "searching"
            }
        }

        query = f"{head_entity} {tail_entity}"
        paper_count = 0

        # Search loop
        for step, papers in self.pubmed.search_and_fetch(
            query,
            batch_size=CONFIG["batch_size"],
            max_batches=CONFIG["max_steps"]
        ):
            if not papers:
                yield {
                    "type": "status",
                    "data": {"message": f"Step {step + 1}: No papers found"}
                }
                continue

            # Send step start
            yield {
                "type": "step_start",
                "data": {
                    "step": step + 1,
                    "papers_count": len(papers),
                    "message": f"Step {step + 1}: Processing {len(papers)} papers..."
                }
            }

            # Extract from all papers in this batch
            extraction_results = []

            for paper in papers:
                paper_count += 1

                yield {
                    "type": "status",
                    "data": {"message": f"Analyzing: {paper.title[:60]}..."}
                }

                # Extract causal relations
                try:
                    result = self.extractor.extract_from_paper(
                        paper, head_entity, tail_entity
                    )
                    extraction_results.append(result)

                    # Log extraction details
                    triples = result.get("triples", [])
                    if result.get("error"):
                        yield {
                            "type": "status",
                            "data": {"message": f"Extraction error: {result.get('error')}"}
                        }
                    elif not triples:
                        yield {
                            "type": "status",
                            "data": {"message": f"No causal relations in PMID:{paper.pmid}"}
                        }

                    # Add edges for extracted triples (with full paper metadata)
                    for triple in triples:
                        edge_count += 1
                        edge = GraphEdge(
                            source="head",
                            target="tail",
                            relation_type=triple.relation_type,
                            confidence=triple.confidence,
                            pmid=paper.pmid,
                            title=paper.title,
                            abstract=paper.abstract,
                            authors=paper.authors,
                            journal=paper.journal,
                            pub_year=paper.pub_year,
                            evidence_text=triple.evidence_text
                        )
                        edges.append(edge)

                        edge_data = edge.to_dict()
                        edge_data["id"] = f"edge_{edge_count}"

                        yield {
                            "type": "add_edge",
                            "data": {
                                "edge": edge_data,
                                "relation": triple.relation_type,
                                "confidence": triple.confidence,
                                "pmid": paper.pmid
                            }
                        }

                except Exception as e:
                    yield {
                        "type": "status",
                        "data": {"message": f"Error processing {paper.pmid}: {str(e)}"}
                    }
                    extraction_results.append({
                        "pmid": paper.pmid,
                        "triples": [],
                        "is_relevant": False,
                        "error": str(e)
                    })

                time.sleep(0.2)  # Small delay for visual effect

            # Update state with all extraction results
            state.update(extraction_results, CONFIG["batch_size"])

            # Get PRM decision
            features = state.get_current_features()
            should_stop, reason, reward = self.prm.should_stop(features)

            # Send step summary
            conclusion = state.get_conclusion()
            confidence = state.get_confidence()

            yield {
                "type": "step_end",
                "data": {
                    "step": step + 1,
                    "conclusion": conclusion,
                    "confidence": confidence,
                    "reward": reward,
                    "supporting": state.supporting_count,
                    "opposing": state.opposing_count,
                    "neutral": state.neutral_count,
                    "papers_total": state.papers_seen,
                    "should_stop": should_stop,
                    "stop_reason": reason
                }
            }

            # Check stopping conditions
            if should_stop:
                yield {
                    "type": "complete",
                    "data": {
                        "conclusion": conclusion,
                        "confidence": confidence,
                        "reason": reason,
                        "steps": step + 1,
                        "papers": state.papers_seen,
                        "evidence": {
                            "supporting": state.supporting_count,
                            "opposing": state.opposing_count,
                            "neutral": state.neutral_count
                        },
                        "message": f"Search complete: {conclusion} (confidence: {confidence:.0%})"
                    }
                }
                return

            # Also check confidence threshold
            if confidence >= 0.8:
                yield {
                    "type": "complete",
                    "data": {
                        "conclusion": conclusion,
                        "confidence": confidence,
                        "reason": "high_confidence",
                        "steps": step + 1,
                        "papers": state.papers_seen,
                        "evidence": {
                            "supporting": state.supporting_count,
                            "opposing": state.opposing_count,
                            "neutral": state.neutral_count
                        },
                        "message": f"Search complete: {conclusion} (confidence: {confidence:.0%})"
                    }
                }
                return

        # Max steps reached
        conclusion = state.get_conclusion()
        confidence = state.get_confidence()

        yield {
            "type": "complete",
            "data": {
                "conclusion": conclusion,
                "confidence": confidence,
                "reason": "max_steps",
                "steps": CONFIG["max_steps"],
                "papers": state.papers_seen,
                "evidence": {
                    "supporting": state.supporting_count,
                    "opposing": state.opposing_count,
                    "neutral": state.neutral_count
                },
                "message": f"Search complete (max steps): {conclusion} (confidence: {confidence:.0%})"
            }
        }


class StreamingKGSearchAgent:
    """Local KG search agent (Neo4j only, no LLM)."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            KG_CONFIG["neo4j_uri"],
            auth=(KG_CONFIG["neo4j_user"], KG_CONFIG["neo4j_password"])
        )

    def _get_top_k_paths(self, head_entity: str, tail_entity: str):
        """
        Return top-k shortest paths between head and tail by exact name match.
        Prefer multi-hop (>=2) paths; fall back to 1-hop if none found.
        """
        max_hops = int(KG_CONFIG["path_max_hops"])
        query_multi_all = f"""
        MATCH (h:Entity {{name:$head}}), (t:Entity {{name:$tail}})
        MATCH p = (h)-[:CAUSAL_RELATION*2..{max_hops}]->(t)
        RETURN p
        ORDER BY length(p) ASC
        LIMIT $k
        """
        query_any_all = f"""
        MATCH (h:Entity {{name:$head}}), (t:Entity {{name:$tail}})
        MATCH p = (h)-[:CAUSAL_RELATION*1..{max_hops}]->(t)
        RETURN p
        ORDER BY length(p) ASC
        LIMIT $k
        """
        query_multi_shortest_undir = f"""
        MATCH (h:Entity {{name:$head}}), (t:Entity {{name:$tail}})
        MATCH p = (h)-[:CAUSAL_RELATION*2..{max_hops}]-(t)
        RETURN p
        ORDER BY length(p) ASC
        LIMIT $k
        """
        query_multi_all_undir = f"""
        MATCH (h:Entity {{name:$head}}), (t:Entity {{name:$tail}})
        MATCH p = (h)-[:CAUSAL_RELATION*2..{max_hops}]-(t)
        RETURN p
        ORDER BY length(p) ASC
        LIMIT $k
        """
        with self.driver.session() as session:
            result = session.run(
                query_multi_all,
                head=head_entity,
                tail=tail_entity,
                k=KG_CONFIG["path_top_k"]
            )
            paths = [r["p"] for r in result]
            if paths:
                return paths, 2, "directed"

            if KG_CONFIG["path_allow_undirected"]:
                result = session.run(
                    query_multi_shortest_undir,
                    head=head_entity,
                    tail=tail_entity,
                    k=KG_CONFIG["path_top_k"]
                )
                paths = [r["p"] for r in result]
                if paths:
                    return paths, 2, "undirected"

            result = session.run(
                query_any_all,
                head=head_entity,
                tail=tail_entity,
                k=KG_CONFIG["path_top_k"]
            )
            paths = [r["p"] for r in result]
            if not paths:
                return [], 1, "directed"
            return paths, 1, "directed"

    def _path_nodes_edges(self, path, head_entity: str, tail_entity: str):
        nodes = []
        edges = []
        node_ids = []
        # Build nodes
        for node in path.nodes:
            entity_id = node.get("entity_id") or node.element_id
            name = node.get("name") or str(entity_id)
            node_id = f"kg_{entity_id}"
            if name.lower() == head_entity.lower():
                node_id = "head"
            elif name.lower() == tail_entity.lower():
                node_id = "tail"
            node_ids.append(node_id)
            nodes.append({
                "id": node_id,
                "label": name,
                "color": "#8d99ae",
                "size": 18,
                "font": {"size": 12, "color": "#fff"}
            })
        # Build edges
        for i, rel in enumerate(path.relationships):
            rel_type = rel.get("relation_type") or rel.type
            confidence = float(rel.get("aggregated_confidence") or 0.5)
            pmid = str(rel.get("earliest_pmid") or "KG_PATH")
            edges.append({
                "source": node_ids[i],
                "target": node_ids[i + 1],
                "relation_type": rel_type,
                "confidence": confidence,
                "pmid": pmid,
                "evidence_text": "KG path preview"
            })
        return nodes, edges

    def _direction_from_relation(self, rel_type: str) -> str:
        if rel_type in {"Treat", "Prevent", "Inhibit"}:
            return "beneficial"
        if rel_type in {"Cause", "Worsen", "Stimulate"}:
            return "harmful"
        if rel_type in {"NoEffect"}:
            return "neutral"
        return "indirect"

    def search_stream(
        self,
        head_entity: str,
        tail_entity: str,
        multi_hop: bool = True
    ) -> Generator[dict, None, None]:
        state = SearchState(head_entity=head_entity, tail_entity=tail_entity)

        nodes = [
            {"id": "head", "label": head_entity, "color": "#ff6b6b", "size": 30, "font": {"size": 16, "color": "#fff"}},
            {"id": "tail", "label": tail_entity, "color": "#4ecdc4", "size": 30, "font": {"size": 16, "color": "#fff"}},
        ]
        edge_count = 0

        yield {
            "type": "init",
            "data": {
                "nodes": nodes,
                "edges": [],
                "head": head_entity,
                "tail": tail_entity,
                "message": f"[KG] Searching: {head_entity} → {tail_entity}",
                "status": "searching"
            }
        }

        if multi_hop:
            # Path preview (top-k shortest)
            try:
                paths, min_hops, path_mode = self._get_top_k_paths(head_entity, tail_entity)
            except Exception as e:
                paths, min_hops, path_mode = [], KG_CONFIG["path_min_hops"], "directed"
                yield {
                    "type": "status",
                    "data": {"message": f"[KG] Path preview error: {e}"}
                }

            if paths:
                seen_nodes = set(["head", "tail"])
                path_nodes = []
                path_edges = []
                edge_count = 0

                for p in paths:
                    nodes_list, edges_list = self._path_nodes_edges(p, head_entity, tail_entity)
                    for n in nodes_list:
                        if n["id"] not in seen_nodes:
                            seen_nodes.add(n["id"])
                            path_nodes.append(n)
                    for e in edges_list:
                        edge_count += 1
                        edge = GraphEdge(
                            source=e["source"],
                            target=e["target"],
                            relation_type=e["relation_type"],
                            confidence=e["confidence"],
                            pmid=e["pmid"],
                            title="KG path",
                            abstract="",
                            authors=[],
                            journal="KG",
                            pub_year=0,
                            evidence_text=e["evidence_text"]
                        )
                        edge_data = edge.to_dict()
                        edge_data["id"] = f"path_{edge_count}"
                        edge_data["dashes"] = True
                        path_edges.append(edge_data)

                yield {
                    "type": "path_preview",
                    "data": {
                        "nodes": path_nodes,
                        "edges": path_edges,
                        "message": f"[KG] Path preview: {len(paths)} path(s), min_hops={min_hops}, mode={path_mode}"
                    }
                }
            else:
                yield {
                    "type": "status",
                    "data": {"message": "[KG] Path preview: none"}
                }
        else:
            yield {
                "type": "status",
                "data": {"message": "[KG] Path preview: disabled (1-hop mode)"}
            }

        yield {
            "type": "status",
            "data": {"message": "[KG] Fetching evidence from KG..."}
        }
        time.sleep(0.2)

        evidences = get_relevant_evidences(
            self.driver,
            head_mesh_id="",
            tail_mesh_id="",
            head_name=head_entity,
            tail_name=tail_entity,
            limit=KG_CONFIG["max_evidences"]
        )

        if not evidences:
            yield {
                "type": "complete",
                "data": {
                    "conclusion": "Uncertain",
                    "confidence": 0.0,
                    "reason": "no_evidence",
                    "steps": 0,
                    "papers": 0,
                    "evidence": {"supporting": 0, "opposing": 0, "neutral": 0},
                    "message": "KG search complete: no evidence found"
                }
            }
            return

        batch_size = KG_CONFIG["batch_size"]
        max_steps = KG_CONFIG["max_steps"]

        for step_idx in range(max_steps):
            start_idx = step_idx * batch_size
            end_idx = start_idx + batch_size
            batch = evidences[start_idx:end_idx]
            if not batch:
                break

            yield {
                "type": "step_start",
                "data": {
                    "step": step_idx + 1,
                    "papers_count": len(batch),
                    "message": f"[KG] Step {step_idx + 1}: {len(batch)} evidences"
                }
            }
            time.sleep(KG_CONFIG["step_warmup_delay"])

            extraction_results = []
            for ev in batch:
                rel_type = ev.get("relation_type", "Associated")
                confidence = float(ev.get("confidence", 0.5) or 0.5)
                direction = self._direction_from_relation(rel_type)
                pmid = str(ev.get("pmid") or f"KG_{step_idx}_{edge_count}")

                triple = {
                    "head_entity": head_entity,
                    "tail_entity": tail_entity,
                    "relation_type": rel_type,
                    "confidence": confidence,
                    "evidence_text": f"KG relation {rel_type} (evidence_count={ev.get('evidence_count')})",
                    "is_causal": rel_type not in {"Associated"},
                    "eval_direction": direction
                }
                extraction_results.append({
                    "pmid": pmid,
                    "pub_year": 0,
                    "triples": [triple],
                    "relevance_score": 1.0,
                    "is_relevant": True
                })

                edge_count += 1
                edge = GraphEdge(
                    source="head",
                    target="tail",
                    relation_type=rel_type,
                    confidence=confidence,
                    pmid=pmid,
                    title=f"KG evidence ({ev.get('evidence_count', 1)})",
                    abstract="",
                    authors=[],
                    journal="KG",
                    pub_year=0,
                    evidence_text=triple["evidence_text"]
                )
                edge_data = edge.to_dict()
                edge_data["id"] = f"edge_{edge_count}"

                yield {
                    "type": "add_edge",
                    "data": {
                        "edge": edge_data,
                        "relation": rel_type,
                        "confidence": confidence,
                        "pmid": pmid
                    }
                }
                time.sleep(KG_CONFIG["edge_delay"])

            state.update(extraction_results, batch_size)
            conclusion = state.get_conclusion()
            confidence = state.get_confidence()

            should_stop = (step_idx + 1) >= max_steps or end_idx >= len(evidences)
            stop_reason = "max_steps" if (step_idx + 1) >= max_steps else "evidence_exhausted"

            yield {
                "type": "step_end",
                "data": {
                    "step": step_idx + 1,
                    "conclusion": conclusion,
                    "confidence": confidence,
                    "reward": 0.0,
                    "supporting": state.supporting_count,
                    "opposing": state.opposing_count,
                    "neutral": state.neutral_count,
                    "papers_total": state.papers_seen,
                    "should_stop": should_stop,
                    "stop_reason": stop_reason
                }
            }
            time.sleep(KG_CONFIG["step_delay"])

            if should_stop:
                yield {
                    "type": "complete",
                    "data": {
                        "conclusion": conclusion,
                        "confidence": confidence,
                        "reason": stop_reason,
                        "steps": step_idx + 1,
                        "papers": state.papers_seen,
                        "evidence": {
                            "supporting": state.supporting_count,
                            "opposing": state.opposing_count,
                            "neutral": state.neutral_count
                        },
                        "message": f"KG search complete: {conclusion} (confidence: {confidence:.0%})"
                    }
                }
                return


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current LLM configuration (without exposing full API key)"""
    return jsonify({
        "api_key_set": bool(LLM_CONFIG.get("api_key")),
        "api_key_preview": LLM_CONFIG.get("api_key", "")[:8] + "..." if LLM_CONFIG.get("api_key") else "",
        "api_base": LLM_CONFIG.get("api_base", ""),
        "model": LLM_CONFIG.get("model", "")
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    """Update LLM configuration"""
    data = request.get_json()

    if data.get("api_key"):
        LLM_CONFIG["api_key"] = data["api_key"]
    if data.get("api_base"):
        LLM_CONFIG["api_base"] = data["api_base"]
    if data.get("model"):
        LLM_CONFIG["model"] = data["model"]

    return jsonify({
        "success": True,
        "message": "Configuration updated",
        "api_key_set": bool(LLM_CONFIG.get("api_key")),
        "api_base": LLM_CONFIG.get("api_base"),
        "model": LLM_CONFIG.get("model")
    })


@app.route('/search')
def search():
    """
    SSE endpoint for streaming search results.

    Query params:
        head: Head entity
        tail: Tail entity
        api_key: (optional) Override API key
        api_base: (optional) Override API base
        model: (optional) Override model
        multi_hop: (optional) "1"/"0" to enable KG path preview
    """
    head = request.args.get('head', '').strip()
    tail = request.args.get('tail', '').strip()

    mode = request.args.get('mode', 'pubmed').strip().lower()
    multi_hop_raw = request.args.get('multi_hop', '1').strip().lower()
    multi_hop = multi_hop_raw in {"1", "true", "yes", "on"}

    # Get optional API overrides from query params
    api_key = request.args.get('api_key', '').strip() or LLM_CONFIG.get("api_key")
    api_base = request.args.get('api_base', '').strip() or LLM_CONFIG.get("api_base")
    model = request.args.get('model', '').strip() or LLM_CONFIG.get("model")

    if not head or not tail:
        return jsonify({"error": "Missing head or tail entity"}), 400

    if mode != "kg" and not api_key:
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'API Key not set. Please configure in Settings.'}})}\n\n"
        return Response(error_gen(), mimetype='text/event-stream')

    def generate():
        try:
            if mode == "kg":
                search_agent = StreamingKGSearchAgent()
                for event in search_agent.search_stream(head, tail, multi_hop=multi_hop):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                search_agent = StreamingSearchAgent(
                    api_key=api_key,
                    api_base=api_base,
                    model=model
                )
                for event in search_agent.search_stream(head, tail):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/examples')
def examples():
    """Return example queries"""
    examples = [
        {"head": "aspirin", "tail": "cardiovascular disease", "expected": "Beneficial"},
        {"head": "hydroxychloroquine", "tail": "COVID-19", "expected": "NoEffect"},
        {"head": "metformin", "tail": "type 2 diabetes", "expected": "Beneficial"},
        {"head": "statins", "tail": "heart disease", "expected": "Beneficial"},
        {"head": "smoking", "tail": "lung cancer", "expected": "Harmful"},
        {"head": "obesity", "tail": "type 2 diabetes", "expected": "Harmful"},
        {"head": "levodopa", "tail": "Parkinson's disease", "expected": "Beneficial"},
        {"head": "SARS-CoV-2", "tail": "COVID-19", "expected": "Harmful"},
    ]
    return jsonify(examples)


if __name__ == '__main__':
    import socket

    # Get server IP
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "unknown"

    print("\n" + "="*50)
    print("  KGSA Web Interface")
    print("="*50)
    print(f"\n  Server IP: {local_ip}")
    print(f"  Hostname:  {hostname}")
    print("\n  Access URLs:")
    print(f"    - Local:    http://localhost:5000")
    print(f"    - Network:  http://{local_ip}:5000")
    print("\n  For remote access via SSH tunnel:")
    print(f"    ssh -L 5000:localhost:5000 user@{local_ip}")
    print("    Then open http://localhost:5000 in your browser")
    print("\n" + "="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
