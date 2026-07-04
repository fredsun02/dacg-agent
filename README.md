# DACG-agent: Drift-Aware Causal-Graph Agent

Code, benchmark, and trained model weights for **"When More Evidence Hurts:
When More Evidence Hurts: Publication-Bias Drift and Principled Stopping for Biomedical Causal Search"** (DACG-agent).

The system incrementally builds a causal knowledge graph from PubMed abstracts
and applies a two-layer stopping policy — a KL-divergence posterior-convergence
monitor and a Bradley–Terry process reward model (PRM) — to halt retrieval
before *evidence drift* (the tendency of a bias-agnostic aggregator to
increasingly misclassify null-effect queries as *Beneficial* as retrieval
deepens) overwhelms the early signal.

## Repository layout

| Path | Contents |
|------|----------|
| `agent/` | Core agent: graph construction, entity resolver, extractor, path inference (`path_inference.py`: Boltzmann exponent β=1.5, per-hop discount ρ=1.0), noisy-OR confidence, PRM |
| `baselines/` | LLM zero-shot (E1) and LLM+RAG (E2) baselines |
| `run_baseline1_singlepass.py` | Single-pass KG-agent baseline (E3): the prevailing external paradigm of LLM-driven medical KG agents (single-pass, no query-time stopping), implemented on a controlled backbone identical to DACG |
| `run_baseline2_rerank.py` | Retrieval-with-reranking baseline (BGE cross-encoder) |
| `run_dacg_flash.py` | Cross-LLM generalization run (glm-4-flash) |
| `evaluation/` | Evaluation harness; recorded flash-stack prediction files, aggregate baseline results (`baselines_on_140.json`), McNemar and bootstrap-CI outputs, and the offline `reproduce.py` inputs |
| `benchmark/` | Cochrane-derived benchmark: `train.json`, `test.json`, `candidates.json`, and `benchmark_v3.json` with gold labels and 656/140/173 splits |
| `trajectories/` | Recorded 20-step retrieval trajectories (offline replay of the stopping policy via `reproduce.py`; illustrative, not the paper's final extraction run) |
| `models/prm_v3/`, `models/prm_v2_clean/` | Trained PRM weights (`graph_prm.pt`) + training reports |
| `config/config.yaml` | Runtime config (API key read from `$LLM_API_KEY`) |

## Setup

```bash
pip install -r requirements.txt
export LLM_API_KEY=...        # your LLM proxy/API key (never commit this)
export NCBI_API_KEY=...       # optional, for live PubMed retrieval
```

The reranking baseline uses `BAAI/bge-reranker-base`, which is **not** bundled
(1.1 GB third-party model). It is downloaded automatically by
`sentence-transformers` on first use, or fetch it manually:

```bash
huggingface-cli download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
```

## Results and reproduction

This repository is released to make the work reproducible. It provides the
**complete agent code**, the **Cochrane-derived benchmark** (queries, gold
labels, and the 656/140/173 train/validation/test splits), the **trained PRM
weights**, the **recorded 20-step retrieval trajectories**, and the **result
tables** reported in the paper (`evaluation/redesign/test_only_table.json` for
the stopping methods and fixed-k baselines; `evaluation/redesign/baselines_on_140.json`
for the external baselines).

**Inspect and replay the stopping policy offline.** The shipped 20-step
trajectories let the stopping mechanism be replayed with a self-contained script
that needs no install, no network, and no API key:

```bash
python reproduce.py
```

This reproduces the central finding on the released trajectories -- NoEffect
accuracy collapses and drift rises as the fixed budget grows, while
KL-convergence stopping recovers accuracy and cuts steps. (These trajectories
are from an earlier extraction pass than the paper's final tables, so it
reproduces the direction and shape of the effect rather than the exact
per-configuration values.)

**Re-run the full pipeline.** The agent builds its causal graph from PubMed
abstracts and extracts triples with an LLM, so the complete end-to-end pipeline
is launched via the top-level `run_*.py` scripts with `LLM_API_KEY` and
`NCBI_API_KEY` set.

Supporting scripts (`evaluation/run_ablation.py`, `scripts/mcnemar_test.py`,
`scripts/bootstrap_ci.py`, `scripts/train_graph_prm.py`) regenerate the
significance tests, confidence intervals, and PRM training.

## Data provenance and caveats

- The benchmark is derived from Cochrane systematic reviews. Cochrane consensus
  labels are themselves meta-analysis products and can carry residual
  publication bias; we treat them as a strong but imperfect gold standard (see
  the paper's Benchmark section).
- Retrieved evidence spans PubMed abstracts published 2010–2025.
- The *Harmful* class is excluded from the main evaluation (abstract-level
  oracle accuracy ≈28%).

## License

MIT (see `LICENSE`). The bundled `benchmark/` data derives from PubMed abstract
metadata and Cochrane-review-derived labels; downstream use should respect the
original sources' terms.
