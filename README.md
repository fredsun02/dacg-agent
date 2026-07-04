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

**What this repository provides:** the complete agent code, the Cochrane-derived
benchmark (queries, gold labels, and 656/140/173 splits), the trained PRM
weights, and the result tables reported in the paper
(`evaluation/redesign/test_only_table.json` for the stopping methods and fixed-k
baselines; `evaluation/redesign/baselines_on_140.json` for the E1/E2 external
baselines).

**On reproduction.** The reported metrics are produced by running the agent
end-to-end: it retrieves abstracts from PubMed and extracts causal triples with
an LLM. The evidence base is therefore a **live external corpus** -- exact
numbers depend on PubMed's contents at query time and on the extraction model,
so a bit-identical re-run is not expected from a clone alone. Full end-to-end
runs require `LLM_API_KEY` and `NCBI_API_KEY` and are launched via the top-level
`run_*.py` scripts.

**Offline inspection of the stopping policy.** So that the core mechanism can be
examined without API access, we ship recorded 20-step retrieval trajectories in
`trajectories/`. The self-contained `reproduce.py` replays these and
demonstrates the paper's central *qualitative* finding -- under fixed-depth
retrieval, NoEffect accuracy collapses and evidence drift rises monotonically as
the budget grows, while a KL-convergence stopping rule recovers NoEffect
accuracy and cuts retrieval steps:

```bash
python reproduce.py     # stdlib only; no install, no network, no API key
```

Note that `reproduce.py` runs on the shipped trajectories, which come from an
earlier extraction pass than the run behind the paper's final tables; it
reproduces the *direction and shape* of the drift effect, not the exact
per-configuration values in the paper.

Supporting scripts (`evaluation/run_ablation.py`, `scripts/mcnemar_test.py`,
`scripts/bootstrap_ci.py`, `scripts/train_graph_prm.py`) regenerate the
significance tests, confidence intervals, and PRM training; some contain
absolute paths from the original cluster and may need the path constants at the
top of the file adjusted to this checkout.

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
