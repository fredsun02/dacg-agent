#!/usr/bin/env python3
"""Baseline 1: single-pass KG-agent (MedKGent paradigm) on the 140-query test set.
Same stack as DACG (GraphKGSAAgent) but max_steps=1, no adaptive stopping.
LLM: glm-4-flash. Resumable: writes after each query."""
import sys, os, json, time, io, contextlib
sys.path.insert(0, '.')
from agent.graph_agent import GraphKGSAAgent

OUT = "evaluation/redesign/baseline1_singlepass_flash.json"
KEY = os.environ['GLM_KEY']
BASE = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4-flash"

test = json.load(open('trajectories/step2_v3_test/trajectories_20260305_053618.json'))
queries = [t for t in test if t['ground_truth'] in {'Beneficial','NoEffect'}]

# resume
done = {}
if os.path.exists(OUT):
    done = {r['query_id']: r for r in json.load(open(OUT))}
    print(f"[resume] {len(done)} already done", flush=True)

agent = GraphKGSAAgent(
    extractor_api_key=KEY, extractor_api_base=BASE, extractor_model=MODEL,
    batch_size=8, max_steps=1, mode="graph_only", enable_multihop=False,
)

results = list(done.values())
for i, q in enumerate(queries):
    qid = q['query_id']
    if qid in done:
        continue
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # silence extractor logs
            res = agent.search(head_entity=q['head'], tail_entity=q['tail'],
                               verbose=False, max_search_time=180)
        d = res.to_dict() if hasattr(res,'to_dict') else res
        rec = {
            'query_id': qid, 'head': q['head'], 'tail': q['tail'],
            'ground_truth': q['ground_truth'],
            'prediction': d.get('conclusion'),
            'confidence': d.get('confidence'),
            'total_evidence': d.get('total_evidence'),
            'direct_paths': d.get('direct_paths'),
            'two_hop_paths': d.get('two_hop_paths'),
            'papers_searched': d.get('papers_searched'),
        }
    except Exception as e:
        rec = {'query_id': qid, 'head': q['head'], 'tail': q['tail'],
               'ground_truth': q['ground_truth'], 'prediction': 'ERROR', 'error': str(e)[:200]}
    results.append(rec)
    json.dump(results, open(OUT,'w'), indent=2)
    dt = time.time()-t0
    print(f"[{len(results)}/140] {qid} GT={q['ground_truth']} -> {rec.get('prediction')} ({dt:.0f}s)", flush=True)
    time.sleep(1)

print(f"[DONE] {len(results)} queries -> {OUT}", flush=True)
