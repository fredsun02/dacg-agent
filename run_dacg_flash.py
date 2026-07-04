#!/usr/bin/env python3
"""DACG full pipeline re-run with glm-4-flash for controlled comparison vs flash baselines.
Multi-step retrieval + two-layer stopping (KL convergence + PRM decline).
Same stack as B1/B2 baselines, only difference = adaptive drift-aware stopping.
Resumable. Reads GLM key from file (never inline)."""
import sys, os, json, time, io, contextlib
sys.path.insert(0, '.')
from agent.graph_agent import GraphKGSAAgent

OUT = "evaluation/redesign/dacg_flash.json"
KEY = open(os.environ['GLM_KEY_FILE']).read().strip()
BASE = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4-flash"
PRM = "models/prm_v3/graph_prm.pt"

test = json.load(open('trajectories/step2_v3_test/trajectories_20260305_053618.json'))
queries = [t for t in test if t['ground_truth'] in {'Beneficial','NoEffect'}]

done = {}
if os.path.exists(OUT):
    done = {r['query_id']: r for r in json.load(open(OUT))}
    print(f"[resume] {len(done)} done", flush=True)

agent = GraphKGSAAgent(
    extractor_api_key=KEY, extractor_api_base=BASE, extractor_model=MODEL,
    prm_model_path=PRM, batch_size=2, max_steps=20, mode="graph_prm", enable_multihop=True,
)

results = list(done.values())
for q in queries:
    qid = q['query_id']
    if qid in done: continue
    t0=time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = agent.search(head_entity=q['head'], tail_entity=q['tail'], verbose=False, max_search_time=300)
        d = res.to_dict() if hasattr(res,'to_dict') else res
        rec = {'query_id':qid,'head':q['head'],'tail':q['tail'],'ground_truth':q['ground_truth'],
               'prediction':d.get('conclusion'),'confidence':d.get('confidence'),
               'total_steps':d.get('total_steps'),'stop_reason':d.get('stop_reason'),
               'total_evidence':d.get('total_evidence')}
    except Exception as e:
        rec = {'query_id':qid,'head':q['head'],'tail':q['tail'],'ground_truth':q['ground_truth'],
               'prediction':'ERROR','error':str(e)[:200]}
    results.append(rec)
    json.dump(results, open(OUT,'w'), indent=2)
    print(f"[{len(results)}/140] {qid} GT={q['ground_truth']} -> {rec.get('prediction')} steps={rec.get('total_steps')} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(1)
print(f"[DONE] {len(results)} -> {OUT}", flush=True)
