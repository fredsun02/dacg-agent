#!/usr/bin/env python3
"""Baseline 2: retrieval + cross-encoder reranking, single LLM judgment.
Per query: PubMed candidate pool -> BGE reranker top-k -> glm-4-flash one-shot verdict.
Resumable. LLM: glm-4-flash."""
import sys, os, json, time, re, requests
sys.path.insert(0, '.')
from agent.pubmed_client import PubMedClient
from sentence_transformers import CrossEncoder

OUT = "evaluation/redesign/baseline2_rerank_flash.json"
KEY = os.environ['GLM_KEY']
API = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4-flash"
POOL = 20   # candidate pool size
TOPK = 5    # rerank keep

test = json.load(open('trajectories/step2_v3_test/trajectories_20260305_053618.json'))
queries = [t for t in test if t['ground_truth'] in {'Beneficial','NoEffect'}]

done = {}
if os.path.exists(OUT):
    done = {r['query_id']: r for r in json.load(open(OUT))}
    print(f"[resume] {len(done)} done", flush=True)

print("[init] loading BGE reranker...", flush=True)
reranker = CrossEncoder('models/bge-reranker-base', max_length=512)
pubmed = PubMedClient()

VERDICT_PROMPT = """You are a biomedical evidence assessor. Given a claim about an intervention-outcome pair and the most relevant PubMed abstracts, classify the causal relationship.

Intervention: {head}
Outcome: {tail}

Most relevant abstracts:
{abstracts}

Classify the effect of the intervention on the outcome as exactly one of:
- Beneficial (intervention improves/treats the outcome)
- NoEffect (no significant effect; null result)
- Harmful (intervention worsens the outcome)

Reply with ONLY the single word label."""

def llm_verdict(head, tail, papers):
    abstracts = "\n\n".join(f"[{i+1}] {p.title}\n{p.abstract[:800]}" for i,p in enumerate(papers))
    prompt = VERDICT_PROMPT.format(head=head, tail=tail, abstracts=abstracts)
    for attempt in range(4):
        try:
            r = requests.post(API, headers={'Authorization':f'Bearer {KEY}','Content-Type':'application/json'},
                json={'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':100}, timeout=60)
            if r.status_code==200:
                txt = r.json()['choices'][0]['message']['content']
                txt = re.sub(r'<think>.*?</think>','',txt,flags=re.DOTALL)
                for lab in ['NoEffect','Beneficial','Harmful']:
                    if lab.lower() in txt.lower(): return lab
                return txt.strip()[:20]
            time.sleep(3*(attempt+1))
        except Exception as e:
            time.sleep(3)
    return 'ERROR'

results = list(done.values())
for q in queries:
    qid = q['query_id']
    if qid in done: continue
    t0 = time.time()
    try:
        pmids = pubmed.search(f"{q['head']} {q['tail']}", max_results=POOL)
        papers = pubmed.fetch_papers(pmids) if pmids else []
        papers = [p for p in papers if p.abstract]
        if not papers:
            rec = {'query_id':qid,'head':q['head'],'tail':q['tail'],'ground_truth':q['ground_truth'],
                   'prediction':'NoEvidence','n_candidates':0}
        else:
            pairs = [[f"{q['head']} {q['tail']}", p.abstract[:512]] for p in papers]
            scores = reranker.predict(pairs)
            ranked = [p for _,p in sorted(zip(scores,papers), key=lambda x:-x[0])][:TOPK]
            pred = llm_verdict(q['head'], q['tail'], ranked)
            rec = {'query_id':qid,'head':q['head'],'tail':q['tail'],'ground_truth':q['ground_truth'],
                   'prediction':pred,'n_candidates':len(papers),'n_reranked':len(ranked)}
    except Exception as e:
        rec = {'query_id':qid,'head':q['head'],'tail':q['tail'],'ground_truth':q['ground_truth'],
               'prediction':'ERROR','error':str(e)[:200]}
    results.append(rec)
    json.dump(results, open(OUT,'w'), indent=2)
    print(f"[{len(results)}/140] {qid} GT={q['ground_truth']} -> {rec.get('prediction')} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(1)

print(f"[DONE] {len(results)} -> {OUT}", flush=True)
