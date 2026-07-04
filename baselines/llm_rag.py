#!/usr/bin/env python3
"""E2 LLM+RAG baseline: retrieve 10 PubMed abstracts then synthesize with LLM."""

import os, sys, json, time, argparse, requests, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.pubmed_client import PubMedClient

SYNTHESIS_PROMPT = """You are a biomedical expert. Based on the following PubMed abstracts, determine the causal relationship between the treatment/intervention and condition.

Treatment/Intervention: {head}
Condition/Outcome: {tail}

=== Retrieved Abstracts ===
{abstracts}
=== End of Abstracts ===

Based on the evidence above, is "{head}" beneficial, harmful, or has no effect for "{tail}"?

Answer with exactly one of: Beneficial, NoEffect, Harmful
Answer:"""


def call_llm(prompt, api_base, api_key, model, max_retries=3):
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 20}
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"  API {resp.status_code}: {resp.text[:100]}")
                time.sleep(2 ** attempt)
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
            for label in ["Beneficial", "NoEffect", "Harmful"]:
                if label.lower() in content.lower():
                    return label
            return content.split()[0] if content else "Unknown"
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2 ** attempt)
    return "Unknown"


def is_correct(pred, gt):
    if pred in ('NoEvidence', 'Uncertain', 'Unknown'):
        pred = 'Uncertain'
    if gt == 'NoEffect' and pred in ('NoEffect', 'Uncertain'):
        return True
    return pred == gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="Stage4/Task1A/benchmark_gold/splits_v3/test.json")
    p.add_argument("--tagged", default="Stage5_Agent/evaluation/redesign/tagged_queries.json")
    p.add_argument("--output", default="Stage5_Agent/evaluation/redesign/e2_rag.json")
    p.add_argument("--api-base", default="https://www.packyapi.com/v1")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-papers", type=int, default=10)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    api_key = os.getenv("PACKY_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set PACKY_API_KEY or LLM_API_KEY")

    with open(root / args.test) as f:
        test_queries = json.load(f)
    with open(root / args.tagged) as f:
        tagged = json.load(f)
    clean_ids = {q["id"] for q in tagged if q.get("in_clean_103")}

    pubmed = PubMedClient()

    results = []
    for i, q in enumerate(test_queries):
        head, tail = q["head_entity"], q["tail_entity"]
        # Step 1: PubMed search
        query_str = f"{head} {tail}"
        pmids = pubmed.search(query_str, max_results=args.max_papers)
        papers = pubmed.fetch_papers(pmids) if pmids else []

        # Step 2: Build abstracts context
        if papers:
            abs_texts = []
            for j, paper in enumerate(papers[:args.max_papers]):
                abs_texts.append(f"[{j+1}] {paper.title}\n{paper.abstract[:500]}")
            abstracts_str = "\n\n".join(abs_texts)
        else:
            abstracts_str = "(No abstracts retrieved)"

        # Step 3: LLM synthesis
        prompt = SYNTHESIS_PROMPT.format(head=head, tail=tail, abstracts=abstracts_str)
        pred = call_llm(prompt, args.api_base, api_key, args.model)
        gt = q["ground_truth"]
        correct = is_correct(pred, gt)

        results.append({"id": q["id"], "head": head, "tail": tail,
                        "ground_truth": gt, "prediction": pred, "correct": correct,
                        "n_papers": len(papers), "in_clean_103": q["id"] in clean_ids})
        print(f"[{i+1}/{len(test_queries)}] {head[:30]} ({len(papers)} papers) -> {pred} (gt={gt}, {'✓' if correct else '✗'})")
        time.sleep(0.5)

    # Summarize
    clean = [r for r in results if r["in_clean_103"]]
    full = results
    for label, subset in [("Clean-103", clean), ("Full-173", full)]:
        n = len(subset)
        acc = sum(r["correct"] for r in subset) / n * 100 if n else 0
        print(f"\n{label} (n={n}): acc={acc:.1f}%")
        for cls in ["Beneficial", "NoEffect", "Harmful"]:
            sub = [r for r in subset if r["ground_truth"] == cls]
            if sub:
                cls_acc = sum(r["correct"] for r in sub) / len(sub) * 100
                print(f"  {cls} (n={len(sub)}): {cls_acc:.1f}%")

    output = {"results": results,
              "clean_103": {"n": len(clean), "acc": round(sum(r["correct"] for r in clean) / len(clean) * 100, 1)},
              "full_173": {"n": len(full), "acc": round(sum(r["correct"] for r in full) / len(full) * 100, 1)}}
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
