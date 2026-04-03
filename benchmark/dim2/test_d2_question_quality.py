"""
Dimension 2 — Test 1: Question Quality
========================================
Measures the quality of questions generated during the dream cycle compared
to a naive baseline (random concept pairing).

Pass criterion: Dream questions score higher on average across specificity,
answerability, and novelty than baseline questions.

Usage:
    python benchmark/dim2/test_d2_question_quality.py \
        --judge-model <ollama-model-name> \
        --out benchmark/dim2/results/d2_question_quality.json
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPUS = [
    {"id": "dna",              "title": "DNA"},
    {"id": "thermodynamics",   "title": "Thermodynamics"},
    {"id": "natural_selection","title": "Natural selection"},
    {"id": "neural_network",   "title": "Artificial neural network"},
    {"id": "game_theory",      "title": "Game theory"},
]

BASELINE_PROMPT = """
Write a single research question that connects the following two ideas:
Idea 1: "{node_a}"
Idea 2: "{node_b}"

Respond ONLY with the question itself. Do not add any extra text or quotes.
"""

JUDGE_PROMPT = """
You are evaluating the quality of a scientific research question.

Question to evaluate:
"{question}"

Please score this question on a 1-5 scale for the following criteria:
1. Specificity (1=vague, 5=precise and well-defined)
2. Answerability (1=impossible to test or answer practically, 5=clear path to empirical testing or logical resolution)
3. Novelty/Interestingness (1=trivial lookup, 5=creative and unexpected connection)

Respond EXACTLY in this JSON format:
{{
  "specificity": <int 1-5>,
  "answerability": <int 1-5>,
  "novelty": <int 1-5>,
  "reasoning": "<1-2 sentence explanation>"
}}
"""

def fetch_wikipedia(title: str) -> str:
    import requests
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "titles": title,
        "prop": "extracts", "format": "json",
        "explaintext": 1, "exsectionformat": "plain",
    }
    resp = requests.get(api, params=params, timeout=20,
                        headers={"User-Agent": "AutoScientist-Benchmark/1.0"})
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")[:8000]
    return ""

def ingest_corpus(brain, ingestor):
    from graph.brain import EdgeSource
    for article in CORPUS:
        print(f"  Ingesting: {article['title']}...")
        text = fetch_wikipedia(article["title"])
        if text:
            ingestor.ingest(text, source=EdgeSource.READING)
            time.sleep(1)

def build_baseline_questions(nodes, model_name, count=15):
    from llm_utils import llm_call
    baseline_qs = []
    print(f"\nGenerating {count} baseline questions...")
    for _ in range(count):
        nid_a, data_a = random.choice(nodes)
        nid_b, data_b = random.choice(nodes)
        prompt = BASELINE_PROMPT.format(node_a=data_a['statement'], node_b=data_b['statement'])
        q = llm_call(prompt, temperature=0.7, model=model_name).strip()
        baseline_qs.append(q)
        time.sleep(0.5)
    return baseline_qs

def judge_questions(questions, model_name):
    from llm_utils import llm_call, require_json
    results = []
    print(f"Evaluating {len(questions)} questions with judge model '{model_name}'...")
    for i, q in enumerate(questions):
        prompt = JUDGE_PROMPT.format(question=q)
        raw = llm_call(prompt, temperature=0.1, model=model_name, role="precise")
        try:
            score = require_json(raw, default={"specificity": 0, "answerability": 0, "novelty": 0})
        except Exception:
             score = {"specificity": 0, "answerability": 0, "novelty": 0}
        
        results.append({
            "question": q,
            "scores": score
        })
        time.sleep(0.3)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--baseline-gen-model", default="llama3.1:8b")
    parser.add_argument("--out", default="benchmark/dim2/results/d2_question_quality.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from dreamer.dreamer import Dreamer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)

    print("=" * 60)
    print("PHASE 1: Ingestion")
    print("=" * 60)

    # Load shared brain and index if available to save time
    brain_path = "benchmark/dim2/shared/brain.json"
    index_path = "benchmark/dim2/shared/embedding_index"
    
    if os.path.exists(brain_path) and os.path.exists(index_path + ".json"):
        print("  Loading shared brain and index...")
        brain.load(brain_path)
        emb_index = EmbeddingIndex.load(index_path)
    else:
        print("  Shared brain not found. Ingesting from scratch...")
        observer = Observer(brain)
        ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)
        ingest_corpus(brain, ingestor)

    observer = Observer(brain)
    dreamer = Dreamer(brain, research_agenda=observer)
    
    nodes = brain.all_nodes()
    if len(nodes) < 10:
        print("Not enough nodes ingested, exiting.")
        return

    print("\n" + "=" * 60)
    print("PHASE 2: Generating Dream Questions")
    print("=" * 60)
    dream_qs = []
    # Run a few dream cycles to accumulate questions
    for i in range(3):
        print(f"  Running Dream Cycle {i+1}/3...")
        log = dreamer.dream(steps=15, run_nrem=False)
        dream_qs.extend(log.questions)
    
    # Take up to 15 questions
    dream_qs = random.sample(dream_qs, min(15, len(dream_qs))) if dream_qs else ["How do things work?"]
    
    print("\n" + "=" * 60)
    print("PHASE 3: Generating Baseline Questions")
    print("=" * 60)
    baseline_qs = build_baseline_questions(nodes, args.baseline_gen_model, count=len(dream_qs))

    print("\n" + "=" * 60)
    print("PHASE 4: Scoring Questions (LLM as Judge)")
    print("=" * 60)
    dream_results = judge_questions(dream_qs, args.judge_model)
    base_results = judge_questions(baseline_qs, args.judge_model)

    def avg(lst, key):
        valid = [r['scores'].get(key, 0) for r in lst if isinstance(r['scores'].get(key), (int, float))]
        return sum(valid) / len(valid) if valid else 0.0

    dream_spec = avg(dream_results, 'specificity')
    dream_ans  = avg(dream_results, 'answerability')
    dream_nov  = avg(dream_results, 'novelty')
    dream_total = dream_spec + dream_ans + dream_nov

    base_spec = avg(base_results, 'specificity')
    base_ans  = avg(base_results, 'answerability')
    base_nov  = avg(base_results, 'novelty')
    base_total = base_spec + base_ans + base_nov

    passed = dream_total > base_total

    report = {
        "test": "D2 — Question Quality",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
        },
        "summary": {
            "num_questions_evaluated_each": len(dream_qs),
            "dream_scores": {
                "specificity": round(dream_spec, 2),
                "answerability": round(dream_ans, 2),
                "novelty": round(dream_nov, 2),
                "total": round(dream_total, 2)
            },
            "baseline_scores": {
                "specificity": round(base_spec, 2),
                "answerability": round(base_ans, 2),
                "novelty": round(base_nov, 2),
                "total": round(base_total, 2)
            },
            "PASS": passed
        },
        "dream_evaluations": dream_results,
        "baseline_evaluations": base_results
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D2: Question Quality")
    print("=" * 60)
    print(f"Dream    | Spec: {dream_spec:.2f} | Ans: {dream_ans:.2f} | Nov: {dream_nov:.2f} | Tot: {dream_total:.2f}")
    print(f"Baseline | Spec: {base_spec:.2f} | Ans: {base_ans:.2f} | Nov: {base_nov:.2f} | Tot: {base_total:.2f}")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")

if __name__ == "__main__":
    main()
