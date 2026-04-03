"""
Dimension 2 — Test 3: Mission Advance Precision
===============================================
Tests whether the mission advance detector (0.5-strength threshold) is well
calibrated. It compares the system's assigned "strength" against an LLM-as-judge
rating of actual relevance to the mission.

Pass criterion: The Pearson correlation between system strength and
judge ratings > 0.60, and correctly flagged advance rate > 75%.

Usage:
    python benchmark/dim2/test_d2_mission_advance.py \
        --judge-model <ollama-model-name> \
        --out benchmark/dim2/results/d2_mission_advance.json
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPUS = [
    {"id": "genetics",         "title": "Genetics"},
    {"id": "epigenetics",      "title": "Epigenetics"},
    {"id": "mutation",         "title": "Mutation"},
    {"id": "thermodynamics",   "title": "Thermodynamics"},
    {"id": "game_theory",      "title": "Game theory"},
]

MISSION = "How do environmental factors influence heritable genetic traits?"

JUDGE_PROMPT = """
You are a domain expert evaluating the relevance of a newly discovered idea to a specific research mission.

Mission Question: "{mission}"

Discovered Idea/Connection:
"{node}"
"{narration}"

Rate the relevance of this idea to advancing the mission on a 0.0 to 1.0 continuous scale.
- 0.1-0.3: Tangential — Relates to the same topic but does not inform the question.
- 0.4-0.6: Relevant context — Provides useful background but is NOT an advance.
- 0.7-0.85: Advancing — ONLY assign if it provides a direct missing piece or evidence to resolve the mission.
- 0.9-1.0: Breakthrough

IMPORTANT: Do NOT assign a rating >= 0.5 unless the idea provides *direct* evidence or a *missing piece* to answer the mission. Shared topic alone must be < 0.5.

Respond EXACTLY in this JSON format:
{{
  "rating": <float 0.0-1.0>,
  "reasoning": "<1 sentence explanation>"
}}
"""

def fetch_wikipedia(title: str) -> str:
    import requests
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "titles": title,
        "prop": "extracts", "format": "json",
    }
    resp = requests.get(api, params=params, timeout=20,
                        headers={"User-Agent": "AutoScientist-Benchmark/2.0"})
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")[:8000]
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--out", default="benchmark/dim2/results/d2_mission_advance.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from dreamer.dreamer import Dreamer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)

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
        from graph.brain import EdgeSource
        for article in CORPUS:
            print(f"  Ingesting: {article['title']}...")
            text = fetch_wikipedia(article["title"])
            if text:
                ingestor.ingest(text, source=EdgeSource.READING)
                time.sleep(1)

    observer = Observer(brain)
    dreamer = Dreamer(brain, research_agenda=observer)

    print("\n" + "=" * 60)
    print("PHASE 2: Setting Mission & Dreaming")
    print("=" * 60)
    
    brain.set_mission(MISSION)
    
    samples = []
    
    for i in range(3):
        print(f"  Running Focused Dream Cycle {i+1}/3...")
        # using run_nrem=False to speed up for benchmarks
        log = dreamer.dream(steps=20, run_nrem=False)
        
        # We need both flagged and non-flagged steps
        for step in log.steps:
            if step.to_id:
                node = brain.get_node(step.to_id)
                if node:
                     adv = step.mission_advance
                     sys_strength = step.mission_strength
                     samples.append({
                         "node_statement": node['statement'],
                         "narration": step.narration,
                         "flagged_advance": adv,
                         "system_strength": sys_strength
                     })

    # Balanced sampling of flagged and unflagged if possible
    flagged = [s for s in samples if s['flagged_advance']]
    unflagged = [s for s in samples if not s['flagged_advance']]
    
    # take up to 10 of each
    test_set = random.sample(flagged, min(10, len(flagged))) + random.sample(unflagged, min(10, len(unflagged)))

    print("\n" + "=" * 60)
    print(f"PHASE 3: Evaluating {len(test_set)} Dream Steps")
    print("=" * 60)

    from llm_utils import llm_call, require_json
    results = []
    sys_scores = []
    judge_scores = []
    correct_flags = 0

    for i, s in enumerate(test_set):
        prompt = JUDGE_PROMPT.format(
            mission=MISSION,
            node=s['node_statement'],
            narration=s['narration']
        )
        raw = llm_call(prompt, temperature=0.1, model=args.judge_model, role="precise")
        try:
             res = require_json(raw, default={"rating": 0.0})
             rating = float(res.get("rating", 0.0))
        except:
             rating = 0.0
             res = {"rating": 0.0}
        
        sys_s = float(s['system_strength'])
        sys_scores.append(sys_s)
        judge_scores.append(rating)
        
        # Check if the binary flagging was correct (using 0.5 threshold)
        sys_flag = sys_s >= 0.5
        judge_flag = rating >= 0.5
        if sys_flag == judge_flag:
            correct_flags += 1

        results.append({
            "step": s,
            "judge_rating": rating,
            "judge_reasoning": res.get("reasoning", "")
        })
        time.sleep(0.3)

    # Calculate Pearson correlation
    correlation = 0.0
    if len(sys_scores) > 1 and np.std(sys_scores) > 0 and np.std(judge_scores) > 0:
        correlation = np.corrcoef(sys_scores, judge_scores)[0, 1]

    flag_acc = correct_flags / max(len(test_set), 1)
    passed = correlation > 0.60 and flag_acc > 0.75

    report = {
        "test": "D2 — Mission Advance Precision",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
        },
        "summary": {
            "samples_evaluated": len(test_set),
            "correlation": round(float(correlation), 3),
            "flagging_accuracy": round(flag_acc, 3),
            "PASS": bool(passed)
        },
        "evaluations": results
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D2: Mission Advance Precision")
    print("=" * 60)
    print(f"Total steps evaluated                : {len(test_set)}")
    print(f"Pearson Correlation (System vs Judge): {correlation:.3f} (pass > 0.60)")
    print(f"Binary Flagging Accuracy             : {flag_acc:.2%} (pass > 75%)")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")

if __name__ == "__main__":
    main()
