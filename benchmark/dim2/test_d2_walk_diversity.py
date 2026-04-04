"""
Dimension 2 — Test 4: Walk Diversity
====================================
Measures the node coverage and revisit rate of the dreamer over multiple 
cycles on a static graph. Ensures the agent naturally explores instead of 
looping endlessly in attractor states.

Pass criterion: Coverage > 40% of graph after 10 cycles, Mean Re-visit Rate < 3.0.

Usage:
    python benchmark/dim2/test_d2_walk_diversity.py \
        --out benchmark/dim2/results/d2_walk_diversity.json
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
    parser.add_argument("--out", default="benchmark/dim2/results/d2_walk_diversity.json")
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from dreamer.dreamer import Dreamer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)
    dreamer = Dreamer(brain, research_agenda=observer)

    from graph.brain import EdgeSource
    print("=" * 60)
    print("PHASE 1: Ingestion (or Graph Loading)")
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
        for article in CORPUS:
            print(f"  Ingesting: {article['title']}...")
            text = fetch_wikipedia(article["title"])
            if text:
                ingestor.ingest(text, source=EdgeSource.READING)
                time.sleep(1)

    total_nodes = len(brain.all_nodes())
    if total_nodes == 0:
        print("No nodes ingested, aborting.")
        return

    print("\n" + "=" * 60)
    print(f"PHASE 2: Running {args.cycles} Dream Cycles")
    print("=" * 60)

    # Track visits
    # visit_counts maps node_id to the number of times it was the target of a dream step
    from collections import defaultdict
    visit_counts = defaultdict(int)
    total_steps = 0
    shared_visited = set()

    for i in range(args.cycles):
        print(f"  Running Walking Dream Cycle {i+1}/{args.cycles}...")
        log = dreamer.dream(steps=8, run_nrem=False, visited_set=shared_visited)
        for step in log.steps:
            if step.to_id:
                visit_counts[step.to_id] += 1
                total_steps += 1

    visited_unique = len(visit_counts)
    coverage = visited_unique / total_nodes
    
    # Calculate mean revisit rate for nodes that were visited AT LEAST once
    if visited_unique > 0:
        mean_revisit_rate = sum(visit_counts.values()) / visited_unique
    else:
        mean_revisit_rate = 0.0

    visited_more_than_once = sum(1 for v in visit_counts.values() if v > 1)

    passed_coverage = coverage > 0.40
    passed_revisit = mean_revisit_rate < 3.0
    passed = passed_coverage and passed_revisit

    report = {
        "test": "D2 — Walk Diversity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "cycles": args.cycles,
            "total_nodes_available": total_nodes
        },
        "summary": {
            "total_nodes": total_nodes,
            "total_steps_taken": total_steps,
            "unique_nodes_visited": visited_unique,
            "coverage_fraction": round(coverage, 3),
            "mean_revisit_rate": round(mean_revisit_rate, 3),
            "visited_more_than_once": visited_more_than_once,
            "PASS": passed
        },
        "visits": dict(visit_counts)
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D2: Walk Diversity")
    print("=" * 60)
    print(f"Graph Size            : {total_nodes} nodes")
    print(f"Dream Steps           : {total_steps} steps over {args.cycles} cycles")
    print(f"Nodes visited at least once: {visited_unique}")
    print(f"Coverage Fraction     : {coverage:.2%} (pass > 40%)")
    print(f"Mean Re-visit Rate    : {mean_revisit_rate:.2f} (pass < 3.00)")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")

if __name__ == "__main__":
    main()
