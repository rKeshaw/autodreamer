"""
Dimension 1 — Test 2: Duplicate Rate
======================================
Ingests the benchmark corpus TWICE, then runs consolidation.
Measures how many near-duplicate nodes survive after consolidation.

Pass criterion: duplicate rate < 5% of total nodes post-consolidation.

A duplicate pair is defined as two nodes with cosine similarity >= 0.80
(the MERGE_NODE threshold from config.py).

Usage:
    python test_d1_duplicate_rate.py \
        --judge-model <ollama-model-name> \
        --out results/d1_duplicate_rate.json
"""

import os
import sys
import json
import time
import argparse
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

DUPLICATE_JUDGMENT_PROMPT = """You are evaluating whether two knowledge graph nodes are
near-duplicates — i.e., they express the same core idea even if worded differently.

Node A:
"{node_a}"

Node B:
"{node_b}"

Cosine similarity (embedding distance): {similarity:.3f}

Are these expressing the SAME conceptual idea?
- YES if: one is a paraphrase, elaboration, or minor rewording of the other
- NO if: they make genuinely different claims, even if related

Respond with JSON:
{{
  "are_duplicates": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence"
}}
Respond ONLY with JSON.
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
    """Ingest all 5 articles once; return all new node IDs."""
    from graph.brain import EdgeSource
    all_ids = []
    for article in CORPUS:
        print(f"  Ingesting: {article['title']}...")
        text = fetch_wikipedia(article["title"])
        if not text:
            print(f"    WARNING: empty text")
            continue
        ids = ingestor.ingest(text, source=EdgeSource.READING) or []
        all_ids.extend(ids)
        time.sleep(1)
    return all_ids


def compute_duplicate_pairs(emb_index, threshold: float = 0.80):
    """Use FAISS to find all pairs above the duplicate threshold."""
    pairs = emb_index.all_pairwise_above(threshold)
    return pairs


def judge_pairs_sample(pairs, brain, model: str, sample_size: int = 30):
    """
    Use LLM judge to validate a random sample of near-duplicate pairs.
    Returns: (true_positives, false_positives, judgments)
    """
    import random
    from llm_utils import llm_call, require_json

    sample = random.sample(pairs, min(sample_size, len(pairs)))
    judgments = []

    for i, (nid_a, nid_b, sim) in enumerate(sample):
        node_a = brain.get_node(nid_a)
        node_b = brain.get_node(nid_b)
        if not node_a or not node_b:
            continue

        print(f"  Judging pair [{i+1}/{len(sample)}] sim={sim:.3f}...")
        prompt = DUPLICATE_JUDGMENT_PROMPT.format(
            node_a=node_a["statement"],
            node_b=node_b["statement"],
            similarity=sim,
        )
        raw = llm_call(prompt, temperature=0.1, model=model, role="precise")
        from llm_utils import require_json
        result = require_json(raw, default={})

        judgments.append({
            "node_a_id": nid_a,
            "node_b_id": nid_b,
            "node_a_stmt": node_a["statement"],
            "node_b_stmt": node_b["statement"],
            "cosine_sim": round(sim, 4),
            "judgment": result,
        })
        time.sleep(0.3)

    true_positives  = sum(1 for j in judgments if j["judgment"].get("are_duplicates"))
    false_positives = sum(1 for j in judgments if not j["judgment"].get("are_duplicates"))
    return true_positives, false_positives, judgments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--out", default="results/d1_duplicate_rate.json")
    parser.add_argument("--duplicate-threshold", type=float, default=0.80)
    parser.add_argument("--judge-sample", type=int, default=30,
                        help="How many duplicate pairs to validate with LLM judge")
    parser.add_argument("--skip-consolidation", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from embedding import embed as shared_embed
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from consolidator.consolidator import Consolidator

    # ── Setup ──
    brain     = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer  = Observer(brain)
    ingestor  = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    # ── Ingest PASS 1 ──
    print("=" * 60)
    print("PASS 1: First ingestion of corpus")
    print("=" * 60)
    ids_pass1 = ingest_corpus(brain, ingestor)
    nodes_after_pass1 = len(brain.all_nodes())
    pairs_after_pass1 = compute_duplicate_pairs(emb_index, args.duplicate_threshold)

    print(f"\nAfter pass 1:")
    print(f"  Nodes: {nodes_after_pass1}")
    print(f"  Duplicate pairs (sim >= {args.duplicate_threshold}): {len(pairs_after_pass1)}")
    print(f"  Duplicate rate: {len(pairs_after_pass1) / max(nodes_after_pass1, 1):.2%}")

    # ── Ingest PASS 2 ──
    print("\n" + "=" * 60)
    print("PASS 2: Second ingestion of same corpus")
    print("=" * 60)
    ids_pass2 = ingest_corpus(brain, ingestor)
    nodes_after_pass2 = len(brain.all_nodes())
    pairs_after_pass2 = compute_duplicate_pairs(emb_index, args.duplicate_threshold)

    print(f"\nAfter pass 2:")
    print(f"  Nodes: {nodes_after_pass2}")
    print(f"  New nodes created (should be ~0): {nodes_after_pass2 - nodes_after_pass1}")
    print(f"  Duplicate pairs (sim >= {args.duplicate_threshold}): {len(pairs_after_pass2)}")
    print(f"  Duplicate rate: {len(pairs_after_pass2) / max(nodes_after_pass2, 1):.2%}")

    # ── Consolidation ──
    if not args.skip_consolidation:
        print("\n" + "=" * 60)
        print("PHASE 3: Running consolidation (merges duplicates)")
        print("=" * 60)
        consolidator = Consolidator(brain, observer=observer, embedding_index=emb_index)
        all_ids = list(set(ids_pass1 + ids_pass2))
        consolidator._merge_duplicates(
            type('Report', (), {
                'merges': 0, 'to_dict': lambda self: {}
            })()
        )
        # Use internal merge method directly for testing
        # Re-import and run properly
        from consolidator.consolidator import ConsolidationReport
        report = ConsolidationReport()
        consolidator._merge_duplicates(report)
        print(f"  Merges performed: {report.merges}")

    nodes_final = len(brain.all_nodes())
    pairs_final = compute_duplicate_pairs(emb_index, args.duplicate_threshold)
    final_dup_rate = len(pairs_final) / max(nodes_final, 1)

    # ── Judge a sample of remaining pairs ──
    print("\n" + "=" * 60)
    print(f"PHASE 4: LLM validation of {args.judge_sample} duplicate pairs")
    print("=" * 60)
    tp, fp, pair_judgments = judge_pairs_sample(
        pairs_final, brain, args.judge_model, args.judge_sample
    )
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # ── Similarity distribution of pairs ──
    sim_values = [p[2] for p in pairs_final]
    sim_buckets = {
        "0.80-0.85": sum(1 for s in sim_values if 0.80 <= s < 0.85),
        "0.85-0.90": sum(1 for s in sim_values if 0.85 <= s < 0.90),
        "0.90-0.95": sum(1 for s in sim_values if 0.90 <= s < 0.95),
        "0.95-1.00": sum(1 for s in sim_values if s >= 0.95),
    }

    report = {
        "test": "D1 — Duplicate Rate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "duplicate_threshold": args.duplicate_threshold,
            "judge_sample_size": args.judge_sample,
        },
        "summary": {
            "nodes_after_pass1": nodes_after_pass1,
            "nodes_after_pass2": nodes_after_pass2,
            "new_nodes_on_reingestion": nodes_after_pass2 - nodes_after_pass1,
            "nodes_final": nodes_final,
            "duplicate_pairs_after_pass1": len(pairs_after_pass1),
            "duplicate_pairs_after_pass2": len(pairs_after_pass2),
            "duplicate_pairs_final": len(pairs_final),
            "duplicate_rate_final": round(final_dup_rate, 4),
            "PASS": final_dup_rate < 0.05,
            "pass_threshold": 0.05,
            "llm_judge": {
                "pairs_validated": len(pair_judgments),
                "true_positives": tp,
                "false_positives": fp,
                "precision": round(precision, 3),
            },
            "similarity_distribution": sim_buckets,
        },
        "pair_judgments": pair_judgments,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D1: Duplicate Rate")
    print("=" * 60)
    print(f"Nodes after pass 1    : {nodes_after_pass1}")
    print(f"Nodes after pass 2    : {nodes_after_pass2}")
    print(f"New nodes on re-ingest: {nodes_after_pass2 - nodes_after_pass1} "
          f"(ideally 0)")
    print(f"Nodes after consolidat: {nodes_final}")
    print(f"Remaining dup pairs   : {len(pairs_final)}")
    print(f"Duplicate rate        : {final_dup_rate:.2%} (pass threshold: <5%)")
    print(f"LLM precision         : {precision:.2%} "
          f"({tp} true dups / {tp+fp} pairs validated)")
    print(f"Similarity distribution: {sim_buckets}")
    verdict = "PASS ✓" if report["summary"]["PASS"] else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
