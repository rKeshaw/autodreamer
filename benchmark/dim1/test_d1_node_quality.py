"""
Dimension 1 — Test 1: Node Quality
====================================
Ingests a fixed benchmark corpus (5 Wikipedia articles on known topics),
then evaluates every extracted node using an LLM-as-judge (70b class model).

Judge rubric (1-5 per node):
  5 — Rich, self-contained conceptual statement; preserves nuance/uncertainty
  4 — Clear conceptual statement; minor completeness issues
  3 — Captures the idea but reads like a summary rather than a thought
  2 — Keyword-like or too vague to stand alone
  1 — Meaningless, hallucinated, or pure noise

Pass criterion: ≥ 80% of nodes score 4 or above.

Usage:
    python test_d1_node_quality.py \
        --judge-model <ollama-model-name> \
        --out results/d1_node_quality.json
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Benchmark corpus ──────────────────────────────────────────────────────────

CORPUS = [
    {
        "id": "dna",
        "title": "DNA",
        "url": "https://en.wikipedia.org/wiki/DNA",
        "domain": "biology",
        "key_concepts": [
            "double helix", "base pairing", "replication",
            "transcription", "genetic code"
        ]
    },
    {
        "id": "thermodynamics",
        "title": "Thermodynamics",
        "url": "https://en.wikipedia.org/wiki/Thermodynamics",
        "domain": "physics",
        "key_concepts": [
            "entropy", "laws of thermodynamics", "heat", "work", "free energy"
        ]
    },
    {
        "id": "natural_selection",
        "title": "Natural Selection",
        "url": "https://en.wikipedia.org/wiki/Natural_selection",
        "domain": "evolutionary_biology",
        "key_concepts": [
            "variation", "fitness", "adaptation", "selection pressure",
            "heritability"
        ]
    },
    {
        "id": "neural_network",
        "title": "Artificial neural network",
        "url": "https://en.wikipedia.org/wiki/Artificial_neural_network",
        "domain": "computer_science",
        "key_concepts": [
            "weights", "activation function", "backpropagation",
            "gradient descent", "overfitting"
        ]
    },
    {
        "id": "game_theory",
        "title": "Game theory",
        "url": "https://en.wikipedia.org/wiki/Game_theory",
        "domain": "economics",
        "key_concepts": [
            "Nash equilibrium", "payoff matrix", "prisoner's dilemma",
            "dominant strategy", "zero-sum"
        ]
    },
]

# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating nodes extracted from a knowledge graph system.
Each node should be a rich, self-contained conceptual statement — NOT a keyword,
NOT a title, NOT a summary. It should read like a thought a scientist would have.

Source article domain: {domain}
Source article title: {title}

Node to evaluate:
"{statement}"

Score this node on a 1-5 scale:

5 — Rich, self-contained conceptual statement. Could be understood without the
    source article. Preserves nuance, uncertainty, or causal structure where
    relevant. Example: "REM sleep appears to loosen associative constraints,
    allowing ideas that were previously unrelated to form novel connections —
    this may explain why insights often occur upon waking."

4 — Clear conceptual statement. Captures the core idea. Minor completeness
    issues (e.g., drops a qualification, slightly generic). Still stands alone.

3 — Captures the idea but reads more like a summary sentence than an
    independent thought. Needs context to be fully meaningful.

2 — Too vague, keyword-like, or so generic it could apply to almost anything.
    Example: "DNA is important for life" — this is not a conceptual statement.

1 — Meaningless, hallucinated content not present in the source, pure noise,
    or a title/heading rather than an idea.

Also check:
- Is the node SELF-CONTAINED (can it stand alone without the source text)?
- Does it go beyond labeling to actually say something about HOW or WHY?
- If it mentions uncertainty or tension, does it preserve that nuance?

Respond with a JSON object:
{{
  "score": <integer 1-5>,
  "self_contained": <true or false>,
  "says_how_or_why": <true or false>,
  "preserves_nuance": <true or false>,
  "reasoning": "<one sentence explaining the score>",
  "improved_version": "<only if score <= 3: how this node could be improved>"
}}

Respond ONLY with JSON. No preamble.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def ingest_article(brain, ingestor, emb_index, article: dict) -> list:
    """Fetch and ingest one article; return list of (node_id, statement)."""
    print(f"  Fetching: {article['title']}...")
    text = fetch_wikipedia(article["title"])
    if not text:
        print(f"    WARNING: empty text for {article['title']}")
        return []

    nodes_before = set(nid for nid, _ in brain.all_nodes())
    from graph.brain import EdgeSource
    new_ids = ingestor.ingest(text, source=EdgeSource.READING) or []
    
    results = []
    for nid in new_ids:
        data = brain.get_node(nid)
        if data:
            results.append({
                "node_id": nid,
                "statement": data["statement"],
                "cluster": data.get("cluster", "unknown"),
                "node_type": data.get("node_type", "concept"),
                "article_id": article["id"],
                "article_title": article["title"],
                "domain": article["domain"],
            })
    print(f"    Extracted {len(results)} nodes")
    return results


def judge_node(node: dict, model: str) -> dict:
    """Call the judge LLM to score a single node."""
    from llm_utils import llm_call, require_json
    
    prompt = JUDGE_PROMPT.format(
        domain=node["domain"],
        title=node["article_title"],
        statement=node["statement"],
    )
    raw = llm_call(prompt, temperature=0.1, model=model, role="precise")
    result = require_json(raw, default={})
    
    if not result or "score" not in result:
        return {
            "score": 1,
            "self_contained": False,
            "says_how_or_why": False,
            "preserves_nuance": False,
            "reasoning": "Judge parse failed",
            "parse_error": True,
        }
    return result


def compute_coverage_score(nodes: list, article: dict) -> float:
    """
    Check what fraction of the article's key concepts appear
    in at least one node statement (simple substring match).
    """
    all_statements = " ".join(n["statement"].lower() for n in nodes)
    hits = sum(
        1 for concept in article["key_concepts"]
        if concept.lower() in all_statements
    )
    return hits / len(article["key_concepts"]) if article["key_concepts"] else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b",
                        help="Ollama model for judging (e.g. llama3.1:70b)")
    parser.add_argument("--out", default="results/d1_node_quality.json")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Load pre-ingested nodes from cache instead of re-ingesting")
    parser.add_argument("--cache", default="results/d1_node_cache.json")
    parser.add_argument("--max-nodes-per-article", type=int, default=50,
                        help="Cap nodes judged per article (cost control)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # ── Setup brain ──
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from embedding import embed as shared_embed
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer,
                        embedding_index=emb_index)

    # ── Ingest or load cache ──
    all_nodes = []

    if args.skip_ingest and os.path.exists(args.cache):
        print(f"Loading cached nodes from {args.cache}...")
        with open(args.cache) as f:
            all_nodes = json.load(f)
        print(f"Loaded {len(all_nodes)} cached nodes")
    else:
        print("=" * 60)
        print("PHASE 1: Ingesting benchmark corpus")
        print("=" * 60)
        for article in CORPUS:
            nodes = ingest_article(brain, ingestor, emb_index, article)
            all_nodes.extend(nodes)
            time.sleep(1)

        # save cache
        with open(args.cache, "w") as f:
            json.dump(all_nodes, f, indent=2)
        print(f"\nTotal nodes ingested: {len(all_nodes)}")
        print(f"Cache saved to: {args.cache}")

    # ── Judge nodes ──
    print("\n" + "=" * 60)
    print(f"PHASE 2: Judging nodes with model: {args.judge_model}")
    print("=" * 60)

    # Group by article and cap
    from itertools import groupby
    nodes_by_article = {}
    for n in all_nodes:
        aid = n["article_id"]
        nodes_by_article.setdefault(aid, []).append(n)

    nodes_to_judge = []
    for aid, nodes in nodes_by_article.items():
        # prefer longer, richer statements for judging
        sorted_nodes = sorted(nodes, key=lambda x: len(x["statement"]), reverse=True)
        nodes_to_judge.extend(sorted_nodes[:args.max_nodes_per_article])

    print(f"Judging {len(nodes_to_judge)} nodes "
          f"({args.max_nodes_per_article} max per article)...")

    judgments = []
    for i, node in enumerate(nodes_to_judge):
        print(f"  [{i+1}/{len(nodes_to_judge)}] {node['statement'][:60]}...")
        judgment = judge_node(node, args.judge_model)
        judgments.append({**node, "judgment": judgment})
        time.sleep(0.3)   # rate limit

    # ── Coverage scores ──
    print("\nPHASE 3: Computing coverage scores...")
    coverage_by_article = {}
    for article in CORPUS:
        article_nodes = [n for n in all_nodes if n["article_id"] == article["id"]]
        coverage_by_article[article["id"]] = {
            "coverage_score": compute_coverage_score(article_nodes, article),
            "key_concepts": article["key_concepts"],
            "node_count": len(article_nodes),
        }

    # ── Compute metrics ──
    print("\nPHASE 4: Computing metrics...")
    
    scores = [j["judgment"].get("score", 1) for j in judgments]
    parse_errors = sum(1 for j in judgments if j["judgment"].get("parse_error"))
    
    score_dist = {str(i): scores.count(i) for i in range(1, 6)}
    pct_4_plus = sum(1 for s in scores if s >= 4) / len(scores) if scores else 0
    pct_self_contained = sum(
        1 for j in judgments if j["judgment"].get("self_contained")
    ) / len(judgments) if judgments else 0
    pct_how_why = sum(
        1 for j in judgments if j["judgment"].get("says_how_or_why")
    ) / len(judgments) if judgments else 0
    pct_nuance = sum(
        1 for j in judgments if j["judgment"].get("preserves_nuance")
    ) / len(judgments) if judgments else 0

    # per-article breakdown
    per_article = {}
    for article in CORPUS:
        art_judgments = [j for j in judgments if j["article_id"] == article["id"]]
        if not art_judgments:
            continue
        art_scores = [j["judgment"].get("score", 1) for j in art_judgments]
        per_article[article["id"]] = {
            "title": article["title"],
            "domain": article["domain"],
            "node_count_ingested": coverage_by_article[article["id"]]["node_count"],
            "node_count_judged": len(art_judgments),
            "mean_score": round(statistics.mean(art_scores), 3),
            "median_score": statistics.median(art_scores),
            "pct_4_plus": round(
                sum(1 for s in art_scores if s >= 4) / len(art_scores), 3),
            "coverage_score": round(
                coverage_by_article[article["id"]]["coverage_score"], 3),
            "score_distribution": {str(i): art_scores.count(i) for i in range(1,6)},
        }

    # worst nodes (score <= 2) — useful for failure analysis
    worst_nodes = [
        {
            "article": j["article_title"],
            "statement": j["statement"],
            "score": j["judgment"].get("score"),
            "reasoning": j["judgment"].get("reasoning", ""),
            "improved_version": j["judgment"].get("improved_version", ""),
        }
        for j in judgments if j["judgment"].get("score", 3) <= 2
    ]
    worst_nodes.sort(key=lambda x: x["score"])

    # best nodes (score == 5) — for the report
    best_nodes = [
        {
            "article": j["article_title"],
            "statement": j["statement"],
            "score": j["judgment"].get("score"),
            "reasoning": j["judgment"].get("reasoning", ""),
        }
        for j in judgments if j["judgment"].get("score", 0) == 5
    ]

    # ── Build report ──
    report = {
        "test": "D1 — Node Quality",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "corpus_size": len(CORPUS),
            "total_nodes_ingested": len(all_nodes),
            "total_nodes_judged": len(nodes_to_judge),
            "max_nodes_per_article": args.max_nodes_per_article,
        },
        "summary": {
            "mean_score": round(statistics.mean(scores), 3) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "std_score": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0,
            "pct_score_4_plus": round(pct_4_plus, 3),
            "pct_self_contained": round(pct_self_contained, 3),
            "pct_says_how_or_why": round(pct_how_why, 3),
            "pct_preserves_nuance": round(pct_nuance, 3),
            "score_distribution": score_dist,
            "parse_errors": parse_errors,
            "PASS": pct_4_plus >= 0.80,
            "pass_threshold": 0.80,
        },
        "per_article": per_article,
        "worst_nodes": worst_nodes[:20],
        "best_nodes": best_nodes[:10],
        "raw_judgments": judgments,
    }

    # ── Save ──
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", 
                exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("RESULTS — D1: Node Quality")
    print("=" * 60)
    print(f"Total nodes judged  : {len(nodes_to_judge)}")
    print(f"Mean score          : {report['summary']['mean_score']}/5")
    print(f"Median score        : {report['summary']['median_score']}/5")
    print(f"Std deviation       : {report['summary']['std_score']}")
    print(f"% score >= 4        : {pct_4_plus:.1%}  (pass threshold: 80%)")
    print(f"% self-contained    : {pct_self_contained:.1%}")
    print(f"% says how/why      : {pct_how_why:.1%}")
    print(f"% preserves nuance  : {pct_nuance:.1%}")
    print(f"Score distribution  : {score_dist}")
    print(f"Parse errors        : {parse_errors}")
    print()
    print("Per-article breakdown:")
    for aid, data in per_article.items():
        status = "✓" if data["pct_4_plus"] >= 0.80 else "✗"
        print(f"  {status} {data['title']:30s}  "
              f"mean={data['mean_score']}  "
              f"4+={data['pct_4_plus']:.0%}  "
              f"cov={data['coverage_score']:.0%}")
    print()
    verdict = "PASS ✓" if report["summary"]["PASS"] else "FAIL ✗"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"\nFull report saved to: {args.out}")

    if worst_nodes:
        print(f"\nTop failure cases ({len(worst_nodes)} nodes scored ≤ 2):")
        for n in worst_nodes[:5]:
            print(f"  [{n['score']}] [{n['article']}] {n['statement'][:80]}...")
            print(f"      Reason: {n['reasoning']}")


if __name__ == "__main__":
    main()
