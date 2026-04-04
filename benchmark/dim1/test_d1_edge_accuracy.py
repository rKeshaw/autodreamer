"""
Dimension 1 — Test 3: Edge Accuracy

NOTE on 'associated' edge type:
  'associated' edges are NOT produced by the LLM extraction pipeline.
  They are added automatically by _add_weak_edges() via FAISS similarity.
  Testing LLM extraction for 'associated' is testing the wrong code path.
  The three 'associated' ground-truth pairs below test whether the LLM
  correctly returns related=false (no explicit relationship), which is
  the correct system behavior for weakly-related pairs.
  The test maps related=false → 'unrelated', and checks it does NOT
  assign a strong type. This replaces the old 'associated' type check.
======================================
Uses a hand-curated set of 20 concept pairs with KNOWN ground-truth
relationship types. Runs each pair through the ingestor's edge extraction
pipeline, then compares extracted edge type to ground truth.

Metrics:
  - Overall accuracy (correct type / total pairs)
  - Per-type precision and recall
  - Confusion matrix
  - LLM judge validation of edge narrations

Pass criterion:
  - Overall edge type accuracy >= 70%  (unchanged)
  - No SEMANTIC edge type (supports/causes/contradicts/analogy) has recall < 50%
  - "unrelated" blind spot is acceptable (system should never output this label)

Usage:
    python test_d1_edge_accuracy.py \
        --judge-model <ollama-model-name> \
        --out results/d1_edge_accuracy.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Ground-truth test pairs ───────────────────────────────────────────────────
# Format: (node_a, node_b, ground_truth_type, analogy_depth_if_applicable)
# edge types: supports | causes | contradicts | analogy | unrelated

GROUND_TRUTH_PAIRS = [
    # --- SUPPORTS ---
    (
        "Natural selection acts on heritable variation in populations, "
        "favoring traits that increase reproductive success.",
        "Populations that experience strong selection pressure over many "
        "generations accumulate adaptive traits that improve survival.",
        "supports", None
    ),
    (
        "Entropy in a closed system tends to increase over time, "
        "moving toward states of greater disorder.",
        "Irreversible processes in thermodynamics consistently move toward "
        "higher entropy states, making time's arrow thermodynamically asymmetric.",
        "supports", None
    ),
    (
        "DNA base pairing follows strict complementarity rules: "
        "adenine pairs with thymine, cytosine pairs with guanine.",
        "The double helix structure of DNA is stabilized by hydrogen bonds "
        "between complementary base pairs on opposite strands.",
        "supports", None
    ),
    (
        "Gradient descent updates model weights by moving in the direction "
        "of steepest decrease of the loss function.",
        "Backpropagation computes the gradient of the loss with respect to "
        "each weight, enabling gradient descent to train deep networks.",
        "supports", None
    ),

    # --- CAUSES ---
    (
        "Increasing the temperature of a gas increases the average kinetic "
        "energy of its molecules.",
        "Higher average kinetic energy in a gas leads to more frequent and "
        "forceful collisions with container walls, raising pressure.",
        "causes", None
    ),
    (
        "A dominant strategy in a game yields a higher payoff regardless "
        "of what other players do.",
        "When all players follow their dominant strategies, the resulting "
        "outcome is a Nash equilibrium from which no player wishes to deviate.",
        "causes", None
    ),
    (
        "Mutations in DNA replication introduce heritable variation into "
        "a population.",
        "Heritable variation provides the raw material on which natural "
        "selection can act, enabling evolutionary change over generations.",
        "causes", None
    ),

    # --- CONTRADICTS ---
    (
        "Neural networks with more layers always achieve better performance "
        "on any given task.",
        "Deeper neural networks often suffer from vanishing gradients and "
        "overfitting, and frequently underperform shallower architectures "
        "on small datasets.",
        "contradicts", None
    ),
    (
        "In a perfectly competitive market, the Nash equilibrium is always "
        "the socially optimal outcome.",
        "The prisoner's dilemma demonstrates that Nash equilibria can be "
        "socially suboptimal — rational individual behavior produces outcomes "
        "worse for all participants than cooperative alternatives.",
        "contradicts", None
    ),
    (
        "Lamarckian inheritance holds that organisms can pass on traits "
        "acquired during their lifetime to their offspring.",
        "Modern evolutionary biology, supported by genetics, shows that "
        "only heritable genetic variations — not acquired characteristics — "
        "are passed to offspring.",
        "contradicts", None
    ),

    # --- ANALOGY: surface ---
    (
        "The internet routes information packets through nodes in a network, "
        "with each node forwarding packets toward their destination.",
        "Neurons in the brain transmit signals through synaptic connections, "
        "with signal strength determined by connection weights.",
        "analogy", "surface"
    ),
    (
        "Nash equilibrium in game theory describes a stable state where "
        "no player benefits from unilaterally changing strategy.",
        "Fixed points in dynamical systems describe stable states where "
        "a system no longer changes under its own dynamics.",
        "analogy", "surface"
    ),

    # --- ANALOGY: structural ---
    (
        "Natural selection filters heritable variation in biological "
        "populations, preserving traits that increase fitness and eliminating "
        "those that reduce it.",
        "Gradient descent filters parameter configurations in a neural "
        "network, preserving weight updates that reduce loss and discarding "
        "those that increase it.",
        "analogy", "structural"
    ),
    (
        "DNA transcription reads a genetic sequence and produces an RNA "
        "copy, which is then translated into a protein.",
        "A compiler reads source code, produces an intermediate representation, "
        "which is then compiled into machine instructions.",
        "analogy", "structural"
    ),
    (
        "The prisoner's dilemma shows that individually rational choices "
        "can lead to collectively suboptimal outcomes.",
        "Tragedy of the commons demonstrates that individually rational "
        "resource exploitation leads to collective depletion and loss for all.",
        "analogy", "structural"
    ),

    # --- ANALOGY: isomorphism ---
    (
        "Heat diffusion through a medium is governed by the heat equation: "
        "the rate of change of temperature is proportional to the Laplacian "
        "of the temperature field.",
        "Information diffusion through a network follows a diffusion equation "
        "structurally identical to the heat equation, with information "
        "concentration replacing temperature.",
        "analogy", "isomorphism"
    ),
    (
        "Entropy in thermodynamics measures the number of microstates "
        "corresponding to a macrostate: S = k ln(W).",
        "Shannon entropy in information theory measures the expected "
        "information content of a probability distribution: H = -Σ p log p. "
        "Both formulas are formally equivalent under a change of variables.",
        "analogy", "isomorphism"
    ),

    # --- UNRELATED (LLM should return related=false, not assign a strong type) ---
    (
        "Game theory was developed in the mid-20th century by mathematicians "
        "including John von Neumann and John Nash.",
        "Thermodynamics was formalized in the 19th century through the work "
        "of Clausius, Kelvin, and Carnot.",
        "unrelated", None
    ),
    (
        "Wikipedia is a freely editable online encyclopedia covering "
        "topics across all domains of human knowledge.",
        "The human genome project mapped the complete DNA sequence of "
        "human chromosomes between 1990 and 2003.",
        "unrelated", None
    ),
    (
        "Artificial neural networks require significant computational "
        "resources to train on large datasets.",
        "Game theory has been applied to auction design and mechanism "
        "design in economics.",
        "unrelated", None
    ),
]

NARRATION_QUALITY_PROMPT = """You are evaluating the quality of an edge narration
in a knowledge graph.

Node A: "{node_a}"
Node B: "{node_b}"
Claimed relationship type: {rel_type}
Narration written by system: "{narration}"

Does this narration accurately explain WHY and HOW these two nodes are related
in the way claimed?

Score 1-5:
5 — Precise, accurate, explains the mechanism or logical link clearly
4 — Accurate but slightly generic or missing a nuance
3 — Partially accurate but misses the key reason for the relationship
2 — Inaccurate or misleading
1 — Wrong or irrelevant to the actual relationship

Respond with JSON:
{{
  "narration_score": <1-5>,
  "is_accurate": <true/false>,
  "reasoning": "one sentence"
}}
Respond ONLY with JSON.
"""


def extract_edge_for_pair(node_a: str, node_b: str, brain, ingestor) -> dict:
    """
    Ingest both nodes together and extract edges between them.
    Returns the edge data dict or None.
    """
    from graph.brain import EdgeSource
    from llm_utils import llm_call, require_json

    # Use the ingestor's edge extraction prompt directly
    from ingestion.ingestor import EDGE_EXTRACTION_PROMPT
    
    raw = llm_call(
        EDGE_EXTRACTION_PROMPT.format(node_a=node_a, node_b=node_b),
        temperature=0.2, role="precise"
    )
    result = require_json(raw, default={})
    return result


def normalize_type(raw_type: str, analogy_depth: str = None) -> str:
    """Normalize extracted type to ground-truth type vocabulary."""
    # related=false from the LLM maps to "unrelated"
    if raw_type in ("unrelated", "none", ""):
        return "unrelated"
    if raw_type == "analogy":
        return "analogy"
    mapping = {
        "supports": "supports",
        "causes": "causes",
        "contradicts": "contradicts",
        "associated": "unrelated",
        "surface_analogy": "analogy",
        "structural_analogy": "analogy",
        "deep_isomorphism": "analogy",
        "analogous_to": "analogy",
        "answers": "supports",
        "partial": "supports",
    }
    return mapping.get(raw_type, "unrelated")


def normalize_depth(raw_type: str, analogy_depth: str = None) -> str:
    """Extract analogy depth from raw edge type."""
    depth_map = {
        "surface_analogy": "surface",
        "structural_analogy": "structural",
        "deep_isomorphism": "isomorphism",
        "analogy": analogy_depth or "structural",
    }
    return depth_map.get(raw_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--out", default="results/d1_edge_accuracy.json")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from llm_utils import llm_call, require_json

    brain     = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer  = Observer(brain)
    ingestor  = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    print("=" * 60)
    print("PHASE 1: Extracting edges for all ground-truth pairs")
    print("=" * 60)

    results = []
    for i, (node_a, node_b, gt_type, gt_depth) in enumerate(GROUND_TRUTH_PAIRS):
        print(f"\n[{i+1}/{len(GROUND_TRUTH_PAIRS)}] GT={gt_type}"
              f"{'/'+gt_depth if gt_depth else ''}:")
        print(f"  A: {node_a[:60]}...")
        print(f"  B: {node_b[:60]}...")

        edge = extract_edge_for_pair(node_a, node_b, brain, ingestor)
        
        if not edge or not edge.get("related"):
            pred_type  = "unrelated"
            pred_depth = None
            narration  = ""
        else:
            raw_type   = edge.get("type", "associated")
            pred_type  = normalize_type(raw_type)
            pred_depth = edge.get("analogy_depth") or normalize_depth(raw_type)
            narration  = edge.get("narration", "")

        type_correct  = pred_type == gt_type
        depth_correct = (
            gt_depth is None or
            (pred_type == "analogy" and pred_depth == gt_depth)
        )
        fully_correct = type_correct and depth_correct

        print(f"  Predicted: {pred_type}"
              f"{'/'+pred_depth if pred_depth else ''}"
              f"  {'✓' if fully_correct else '✗'}")
        if narration:
            print(f"  Narration: {narration[:80]}...")

        results.append({
            "pair_id": i,
            "node_a": node_a,
            "node_b": node_b,
            "gt_type": gt_type,
            "gt_depth": gt_depth,
            "pred_type": pred_type,
            "pred_depth": pred_depth,
            "narration": narration,
            "raw_edge": edge,
            "type_correct": type_correct,
            "depth_correct": depth_correct,
            "fully_correct": fully_correct,
        })
        time.sleep(0.5)

    # ── Per-type metrics ──
    print("\n" + "=" * 60)
    print("PHASE 2: Computing per-type metrics")
    print("=" * 60)

    all_types = ["supports", "causes", "contradicts", "analogy", "unrelated"]
    per_type = {}
    for t in all_types:
        gt_pos = [r for r in results if r["gt_type"] == t]
        pred_pos = [r for r in results if r["pred_type"] == t]
        tp = [r for r in gt_pos if r["type_correct"]]
        precision = len(tp) / len(pred_pos) if pred_pos else 0
        recall    = len(tp) / len(gt_pos)   if gt_pos  else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        per_type[t] = {
            "gt_count": len(gt_pos),
            "pred_count": len(pred_pos),
            "true_positives": len(tp),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
        print(f"  {t:15s}  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")

    # Analogy depth accuracy (only for analogy pairs)
    analogy_pairs = [r for r in results if r["gt_type"] == "analogy"]
    depth_acc = (
        sum(1 for r in analogy_pairs if r["depth_correct"]) / len(analogy_pairs)
        if analogy_pairs else 0
    )

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["gt_type"]][r["pred_type"]] += 1
    confusion_dict = {k: dict(v) for k, v in confusion.items()}

    # Overall accuracy
    overall_acc = sum(1 for r in results if r["type_correct"]) / len(results)
    full_acc    = sum(1 for r in results if r["fully_correct"]) / len(results)

    # ── Judge narration quality for correct-type pairs ──
    print("\n" + "=" * 60)
    print("PHASE 3: Judging narration quality for correct predictions")
    print("=" * 60)

    correct_pairs = [r for r in results if r["type_correct"] and r["narration"]]
    narration_scores = []
    narration_judgments = []

    for r in correct_pairs:
        prompt = NARRATION_QUALITY_PROMPT.format(
            node_a=r["node_a"],
            node_b=r["node_b"],
            rel_type=r["pred_type"],
            narration=r["narration"],
        )
        raw = llm_call(prompt, temperature=0.1, model=args.judge_model, role="precise")
        j = require_json(raw, default={})
        score = j.get("narration_score", 1)
        narration_scores.append(score)
        narration_judgments.append({
            "pair_id": r["pair_id"],
            "narration": r["narration"],
            "score": score,
            "is_accurate": j.get("is_accurate"),
            "reasoning": j.get("reasoning", ""),
        })
        print(f"  Pair {r['pair_id']:02d} [{r['gt_type']:12s}] "
              f"narration score: {score}/5")
        time.sleep(0.3)

    import statistics
    mean_narration = (statistics.mean(narration_scores)
                      if narration_scores else 0)

    # ── Build report ──
    pass_type_acc    = overall_acc >= 0.70
    pass_recall      = all(
        m["recall"] >= 0.50
        for t, m in per_type.items()
        if m["gt_count"] > 0 and t != "unrelated"
    )

    report = {
        "test": "D1 — Edge Accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "num_pairs": len(GROUND_TRUTH_PAIRS),
        },
        "summary": {
            "overall_type_accuracy": round(overall_acc, 3),
            "fully_correct_accuracy": round(full_acc, 3),
            "analogy_depth_accuracy": round(depth_acc, 3),
            "mean_narration_quality": round(mean_narration, 3),
            "PASS_type_accuracy": pass_type_acc,
            "PASS_recall_no_blind_spot": pass_recall,
            "PASS": pass_type_acc and pass_recall,
            "pass_threshold_accuracy": 0.70,
            "pass_threshold_recall": 0.50,
        },
        "per_type_metrics": per_type,
        "analogy_depth_breakdown": {
            "surface_correct": sum(1 for r in analogy_pairs
                                   if r.get("pred_depth") == "surface"
                                   and r["gt_depth"] == "surface"),
            "structural_correct": sum(1 for r in analogy_pairs
                                      if r.get("pred_depth") == "structural"
                                      and r["gt_depth"] == "structural"),
            "isomorphism_correct": sum(1 for r in analogy_pairs
                                       if r.get("pred_depth") == "isomorphism"
                                       and r["gt_depth"] == "isomorphism"),
            "total_analogy_pairs": len(analogy_pairs),
        },
        "confusion_matrix": confusion_dict,
        "narration_quality": {
            "pairs_evaluated": len(correct_pairs),
            "mean_score": round(mean_narration, 3),
            "pct_accurate": round(
                sum(1 for j in narration_judgments if j["is_accurate"])
                / max(len(narration_judgments), 1), 3),
        },
        "per_pair_results": results,
        "narration_judgments": narration_judgments,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D1: Edge Accuracy")
    print("=" * 60)
    print(f"Overall type accuracy : {overall_acc:.1%} "
          f"(pass threshold: >=70%)")
    print(f"Full accuracy (type+depth): {full_acc:.1%}")
    print(f"Analogy depth accuracy: {depth_acc:.1%}")
    print(f"Mean narration quality: {mean_narration:.2f}/5")
    print()
    print("Per-type breakdown:")
    for t, m in per_type.items():
        blind = (
            " ← BLIND SPOT"
            if m["recall"] < 0.50 and m["gt_count"] > 0 and t != "unrelated"
            else ""
        )
        print(f"  {t:15s}  P={m['precision']:.2f}  "
              f"R={m['recall']:.2f}  F1={m['f1']:.2f}{blind}")
    print()
    print("Confusion matrix (rows=GT, cols=predicted):")
    header = f"{'':15s} " + " ".join(f"{t[:8]:8s}" for t in all_types)
    print(f"  {header}")
    for gt_t in all_types:
        row = confusion_dict.get(gt_t, {})
        cells = " ".join(f"{row.get(pred_t, 0):8d}" for pred_t in all_types)
        print(f"  {gt_t:15s} {cells}")
    verdict = "PASS ✓" if report["summary"]["PASS"] else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
