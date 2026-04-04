"""
Dimension 1 — Test 5: Contradiction Detection Accuracy
=======================================================
Injects 10 genuinely contradicting statement pairs and 10 compatible
statement pairs into the ingestor. Measures whether the system correctly
fires the contradiction check and creates CONTRADICTS edges for the
contradicting pairs, and does NOT create them for the compatible pairs.

Metrics:
  - True positive rate  (contradictions caught)
  - True negative rate  (compatible pairs not wrongly flagged)
  - Precision / Recall / F1

Pass criterion:
  - Precision >= 0.80 (not too many false alarms)
  - Recall    >= 0.70 (catches most real contradictions)

Usage:
    python test_d1_contradiction_detection.py \
        --judge-model <ollama-model-name> \
        --out results/d1_contradiction_detection.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Test data ─────────────────────────────────────────────────────────────────

CONTRADICTING_PAIRS = [
    # Scientific contradictions
    (
        "Heavier objects fall faster than lighter objects in a vacuum, "
        "as Aristotle argued.",
        "In a vacuum, all objects fall at the same rate regardless of mass, "
        "as Galileo demonstrated — gravity accelerates all objects equally."
    ),
    (
        "Acquired characteristics — skills and features an organism develops "
        "during its lifetime — can be inherited by its offspring.",
        "Only heritable genetic variations encoded in DNA can be passed to "
        "offspring; traits acquired during an individual's lifetime cannot "
        "be directly transmitted."
    ),
    (
        "Neural networks with more hidden layers always achieve higher "
        "accuracy on any classification task.",
        "Adding more layers to a neural network frequently causes vanishing "
        "gradients and overfitting, leading to worse generalization on "
        "tasks with limited training data."
    ),
    (
        "The Nash equilibrium always produces the socially optimal outcome "
        "in any strategic interaction between rational agents.",
        "The prisoner's dilemma is a canonical example of a Nash equilibrium "
        "that is Pareto suboptimal — all players would be better off "
        "deviating to a cooperative outcome."
    ),
    (
        "DNA is a single-stranded molecule that carries genetic information "
        "in a linear sequence of nucleotides.",
        "DNA is a double-stranded helical molecule; the two strands are held "
        "together by hydrogen bonds between complementary base pairs and run "
        "antiparallel to each other."
    ),
    (
        "Entropy in an isolated system can spontaneously decrease, allowing "
        "order to emerge from disorder without external energy input.",
        "The second law of thermodynamics states that entropy in an isolated "
        "system never decreases — spontaneous processes always increase or "
        "maintain total entropy."
    ),
    (
        "Gradient descent guarantees finding the global minimum of the "
        "loss function for any neural network architecture.",
        "Gradient descent is a local optimization method and is not guaranteed "
        "to find the global minimum; non-convex loss landscapes mean it "
        "frequently converges to local minima or saddle points."
    ),
    (
        "Natural selection always produces optimal organisms — evolution "
        "reliably converges on the best possible solution to any "
        "environmental challenge.",
        "Natural selection is constrained by historical contingency, "
        "genetic drift, developmental constraints, and pleiotropy; "
        "it produces organisms that are 'good enough,' not optimal."
    ),
    (
        "Information cannot be destroyed; it is always conserved in any "
        "physical process, including black hole formation.",
        "The black hole information paradox arises because Hawking radiation "
        "appears to destroy information, contradicting the principle of "
        "unitarity in quantum mechanics."
    ),
    (
        "The central dogma of molecular biology states that information "
        "flows from protein back to DNA, allowing proteins to modify "
        "the genome directly.",
        "The central dogma states information flows from DNA to RNA to protein "
        "and never in reverse from protein to nucleic acid; reverse "
        "transcriptase transfers RNA back to DNA, but protein to DNA "
        "transfer has not been observed."
    ),
]

COMPATIBLE_PAIRS = [
    # Related but not contradicting
    (
        "DNA replication is semi-conservative: each new double helix "
        "consists of one original strand and one newly synthesized strand.",
        "During DNA replication, helicase unwinds the double helix and "
        "DNA polymerase synthesizes the new complementary strand in the "
        "5' to 3' direction."
    ),
    (
        "Natural selection acts on phenotypic variation, favoring "
        "individuals whose traits increase reproductive success.",
        "Genetic drift is a random process that changes allele frequencies "
        "in small populations, independent of natural selection."
    ),
    (
        "In game theory, a Nash equilibrium is a set of strategies "
        "from which no player has an incentive to deviate unilaterally.",
        "The minimax theorem states that in zero-sum games, the minimax "
        "strategy profile coincides with the Nash equilibrium."
    ),
    (
        "Backpropagation computes gradients by applying the chain rule "
        "backward through the computational graph of a neural network.",
        "Stochastic gradient descent uses random mini-batches to estimate "
        "the gradient, making it computationally efficient for large datasets."
    ),
    (
        "The first law of thermodynamics states that energy is conserved: "
        "it can be converted between forms but not created or destroyed.",
        "Free energy, defined as the energy available to do useful work, "
        "decreases in spontaneous processes at constant temperature and pressure."
    ),
    (
        "Mutations in protein-coding genes can alter the amino acid "
        "sequence of the resulting protein, potentially changing its function.",
        "Gene regulatory regions control when and where a gene is expressed, "
        "and mutations in these regions can affect gene expression without "
        "changing the protein sequence."
    ),
    (
        "The prisoner's dilemma illustrates how individually rational "
        "decisions can lead to collectively suboptimal outcomes.",
        "Repeated games and reputation mechanisms can enable cooperation "
        "to emerge in prisoner's dilemma situations over time."
    ),
    (
        "Entropy measures the number of possible microstates consistent "
        "with a given macrostate of a thermodynamic system.",
        "Shannon entropy in information theory is a measure of uncertainty "
        "in a probability distribution, formally analogous to thermodynamic entropy."
    ),
    (
        "Overfitting occurs when a model learns the noise in training data, "
        "performing well on training data but poorly on unseen examples.",
        "Regularization techniques such as L2 weight decay penalize large "
        "weights and reduce overfitting by constraining model complexity."
    ),
    (
        "Game theory assumes rational agents who maximize their own payoff "
        "given beliefs about other players' strategies.",
        "Behavioral economics documents systematic deviations from rational "
        "game-theoretic predictions in human decision-making."
    ),
]

VALIDATE_CONTRADICTION_PROMPT = """You are validating whether a knowledge graph system
correctly identified a contradiction between two statements.

Statement A:
"{stmt_a}"

Statement B:
"{stmt_b}"

The system classified these as: {classification}

Is this classification CORRECT?
- A genuine contradiction means A and B make mutually exclusive claims —
  if A is true, B must be false.
- Two statements about the same topic that don't directly conflict are NOT contradictions.

Respond with JSON:
{{
  "correct_classification": true or false,
  "is_genuine_contradiction": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence"
}}
Respond ONLY with JSON.
"""


def run_contradiction_test(brain, ingestor, pairs: list,
                           expected_label: str) -> list:
    """
    Ingest each pair, check whether CONTRADICTS edge exists.
    Returns list of results with detected, expected, correct fields.
    """
    from graph.brain import EdgeSource, EdgeType

    results = []
    for i, (stmt_a, stmt_b) in enumerate(pairs):
        nodes_before = set(nid for nid, _ in brain.all_nodes())
        
        # Ingest both statements together as a single text block
        combined = f"{stmt_a}\n\n{stmt_b}"
        new_ids = ingestor.ingest(combined, source=EdgeSource.RESEARCH) or []

        # Find the two main nodes from this ingest
        all_current = {nid for nid, _ in brain.all_nodes()}
        new_this = list(all_current - nodes_before)

        # Check if a CONTRADICTS edge exists between any of the new nodes
        contradiction_found = False
        contradiction_details = []
        for nid_a in new_this:
            for nid_b in new_this:
                if nid_a == nid_b:
                    continue
                edge = brain.get_edge(nid_a, nid_b)
                if edge and edge.get("type") == EdgeType.CONTRADICTS.value:
                    contradiction_found = True
                    contradiction_details.append({
                        "from": nid_a,
                        "to": nid_b,
                        "narration": edge.get("narration", ""),
                    })

        correct = (
            (expected_label == "contradicts" and contradiction_found) or
            (expected_label == "compatible"  and not contradiction_found)
        )

        results.append({
            "pair_id": i,
            "stmt_a": stmt_a,
            "stmt_b": stmt_b,
            "expected": expected_label,
            "contradiction_detected": contradiction_found,
            "correct": correct,
            "new_nodes_created": len(new_this),
            "contradiction_details": contradiction_details,
        })
        status = "✓" if correct else "✗"
        detected = "DETECTED" if contradiction_found else "NOT DETECTED"
        print(f"  [{i+1:02d}] {status} Expected={expected_label:12s} "
              f"Got={detected}")
        time.sleep(0.5)

    return results


def validate_with_llm(results: list, model: str) -> list:
    """Ask LLM judge to validate the system's classification decisions."""
    from llm_utils import llm_call, require_json

    validated = []
    for r in results:
        classification = (
            "CONTRADICTION" if r["contradiction_detected"]
            else "COMPATIBLE (no contradiction)"
        )
        prompt = VALIDATE_CONTRADICTION_PROMPT.format(
            stmt_a=r["stmt_a"],
            stmt_b=r["stmt_b"],
            classification=classification,
        )
        raw = llm_call(prompt, temperature=0.1, model=model, role="precise")
        j = require_json(raw, default={})
        validated.append({
            **r,
            "llm_validation": {
                "correct_classification": j.get("correct_classification"),
                "is_genuine_contradiction": j.get("is_genuine_contradiction"),
                "confidence": j.get("confidence", 0.5),
                "reasoning": j.get("reasoning", ""),
            }
        })
        print(f"  Pair {r['pair_id']:02d} [{r['expected']:12s}] "
              f"LLM says classification correct: "
              f"{j.get('correct_classification')}")
        time.sleep(0.3)
    return validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--out", default="results/d1_contradiction_detection.json")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    brain     = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer  = Observer(brain)
    ingestor  = Ingestor(brain, research_agenda=observer,
                          embedding_index=emb_index)

    # ── Phase 1: Test contradicting pairs ──
    print("=" * 60)
    print("PHASE 1: Injecting CONTRADICTING pairs")
    print("=" * 60)
    contra_results = run_contradiction_test(
        brain, ingestor, CONTRADICTING_PAIRS, "contradicts"
    )

    # ── Phase 2: Test compatible pairs ──
    print("\n" + "=" * 60)
    print("PHASE 2: Injecting COMPATIBLE pairs")
    print("=" * 60)
    compat_results = run_contradiction_test(
        brain, ingestor, COMPATIBLE_PAIRS, "compatible"
    )

    all_results = contra_results + compat_results

    # ── Phase 3: LLM validation ──
    print("\n" + "=" * 60)
    print("PHASE 3: LLM validation of classifications")
    print("=" * 60)
    all_results = validate_with_llm(all_results, args.judge_model)

    # ── Compute metrics ──
    tp = sum(1 for r in contra_results if r["contradiction_detected"])
    fn = sum(1 for r in contra_results if not r["contradiction_detected"])
    fp = sum(1 for r in compat_results if r["contradiction_detected"])
    tn = sum(1 for r in compat_results if not r["contradiction_detected"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = (tp + tn) / len(all_results)

    # LLM-validated metrics
    llm_correct = sum(
        1 for r in all_results
        if r.get("llm_validation", {}).get("correct_classification")
    )
    llm_validation_rate = llm_correct / len(all_results) if all_results else 0

    # Failure analysis
    missed = [r for r in contra_results if not r["contradiction_detected"]]
    false_alarms = [r for r in compat_results if r["contradiction_detected"]]

    report = {
        "test": "D1 — Contradiction Detection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "num_contradicting_pairs": len(CONTRADICTING_PAIRS),
            "num_compatible_pairs": len(COMPATIBLE_PAIRS),
        },
        "summary": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "accuracy": round(accuracy, 3),
            "llm_validation_agreement": round(llm_validation_rate, 3),
            "PASS_precision": precision >= 0.80,
            "PASS_recall": recall >= 0.70,
            "PASS": precision >= 0.80 and recall >= 0.70,
            "pass_threshold_precision": 0.80,
            "pass_threshold_recall": 0.70,
        },
        "missed_contradictions": [
            {"stmt_a": r["stmt_a"][:100], "stmt_b": r["stmt_b"][:100]}
            for r in missed
        ],
        "false_alarms": [
            {"stmt_a": r["stmt_a"][:100], "stmt_b": r["stmt_b"][:100]}
            for r in false_alarms
        ],
        "all_results": all_results,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D1: Contradiction Detection")
    print("=" * 60)
    print(f"True positives  (caught contradictions): {tp}/{len(CONTRADICTING_PAIRS)}")
    print(f"True negatives  (compatible left clean) : {tn}/{len(COMPATIBLE_PAIRS)}")
    print(f"False positives (wrong contradictions)  : {fp}")
    print(f"False negatives (missed contradictions) : {fn}")
    print(f"Precision : {precision:.2%} ({'✓' if precision >= 0.80 else '✗'} threshold: >=80%)")
    print(f"Recall    : {recall:.2%} ({'✓' if recall >= 0.70 else '✗'} threshold: >=70%)")
    print(f"F1        : {f1:.2%}")
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"LLM validation agreement: {llm_validation_rate:.2%}")
    if missed:
        print(f"\nMissed contradictions ({len(missed)}):")
        for r in missed[:3]:
            print(f"  A: {r['stmt_a'][:70]}...")
            print(f"  B: {r['stmt_b'][:70]}...")
    if false_alarms:
        print(f"\nFalse alarms ({len(false_alarms)}):")
        for r in false_alarms[:3]:
            print(f"  A: {r['stmt_a'][:70]}...")
            print(f"  B: {r['stmt_b'][:70]}...")
    verdict = "PASS ✓" if report["summary"]["PASS"] else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
