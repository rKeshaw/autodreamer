"""
Dimension 2 - Test 7: Deferred Insight Promotion Quality
========================================================
Benchmarks the revised-roadmap slice for delayed promotion quality from
the InsightBuffer. The test seeds the buffer with controlled dream-like
deferred analogies, adds bridge context, and measures whether promoted
connections are genuinely meaningful.

Pass criterion:
- at least 50% of deferred pairs promote after added context
- at least 67% of promoted connections are judged genuine

Usage:
    python benchmark/dim2/test_d2_buffer_promotion_quality.py         --judge-model llama3.1:70b         --out benchmark/dim2/results/d2_buffer_promotion_quality.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFERRED_CASES = [
    {
        'id': 'selection_gradient',
        'statement_a': 'Natural selection accumulates beneficial variants through differential reproduction over generations.',
        'statement_b': 'Gradient descent accumulates parameter updates that reduce loss over many optimization steps.',
        'claim': 'Natural selection and gradient descent both implement iterative search by retaining better-performing configurations and suppressing worse ones.',
        'bridge': 'Selection and gradient descent are both iterative optimization processes that preserve fitter states while local feedback steers later updates.',
    },
    {
        'id': 'mutation_exploration',
        'statement_a': 'Mutation introduces stochastic variation into populations before selection filters outcomes.',
        'statement_b': 'Exploration noise injects stochastic perturbations into reinforcement learning policies before reward filters outcomes.',
        'claim': 'Mutation and exploration noise both widen a search process before an external fitness signal filters the resulting trajectories.',
        'bridge': 'Both mutation and exploration noise increase state-space coverage before a later selection signal determines which trajectories persist.',
    },
    {
        'id': 'proofreading_parity',
        'statement_a': 'DNA proofreading detects local copying mismatches and prevents error propagation during replication.',
        'statement_b': 'Parity checks detect local transmission mismatches and prevent error propagation in communication channels.',
        'claim': 'DNA proofreading and parity checking both perform local mismatch detection to stop downstream corruption in a copying channel.',
        'bridge': 'Proofreading in replication and parity checks in communication both insert local error-detection structure into a noisy copying process.',
    },
]

JUDGE_PROMPT = """
You are evaluating a connection that was promoted from a delayed-insight buffer.

Concept A: "{node_a}"
Concept B: "{node_b}"
Promoted connection type: {edge_type}
Promoted narration: "{narration}"

Is this promoted connection a genuine and meaningful relationship,
rather than a vague or superficial association?

Respond EXACTLY in JSON:
{{
  "genuine": true or false,
  "reasoning": "one or two sentences"
}}
"""


def _judge_promotion(item, model):
    from llm_utils import llm_call, require_json

    prompt = JUDGE_PROMPT.format(
        node_a=item['statement_a'],
        node_b=item['statement_b'],
        edge_type=item['edge_type'],
        narration=item['narration'],
    )
    raw = llm_call(prompt, temperature=0.1, model=model, role='precise')
    result = require_json(raw, default={'genuine': False, 'reasoning': 'Judge parse failed'})
    if 'genuine' not in result:
        result['genuine'] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge-model', default='llama3.1:70b')
    parser.add_argument('--out', default='benchmark/dim2/results/d2_buffer_promotion_quality.json')
    parser.add_argument('--cycles', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import insight_buffer as insight_buffer_mod
    from graph.brain import Brain, Node, NodeType, NodeStatus, Edge, EdgeType, EdgeSource
    from embedding_index import EmbeddingIndex
    from embedding import embed as shared_embed
    from critic.critic import Critic, CandidateThought
    from insight_buffer import InsightBuffer

    insight_buffer_mod.BUFFER_PATH = 'benchmark/dim2/results/_tmp_insight_buffer.json'

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    buffer = InsightBuffer(brain, embedding_index=emb_index)
    buffer.pending = []
    critic = Critic(brain, embedding_index=emb_index, insight_buffer=buffer)

    pair_records = []

    print('=' * 60)
    print('PHASE 1: Seeding deferred dream-like insights')
    print('=' * 60)
    for case in DEFERRED_CASES:
        node_a = Node(
            statement=case['statement_a'],
            node_type=NodeType.CONCEPT,
            cluster='benchmark_a',
            status=NodeStatus.UNCERTAIN,
            importance=0.7,
        )
        node_b = Node(
            statement=case['statement_b'],
            node_type=NodeType.CONCEPT,
            cluster='benchmark_b',
            status=NodeStatus.UNCERTAIN,
            importance=0.7,
        )
        node_a_id = brain.add_node(node_a)
        node_b_id = brain.add_node(node_b)
        emb_index.add(node_a_id, shared_embed(node_a.statement))
        emb_index.add(node_b_id, shared_embed(node_b.statement))

        candidate = CandidateThought(
            claim=case['claim'],
            source_module='dreamer',
            proposed_type='structural_analogy',
            importance=0.55,
            edge_type='structural',
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            crosses_domains=True,
        )
        critic.route_deferred(candidate)

        bridge_node = Node(
            statement=case['bridge'],
            node_type=NodeType.SYNTHESIS,
            cluster='benchmark_bridge',
            status=NodeStatus.UNCERTAIN,
            importance=0.8,
        )
        bridge_id = brain.add_node(bridge_node)
        emb_index.add(bridge_id, shared_embed(bridge_node.statement))

        support_edge_a = Edge(
            type=EdgeType.ASSOCIATED,
            narration='Bridge context added for delayed promotion benchmark',
            weight=0.72,
            confidence=0.7,
            source=EdgeSource.CONSOLIDATION,
        )
        support_edge_b = Edge(
            type=EdgeType.ASSOCIATED,
            narration='Bridge context added for delayed promotion benchmark',
            weight=0.72,
            confidence=0.7,
            source=EdgeSource.CONSOLIDATION,
        )
        brain.add_edge(node_a_id, bridge_id, support_edge_a)
        brain.add_edge(node_b_id, bridge_id, support_edge_b)

        pair_records.append({
            'case_id': case['id'],
            'node_a_id': node_a_id,
            'node_b_id': node_b_id,
            'statement_a': case['statement_a'],
            'statement_b': case['statement_b'],
            'claim': case['claim'],
            'bridge': case['bridge'],
            'promotion_cycle': None,
        })

    print('\n' + '=' * 60)
    print('PHASE 2: Evaluating the insight buffer over multiple cycles')
    print('=' * 60)
    cycle_stats = []
    for cycle in range(1, args.cycles + 1):
        stats = buffer.evaluate_all()
        stats['cycle'] = cycle
        cycle_stats.append(stats)
        for item in pair_records:
            if item['promotion_cycle'] is not None:
                continue
            if brain.graph.has_edge(item['node_a_id'], item['node_b_id']) or brain.graph.has_edge(item['node_b_id'], item['node_a_id']):
                item['promotion_cycle'] = cycle
        if all(item['promotion_cycle'] is not None for item in pair_records):
            break
        time.sleep(0.2)

    print('\n' + '=' * 60)
    print('PHASE 3: Judging promoted delayed insights')
    print('=' * 60)
    promoted = []
    genuine_count = 0
    for item in pair_records:
        edge = None
        if brain.graph.has_edge(item['node_a_id'], item['node_b_id']):
            edge = brain.get_edge(item['node_a_id'], item['node_b_id'])
        elif brain.graph.has_edge(item['node_b_id'], item['node_a_id']):
            edge = brain.get_edge(item['node_b_id'], item['node_a_id'])
        if not edge:
            continue
        promoted_item = {
            'case_id': item['case_id'],
            'statement_a': item['statement_a'],
            'statement_b': item['statement_b'],
            'edge_type': edge.get('type', 'associated'),
            'narration': edge.get('narration', ''),
            'promotion_cycle': item['promotion_cycle'],
            'confidence': edge.get('confidence', 0.0),
        }
        judgment = _judge_promotion(promoted_item, args.judge_model)
        promoted_item['judgment'] = judgment
        promoted.append(promoted_item)
        if judgment.get('genuine'):
            genuine_count += 1
        time.sleep(0.2)

    promotion_rate = len(promoted) / max(len(pair_records), 1)
    genuine_fraction = genuine_count / max(len(promoted), 1)
    promotion_cycles = [p['promotion_cycle'] for p in promoted if p.get('promotion_cycle') is not None]
    mean_promotion_cycle = sum(promotion_cycles) / max(len(promotion_cycles), 1)

    passed = promotion_rate >= 0.50 and genuine_fraction >= 0.67

    report = {
        'test': 'D2 - Deferred Insight Promotion Quality',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'judge_model': args.judge_model,
            'deferred_cases': len(DEFERRED_CASES),
            'cycles_attempted': args.cycles,
        },
        'summary': {
            'deferred_pairs': len(pair_records),
            'promoted_pairs': len(promoted),
            'promotion_rate': round(promotion_rate, 3),
            'genuine_promotions': genuine_count,
            'genuine_fraction': round(genuine_fraction, 3),
            'mean_promotion_cycle': round(mean_promotion_cycle, 3),
            'remaining_buffer_size': buffer.size,
            'PASS': passed,
        },
        'cycle_stats': cycle_stats,
        'promoted_pairs_detail': promoted,
        'deferred_pairs_detail': pair_records,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 60)
    print('RESULTS - D2: Deferred Insight Promotion Quality')
    print('=' * 60)
    print(f"Deferred pairs         : {len(pair_records)}")
    print(f"Promoted pairs         : {len(promoted)}")
    print(f"Promotion rate         : {promotion_rate:.2%} (pass >= 50%)")
    print(f"Genuine promotion rate : {genuine_fraction:.2%} (pass >= 67%)")
    print(f"Mean promotion cycle   : {mean_promotion_cycle:.2f}")
    verdict = 'PASS' if passed else 'FAIL'
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == '__main__':
    main()
