"""
Dimension 2 - Test 6: Critic Precision Lift
===========================================
Measures whether System 2 review improves the validity of deep dream
insights without rejecting too many genuinely valuable ones.

This test implements the revised-roadmap slices:
- raw deep-insight validity
- critic-accepted deep-insight validity
- precision lift from critic review
- false-negative rate on valuable dream insights

Pass criterion:
- accepted validity >= raw validity
- accepted validity >= 65%
- false-negative rate <= 40%

Usage:
    python benchmark/dim2/test_d2_critic_lift.py         --judge-model llama3.1:70b         --out benchmark/dim2/results/d2_critic_lift.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MISSION = 'How do biological and artificial learning systems balance exploration with stability?'

JUDGE_PROMPT = """
You are evaluating the quality of a dream-derived scientific insight.

Concept A: "{node_a}"
Concept B: "{node_b}"
Insight claim: "{claim}"
Claimed depth: {depth}

Does this claim represent a genuine structural or isomorphic connection,
rather than surface-level shared vocabulary?

Respond EXACTLY in JSON:
{{
  "genuine": true or false,
  "reasoning": "one or two sentences"
}}
"""


def _load_shared_graph():
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex

    brain = Brain()
    brain_path = 'benchmark/dim2/shared/brain.json'
    index_path = 'benchmark/dim2/shared/embedding_index'
    if not (os.path.exists(brain_path) and os.path.exists(index_path + '.json')):
        raise FileNotFoundError('Shared Dimension 2 graph not found. Run prep_d2_graph.py first.')
    brain.load(brain_path)
    emb_index = EmbeddingIndex.load(index_path)
    return brain, emb_index


def _judge_insight(insight, model):
    from llm_utils import llm_call, require_json

    prompt = JUDGE_PROMPT.format(
        node_a=insight['from'],
        node_b=insight['to'],
        claim=insight['claim'],
        depth=insight['depth'],
    )
    raw = llm_call(prompt, temperature=0.1, model=model, role='precise')
    result = require_json(raw, default={'genuine': False, 'reasoning': 'Judge parse failed'})
    if 'genuine' not in result:
        result['genuine'] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge-model', default='llama3.1:70b')
    parser.add_argument('--out', default='benchmark/dim2/results/d2_critic_lift.json')
    parser.add_argument('--cycles', type=int, default=6)
    parser.add_argument('--min-insights', type=int, default=6)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from observer.observer import Observer
    from dreamer.dreamer import Dreamer
    from critic.critic import Critic, CandidateThought

    dream_brain, _ = _load_shared_graph()
    dream_brain.set_mission(MISSION)
    dream_observer = Observer(dream_brain)
    dreamer = Dreamer(dream_brain, research_agenda=dream_observer)

    raw_insights = []
    seen = set()

    print('=' * 60)
    print('PHASE 1: Collecting raw deep dream insights')
    print('=' * 60)
    for cycle in range(args.cycles):
        print(f'  Dream cycle {cycle + 1}/{args.cycles}...')
        log = dreamer.dream(steps=20, temperature=0.9, run_nrem=False)
        for ins in log.insights:
            if ins.get('depth') not in ['structural', 'isomorphism']:
                continue
            key = (ins.get('from_node_id'), ins.get('to_node_id'), ins.get('narration'))
            if key in seen:
                continue
            seen.add(key)
            raw_insights.append({
                'from': ins['from'],
                'to': ins['to'],
                'from_node_id': ins['from_node_id'],
                'to_node_id': ins['to_node_id'],
                'claim': ins['narration'],
                'depth': ins['depth'],
            })
        if len(raw_insights) >= args.min_insights:
            break

    print('\n' + '=' * 60)
    print('PHASE 2: Judging raw deep insights')
    print('=' * 60)
    evaluations = []
    genuine_raw_count = 0
    for ins in raw_insights:
        raw_judgment = _judge_insight(ins, args.judge_model)
        if raw_judgment.get('genuine'):
            genuine_raw_count += 1
        evaluations.append({
            'raw_insight': ins,
            'raw_judgment': raw_judgment,
        })
        time.sleep(0.2)

    print('\n' + '=' * 60)
    print('PHASE 3: Running System 2 review on the same raw insights')
    print('=' * 60)
    critic_brain, critic_index = _load_shared_graph()
    critic = Critic(critic_brain, embedding_index=critic_index)

    accepted_count = 0
    accepted_genuine_count = 0
    verdict_counts = {'accept': 0, 'refine': 0, 'reject': 0, 'defer': 0}
    false_negatives = 0

    for ev in evaluations:
        raw_insight = ev['raw_insight']
        depth = raw_insight['depth']
        candidate = CandidateThought(
            claim=raw_insight['claim'],
            source_module='dreamer',
            proposed_type='structural_analogy' if depth == 'structural' else 'deep_isomorphism',
            importance=0.75 if depth == 'structural' else 0.85,
            edge_type=depth,
            node_a_id=raw_insight['from_node_id'],
            node_b_id=raw_insight['to_node_id'],
            crosses_domains=True,
        )
        critic_log = critic.evaluate_with_refinement(candidate)
        ev['critic_log'] = critic_log.to_dict()
        verdict = critic_log.verdict.value
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        accepted_claim = critic_log.refinement_note or candidate.claim
        if verdict == 'accept':
            accepted_count += 1
            accepted_insight = dict(raw_insight)
            accepted_insight['claim'] = accepted_claim
            accepted_judgment = _judge_insight(accepted_insight, args.judge_model)
            ev['accepted_claim'] = accepted_claim
            ev['accepted_judgment'] = accepted_judgment
            if accepted_judgment.get('genuine'):
                accepted_genuine_count += 1
        elif ev['raw_judgment'].get('genuine'):
            false_negatives += 1
        time.sleep(0.2)

    raw_validity = genuine_raw_count / max(len(raw_insights), 1)
    accepted_validity = accepted_genuine_count / max(accepted_count, 1)
    precision_lift = accepted_validity - raw_validity if accepted_count else -raw_validity
    false_negative_rate = false_negatives / max(genuine_raw_count, 1)

    passed = (
        len(raw_insights) >= args.min_insights
        and accepted_count > 0
        and accepted_validity >= raw_validity
        and accepted_validity >= 0.65
        and false_negative_rate <= 0.40
    )

    report = {
        'test': 'D2 - Critic Precision Lift',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'judge_model': args.judge_model,
            'cycles_attempted': args.cycles,
            'min_insights': args.min_insights,
            'mission': MISSION,
        },
        'summary': {
            'raw_deep_insights': len(raw_insights),
            'raw_genuine_count': genuine_raw_count,
            'raw_validity': round(raw_validity, 3),
            'accepted_count': accepted_count,
            'accepted_genuine_count': accepted_genuine_count,
            'accepted_validity': round(accepted_validity, 3),
            'precision_lift': round(precision_lift, 3),
            'false_negative_rate': round(false_negative_rate, 3),
            'verdict_breakdown': verdict_counts,
            'PASS': passed,
        },
        'evaluations': evaluations,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 60)
    print('RESULTS - D2: Critic Precision Lift')
    print('=' * 60)
    print(f"Raw deep insights       : {len(raw_insights)}")
    print(f"Raw validity            : {raw_validity:.2%}")
    print(f"Accepted insights       : {accepted_count}")
    print(f"Accepted validity       : {accepted_validity:.2%}")
    print(f"Precision lift          : {precision_lift:+.2%}")
    print(f"False-negative rate     : {false_negative_rate:.2%}")
    verdict = 'PASS' if passed else 'FAIL'
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == '__main__':
    main()
