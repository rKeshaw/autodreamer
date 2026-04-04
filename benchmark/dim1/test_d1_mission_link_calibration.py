"""
Dimension 1 - Test 7: Mission-Link Calibration
===============================================
Measures whether newly ingested nodes that are actually relevant to the
active mission are linked with `toward_mission` edges at the right rate.

Pass criterion:
- Precision >= 75%
- Recall >= 70%

Usage:
    python benchmark/dim1/test_d1_mission_link_calibration.py         --out benchmark/dim1/results/d1_mission_link_calibration.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MISSION = 'How does sleep support memory consolidation?'

SAMPLES = [
    {
        'label': 'relevant',
        'text': 'During slow-wave sleep, hippocampal replay reactivates recent experiences and helps transfer them into cortical storage.',
    },
    {
        'label': 'relevant',
        'text': 'Sleep spindles are associated with stronger overnight retention of declarative memories after learning.',
    },
    {
        'label': 'relevant',
        'text': 'Targeted memory reactivation during sleep can selectively strengthen recently encoded associations.',
    },
    {
        'label': 'relevant',
        'text': 'Disrupting post-learning sleep reduces later recall accuracy even when wake practice is unchanged.',
    },
    {
        'label': 'irrelevant',
        'text': 'Plate tectonics recycles oceanic crust through subduction and mantle convection.',
    },
    {
        'label': 'irrelevant',
        'text': 'Photosynthesis converts light energy into chemical energy in chloroplasts.',
    },
    {
        'label': 'irrelevant',
        'text': 'Auction design changes bidding incentives in spectrum markets.',
    },
    {
        'label': 'irrelevant',
        'text': 'CRISPR systems enable targeted genome editing using guide RNAs and nuclease proteins.',
    },
]


def _has_mission_link(brain, node_id, mission_id):
    if brain.graph.has_edge(node_id, mission_id):
        edge = brain.get_edge(node_id, mission_id) or {}
        if edge.get('type') == 'toward_mission':
            return True, edge.get('weight', 0.0)
    if brain.graph.has_edge(mission_id, node_id):
        edge = brain.get_edge(mission_id, node_id) or {}
        if edge.get('type') == 'toward_mission':
            return True, edge.get('weight', 0.0)
    return False, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default='benchmark/dim1/results/d1_mission_link_calibration.json'
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain, EdgeSource
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    brain = Brain()
    mission_id = brain.set_mission(MISSION)
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    evaluations = []

    print('=' * 60)
    print('PHASE 1: Ingesting labeled mission-relevance samples')
    print('=' * 60)
    for idx, sample in enumerate(SAMPLES, start=1):
        print(f"  [{idx}/{len(SAMPLES)}] {sample['label']}: {sample['text'][:80]}...")
        new_ids = ingestor.ingest(sample['text'], source=EdgeSource.READING) or []
        for nid in new_ids:
            node = brain.get_node(nid)
            linked, strength = _has_mission_link(brain, nid, mission_id)
            evaluations.append({
                'node_id': nid,
                'statement': (node or {}).get('statement', ''),
                'ground_truth_relevant': sample['label'] == 'relevant',
                'mission_linked': linked,
                'link_strength': round(strength, 3),
                'source_text': sample['text'],
            })

    tp = sum(1 for e in evaluations if e['ground_truth_relevant'] and e['mission_linked'])
    fp = sum(1 for e in evaluations if (not e['ground_truth_relevant']) and e['mission_linked'])
    tn = sum(1 for e in evaluations if (not e['ground_truth_relevant']) and (not e['mission_linked']))
    fn = sum(1 for e in evaluations if e['ground_truth_relevant'] and (not e['mission_linked']))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    relevant_strengths = [e['link_strength'] for e in evaluations if e['ground_truth_relevant'] and e['mission_linked']]
    irrelevant_strengths = [e['link_strength'] for e in evaluations if (not e['ground_truth_relevant']) and e['mission_linked']]

    passed = precision >= 0.75 and recall >= 0.70

    report = {
        'test': 'D1 - Mission-Link Calibration',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'mission': MISSION,
            'samples': len(SAMPLES),
        },
        'summary': {
            'nodes_evaluated': len(evaluations),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'mean_relevant_link_strength': round(sum(relevant_strengths) / max(len(relevant_strengths), 1), 3),
            'mean_irrelevant_link_strength': round(sum(irrelevant_strengths) / max(len(irrelevant_strengths), 1), 3),
            'PASS': passed,
        },
        'evaluations': evaluations,
        'false_links': [e for e in evaluations if (not e['ground_truth_relevant']) and e['mission_linked']],
        'missed_links': [e for e in evaluations if e['ground_truth_relevant'] and (not e['mission_linked'])],
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 60)
    print('RESULTS - D1: Mission-Link Calibration')
    print('=' * 60)
    print(f"Nodes evaluated  : {len(evaluations)}")
    print(f"Precision        : {precision:.2%} (pass >= 75%)")
    print(f"Recall           : {recall:.2%} (pass >= 70%)")
    print(f"F1               : {f1:.2%}")
    verdict = 'PASS' if passed else 'FAIL'
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == '__main__':
    main()
