"""
Dimension 1 - Test 6: Graph/Index Consistency
=============================================
Verifies that the shared semantic state stays synchronized across the
three main knowledge-entry paths introduced in the revised architecture:
direct ingestion, Reader absorption, and Researcher ingestion.

Pass criterion:
- 100% of newly created nodes exist in the graph
- 100% of newly created nodes exist in the shared embedding index
- >= 95% of newly created nodes are immediately self-queryable at rank 1

Usage:
    python benchmark/dim1/test_d1_graph_index_consistency.py         --out benchmark/dim1/results/d1_graph_index_consistency.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DIRECT_INGEST_TEXT = """
Hippocampal replay reactivates recently encoded experiences during sleep.
Replay compresses waking trajectories into burst-like sequences that can drive cortical plasticity.
Systems consolidation gradually shifts memory dependence from the hippocampus toward distributed cortical representations.
"""

READER_TEXT = """
Sleep spindles coordinate hippocampal sharp-wave ripples with cortical up states.
This temporal coupling appears to stabilize newly encoded declarative memories.
Disrupting spindle timing weakens overnight retention even when total sleep duration is preserved.
"""

RESEARCH_QUESTION = "What mechanisms improve the fidelity of memory consolidation during sleep?"
RESEARCH_RESULT_TEXT = """
Targeted memory reactivation during slow-wave sleep can selectively strengthen recently learned associations.
Closed-loop auditory stimulation increases slow oscillation amplitude and can improve later recall.
Coordinating replay events with cortical excitability windows appears to improve consolidation fidelity.
"""


def _phase_metrics(phase_name, node_ids, brain, emb_index):
    checks = []
    for nid in node_ids:
        node = brain.get_node(nid)
        in_graph = node is not None
        in_index = emb_index.has(nid)
        has_embedding = emb_index.get_embedding(nid) is not None
        self_rank1 = False
        top_score = None

        if has_embedding:
            matches = emb_index.query(
                emb_index.get_embedding(nid),
                threshold=-1.0,
                top_k=1,
            )
            if matches:
                self_rank1 = matches[0][0] == nid
                top_score = round(matches[0][1], 4)

        checks.append({
            'node_id': nid,
            'statement': (node or {}).get('statement', ''),
            'in_graph': in_graph,
            'in_index': in_index,
            'has_embedding': has_embedding,
            'self_query_rank1': self_rank1,
            'top_match_score': top_score,
        })

    total = len(checks)
    graph_ok = sum(1 for c in checks if c['in_graph'])
    index_ok = sum(1 for c in checks if c['in_index'])
    emb_ok = sum(1 for c in checks if c['has_embedding'])
    query_ok = sum(1 for c in checks if c['self_query_rank1'])

    return {
        'phase': phase_name,
        'nodes_created': total,
        'graph_presence_fraction': round(graph_ok / max(total, 1), 3),
        'index_presence_fraction': round(index_ok / max(total, 1), 3),
        'embedding_presence_fraction': round(emb_ok / max(total, 1), 3),
        'self_query_rank1_fraction': round(query_ok / max(total, 1), 3),
        'checks': checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default='benchmark/dim1/results/d1_graph_index_consistency.json'
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain, EdgeSource
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from reader.reader import Reader, ReadingEntry
    from researcher.researcher import Researcher

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)
    reader = Reader(
        brain,
        observer=observer,
        ingestor=ingestor,
        embedding_index=emb_index,
    )
    researcher = Researcher(
        brain,
        observer=observer,
        ingestor=ingestor,
        embedding_index=emb_index,
    )

    print('=' * 60)
    print('PHASE 1: Direct ingestion path')
    print('=' * 60)
    direct_ids = ingestor.ingest(DIRECT_INGEST_TEXT, source=EdgeSource.CONVERSATION) or []
    direct_metrics = _phase_metrics('direct_ingestion', direct_ids, brain, emb_index)

    print('\n' + '=' * 60)
    print('PHASE 2: Reader absorption path')
    print('=' * 60)
    entry = ReadingEntry(
        url='benchmark://graph-index-consistency/reader',
        title='Synthetic reader benchmark note',
        source_type='text',
        raw_text=READER_TEXT,
        added_reason='Synthetic benchmark note',
        added_by='benchmark',
    )
    absorb_result = reader.absorb_entry(entry)
    reader_metrics = _phase_metrics('reader_absorption', absorb_result.node_ids, brain, emb_index)

    print('\n' + '=' * 60)
    print('PHASE 3: Researcher ingestion path')
    print('=' * 60)
    researcher._generate_queries = lambda question, n: [question]
    researcher._web_search = lambda query: [
        (
            'Synthetic benchmark finding',
            RESEARCH_RESULT_TEXT,
            'benchmark://graph-index-consistency/research',
        )
    ]
    researcher._arxiv_search = lambda query: []
    researcher._filter_relevant = lambda question, results: results
    research_entry = researcher._research_question(RESEARCH_QUESTION)
    research_metrics = _phase_metrics('researcher_ingestion', research_entry.node_ids, brain, emb_index)

    all_checks = (
        direct_metrics['checks']
        + reader_metrics['checks']
        + research_metrics['checks']
    )
    total_nodes = len(all_checks)
    graph_ok = sum(1 for c in all_checks if c['in_graph'])
    index_ok = sum(1 for c in all_checks if c['in_index'])
    emb_ok = sum(1 for c in all_checks if c['has_embedding'])
    query_ok = sum(1 for c in all_checks if c['self_query_rank1'])

    graph_fraction = graph_ok / max(total_nodes, 1)
    index_fraction = index_ok / max(total_nodes, 1)
    embedding_fraction = emb_ok / max(total_nodes, 1)
    query_fraction = query_ok / max(total_nodes, 1)

    passed = (
        total_nodes > 0
        and graph_fraction == 1.0
        and index_fraction == 1.0
        and embedding_fraction == 1.0
        and query_fraction >= 0.95
    )

    report = {
        'test': 'D1 - Graph/Index Consistency',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'summary': {
            'nodes_checked': total_nodes,
            'graph_presence_fraction': round(graph_fraction, 3),
            'index_presence_fraction': round(index_fraction, 3),
            'embedding_presence_fraction': round(embedding_fraction, 3),
            'self_query_rank1_fraction': round(query_fraction, 3),
            'graph_nodes_total': len(brain.all_nodes()),
            'index_size_total': emb_index.size,
            'PASS': passed,
        },
        'phases': {
            'direct_ingestion': direct_metrics,
            'reader_absorption': reader_metrics,
            'researcher_ingestion': research_metrics,
        },
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 60)
    print('RESULTS - D1: Graph/Index Consistency')
    print('=' * 60)
    print(f"Nodes checked             : {total_nodes}")
    print(f"Graph presence fraction   : {graph_fraction:.2%} (pass = 100%)")
    print(f"Index presence fraction   : {index_fraction:.2%} (pass = 100%)")
    print(f"Embedding presence        : {embedding_fraction:.2%} (pass = 100%)")
    print(f"Immediate self-query rate : {query_fraction:.2%} (pass >= 95%)")
    verdict = 'PASS' if passed else 'FAIL'
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == '__main__':
    main()
