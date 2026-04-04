"""
Dimension 1 - Report Aggregator
===============================
Reads all Dimension 1 test result files and produces a unified benchmark
report with an overall pass/fail verdict.
"""

import os
import json
import time
import argparse

D1_TESTS = [
    ('d1_node_quality', 'Node Quality'),
    ('d1_duplicate_rate', 'Duplicate Rate'),
    ('d1_edge_accuracy', 'Edge Accuracy'),
    ('d1_cluster_coherence', 'Cluster Coherence'),
    ('d1_contradiction_detection', 'Contradiction Detection'),
    ('d1_graph_index_consistency', 'Graph/Index Consistency'),
    ('d1_mission_link_calibration', 'Mission-Link Calibration'),
]


def load_result(results_dir: str, test_id: str) -> dict:
    path = os.path.join(results_dir, f'{test_id}.json')
    if not os.path.exists(path):
        return {'_missing': True, 'test': test_id}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def extract_key_metrics(test_id: str, data: dict) -> dict:
    if data.get('_missing'):
        return {'status': 'MISSING', 'pass': None}

    s = data.get('summary', {})
    if test_id == 'd1_node_quality':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'mean_score': s.get('mean_score'), 'pct_score_4_plus': s.get('pct_score_4_plus'), 'pct_self_contained': s.get('pct_self_contained'), 'pct_says_how_or_why': s.get('pct_says_how_or_why'), 'nodes_judged': data.get('config', {}).get('total_nodes_judged')}
    if test_id == 'd1_duplicate_rate':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'duplicate_rate_final': s.get('duplicate_rate_final'), 'new_nodes_on_reingestion': s.get('new_nodes_on_reingestion'), 'llm_precision': s.get('llm_judge', {}).get('precision')}
    if test_id == 'd1_edge_accuracy':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'overall_type_accuracy': s.get('overall_type_accuracy'), 'analogy_depth_accuracy': s.get('analogy_depth_accuracy'), 'mean_narration_quality': s.get('mean_narration_quality'), 'per_type': data.get('per_type_metrics', {})}
    if test_id == 'd1_cluster_coherence':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'mean_intra_similarity': s.get('global_mean_intra_similarity'), 'mean_inter_similarity': s.get('global_mean_inter_similarity'), 'mean_silhouette': s.get('mean_silhouette_score'), 'assignment_accuracy': s.get('cluster_assignment_accuracy')}
    if test_id == 'd1_contradiction_detection':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'precision': s.get('precision'), 'recall': s.get('recall'), 'f1': s.get('f1'), 'accuracy': s.get('accuracy')}
    if test_id == 'd1_graph_index_consistency':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'nodes_checked': s.get('nodes_checked'), 'graph_presence_fraction': s.get('graph_presence_fraction'), 'index_presence_fraction': s.get('index_presence_fraction'), 'self_query_rank1_fraction': s.get('self_query_rank1_fraction')}
    if test_id == 'd1_mission_link_calibration':
        return {'status': 'PASS' if s.get('PASS') else 'FAIL', 'pass': s.get('PASS'), 'nodes_evaluated': s.get('nodes_evaluated'), 'precision': s.get('precision'), 'recall': s.get('recall'), 'f1': s.get('f1')}
    return {'status': 'UNKNOWN'}


def print_report(report: dict):
    print('' + '=' * 70)
    print('DIMENSION 1 - KNOWLEDGE GRAPH QUALITY - BENCHMARK REPORT')
    print('=' * 70)
    print(f"Generated: {report['timestamp']}")
    print()
    for test_id, test_name in D1_TESTS:
        m = report['tests'][test_id]
        status = m.get('status', 'MISSING')
        print(f"  [{status}] {test_name}")
        if test_id == 'd1_node_quality':
            print(f"      Mean score       : {m.get('mean_score')}/5")
            print(f"      % score >= 4     : {m.get('pct_score_4_plus', 0):.0%} (threshold: 80%)")
            print(f"      % self-contained : {m.get('pct_self_contained', 0):.0%}")
            print(f"      % says how/why   : {m.get('pct_says_how_or_why', 0):.0%}")
            print(f"      Nodes judged     : {m.get('nodes_judged')}")
        elif test_id == 'd1_duplicate_rate':
            print(f"      Duplicate rate   : {m.get('duplicate_rate_final', 0):.1%} (threshold: <5%)")
            print(f"      New nodes on re-ingest: {m.get('new_nodes_on_reingestion', '?')} (ideal: 0)")
            print(f"      LLM pair precision    : {m.get('llm_precision', 0):.0%}")
        elif test_id == 'd1_edge_accuracy':
            print(f"      Type accuracy    : {m.get('overall_type_accuracy', 0):.0%} (threshold: 70%)")
            print(f"      Depth accuracy   : {m.get('analogy_depth_accuracy', 0):.0%}")
            print(f"      Narration quality: {m.get('mean_narration_quality', 0):.2f}/5")
        elif test_id == 'd1_cluster_coherence':
            print(f"      Intra-cluster sim: {m.get('mean_intra_similarity', 0):.4f} (threshold: >=0.55)")
            print(f"      Inter-cluster sim: {m.get('mean_inter_similarity', 0):.4f} (threshold: <=0.40)")
            print(f"      Silhouette score : {m.get('mean_silhouette', 0):.4f}")
            print(f"      Assignment acc   : {m.get('assignment_accuracy', 0):.0%} (threshold: 75%)")
        elif test_id == 'd1_contradiction_detection':
            print(f"      Precision        : {m.get('precision', 0):.0%} (threshold: 80%)")
            print(f"      Recall           : {m.get('recall', 0):.0%} (threshold: 70%)")
            print(f"      F1               : {m.get('f1', 0):.0%}")
        elif test_id == 'd1_graph_index_consistency':
            print(f"      Nodes checked    : {m.get('nodes_checked')}")
            print(f"      Graph presence   : {m.get('graph_presence_fraction', 0):.0%} (threshold: 100%)")
            print(f"      Index presence   : {m.get('index_presence_fraction', 0):.0%} (threshold: 100%)")
            print(f"      Self-query @1    : {m.get('self_query_rank1_fraction', 0):.0%} (threshold: 95%)")
        elif test_id == 'd1_mission_link_calibration':
            print(f"      Nodes evaluated  : {m.get('nodes_evaluated')}")
            print(f"      Precision        : {m.get('precision', 0):.0%} (threshold: 75%)")
            print(f"      Recall           : {m.get('recall', 0):.0%} (threshold: 70%)")
            print(f"      F1               : {m.get('f1', 0):.0%}")
        print()
    print('-' * 70)
    passed = report['summary']['tests_passed']
    total = report['summary']['tests_total']
    verdict = 'PASS' if report['summary']['PASS'] else 'FAIL'
    print(f'OVERALL D1 VERDICT: {verdict}  ({passed}/{total} tests passed)')
    print('=' * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out', default='results/d1_report.json')
    args = parser.parse_args()
    test_data = {}
    test_metrics = {}
    for test_id, _ in D1_TESTS:
        data = load_result(args.results_dir, test_id)
        test_data[test_id] = data
        test_metrics[test_id] = extract_key_metrics(test_id, data)
    passed = sum(1 for m in test_metrics.values() if m.get('pass') is True)
    total = len(D1_TESTS)
    missing = sum(1 for d in test_data.values() if d.get('_missing'))
    report = {
        'dimension': 'D1 - Knowledge Graph Quality',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'summary': {'tests_total': total, 'tests_passed': passed, 'tests_failed': total - passed - missing, 'tests_missing': missing, 'PASS': passed == total, 'pass_rate': round(passed / total, 3)},
        'tests': test_metrics,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print_report(report)
    print(f"Full report saved to: {args.out}")


if __name__ == '__main__':
    main()