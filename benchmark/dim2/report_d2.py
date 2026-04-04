"""
Dimension 2 - Reporting Script
==============================
Aggregates the JSON results from the revised Dimension 2 suite into a
single markdown summary report.
"""

import json
from pathlib import Path


D2_TESTS = [
    ('d2_question_quality.json', 'Question Quality'),
    ('d2_insight_validity.json', 'Insight Validity'),
    ('d2_mission_advance.json', 'Mission Advance Precision'),
    ('d2_walk_diversity.json', 'Walk Diversity'),
    ('d2_nrem_effectiveness.json', 'NREM Effectiveness'),
    ('d2_critic_lift.json', 'Critic Precision Lift'),
    ('d2_buffer_promotion_quality.json', 'Deferred Insight Promotion Quality'),
]


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _key_metric(filename: str, summary: dict) -> str:
    if filename == 'd2_question_quality.json':
        total = summary.get('dream_scores', {}).get('total', 0)
        return f'Dream composite score: {total}'
    if filename == 'd2_insight_validity.json':
        return f"Raw validity: {summary.get('genuine_fraction', 0) * 100:.1f}%"
    if filename == 'd2_mission_advance.json':
        return f"System/judge corr: {summary.get('correlation', 0):.2f}"
    if filename == 'd2_walk_diversity.json':
        return f"Coverage: {summary.get('coverage_fraction', 0) * 100:.1f}%"
    if filename == 'd2_nrem_effectiveness.json':
        return (
            f"Reinf vs unreinf importance: "
            f"{summary.get('mean_importance_reinforced', 0)} > "
            f"{summary.get('mean_importance_unreinforced', 0)}"
        )
    if filename == 'd2_critic_lift.json':
        return (
            f"Accepted validity: {summary.get('accepted_validity', 0) * 100:.1f}% "
            f"(lift {summary.get('precision_lift', 0):+.3f})"
        )
    if filename == 'd2_buffer_promotion_quality.json':
        return (
            f"Promotion rate: {summary.get('promotion_rate', 0) * 100:.1f}% | "
            f"Genuine promotions: {summary.get('genuine_fraction', 0) * 100:.1f}%"
        )
    return 'N/A'


def main():
    results_dir = Path('benchmark/dim2/results')
    if not results_dir.exists():
        print(f'Results directory not found: {results_dir}')
        return

    report_lines = [
        '# Dimension 2: Dream Cycle Effectiveness Benchmark Report',
        '',
        'This report aggregates the revised Dimension 2 suite, including',
        'the original Dreamer-quality tests plus the critic-aware and',
        'insight-buffer-aware extensions added by the new roadmap.',
        '',
        '## Summary',
        '',
    ]

    all_results = []
    passed_tests = 0
    missing_tests = 0

    for filename, label in D2_TESTS:
        data = _load_json(results_dir / filename)
        if data is None:
            all_results.append((filename, label, None))
            missing_tests += 1
            continue
        summary = data.get('summary', {})
        if summary.get('PASS'):
            passed_tests += 1
        all_results.append((filename, label, data))

    report_lines.append(f'**Overall Pipeline Status:** {passed_tests} / {len(D2_TESTS)} Tests Passed')
    report_lines.append('')
    report_lines.append('| Test Component | Status | Key Metric |')
    report_lines.append('| -------------- | ------ | ---------- |')

    for filename, label, data in all_results:
        if data is None:
            report_lines.append(f'| {label} | MISSING | No result file found |')
            continue
        status = 'PASS' if data.get('summary', {}).get('PASS') else 'FAIL'
        metric = _key_metric(filename, data.get('summary', {}))
        report_lines.append(f'| {label} | {status} | {metric} |')

    report_lines.append('')
    report_lines.append('## Detailed Results')
    report_lines.append('')

    for _, label, data in all_results:
        report_lines.append(f'### {label}')
        if data is None:
            report_lines.append('Result file missing.')
            report_lines.append('')
            continue
        status = 'PASS' if data.get('summary', {}).get('PASS') else 'FAIL'
        report_lines.append(f'**Verdict:** {status}')
        report_lines.append('')
        for key, value in data.get('summary', {}).items():
            if key == 'PASS':
                continue
            if isinstance(value, dict):
                report_lines.append(f'- **{key}**:')
                for sub_key, sub_value in value.items():
                    report_lines.append(f'  - {sub_key}: {sub_value}')
            else:
                report_lines.append(f'- **{key}**: {value}')
        report_lines.append('')
        report_lines.append('---')
        report_lines.append('')

    out_path = results_dir / 'report_d2_summary.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report_lines))

    print(f'Dimension 2 aggregate report generated at: {out_path}')
    if missing_tests:
        print(f'Missing result files: {missing_tests}')


if __name__ == '__main__':
    main()
