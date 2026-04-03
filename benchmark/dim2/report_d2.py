"""
Dimension 2 — Reporting Script
==============================
Aggregates the JSON results from the 5 Dimension 2 tests into a single
markdown summary report.
"""

import json
import glob
from pathlib import Path

def main():
    results_dir = Path("benchmark/dim2/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    json_files = glob.glob(str(results_dir / "*.json"))
    if not json_files:
        print("No JSON result files found in benchmark/dim2/results/")
        return

    report_lines = [
        "# Dimension 2: Dream Cycle Effectiveness Benchmark Report\n",
        "This report aggregates the results of the 5 objective tests designed to evaluate ",
        "the generative and organizational capabilities of the Dreamer module.\n",
        "## Summary\n"
    ]

    all_results = {}
    passed_tests = 0

    for jpath in sorted(json_files):
        with open(jpath, 'r') as f:
            data = json.load(f)
            test_name = data.get("test", Path(jpath).name)
            summary = data.get("summary", {})
            passed = summary.get("PASS", False)
            all_results[test_name] = {
                "summary": summary,
                "passed": passed
            }
            if passed:
                passed_tests += 1

    report_lines.append(f"**Overall Pipeline Status:** {passed_tests} / {len(all_results)} Tests Passed\n")
    report_lines.append("| Test Component | Status | Key Metric |")
    report_lines.append("| -------------- | ------ | ---------- |")

    # Build summary table
    for test_name, res in all_results.items():
        status = "✅ PASS" if res["passed"] else "❌ FAIL"
        summary = res["summary"]
        
        metric = ""
        if "Question Quality" in test_name:
             t = summary.get('dream_scores', {}).get('total', 0)
             metric = f"Dream score: {t} vs Baseline"
        elif "Insight Validity" in test_name:
             f = summary.get('genuine_fraction', 0)
             metric = f"Validity rate: {f*100:.1f}%"
        elif "Mission Advance" in test_name:
             c = summary.get('correlation', 0)
             metric = f"Sys/Judge corr: {c:.2f}"
        elif "Walk Diversity" in test_name:
             cov = summary.get('coverage_fraction', 0)
             metric = f"Coverage: {cov*100:.1f}%"
        elif "NREM Effectiveness" in test_name:
             mr = summary.get('mean_importance_reinforced', 0)
             mu = summary.get('mean_importance_unreinforced', 0)
             metric = f"Reinforced imp > Unreinforced: {mr} > {mu}"
             
        report_lines.append(f"| {test_name} | {status} | {metric} |")

    report_lines.append("\n## Detailed Results\n")

    for test_name, res in all_results.items():
        report_lines.append(f"### {test_name}")
        status = "✅ PASS" if res["passed"] else "❌ FAIL"
        report_lines.append(f"**Verdict:** {status}\n")
        
        for k, v in res["summary"].items():
            if k == "PASS": continue
            if isinstance(v, dict):
                 report_lines.append(f"- **{k}**:")
                 for sub_k, sub_v in v.items():
                      report_lines.append(f"  - {sub_k}: {sub_v}")
            else:
                 report_lines.append(f"- **{k}**: {v}")
        report_lines.append("\n---\n")

    out_path = results_dir / "report_d2_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Dimension 2 aggregate report generated at: {out_path}")

if __name__ == "__main__":
    main()
