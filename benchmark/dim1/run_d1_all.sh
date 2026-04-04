#!/usr/bin/env bash
# =============================================================================
# run_d1_all.sh - Run all Dimension 1 tests sequentially
# =============================================================================
# Usage:
#   cd /path/to/autoscientist
#   bash benchmark/dim1/run_d1_all.sh --judge-model llama3.1:70b
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

JUDGE_MODEL="llama3.1:70b"
SKIP_INGEST=""
RESULTS_DIR="benchmark/dim1/results"

while [[ $# -gt 0 ]]; do
  case $1 in
    --judge-model)
      JUDGE_MODEL="$2"; shift 2 ;;
    --skip-ingest)
      SKIP_INGEST="--skip-ingest"; shift ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " AutoScientist Benchmark - Dimension 1: Knowledge Graph Quality"
echo " Judge model : $JUDGE_MODEL"
echo " Results dir : $RESULTS_DIR"
echo "============================================================"
echo ""

echo ">>> [1/7] Node Quality"
python benchmark/dim1/test_d1_node_quality.py   --judge-model "$JUDGE_MODEL"   --out "$RESULTS_DIR/d1_node_quality.json"   --cache "$RESULTS_DIR/d1_node_cache.json"   $SKIP_INGEST
echo ""

echo ">>> [2/7] Duplicate Rate"
python benchmark/dim1/test_d1_duplicate_rate.py   --judge-model "$JUDGE_MODEL"   --out "$RESULTS_DIR/d1_duplicate_rate.json"
echo ""

echo ">>> [3/7] Edge Accuracy"
python benchmark/dim1/test_d1_edge_accuracy.py   --judge-model "$JUDGE_MODEL"   --out "$RESULTS_DIR/d1_edge_accuracy.json"
echo ""

echo ">>> [4/7] Cluster Coherence"
python benchmark/dim1/test_d1_cluster_coherence.py   --judge-model "$JUDGE_MODEL"   --out "$RESULTS_DIR/d1_cluster_coherence.json"   $SKIP_INGEST
echo ""

echo ">>> [5/7] Contradiction Detection"
python benchmark/dim1/test_d1_contradiction_detection.py   --judge-model "$JUDGE_MODEL"   --out "$RESULTS_DIR/d1_contradiction_detection.json"
echo ""

echo ">>> [6/7] Graph/Index Consistency"
python benchmark/dim1/test_d1_graph_index_consistency.py   --out "$RESULTS_DIR/d1_graph_index_consistency.json"
echo ""

echo ">>> [7/7] Mission-Link Calibration"
python benchmark/dim1/test_d1_mission_link_calibration.py   --out "$RESULTS_DIR/d1_mission_link_calibration.json"
echo ""

echo ">>> Generating Dimension 1 aggregate report..."
python benchmark/dim1/report_d1.py   --results-dir "$RESULTS_DIR"   --out "$RESULTS_DIR/d1_report.json"

echo ""
echo "============================================================"
echo " Done. All results in: $RESULTS_DIR/"
echo "============================================================"
