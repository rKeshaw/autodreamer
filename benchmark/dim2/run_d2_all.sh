#!/bin/bash
# Dimension 2: Dream Cycle Effectiveness Benchmark Runner
#
# Runs all 5 objective tests sequentially.
# Requirements:
# - A running Ollama instance with a strong judge model (default: llama3.1:70b)
# - For question generation, a baseline model (default: llama3.1:8b)

set -e

JUDGE_MODEL=${1:-"llama3.1"}
GEN_MODEL=${2:-"llama3.1"}

echo "================================================================="
echo "  AutoScientist Benchmark — Dimension 2: Dream Cycle"
echo "  Judge Model:    $JUDGE_MODEL"
echo "  Baseline Model: $GEN_MODEL"
echo "================================================================="
echo ""

mkdir -p benchmark/dim2/results

echo "--> [0/5] Prepping Shared Brain Graph..."
python benchmark/dim2/prep_d2_graph.py
echo ""

echo "--> [1/5] Running D2 Test 1: Question Quality..."
python benchmark/dim2/test_d2_question_quality.py \
    --judge-model "$JUDGE_MODEL" \
    --baseline-gen-model "$GEN_MODEL" \
    --out benchmark/dim2/results/d2_question_quality.json
echo ""

echo "--> [2/5] Running D2 Test 2: Insight Validity..."
python benchmark/dim2/test_d2_insight_validity.py \
    --judge-model "$JUDGE_MODEL" \
    --out benchmark/dim2/results/d2_insight_validity.json
echo ""

echo "--> [3/5] Running D2 Test 3: Mission Advance Precision..."
python benchmark/dim2/test_d2_mission_advance.py \
    --judge-model "$JUDGE_MODEL" \
    --out benchmark/dim2/results/d2_mission_advance.json
echo ""

echo "--> [4/5] Running D2 Test 4: Walk Diversity..."
python benchmark/dim2/test_d2_walk_diversity.py \
    --out benchmark/dim2/results/d2_walk_diversity.json \
    --cycles 4
echo ""

echo "--> [5/5] Running D2 Test 5: NREM Effectiveness..."
python benchmark/dim2/test_d2_nrem_effectiveness.py \
    --judge-model "$JUDGE_MODEL" \
    --out benchmark/dim2/results/d2_nrem_effectiveness.json
echo ""

echo "--> Generating aggregate report..."
python benchmark/dim2/report_d2.py

echo ""
echo "================================================================="
echo "  Dimension 2 Benchmark Complete."
echo "  Summary available in: benchmark/dim2/results/report_d2_summary.md"
echo "================================================================="
