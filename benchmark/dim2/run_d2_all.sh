#!/bin/bash
# Dimension 2: Dream Cycle Effectiveness Benchmark Runner
#
# Runs all 7 objective tests sequentially.

set -e

JUDGE_MODEL=${1:-"llama3.1"}
GEN_MODEL=${2:-"llama3.1"}

echo "================================================================="
echo "  AutoScientist Benchmark - Dimension 2: Dream Cycle"
echo "  Judge Model:    $JUDGE_MODEL"
echo "  Baseline Model: $GEN_MODEL"
echo "================================================================="
echo ""

mkdir -p benchmark/dim2/results

echo "--> [0/7] Prepping Shared Brain Graph..."
python benchmark/dim2/prep_d2_graph.py
echo ""

echo "--> [1/7] Running D2 Test 1: Question Quality..."
python benchmark/dim2/test_d2_question_quality.py     --judge-model "$JUDGE_MODEL"     --baseline-gen-model "$GEN_MODEL"     --out benchmark/dim2/results/d2_question_quality.json
echo ""

echo "--> [2/7] Running D2 Test 2: Insight Validity..."
python benchmark/dim2/test_d2_insight_validity.py     --judge-model "$JUDGE_MODEL"     --out benchmark/dim2/results/d2_insight_validity.json
echo ""

echo "--> [3/7] Running D2 Test 3: Mission Advance Precision..."
python benchmark/dim2/test_d2_mission_advance.py     --judge-model "$JUDGE_MODEL"     --out benchmark/dim2/results/d2_mission_advance.json
echo ""

echo "--> [4/7] Running D2 Test 4: Walk Diversity..."
python benchmark/dim2/test_d2_walk_diversity.py     --out benchmark/dim2/results/d2_walk_diversity.json     --cycles 4
echo ""

echo "--> [5/7] Running D2 Test 5: NREM Effectiveness..."
python benchmark/dim2/test_d2_nrem_effectiveness.py     --judge-model "$JUDGE_MODEL"     --out benchmark/dim2/results/d2_nrem_effectiveness.json
echo ""

echo "--> [6/7] Running D2 Test 6: Critic Precision Lift..."
python benchmark/dim2/test_d2_critic_lift.py     --judge-model "$JUDGE_MODEL"     --out benchmark/dim2/results/d2_critic_lift.json
echo ""

echo "--> [7/7] Running D2 Test 7: Deferred Insight Promotion Quality..."
python benchmark/dim2/test_d2_buffer_promotion_quality.py     --judge-model "$JUDGE_MODEL"     --out benchmark/dim2/results/d2_buffer_promotion_quality.json
echo ""

echo "--> Generating aggregate report..."
python benchmark/dim2/report_d2.py

echo ""
echo "================================================================="
echo "  Dimension 2 Benchmark Complete."
echo "  Summary available in: benchmark/dim2/results/report_d2_summary.md"
echo "================================================================="
