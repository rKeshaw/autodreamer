# Dimension 2 - Dream Cycle Effectiveness Benchmark

## Revised Position In The Roadmap

Dimension 2 now covers both halves of the revised Dreamer benchmark:
- the raw Dreamer output quality
- the post-critic and post-buffer quality slices added by the new architecture

In the current implementation, D2 answers:
**Does dreaming add useful scientific structure, and do Critic review and delayed promotion improve the precision of dream-derived claims?**

## Current Implemented Scope

The revised D2 suite now includes seven tests:

| Test | File | What It Measures |
|------|------|------------------|
| T1 | `test_d2_question_quality.py` | Scientific usefulness of dream-generated questions |
| T2 | `test_d2_insight_validity.py` | Raw validity of deep dream insights |
| T3 | `test_d2_mission_advance.py` | Calibration of mission-advance detection |
| T4 | `test_d2_walk_diversity.py` | Coverage and revisit behavior during graph walks |
| T5 | `test_d2_nrem_effectiveness.py` | Whether NREM reinforcement strengthens important edges |
| T6 | `test_d2_critic_lift.py` | Raw-vs-critic validity, critic precision lift, and critic false negatives |
| T7 | `test_d2_buffer_promotion_quality.py` | Quality of deferred insights promoted later by the InsightBuffer |

## What Changed Relative To The Earlier Suite

The old suite only covered the raw Dreamer. The revised roadmap required four
additional slices:
1. raw vs post-critic deep-insight comparison
2. critic precision lift on dream insights
3. critic false-negative analysis
4. deferred insight promotion quality from the `InsightBuffer`

Those slices are now implemented through T6 and T7.

## Running

Run the full revised D2 suite:

```bash
bash benchmark/dim2/run_d2_all.sh "llama3.1" "llama3.1"
```

Run only the critic-aware extension:

```bash
python benchmark/dim2/test_d2_critic_lift.py   --judge-model "llama3.1:70b"   --out benchmark/dim2/results/d2_critic_lift.json
```

Run only the delayed-promotion benchmark:

```bash
python benchmark/dim2/test_d2_buffer_promotion_quality.py   --judge-model "llama3.1:70b"   --out benchmark/dim2/results/d2_buffer_promotion_quality.json
```

Aggregate results after the suite runs:

```bash
python benchmark/dim2/report_d2.py
```

## Notes On The New Tests

`test_d2_critic_lift.py` keeps the raw Dreamer generation stage separate from
the System 2 evaluation stage. That lets the benchmark answer the exact revised
roadmap question: does the Critic make deep dream insights more trustworthy?

`test_d2_buffer_promotion_quality.py` uses controlled deferred dream-like cases
plus bridge context so the delayed-promotion mechanism can be tested
reproducibly, without depending on random future article ingestion to trigger a
promotion at the right moment.

## Interpretation

Read the revised D2 results together:
- If T2 is weak but T6 is strong, the Dreamer is creative but the Critic is doing useful filtering.
- If T6 shows no lift, System 2 may be adding latency without improving deep insight quality.
- If T7 is weak, the buffer may be storing plausible near-misses but promoting them unreliably.
- If T4 and T5 are strong while T2 and T6 are weak, the Dreamer may be useful for exploration but not yet for deep analogical insight generation.
