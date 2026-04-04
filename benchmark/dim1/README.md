# Dimension 1 - Knowledge Graph Quality Benchmark

## Revised Position In The Roadmap

Dimension 1 now covers both the original graph-quality core and the two
structural integrity checks added by the revised roadmap.

In the current implementation, D1 answers:
**Is the knowledge graph itself good enough to support the rest of the system,
and is the shared semantic state staying coherent as new knowledge enters?**

## Current Implemented Scope

The Dimension 1 suite now includes seven tests:

| Test | File | What It Measures | Pass Criterion |
|------|------|------------------|----------------|
| T1: Node Quality | `test_d1_node_quality.py` | Are extracted nodes rich conceptual statements rather than keyword noise? | >= 80% nodes score 4+/5 |
| T2: Duplicate Rate | `test_d1_duplicate_rate.py` | Does the dedup pipeline merge near-duplicates after re-ingestion? | < 5% duplicate pairs post-consolidation |
| T3: Edge Accuracy | `test_d1_edge_accuracy.py` | Are extracted relationship types scientifically correct? | >= 70% type accuracy |
| T4: Cluster Coherence | `test_d1_cluster_coherence.py` | Are cluster assignments semantically meaningful? | Intra >= 0.55, inter <= 0.40, assignment >= 75% |
| T5: Contradiction Detection | `test_d1_contradiction_detection.py` | Does the graph distinguish contradiction from compatibility? | Precision >= 80%, Recall >= 70% |
| T6: Graph/Index Consistency | `test_d1_graph_index_consistency.py` | Do ingestion, Reader, and Researcher all write into the same embedding index immediately? | 100% graph/index presence, >= 95% self-query @1 |
| T7: Mission-Link Calibration | `test_d1_mission_link_calibration.py` | Are mission-relevant new nodes linked to the active mission at the right rate? | Precision >= 75%, Recall >= 70% |

## What D1 Covers Now

D1 is now fully aligned with the revised roadmap items that belong in the
graph-quality layer:
- node quality
- duplicate control
- edge accuracy
- cluster coherence
- contradiction detection
- graph/index consistency
- mission-link calibration

What still belongs elsewhere:
- critic-mediated claim quality
- thinker reasoning quality
- insight-buffer promotion quality as an idea-quality benchmark
- end-to-end comparative ablations

## Running

Run the full Dimension 1 suite:

```bash
bash benchmark/dim1/run_d1_all.sh --judge-model llama3.1:70b
```

Run an individual test:

```bash
python benchmark/dim1/test_d1_graph_index_consistency.py   --out benchmark/dim1/results/d1_graph_index_consistency.json
```

```bash
python benchmark/dim1/test_d1_mission_link_calibration.py   --out benchmark/dim1/results/d1_mission_link_calibration.json
```

Aggregate results after the suite runs:

```bash
python benchmark/dim1/report_d1.py   --results-dir benchmark/dim1/results   --out benchmark/dim1/results/d1_report.json
```

## Notes On The New Tests

`test_d1_graph_index_consistency.py` uses three paths intentionally:
- direct `Ingestor.ingest(...)`
- `Reader.absorb_entry(...)` with a synthetic text entry
- `Researcher._research_question(...)` with a deterministic synthetic result

That makes the integration test reproducible without depending on external search.

`test_d1_mission_link_calibration.py` uses a labeled mini-corpus of clearly
relevant and clearly irrelevant statements for a fixed mission. This keeps the
benchmark interpretable while still exercising the real mission-linking path in
the ingestor.
