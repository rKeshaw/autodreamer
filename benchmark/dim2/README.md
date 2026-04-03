# Dimension 2: Dream Cycle Effectiveness

This directory contains the testing and benchmarking suite for Dimension 2 of the AutoScientist, focused on the **Dreamer** component. The Dreamer is responsible for offline graph walks, question generation, analogical insights, and connection building.

## Methodology

Evaluating a cognitive simulation like the Dreamer requires quantitative metrics combined with qualitative judgments. The benchmarking suite measures five specific aspects of the Dream Cycle. Where human judgment is fundamentally required (e.g., evaluating the quality of a generated insight), we utilize an **LLM-as-judge** pattern, comparing the Dreamer's outputs against randomized baselines or thresholds.

All tests output detailed JSON reports into `benchmark/dim2/results/`. An aggregate Markdown report is generated when running the full suite.

## The Tests

### 1. `test_d2_question_quality.py`
**Goal:** Prove the Dreamer generates scientifically useful, precise questions.
**Method:** Runs a dream walk and extracts generated questions. We then pair random nodes in the graph and ask a baseline generator model to form a simple question. The LLM Judge rates both sets on:
- Specificity (1-5)
- Answerability (1-5) 
- Novelty (1-5)
**Pass Criteria:** The average score for the Dreamer questions must exceed the baseline average.

### 2. `test_d2_insight_validity.py`
**Goal:** Validate that claimed "structural" or "isomorphism" insights represent genuine scientific parallels, rather than superficial vocabulary overlaps.
**Method:** Triggers dream cycles to extract insights flagged as deep links. The LLM Judge reviews the source statement, target statement, and narration to vote `genuine` (True) or False. 
**Pass Criteria:** 60% or more of the proposed structured insights must be judged as genuine.

### 3. `test_d2_mission_advance.py`
**Goal:** Verify the system correctly flags moments where the current active research mission makes a substantial leap.
**Method:** Sets a mock mission about genetics, triggers dreaming, and collects steps. Some steps trigger the internal "mission advance detector" (strength > 0.50). The script measures the Pearson correlation between the system's generated strength score and the LLM Judge's assessment of actual relevance.
**Pass Criteria:** Correlation > 0.60 AND the binary flagging accuracy (was it an advance or not?) > 75%.

### 4. `test_d2_walk_diversity.py`
**Goal:** Ensure the Dreamer does not get trapped in loop attractors.
**Method:** A pure quantitative test. Runs 10 cycles of graph walks and tallies node visits. 
**Pass Criteria:** The Dreamer must cover at least 40% of the nodes, and its average revisit rate must remain under 3.0 visits per touched node.

### 5. `test_d2_nrem_effectiveness.py`
**Goal:** Measure the effectiveness of sleep "consolidation", specifically the proximal reinforcement phase that cements important pathways heavily used during recent cycles.
**Method:** Triggers dream walks followed by an NREM pass. Collects edges the system chose to boost (weight > 0.60) versus ones it ignored. The LLM Judge rates the "conceptual scientific importance" of the connections.
**Pass Criteria:** The mean importance score of NREM reinforced edges must be strictly greater than that of unreinforced edges.

---

## Running the Benchmarks

To run the full suite using default judge models (recommended: `llama3.1:70b` for judge and a fast 8B model for gen/baseline):

```bash
cd /kali-linux/home/Keshaw/temp/autoscientist
bash benchmark/dim2/run_d2_all.sh "llama3.1" "llama3.1"
```

To run an individual test:
```bash
python benchmark/dim2/test_d2_question_quality.py --judge-model "llama3.1" --baseline-gen-model "llama3.1"
```

Results are saved to `benchmark/dim2/results/`.
