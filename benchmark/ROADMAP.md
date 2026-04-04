# AutoScientist Benchmark Roadmap (Revised)

This roadmap supersedes the earlier benchmark outline that was written before
`Thinker`, `Critic`, and the newer shared-state integrations became core parts
of the runtime.

The original benchmark plan is still directionally correct, especially for:
- graph quality
- dream quality
- research/reading quality
- consolidation quality
- end-to-end behavior

What changed is the architecture. AutoScientist is no longer just:
- ingestion -> graph -> dream -> consolidate

It is now more accurately:
- acquisition (`Reader`, `Researcher`, `Ingestor`)
- generative cognition (`Dreamer`, `Thinker`)
- adversarial scrutiny (`Critic`)
- delayed incubation (`InsightBuffer`)
- monitoring and mission tracking (`Observer`)
- test/execution (`Sandbox`)
- explanation and human interface (`Notebook`, `Conversation`, `GUI`)

That means the benchmark strategy must now evaluate not only output quality, but
also whether the newer cognitive control layers improve precision without
destroying novelty.

## Current Status

Implemented benchmark suites:
- `benchmark/dim1` - Knowledge Graph Quality
- `benchmark/dim2` - Dream Cycle Effectiveness

Not yet implemented, but now part of the revised roadmap:
- `dim3` - Thinker / Structured Reasoning Quality
- `dim4` - Critic / System 2 Quality
- `dim5` - Research & Reading Acquisition
- `dim6` - Consolidation & Insight Buffer Quality
- `dim7` - Observer & Mission Tracking
- `dim8` - Sandbox Hypothesis Testing
- `dim9` - Conversation & Groundedness
- `dim10` - End-to-End + Ablation Studies
- `dim11` - Technical / Operational Reliability

## Dimension Overview

## D1 - Knowledge Graph Quality

Purpose:
Validate the substrate that every other module depends on.

Keep from the original plan:
- node quality
- duplicate rate
- edge accuracy
- cluster coherence
- contradiction detection

Add in the revised plan:
- graph/index consistency after ingestion, reading, and research
- mission-link calibration for newly added nodes
- provenance-aware analysis where useful (reading vs research vs conversation)

Interpretation:
D1 remains valid with only modest revision. It is still the most stable and
least architecture-sensitive benchmark dimension.

## D2 - Dream Cycle Effectiveness

Purpose:
Measure whether dreaming produces useful questions, valid insights, and mission
progress rather than just stylistic novelty.

Keep from the original plan:
- question quality
- insight validity
- mission advance calibration
- walk diversity
- NREM effectiveness

Major revision required:
Dream outputs now need to be evaluated in two layers:
- raw dream outputs
- critic-reviewed / critic-accepted outputs

New D2 metrics should include:
- raw deep-insight validity
- critic-accepted deep-insight validity
- precision lift from critic review
- false-negative rate of critic review on valuable dream insights
- quality of dream insights deferred to the `InsightBuffer` and promoted later

Interpretation:
D2 is still valid, but the current suite should be treated as only the first
half of the Dreamer benchmark until the critic-aware slices are added.

## D3 - Thinker / Structured Reasoning Quality

Purpose:
Evaluate deliberate reasoning, decomposition, and answer-oriented thinking.

Core metrics:
- pattern selection accuracy
- reductive sub-question sufficiency
- insight actionability
- cross-round coherence
- mission-answer specificity improvement across thinking rounds
- usefulness of thinker-generated sub-questions for later research

Why it is now separate:
`Thinker` is no longer a side feature. It is a core cognitive module and should
not be hidden inside end-to-end tests.

## D4 - Critic / System 2 Quality

Purpose:
Measure whether the critic improves the system's claims rather than just adding
latency and friction.

Core metrics:
- review activation calibration
- verdict accuracy (`accept` / `refine` / `reject` / `defer`)
- novelty-check accuracy
- refinement quality lift
- defer quality
- bypass safety for low-stakes outputs

This is now essential because the critic changes both `Dreamer` and `Thinker`
behavior in meaningful ways.

## D5 - Research & Reading Acquisition

Purpose:
Measure the quality of external knowledge acquisition.

Core metrics:
- retrieval relevance
- extraction signal-to-noise ratio
- resolution rate for agenda questions
- reading list quality
- index freshness after ingestion
- thinker-question vs observer-question retrieval quality

## D6 - Consolidation & Insight Buffer Quality

Purpose:
Evaluate whether the system deepens and cleans its knowledge over time.

Core metrics:
- synthesis genuineness
- abstraction quality
- gap inference accuracy
- decay calibration
- delayed insight promotion quality
- time-to-promotion behavior for the insight buffer

## D7 - Observer & Mission Tracking

Purpose:
Measure whether self-monitoring highlights the right events.

Core metrics:
- emergence precision / recall
- question deduplication accuracy
- incubation-age calibration
- mission advance recall
- mission-link coverage on high-value nodes

## D8 - Sandbox Hypothesis Testing

Purpose:
Evaluate testability judgment, code generation, execution success, and graph
integration of empirical results.

Core metrics:
- testability classification accuracy
- code execution success rate
- verdict calibration against known ground truth cases
- graph integration quality
- whether critic/thinker improve the quality of hypotheses sent to sandbox

## D9 - Conversation & Groundedness

Purpose:
Evaluate the human-facing interface for faithfulness and useful interaction.

Core metrics:
- groundedness against graph content
- retrieved-node usefulness
- idea-ingestion fidelity
- multi-turn coherence
- running-hypothesis grounding

## D10 - End-to-End + Ablations

Purpose:
Measure the whole system over time and determine which modules matter.

Revised ablation set:
- full pipeline
- no Dreamer
- no Thinker
- no Critic
- no Dreamer + no Thinker
- naive ingestion baseline

Key metrics:
- knowledge accumulation trajectory
- mission progress over cycles
- self-correction rate
- mission suspension / resumption behavior

## D11 - Technical & Operational Reliability

Purpose:
Measure whether the system is operationally trustworthy at scale.

Core metrics:
- phase throughput vs graph size
- LLM call counts by phase
- RAM footprint
- persistence reliability
- concurrent-access stability
- recovery from interruption
- graph/index consistency after crash/restart
- GUI-path vs scheduler-path parity

## Recommended Evaluation Order

Given the current state of the repo, the practical implementation order should be:

1. Finish revising `dim1`
2. Finish revising `dim2`
3. Build `dim3` for `Thinker`
4. Build `dim4` for `Critic`
5. Build `dim10` ablations early, even before every other module suite exists
6. Fill in the remaining module-specific dimensions

Reason:
The largest new unknowns introduced by the recent architecture changes are:
- whether `Thinker` adds real value
- whether `Critic` improves precision enough to justify its cost
- whether the full system is better than simpler ablations

## Notes for the Existing Suites

### `dim1`
Treat the current implementation as still valid and worth keeping.
It maps cleanly to the revised roadmap.

### `dim2`
Treat the current implementation as partially complete.
It still measures raw dream quality well, but it does not fully measure the
new critic-mediated dream pipeline yet. The revised D2 target is:
- raw dream metrics
- post-critic dream metrics
- precision-lift analysis

## Reporting Structure

The final benchmark report should still preserve the original 3-tier structure:

- Tier 1 - Core claims from the README
- Tier 2 - Comparative performance vs ablations/baselines
- Tier 3 - Failure mode catalog

That structure still holds. The major revision is simply that `Thinker` and
`Critic` are now part of the core claims rather than optional extras.
