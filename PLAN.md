# Pivot AutoScientist From Graph-First Memory To Evidence-Grounded Scientist Workbench

## Summary
- Reframe the system so the LLM is the primary reasoning substrate and the graph is a derived ledger of grounded research state.
- Use model prior knowledge transiently during thinking, planning, and hypothesis generation, but do not persist it as settled fact unless supported by external evidence or repeated grounded confirmation.
- Optimize the first refactor for `Thinker`, `Conversation`, and `Researcher`; keep `Dreamer`, `Consolidator`, and the autonomous scheduler as secondary layers that consume the new research state rather than define it.

## Key Changes
- Replace the current “extract everything into graph concepts” center of gravity with a smaller artifact model:
  - `source`
  - `evidence_claim`
  - `question`
  - `hypothesis`
  - `answer`
  - `task`
- Add explicit epistemic status on persisted artifacts:
  - `grounded`
  - `prior`
  - `speculative`
  - `contradicted`
  - `open`
- Refactor ingestion so new text produces source-bound evidence claims and contradictions first, not dense general semantic memory. Mission-irrelevant textbook-like background concepts should usually stay out of permanent storage.
- Refactor reasoning so `Thinker` and `Conversation` operate on a `ScientistWorkspace` built from:
  - mission
  - active questions
  - retrieved grounded evidence
  - current working hypotheses
  - explicit instruction that model background knowledge may be used as prior but must be labeled as such
- Require every nontrivial reasoning output to separate:
  - grounded conclusions
  - prior-based conjectures
  - unresolved gaps
  - next research actions
- Refactor `Researcher` so model prior knowledge is used to propose search angles, discriminating experiments, and candidate mechanisms, then external acquisition is used to confirm, revise, or reject them.
- Keep the graph, but narrow its role to provenance, contradiction structure, mission links, reusable evidence, and unresolved research state. Do not keep spending most of the system budget on exhaustive associative edge generation and broad background clustering.
- Keep `Critic`, but retarget it from “does this graph claim look good?” to “is this conclusion properly grounded, clearly separated from prior knowledge, and scientifically actionable?”
- Defer deep changes to `Dreamer` and `Consolidator` until the new workspace exists. In the first milestone they should not create settled knowledge without grounded evidence.

## Interfaces
- Introduce a persistent artifact schema with fields equivalent to:
  - `id`, `kind`, `text`, `status`, `confidence`, `source_ids`, `mission_relevance`, `created_by`
- Introduce a transient `ScientistWorkspace` containing:
  - mission
  - active evidence pack
  - active priors
  - working hypotheses
  - contradictions
  - next-step agenda
- Introduce a structured `ReasoningResult` carrying:
  - `grounded_claims`
  - `prior_claims`
  - `hypotheses`
  - `open_questions`
  - `next_actions`
  - optional source references
- Keep backward compatibility by reading the current graph as legacy state, but make the new pipeline write the narrower artifact model for all new work.

## Test Plan
- Add new first-class benchmarks for:
  - prior-vs-evidence separation
  - grounded answer faithfulness
  - hypothesis usefulness
  - research-plan quality
  - contradiction preservation under new evidence
  - graph growth efficiency on long texts
- Keep D1-style graph hygiene tests, but treat them as storage-quality checks rather than the main measure of scientific intelligence.
- Reuse D3, D4, and D5 ideas, but score outputs on whether they use prior knowledge productively without laundering it into unsupported fact.
- Add acceptance scenarios where the system starts with little or no stored graph and must still:
  - propose strong subquestions
  - form a plausible working hypothesis
  - distinguish sourced support from background intuition
  - improve after reading new evidence

## Assumptions
- Default architecture choice: the graph becomes a derived evidence ledger, not the scientist’s whole mind.
- Default policy for model prior knowledge: use it transiently; persist it only as `prior` or `hypothesis`, never as settled `grounded` knowledge without support.
- Default product priority: scientist workbench first, autonomous cognitive theater second.
- `Dreamer`/`Consolidator` remain in the repo, but they are not the primary path for the next milestone.
