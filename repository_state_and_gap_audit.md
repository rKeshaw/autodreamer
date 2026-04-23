# Phase-Wise Implementation Plan

Date: 2026-04-20

## Implementation Target

Build an autonomous scientist system where:

- the LLM provides scientific prior knowledge and hypothesis-generation ability
- external reading, research, and experiments inject grounded evidence
- the system maintains strict epistemic separation between prior and grounded claims
- orchestration is reliable, testable, and verifiably wired end to end

This plan is implementation-first and intended to drive coding directly.

## Delivery Principles

- Keep the target stack explicit: LLM prior + grounded evidence, not graph-only cognition.
- Prefer reliability and verification over feature count.
- Wire modules only after they have deterministic checks.
- Preserve backward compatibility where possible, but not at the cost of epistemic correctness.
- Ship by phases with gated exit criteria.

## Phase Map

### Phase 1: Reliability Baseline And Smoke Correctness

Goal:

- establish deterministic baseline behavior and remove known correctness errors

Scope:

- fix LLM smoke cycle mismatch in [smoke_workbench.py](smoke_workbench.py)
- split smoke into clearly scoped checks:
  - structural deterministic smoke
  - optional live-model smoke
- add baseline tests scaffold under tests

Primary files:

- [smoke_workbench.py](smoke_workbench.py)
- [requirements.txt](requirements.txt)
- new tests files

Exit criteria:

- structural smoke passes consistently
- no impossible smoke assertions remain
- pytest runs at least baseline smoke tests

Verification for this phase:

- python smoke_workbench.py --mode structural
- python -m pytest -q

---

### Phase 2: Literature Intake Depth (arXiv Resolution + PDF Path)

Goal:

- ensure intake pipeline can move from query intent to concrete paper artifacts

Scope:

- split reading-list arXiv types:
  - arxiv_query
  - arxiv_paper
- resolve arXiv query entries into specific IDs/URLs before absorb
- implement PDF ingestion path in reader execution flow
- add section segmentation pipeline for full papers

Primary files:

- [reader/reader.py](reader/reader.py)
- [researcher/researcher.py](researcher/researcher.py)
- [ingestion/ingestor.py](ingestion/ingestor.py)

Exit criteria:

- arXiv query item is auto-resolved to paper entry before absorption
- PDF source_type executes real parse branch
- extracted sections are available for downstream ingestion

Verification for this phase:

- add arxiv_query entry and confirm conversion to arxiv_paper
- absorb one known PDF and confirm non-empty section-level extracts
- run regression smoke after changes

---

### Phase 3: Citation-Grade Provenance

Goal:

- make each persisted claim traceable to exact source spans

Scope:

- stop relying on large combined multi-source ingestion blobs for core evidence
- ingest findings per source and per span
- attach provenance fields to claims:
  - source_id
  - source_ref
  - section label
  - span offsets or quote text
  - extraction confidence
- expose citation-ready references for notebook and publication outputs

Primary files:

- [researcher/researcher.py](researcher/researcher.py)
- [reader/reader.py](reader/reader.py)
- [ingestion/ingestor.py](ingestion/ingestor.py)
- [scientist_workspace.py](scientist_workspace.py)

Exit criteria:

- every new evidence claim has source anchor metadata
- publication/notebook can reference exact supporting spans

Verification for this phase:

- inspect a random sample of ingested evidence nodes and verify span metadata presence
- generate a notebook synthesis entry and verify citation anchors are surfaced

---

### Phase 4: Experiment Loop As First-Class Science

Goal:

- upgrade experimenter from fallback script execution to reproducible experiment workflow

Scope:

- add structured experiment plan artifact:
  - hypothesis
  - protocol
  - controls
  - metrics
  - success criteria
- require multiple runs for stochastic tests
- persist run artifacts in per-run folders:
  - code
  - stdout/stderr
  - metrics json
  - environment metadata (versions, seed)
- separate experiment generation from interpretation

Primary files:

- [experimenter/experimenter.py](experimenter/experimenter.py)
- [conductor.py](conductor.py)
- [graph/brain.py](graph/brain.py)

Exit criteria:

- experiments produce reproducible artifact bundles
- verdicts reference measured metrics, not only narrative summaries

Verification for this phase:

- run a controlled known hypothesis twice and compare metric consistency
- verify experiment artifacts written under virtual_lab
- verify graph updates include experiment-derived empirical claims

---

### Phase 5: Cadence Wiring In Conductor

Goal:

- wire daily, weekly, and monthly lifecycle behaviors into actual runtime

Scope:

- daily cadence:
  - morning entry
  - field notes
  - evening entry
- weekly cadence:
  - observer reflection week
  - dead-end summary on pivot trigger
- periodic cadence:
  - synthesis essay
  - publication event
- keep cadence state explicit and persistent

Primary files:

- [conductor.py](conductor.py)
- [observer/observer.py](observer/observer.py)
- [notebook/notebook.py](notebook/notebook.py)
- [notebook/publisher.py](notebook/publisher.py)

Exit criteria:

- all declared cadence hooks are executed by conductor according to schedule
- pivot path updates mission and writes notebook dead-end entry

Verification for this phase:

- run deterministic cycle simulation to at least 35 cycles with mocked LLM calls
- assert each cadence hook execution count matches schedule

---

### Phase 6: Publication Safety And Graph Synchronization

Goal:

- prevent state corruption from publication pruning

Scope:

- publish from frozen snapshot, not live mutable state
- replace direct node deletion in publisher with synchronized prune API
- synchronize any pruning with:
  - embedding index
  - observer agenda references
  - notebook references
  - insight buffer references

Primary files:

- [notebook/publisher.py](notebook/publisher.py)
- [graph/brain.py](graph/brain.py)
- [embedding_index.py](embedding_index.py)
- [observer/observer.py](observer/observer.py)
- [insight_buffer.py](insight_buffer.py)

Exit criteria:

- publication can run without dangling references or index inconsistency

Verification for this phase:

- run publication on test graph snapshot
- validate no missing-node reference errors in observer, notebook, or index lookups

---

### Phase 7: Lab Meeting Depth Upgrade

Goal:

- move lab meeting from one-off personas to persistent review process

Scope:

- add persistent reviewer roster with specialties
- store critique history and unresolved objections
- score closure quality of generated tasks against prior objections

Primary files:

- [critic/lab_meeting.py](critic/lab_meeting.py)
- [observer/observer.py](observer/observer.py)
- [conductor.py](conductor.py)

Exit criteria:

- recurring meetings demonstrate stateful reviewer memory and tracked objection closure

Verification for this phase:

- simulate three consecutive meetings on one hypothesis and verify objection lifecycle transitions

---

### Phase 8: Formal Structural Matching For Isomorphism

Goal:

- reduce dependence on LLM-only isomorphism judgments

Scope:

- define explicit structural representation:
  - entities/roles
  - relations
  - constraints
  - update rules
  - objective
- implement deterministic structural matcher
- use LLM only for abstraction assistance and explanation polishing

Primary files:

- [thinker/isomorphism.py](thinker/isomorphism.py)
- [graph/brain.py](graph/brain.py)

Exit criteria:

- isomorphism edges require deterministic matcher pass before persistence

Verification for this phase:

- run matcher unit tests on known positive and negative structural pairs
- verify no isomorphism edge is written when matcher fails

---

### Phase 9: Runtime Shell And Control Surface

Goal:

- expose system as an operable service rather than module-only runtime

Scope:

- implement service shell aligned with listed dependencies
- provide API endpoints for:
  - cycle control
  - mission control
  - notebook retrieval
  - conversation session
  - status/health
- add background scheduler for cadence execution

Primary files:

- new app entrypoint
- [conversation/conversation.py](conversation/conversation.py)
- [conductor.py](conductor.py)
- [requirements.txt](requirements.txt)

Exit criteria:

- service can start, run cycles, and return state over API

Verification for this phase:

- API smoke test for each endpoint
- scheduler trigger test for at least one cadence event

---

### Phase 10: Evaluation Harness (Dimension-Wise)

Goal:

- measure scientist quality and progress in staged benchmark dimensions

Scope:

- implement benchmark dimensions in this order:
  - Dim 1: prior-vs-grounded separation fidelity
  - Dim 2: literature depth and provenance quality
  - Dim 3: contradiction handling and correction quality
  - Dim 4: experiment quality and reproducibility
  - Dim 5: mission progress and pivot quality
  - Dim 6: synthesis/publication quality
- define metrics and pass thresholds per dimension
- add regression gates in CI or equivalent runner

Primary files:

- new evaluation harness package
- tests and fixtures

Exit criteria:

- each dimension has executable benchmark set and baseline score
- regression run reports pass/fail per dimension

Verification for this phase:

- execute full dimension suite and record baseline report

## Phase Dependencies

- Phase 1 before all others.
- Phase 2 before Phase 3.
- Phase 3 before publication-quality outputs.
- Phase 4 and Phase 5 should complete before serious long-run autonomous tests.
- Phase 6 must complete before enabling periodic publication in production runs.
- Phase 10 should begin in parallel with Phase 3+, then tighten gates after Phase 5.

## Final Wiring Verification Gate

This section is the required end verification to ensure the full system is wired properly.

### A) Conductor Wiring Checklist

Verify in runtime, not only by static inspection:

- reader path executes
- thinker path executes
- researcher path executes
- experimenter path executes on lacks_evidence branch
- anomaly exploitation path executes on contradicted branch
- lab meeting executes on configured cadence
- isomorphism executes during consolidation
- observer reflection executes on weekly cadence
- notebook daily and periodic entries execute on schedule
- publication executes on configured cadence

Pass condition:

- each path is observed at least once in a controlled cycle simulation with deterministic fixtures

### B) State Integrity Checklist

- no dangling node references after publication or pruning
- embedding index remains in sync after state mutations
- observer agenda references valid nodes
- notebook references valid mission and node context

Pass condition:

- integrity check script returns zero errors

### C) Epistemic Integrity Checklist

- grounded claims include citation/provenance anchors
- prior claims are not promoted to grounded without support
- contradicted hypotheses are marked and linked correctly

Pass condition:

- dimension checks for epistemic separation pass

### D) Reproducibility Checklist

- experiment runs emit reproducible artifact bundles
- reruns of same protocol produce bounded metric variance

Pass condition:

- reproducibility tests pass configured tolerance thresholds

### E) End-To-End Acceptance Run

Run a controlled multi-cycle scenario (minimum 35 cycles with deterministic model stubs where needed).

Required observed events:

- at least one hypothesis contradiction
- at least one anomaly-driven new hypothesis
- at least one experiment artifact bundle
- at least one weekly reflection decision
- at least one publication event from snapshot

Final go/no-go rule:

- system is implementation-complete only when A through E all pass

## Deliverable At Plan Completion

Produce a final implementation report containing:

- completed phases and commit references
- benchmark scores by dimension (Dim 1 through Dim 6)
- wiring verification evidence for all gate items
- known residual risks and next-phase recommendations