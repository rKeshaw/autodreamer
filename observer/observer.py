import json
import time
import numpy as np
from dataclasses import dataclass, field
from graph.brain import Brain, EdgeType, NodeStatus, NodeType
from dreamer.dreamer import DreamLog, DreamStep
from config import THRESHOLDS
from embedding import embed as shared_embed
from persistence import atomic_write_json
from llm_utils import llm_call
from scientist_workspace import ArtifactStatus
from scientific_rigor import deterministic_progress_stage, normalize_text

# ── Config ────────────────────────────────────────────────────────────────────

WEAK_EDGE_REPEAT_THRESHOLD = 3
QUESTION_REPEAT_THRESHOLD  = 2
COHERENCE_THRESHOLD        = THRESHOLDS.COHERENCE
INCUBATION_EMERGENCE_AGE   = 5
SIMILARITY_HIGH            = THRESHOLDS.QUESTION_DEDUP_HIGH
SIMILARITY_MID             = THRESHOLDS.QUESTION_DEDUP_LOW
MAX_EMERGENCES_PER_TYPE    = 2
EMERGENCE_COOLDOWN_HOURS   = 24
MAX_UNRESOLVED_AGENDA      = 60
MAX_RESOLVED_AGENDA        = 30
STALE_INCUBATION_AGE       = 10
STALE_PRIORITY_CUTOFF      = 0.25

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgendaItem:
    text:             str
    item_type:        str   = "question"
    source_step:      int   = 0
    dream_cycle:      int   = 0
    count:            int   = 1
    resolved:         bool  = False
    resolution_grade: str   = ""
    priority:         float = 0.5
    incubation_age:   int   = 0
    node_id:          str   = ""
    partial_leads:    list  = field(default_factory=list)
    answer_node_id:   str   = ""

@dataclass
class MissionAdvance:
    node_id:     str
    explanation: str
    strength:    float
    cycle:       int
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self):
        return self.__dict__

@dataclass
class EmergenceSignal:
    signal:    str
    type:      str
    cycle:     int
    node_ids:  list  = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return self.__dict__

# ── Prompts ───────────────────────────────────────────────────────────────────

QUESTION_SIMILAR_PROMPT = """
Are these two questions asking the SAME THING, even if worded differently?

"Same" means: if one were fully answered, the other would also be fully answered.
"Different" means: they could have separate answers, even if they cover related topics.

Q1: {q1}
Q2: {q2}

Examples:
- "What role does X play in Y?" vs "How does X influence Y?" → YES (same question)
- "What role does X play in Y?" vs "Is X or Z more important for Y?" → NO (related but distinct)

Respond ONLY "yes" or "no".
"""

EMERGENCE_PROMPT = """
You are monitoring the dream log of a scientific mind.

Central question: "{mission}"
Event type: {type}
Detail: {detail}

Write ONE short, sharp sentence (under 20 words) describing what is forming.
Be precise, not dramatic.
"""

COHERENCE_PROMPT = """
Rate the conceptual coherence of this connection between ideas from different domains.

Idea A: {node_a}
Idea B: {node_b}
Connection: {narration}
Insight depth claimed: {depth}

Scoring rubric (0.0 to 1.0):

- 0.0-0.2: Incoherent — the claimed connection doesn't actually hold, or is a stretch.
  Example: "Stars are hot" → "Hot dogs are food" via shared word "hot" = 0.1 (spurious).

- 0.2-0.4: Surface — shared vocabulary/theme but no deeper structure.
  Example: "Networks of neurons" → "Social networks" via "both are networks" = 0.3.

- 0.4-0.6: Moderate — a plausible analogical mapping with some structural overlap.
  Example: "Ant colony optimization" → "Neural network learning" via shared emergence = 0.5.

- 0.6-0.8: Strong structural — clear relational correspondence (A:B::X:Y).
  Example: "Natural selection" → "Gradient descent" via shared optimization-via-iterative-selection = 0.7.

- 0.8-1.0: Isomorphic — formal mathematical or logical equivalence.
  Example: "Heat equation" → "Diffusion equation" via identical PDEs = 0.95.

Cross-check the claimed depth:
- If depth="isomorphism" but the connection is really just shared vocabulary → score LOW (0.1-0.3).
- If depth="surface" but there's actually a deep structural parallel → score HIGHER than surface would suggest.

Respond with ONLY a float 0.0 to 1.0. No other text.
"""

HYPOTHESIS_ADVANCE_PROMPT = """
Hypothesis: "{hypothesis}"
New finding: "{candidate}"
Explanation: "{explanation}"

Has this meaningfully advanced the hypothesis?
Respond ONLY "yes" or "no".
"""

MISSION_SUMMARY_PROMPT = """
Central research question: "{mission}"

These are the most significant advances toward this question made so far:
{advances}

Grounded evidence currently available:
{grounded_evidence}

Working hypotheses and prior claims:
{hypotheses}

These are the main open tensions still unresolved:
{contradictions}

Current next tasks:
{next_tasks}

In 3-4 sentences, summarize how close the mind is to answering the central question.
What is the current best partial answer? What is the key remaining gap?
Write like a scientist assessing their own progress.
"""


PIVOT_PROMPT = """
You are the scientific orchestrator evaluating the overarching research agenda.

Our current mission was:
"{old_mission}"

We have encountered sweeping failure. The ratio of contradicted or dead-end 
hypotheses to successful ones is overwhelming. Here are the most prominent 
anomalies and contradicted ideas we found instead:
{anomalies}

Your job is to generate a COMPLETELY NEW MISSION that abandons the old paradigm 
entirely and specifically leans into these anomalies. Treat the failures as the 
starting point of a new field. We must pivot.

Return the new mission statement exactly:
{{
  "new_mission_statement": "A single bold central research question.",
  "justification": "Why this pivot makes our failures irrelevant."
}}
"""

# ── Observer ──────────────────────────────────────────────────────────────────

class Observer:
    def __init__(self, brain: Brain):
        self.brain                 = brain
        self.agenda: list[AgendaItem]          = []
        self.agenda_embeddings: list           = []
        self.emergence_feed: list[EmergenceSignal] = []
        self.mission_advances: list[MissionAdvance]= []
        self.edge_traversal_counts: dict       = {}
        self.cycle_count                       = 0
        self._cycle_emergence_counts: dict     = {}
        self._emergence_last_fired: dict       = {}
        self.lab_meeting = None

    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        return llm_call(prompt, temperature=temperature, role="precise")

    def _embed(self, text: str) -> np.ndarray:
        return shared_embed(text)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _mission_text(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text or "").lower().split())

    def _blocking_objection_count(self, node_id: str) -> int:
        if not self.lab_meeting or not hasattr(self.lab_meeting, "get_hypothesis_state"):
            return 0
        try:
            state = self.lab_meeting.get_hypothesis_state(node_id)
        except Exception:
            return 0
        return int(state.get("blocking_objection_count", 0) or 0)

    def _node_supports_mission_advance(self, node_id: str, strength: float) -> bool:
        node = self.brain.get_node(node_id)
        if not node:
            return True
        node_type = node.get("node_type")
        epistemic = node.get("epistemic_status")
        status = node.get("status")
        if epistemic in {
            ArtifactStatus.CONTRADICTED.value,
            ArtifactStatus.SPECULATIVE.value,
            ArtifactStatus.LACKS_EVIDENCE.value,
        }:
            return False
        if node_type in {
            NodeType.EVIDENCE_CLAIM.value,
            NodeType.EMPIRICAL.value,
            NodeType.ANSWER.value,
        }:
            return epistemic == ArtifactStatus.GROUNDED.value
        if node_type == NodeType.HYPOTHESIS.value:
            if status != NodeStatus.UNCERTAIN.value:
                return False
            if self._blocking_objection_count(node_id) > 0:
                return False
            incoming = list(self.brain.graph.in_edges(node_id, data=True))
            return any(
                self.brain.get_node(source_id) and
                self.brain.get_node(source_id).get("epistemic_status") == ArtifactStatus.GROUNDED.value and
                edge.get("type") in {
                    EdgeType.CONFIRMED_BY.value,
                    EdgeType.SUPPORTS.value,
                    EdgeType.EMPIRICALLY_TESTED.value,
                }
                for source_id, _, edge in incoming
            ) or strength < 0.7
        return strength < 0.7

    def _render_emergence_signal(self, type: str, detail: str) -> str:
        detail = normalize_text(detail)
        templates = {
            "mission_advance": "A grounded result materially narrows the mission.",
            "hypothesis_advanced": "A hypothesis gained mechanism-specific support.",
            "recurring_question": f"Recurring unresolved question: {detail[:96]}",
            "long_incubation": f"Persistent unresolved issue: {detail[:96]}",
            "repeated_weak_edge": "A weak association is being revisited without strong support.",
            "cross_cluster_insight": f"Cross-cluster mapping survived coherence screening: {detail[:96]}",
            "contradiction_circled": f"An active contradiction remains unresolved: {detail[:96]}",
            "incubation_resolved": f"Long-incubated question received a grounded answer: {detail[:96]}",
        }
        return templates.get(type, detail[:120] or "Scientific state changed.")

    # ── Similarity ────────────────────────────────────────────────────────────

    def _items_similar(self, q1: str, emb1: np.ndarray,
                       q2: str, emb2: np.ndarray) -> bool:
        sim = self._cosine(emb1, emb2)
        if sim > SIMILARITY_HIGH:
            return True
        if sim < SIMILARITY_MID:
            return False
        raw = self._llm(QUESTION_SIMILAR_PROMPT.format(q1=q1, q2=q2), temperature=0.1)
        return raw.lower().startswith('yes')

    # ── Agenda ────────────────────────────────────────────────────────────────

    def add_to_agenda(self, text: str, item_type: str = "question",
                      cycle: int = 0, step: int = 0,
                      node_id: str = "") -> AgendaItem:
        new_emb = self._embed(text)
        for i, existing in enumerate(self.agenda):
            if self._items_similar(
                text, new_emb,
                existing.text, self.agenda_embeddings[i]
            ):
                existing.count   += 1
                existing.priority = min(1.0, existing.priority + 0.15)
                if existing.count >= QUESTION_REPEAT_THRESHOLD:
                    self._flag_emergence(
                        type   = "recurring_question",
                        detail = (f"Question recurred {existing.count}x: "
                                  f"{existing.text}"),
                        cycle  = cycle
                    )
                return existing

        item = AgendaItem(
            text=text, item_type=item_type,
            source_step=step, dream_cycle=cycle, node_id=node_id
        )
        self.agenda.append(item)
        self.agenda_embeddings.append(new_emb)
        return item

    def get_prioritized_questions(self, n: int = 10) -> list:
        unresolved = [i for i in self.agenda if not i.resolved]
        return sorted(unresolved, key=lambda i: i.priority, reverse=True)[:n]

    def _prune_agenda(self):
        """Prevent agenda bloat by keeping only high-value active items."""
        if not self.agenda:
            return

        items = list(zip(self.agenda, self.agenda_embeddings))

        # Mark stale unresolved items as resolved when they have remained low
        # priority for many cycles without producing leads.
        stale_marked = 0
        for item, _ in items:
            if item.resolved:
                continue
            if (item.incubation_age >= STALE_INCUBATION_AGE and
                    item.priority < STALE_PRIORITY_CUTOFF and
                    not item.partial_leads):
                item.resolved = True
                item.resolution_grade = "stale_pruned"
                stale_marked += 1

        unresolved = [p for p in items if not p[0].resolved]
        resolved = [p for p in items if p[0].resolved]

        dropped = 0
        if len(unresolved) > MAX_UNRESOLVED_AGENDA:
            unresolved = sorted(
                unresolved,
                key=lambda p: (p[0].priority, p[0].count, -p[0].incubation_age),
                reverse=True,
            )
            dropped = len(unresolved) - MAX_UNRESOLVED_AGENDA
            unresolved = unresolved[:MAX_UNRESOLVED_AGENDA]

        if len(resolved) > MAX_RESOLVED_AGENDA:
            resolved = sorted(
                resolved,
                key=lambda p: (p[0].priority, p[0].count),
                reverse=True,
            )[:MAX_RESOLVED_AGENDA]

        new_items = unresolved + resolved
        self.agenda = [item for item, _ in new_items]
        self.agenda_embeddings = [emb for _, emb in new_items]

        if stale_marked or dropped:
            print(
                f"   Agenda pruned: stale_marked={stale_marked}, "
                f"dropped_unresolved={dropped}"
            )

    def record_answer(self, question_text: str, answer_node_id: str,
                      explanation: str, grade: str = "strong"):
        for item in self.agenda:
            if item.text != question_text:
                continue
            if grade == "strong":
                item.resolved         = True
                item.resolution_grade = grade
                item.answer_node_id   = answer_node_id
                print(f"  ✓ Resolved [{grade}]: {question_text}")
                if item.incubation_age >= 2:
                    self._flag_emergence(
                        type     = "incubation_resolved",
                        detail   = (f"After {item.incubation_age} cycles: "
                                    f"{question_text}"),
                        cycle    = self.cycle_count,
                        node_ids = [answer_node_id]
                    )
                if item.item_type == "hypothesis":
                    node = self.brain.get_node(answer_node_id)
                    if node:
                        adv = self._llm(HYPOTHESIS_ADVANCE_PROMPT.format(
                            hypothesis  = item.text,
                            candidate   = node['statement'],
                            explanation = explanation
                        ), temperature=0.1)
                        if adv.lower().startswith('yes'):
                            self._flag_emergence(
                                type     = "hypothesis_advanced",
                                detail   = f"Hypothesis advanced: {item.text}",
                                cycle    = self.cycle_count,
                                node_ids = [answer_node_id]
                            )
            elif grade == "partial":
                if answer_node_id not in item.partial_leads:
                    item.partial_leads.append(answer_node_id)
                item.priority = min(1.0, item.priority + 0.1)
                print(f"  ~ Partial lead: {question_text}")
            break

    # ── Mission tracking ──────────────────────────────────────────────────────

    def record_mission_advance(self, node_id: str, explanation: str,
                               strength: float):
        """Record a significant advance toward the central question."""
        if not self._node_supports_mission_advance(node_id, strength):
            print("  · Mission advance rejected: evidence not yet strong enough.")
            return
        normalized_explanation = self._normalize_text(explanation)
        for advance in reversed(self.mission_advances[-12:]):
            if advance.cycle != self.cycle_count:
                continue
            if advance.node_id != node_id:
                continue
            if self._normalize_text(advance.explanation) != normalized_explanation:
                continue
            if strength > advance.strength:
                advance.strength = strength
                advance.explanation = explanation
            return
        advance = MissionAdvance(
            node_id     = node_id,
            explanation = explanation,
            strength    = strength,
            cycle       = self.cycle_count
        )
        self.mission_advances.append(advance)
        print(f"  ★ Mission advance recorded (strength={strength:.2f})")
        self.brain.spike_dopamine(0.3)

        if strength > 0.75:
            self._flag_emergence(
                type     = "mission_advance",
                detail   = (f"Strong advance toward central question "
                            f"(strength={strength:.2f}): {explanation}"),
                cycle    = self.cycle_count,
                node_ids = [node_id]
            )

    def get_mission_progress_summary(self) -> str:
        """Generate a summary of progress toward the central question."""
        mission = self.brain.get_mission()
        if not mission:
            return "No central question set."

        top_advances = sorted(
            self.mission_advances,
            key=lambda a: a.strength, reverse=True
        )[:5]

        workspace = self.brain.build_workspace()
        grounded_evidence = "\n".join(
            node.prompt_line() for node in workspace.grounded_evidence[:5]
        ) or "none yet"
        hypotheses = "\n".join(
            node.prompt_line()
            for node in (workspace.working_hypotheses + workspace.prior_claims)[:5]
        ) or "none yet"
        contradictions = "\n".join(
            f"- {item}" for item in workspace.contradictions[:4]
        ) or "none"
        next_tasks = "\n".join(
            node.prompt_line() for node in workspace.next_tasks[:4]
        ) or "none"
        blocker_count = sum(
            self._blocking_objection_count(node.id)
            for node in workspace.working_hypotheses[:4]
        )
        stage = deterministic_progress_stage(
            grounded_count=len(workspace.grounded_evidence),
            supported_hypotheses=len(workspace.working_hypotheses),
            blocker_count=blocker_count,
        )
        advances_text = "\n".join(
            f"- ({a.strength:.2f}) {a.explanation}"
            for a in top_advances
        ) or "- none yet"
        return "\n".join([
            f"Mission: {mission['question']}",
            f"Stage: {stage}",
            "Best grounded evidence:",
            grounded_evidence if grounded_evidence != "none yet" else "- none yet",
            "Current supported hypotheses:",
            hypotheses if hypotheses != "none yet" else "- none yet",
            "Material advances:",
            advances_text,
            "Open tensions:",
            contradictions if contradictions != "none" else "- none",
            "Next tasks:",
            next_tasks if next_tasks != "none" else "- none",
            f"Review blockers: {blocker_count}",
        ])

    # ── Incubation ────────────────────────────────────────────────────────────

    def increment_incubation(self):
        for item in self.agenda:
            if not item.resolved:
                item.incubation_age += 1
                item.priority = min(1.0,
                    item.priority + (item.incubation_age * 0.02))
                if item.node_id and self.brain.get_node(item.node_id):
                    self.brain.update_node(
                        item.node_id,
                        incubation_age=item.incubation_age)
                if item.incubation_age == INCUBATION_EMERGENCE_AGE:
                    self._flag_emergence(
                        type   = "long_incubation",
                        detail = (f"Unresolved {item.incubation_age} cycles: "
                                  f"{item.text}"),
                        cycle  = self.cycle_count
                    )

    # ── Edge traversal ────────────────────────────────────────────────────────

    def _track_edge_traversals(self, steps: list, cycle: int):
        for step in steps:
            key = (step.from_id, step.to_id)
            self.edge_traversal_counts[key] = \
                self.edge_traversal_counts.get(key, 0) + 1
            if self.edge_traversal_counts[key] >= WEAK_EDGE_REPEAT_THRESHOLD:
                edge = self.brain.get_edge(step.from_id, step.to_id)
                if edge and edge.get('type') == EdgeType.ASSOCIATED.value:
                    self._flag_emergence(
                        type     = "repeated_weak_edge",
                        detail   = (f"Weak edge traversed "
                                    f"{self.edge_traversal_counts[key]}x"),
                        cycle    = cycle,
                        node_ids = [step.from_id, step.to_id]
                    )
                    self.brain.update_edge(
                        step.from_id, step.to_id,
                        weight=min(0.9, edge.get('weight', 0.3) + 0.1))

    # ── Contradiction monitoring ──────────────────────────────────────────────

    def _check_contradictions(self, cycle: int):
        edges = list(self.brain.graph.edges(data=True))
        for u, v, data in edges:
            if data.get('type') != EdgeType.CONTRADICTS.value:
                continue
            nu = self.brain.get_node(u)
            nv = self.brain.get_node(v)
            if not nu or not nv:
                continue
            if (time.time() - nu.get('activated_at', 0) < 3600 and
                    time.time() - nv.get('activated_at', 0) < 3600):
                self._flag_emergence(
                    type     = "contradiction_circled",
                    detail   = (f"Contradiction circled: "
                                f"{nu['statement']} vs "
                                f"{nv['statement']}"),
                    cycle    = cycle,
                    node_ids = [u, v]
                )

    # ── Cross-cluster insights ────────────────────────────────────────────────

    def _check_cross_cluster_insights(self, steps: list, cycle: int):
        for step in steps:
            if not step.is_insight:
                continue
            nf = self.brain.get_node(step.from_id)
            nt = self.brain.get_node(step.to_id)
            if not nf or not nt:
                continue
            if nf.get('cluster') == nt.get('cluster'):
                continue
            try:
                score = float(self._llm(COHERENCE_PROMPT.format(
                    node_a   = nf['statement'],
                    node_b   = nt['statement'],
                    narration= step.narration,
                    depth    = step.insight_depth
                ), temperature=0.2))
            except ValueError:
                continue
            if score >= COHERENCE_THRESHOLD:
                self._flag_emergence(
                    type     = "cross_cluster_insight",
                    detail   = (f"[{nf['cluster']} ↔ {nt['cluster']}] "
                                f"depth={step.insight_depth} "
                                f"score={score:.2f}: {step.narration}"),
                    cycle    = cycle,
                    node_ids = [step.from_id, step.to_id]
                )

    # ── Emergence ─────────────────────────────────────────────────────────────

    def _flag_emergence(self, type: str, detail: str,
                        cycle: int, node_ids: list = None):
        last_fired = self._emergence_last_fired.get(type, 0)
        if time.time() - last_fired < EMERGENCE_COOLDOWN_HOURS * 3600:
            return
        count = self._cycle_emergence_counts.get(type, 0)
        if count >= MAX_EMERGENCES_PER_TYPE:
            return
        signal_text = self._render_emergence_signal(type, detail)
        signal = EmergenceSignal(
            signal=signal_text, type=type,
            cycle=cycle, node_ids=node_ids or []
        )
        self.emergence_feed.append(signal)
        self._cycle_emergence_counts[type] = count + 1
        self._emergence_last_fired[type] = time.time()
        print(f"\n  ◆ EMERGENCE [{type}]: {signal_text}")

    # ── Main observe ──────────────────────────────────────────────────────────

    def _observe_internal(self, log: DreamLog, *, increment_cycle: bool,
                          increment_incubation: bool):
        if increment_cycle:
            self.cycle_count += 1
            self._cycle_emergence_counts = {}
        cycle = self.cycle_count
        print(f"\n── Observer cycle {cycle} ──")

        # ingest questions
        for i, qtext in enumerate(log.questions):
            self.add_to_agenda(text=qtext, item_type="question",
                               cycle=cycle, step=i)

        # process answer matches
        for answer in log.answers:
            self.record_answer(
                question_text  = answer['question'],
                answer_node_id = answer.get('node', ''),
                explanation    = answer['explanation'],
                grade          = answer['grade']
            )

        # process mission advances
        for adv in log.mission_advances:
            self.record_mission_advance(
                adv['node'], adv['explanation'], adv['strength'])

        self._track_edge_traversals(log.steps, cycle)
        self._check_contradictions(cycle)
        self._check_cross_cluster_insights(log.steps, cycle)
        if increment_incubation:
            self.increment_incubation()
        self._prune_agenda()

        resolved = sum(1 for i in self.agenda if i.resolved)
        print(f"   Agenda: {len(self.agenda)} ({resolved} resolved)")
        print(f"   Mission advances total: {len(self.mission_advances)}")
        print(f"   Emergences this cycle: "
              f"{sum(self._cycle_emergence_counts.values())} "
              f"total: {len(self.emergence_feed)}")
        print(f"── Observer done ──\n")


    def reflection_week(self) -> dict | None:
        """Evaluate the overall failure ratio and pivot mission if necessary."""
        print(f"\n  ── 🧭 Weekly Reflection (Cycle {self.cycle_count}) ──")
        
        # Calculate failure vs success among investigated hypotheses
        hypotheses = self.brain.nodes_by_type("hypothesis")
        
        failed = []
        succeeded_or_working = []
        
        from graph.brain import NodeStatus, ArtifactStatus
        for nid, data in hypotheses:
            status = data.get("status")
            if status in [NodeStatus.CONTRADICTED.value, NodeStatus.LACKS_EVIDENCE.value]:
                failed.append((nid, data))
            elif status in [NodeStatus.SETTLED.value, NodeStatus.UNCERTAIN.value]: # uncertain means working/promoted
                succeeded_or_working.append((nid, data))
                
        total_investigated = len(failed) + len(succeeded_or_working)
        
        if total_investigated == 0:
            print("     Insufficient hypothesis testing data. No pivot required.")
            return None
            
        failure_ratio = len(failed) / total_investigated
        unresolved_review_blockers = sum(
            self._blocking_objection_count(nid) for nid, _ in hypotheses
        )
        print(f"     Failure Ratio: {len(failed)}/{total_investigated} ({failure_ratio*100:.1f}%)")
        if unresolved_review_blockers:
            print(f"     Review blockers still open: {unresolved_review_blockers}")
        
        # Threshold for pivot: 80% failure (0.8) among tested concepts
        # For testing purposes, we'll allow an easy trigger if we pass a minimum threshold or override
        if failure_ratio >= 0.8 and len(failed) >= 3:
            print("  ‼️  CRITICAL FAILURE THRESHOLD REACHED. PIVOT TRIGGERED ‼️")
            
            anomalies = [data.get('statement', '') for _, data in sorted(
                failed, key=lambda x: x[1].get('importance', 0), reverse=True
            )[:5]]
            
            old_mission_question = self._mission_text()
            
            from llm_utils import require_json
            raw_pivot = self._llm(PIVOT_PROMPT.format(
                old_mission=old_mission_question,
                anomalies="\n".join(f"- {a}" for a in anomalies)
            ), temperature=0.7)
            
            p_data = require_json(raw_pivot, default={
                "new_mission_statement": "Are these anomalies a unified phenomenon?",
                "justification": "Graceful LLM parsing failure fallback."
            })
            
            new_mission = p_data.get("new_mission_statement", "")
            if not new_mission:
                new_mission = "Investigate the recent anomalies."
                
            print(f"     Old Mission: {old_mission_question[:60]}...")
            print(f"     New Mission: {new_mission[:60]}...\n")
            
            # Archive old mission in brain
            if self.brain.mission:
                old_m_id = self.brain.mission.get("id")
                if old_m_id:
                    self.brain.update_node(old_m_id, status=NodeStatus.CONTRADICTED.value, epistemic_status=ArtifactStatus.CONTRADICTED.value)
            
            old_mission_context = (self.brain.mission or {}).get("context", "")
            
            # Set new mission
            self.brain.set_mission(new_mission, context=f"Pivot from failures. Justification: {p_data.get('justification', '')}")
            
            return {
                "pivot_triggered": True,
                "old_mission": old_mission_question,
                "new_mission": new_mission,
                "anomalies": anomalies,
                "justification": p_data.get("justification", "")
            }
                
        if unresolved_review_blockers:
            print("     Research remains viable, but review blockers still prevent strong scientific promotion.")
            return None
        print("     Research remains viable. No pivot required.")
        return None

    def observe(self, log: DreamLog):
        self._observe_internal(
            log, increment_cycle=True, increment_incubation=True
        )

    def observe_supplemental(self, log: DreamLog):
        self._observe_internal(
            log, increment_cycle=False, increment_incubation=False
        )

    # ── Reference cleanup ───────────────────────────────────────────────────

    def remove_node_references(self, removed_node_ids: list[str] | set[str]) -> dict:
        """Remove dangling references to pruned graph nodes."""
        removed = set(removed_node_ids or [])
        if not removed:
            return {"cleaned": 0}

        cleaned = 0

        for item in self.agenda:
            if item.node_id in removed:
                item.node_id = ""
                cleaned += 1
            if item.answer_node_id in removed:
                item.answer_node_id = ""
                cleaned += 1
            before = len(item.partial_leads)
            item.partial_leads = [nid for nid in item.partial_leads if nid not in removed]
            cleaned += max(0, before - len(item.partial_leads))

        before_adv = len(self.mission_advances)
        self.mission_advances = [a for a in self.mission_advances if a.node_id not in removed]
        cleaned += max(0, before_adv - len(self.mission_advances))

        for signal in self.emergence_feed:
            before = len(signal.node_ids)
            signal.node_ids = [nid for nid in signal.node_ids if nid not in removed]
            cleaned += max(0, before - len(signal.node_ids))

        before_edges = len(self.edge_traversal_counts)
        self.edge_traversal_counts = {
            key: val for key, val in self.edge_traversal_counts.items()
            if key[0] not in removed and key[1] not in removed
        }
        cleaned += max(0, before_edges - len(self.edge_traversal_counts))

        return {
            "cleaned": cleaned,
            "agenda_items": len(self.agenda),
            "mission_advances": len(self.mission_advances),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "data/observer.json"):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        data = {
            "cycle_count":     self.cycle_count,
            "agenda":          [i.__dict__ for i in self.agenda],
            "emergences":      [e.to_dict() for e in self.emergence_feed],
            "mission_advances":[a.to_dict() for a in self.mission_advances],
            "edge_traversal_counts": {
                f"{k[0]}|{k[1]}": v
                for k, v in self.edge_traversal_counts.items()
            }
        }
        data["emergence_last_fired"] = self._emergence_last_fired
        atomic_write_json(path, data)
        print(f"Observer saved — {len(self.agenda)} items, "
              f"{len(self.emergence_feed)} emergences, "
              f"{len(self.mission_advances)} mission advances")

    def load(self, path: str = "data/observer.json"):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.cycle_count = data.get('cycle_count', 0)
            self.agenda      = [AgendaItem(**i) for i in data.get('agenda', [])]
            self.agenda_embeddings = [
                self._embed(i.text) for i in self.agenda
            ]
            self.emergence_feed = [
                EmergenceSignal(**e) for e in data.get('emergences', [])
            ]
            self.mission_advances = [
                MissionAdvance(**a)
                for a in data.get('mission_advances', [])
            ]
            self.edge_traversal_counts = {
                tuple(k.split('|')): v
                for k, v in data.get('edge_traversal_counts', {}).items()
            }
            self._emergence_last_fired = data.get("emergence_last_fired", {})
            print(f"Observer loaded — {len(self.agenda)} items, "
                  f"{len(self.emergence_feed)} emergences, "
                  f"{len(self.mission_advances)} mission advances")
        except FileNotFoundError:
            print("No observer state — starting fresh")
