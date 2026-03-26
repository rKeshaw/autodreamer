import json
import time
import numpy as np
from dataclasses import dataclass, field
from ollama import Client
from graph.brain import Brain, EdgeType, NodeType
from dreamer.dreamer import DreamLog, DreamStep
from config import THRESHOLDS
from embedding import embed as shared_embed
from persistence import atomic_write_json

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL               = "llama3.1:8b"
WEAK_EDGE_REPEAT_THRESHOLD = 3
QUESTION_REPEAT_THRESHOLD  = 2
COHERENCE_THRESHOLD        = THRESHOLDS.COHERENCE
INCUBATION_EMERGENCE_AGE   = 5
SIMILARITY_HIGH            = THRESHOLDS.QUESTION_DEDUP_HIGH
SIMILARITY_MID             = THRESHOLDS.QUESTION_DEDUP_LOW
MAX_EMERGENCES_PER_TYPE    = 2
EMERGENCE_COOLDOWN_HOURS   = 24

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
Are these two questions essentially the same, even if worded differently?
Q1: {q1}
Q2: {q2}
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

Respond with ONLY a float 0.0 to 1.0.
Weigh depth heavily: isomorphism=0.8-1.0, structural=0.5-0.8, surface=0.1-0.5.
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

These are the strongest insights found:
{insights}

These are the main open tensions still unresolved:
{contradictions}

In 3-4 sentences, summarize how close the mind is to answering the central question.
What is the current best partial answer? What is the key remaining gap?
Write like a scientist assessing their own progress.
"""

# ── Observer ──────────────────────────────────────────────────────────────────

class Observer:
    def __init__(self, brain: Brain):
        self.brain                 = brain
        self.llm                   = Client()
        self.agenda: list[AgendaItem]          = []
        self.agenda_embeddings: list           = []
        self.emergence_feed: list[EmergenceSignal] = []
        self.mission_advances: list[MissionAdvance]= []
        self.edge_traversal_counts: dict       = {}
        self.cycle_count                       = 0
        self._cycle_emergence_counts: dict     = {}
        self._emergence_last_fired: dict       = {}

    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        response = self.llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    def _embed(self, text: str) -> np.ndarray:
        return shared_embed(text)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _mission_text(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

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
        advance = MissionAdvance(
            node_id     = node_id,
            explanation = explanation,
            strength    = strength,
            cycle       = self.cycle_count
        )
        self.mission_advances.append(advance)
        print(f"  ★ Mission advance recorded (strength={strength:.2f})")

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

        # get strongest insights from dream logs
        insights = []
        try:
            import os
            dream_files = [
                fname for fname in sorted(os.listdir("logs"), reverse=True)
                if fname.startswith("dream_cycle") and fname.endswith(".json")
            ]
            if not dream_files:
                print("Observer: no dream_cycle*.json files found for mission summary.")
            for fname in dream_files[:5]:
                with open(f"logs/{fname}") as f:
                    d = json.load(f)
                for ins in d.get("insights", []):
                    if ins.get("depth") in ["structural", "isomorphism"]:
                        insights.append(ins.get("narration", ""))
        except Exception:
            pass

        # get active contradictions
        contradictions = []
        for u, v, data in list(self.brain.graph.edges(data=True))[:50]:
            if data.get('type') == 'contradicts':
                nu = self.brain.get_node(u)
                nv = self.brain.get_node(v)
                if nu and nv:
                    contradictions.append(
                        f"{nu['statement']} vs {nv['statement']}")
                if len(contradictions) >= 3:
                    break

        return self._llm(MISSION_SUMMARY_PROMPT.format(
            mission       = mission['question'],
            advances      = "\n".join(
                f"- ({a.strength:.2f}) {a.explanation}"
                for a in top_advances) or "none yet",
            insights      = "\n".join(f"- {i}" for i in insights) or "none yet",
            contradictions= "\n".join(f"- {c}" for c in contradictions) or "none"
        ), temperature=0.4)

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
        signal_text = self._llm(EMERGENCE_PROMPT.format(
            mission=self._mission_text(),
            type=type, detail=detail
        ), temperature=0.5)
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

        resolved = sum(1 for i in self.agenda if i.resolved)
        print(f"   Agenda: {len(self.agenda)} ({resolved} resolved)")
        print(f"   Mission advances total: {len(self.mission_advances)}")
        print(f"   Emergences this cycle: "
              f"{sum(self._cycle_emergence_counts.values())} "
              f"total: {len(self.emergence_feed)}")
        print(f"── Observer done ──\n")

    def observe(self, log: DreamLog):
        self._observe_internal(
            log, increment_cycle=True, increment_incubation=True
        )

    def observe_supplemental(self, log: DreamLog):
        self._observe_internal(
            log, increment_cycle=False, increment_incubation=False
        )

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
