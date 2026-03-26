import json
import time
import os
from dataclasses import dataclass, field
from ollama import Client
from graph.brain import Brain, NodeType, EdgeType
from persistence import atomic_write_json

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL   = "llama3.1:8b"
NOTEBOOK_PATH  = "data/notebook.json"
SCIENTIST_NAME = "THE SCIENTIST"

# ── Entry types ───────────────────────────────────────────────────────────────

ENTRY_MORNING       = "morning"      # after dream cycle
ENTRY_FIELD_NOTES   = "field_notes"  # after research day
ENTRY_EVENING       = "evening"      # after consolidation
ENTRY_HYPOTHESIS    = "hypothesis"   # running best answer to mission
ENTRY_BREAKTHROUGH  = "breakthrough" # flagged manually or by strong emergence

# ── Prompts ───────────────────────────────────────────────────────────────────

MORNING_ENTRY_PROMPT = """
You are {name}, a scientist keeping a research journal.

Your central research question:
"{mission}"

Last night's dream cycle summary:
{dream_summary}

Mission advances found during dreaming:
{mission_advances}

Key insights (with depth):
{insights}

Questions generated:
{questions}

Write a morning notebook entry. Be specific, honest, and direct.
Address:
1. What the dream revealed about the central question
2. Whether any connections crossed from analogy into something deeper
3. What you now most urgently need to investigate
4. Your current emotional/intellectual state regarding the question

Keep it to 4-6 sentences. Sign off as: — {name}
"""

FIELD_NOTES_PROMPT = """
You are {name}, a scientist keeping a research journal.

Your central research question:
"{mission}"

Today's research findings:
{findings}

Questions resolved today:
{resolved}

New questions opened:
{new_questions}

Write a field notes entry. Be specific about what you actually found —
not what you hoped to find. Note any surprises. Note any disappointments.
Note any moment where the central question came into sharper focus or
became more complicated.

Keep it to 4-5 sentences. Sign off as: — {name}
"""

EVENING_ENTRY_PROMPT = """
You are {name}, a scientist keeping a research journal.

Your central research question:
"{mission}"

Today's consolidation results:
- Nodes merged (near-duplicates resolved): {merges}
- New synthesis nodes created: {syntheses}
- Abstraction nodes created: {abstractions}
- Gap nodes inferred: {gaps}
- Contradictions still active: {contradictions}

Current brain state: {brain_stats}

Write an evening reflection. What did today add to your understanding?
What tensions remain? What does the mind seem to be building toward?
Be honest if progress was slow. Be precise if something clicked.

Keep it to 4-5 sentences. Sign off as: — {name}
"""

RUNNING_HYPOTHESIS_PROMPT = """
You are {name}, a scientist attempting to answer:

"{mission}"

Here is everything the mind has accumulated so far:

Mission progress synthesis:
{progress_summary}

Most significant mission advances:
{advances}

Strongest structural/isomorphic insights found:
{insights}

Current active hypotheses in the graph:
{hypotheses}

Key contradictions still unresolved:
{contradictions}

Based on all of this, write the current best partial answer to the central question.
This is a working hypothesis — not a conclusion. Be specific about what is
supported, what remains uncertain, and what would need to be true for this
to be the correct answer.

Write 5-7 sentences. Label it clearly as a working hypothesis.
Sign off as: — {name}
"""

BREAKTHROUGH_PROMPT = """
You are {name}, a scientist keeping a research journal.

Your central research question:
"{mission}"

Something significant just happened:
{detail}

Write a brief, excited but precise breakthrough note.
What happened? What does it mean for the central question?
What must be done next?

Keep it to 3-4 sentences. Sign off as: — {name}
"""

# ── Notebook entry ────────────────────────────────────────────────────────────

@dataclass
class NotebookEntry:
    entry_type:  str
    content:     str
    cycle:       int
    timestamp:   float = field(default_factory=time.time)
    tags:        list  = field(default_factory=list)

    def to_dict(self):
        return self.__dict__

# ── Notebook ──────────────────────────────────────────────────────────────────

class Notebook:
    def __init__(self, brain: Brain, observer=None,
                 scientist_name: str = SCIENTIST_NAME):
        self.brain          = brain
        self.observer       = observer
        self.llm            = Client()
        self.name           = scientist_name
        self.entries: list[NotebookEntry] = []
        self.running_hypothesis: str = ""
        self._load()

    def _llm(self, prompt: str, temperature: float = 0.6) -> str:
        response = self.llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    def _mission(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

    def _add_entry(self, entry_type: str, content: str,
                   cycle: int, tags: list = None) -> NotebookEntry:
        entry = NotebookEntry(
            entry_type = entry_type,
            content    = content,
            cycle      = cycle,
            tags       = tags or []
        )
        self.entries.append(entry)
        self._save()
        return entry

    # ── Entry writers ─────────────────────────────────────────────────────────

    def write_morning_entry(self, dream_log, cycle: int) -> str:
        """Write a morning entry after a dream cycle."""
        mission_advances = "\n".join(
            f"- ({a['strength']:.2f}) {a['explanation']}"
            for a in dream_log.mission_advances
        ) or "none"

        insights = "\n".join(
            f"- [{i['depth']}] {i['narration']}"
            for i in dream_log.insights
        ) or "none"

        questions = "\n".join(
            f"- {q}" for q in dream_log.questions[:8]
        ) or "none"

        content = self._llm(MORNING_ENTRY_PROMPT.format(
            name             = self.name,
            mission          = self._mission(),
            dream_summary    = dream_log.summary,
            mission_advances = mission_advances,
            insights         = insights,
            questions        = questions
        ))

        entry = self._add_entry(
            ENTRY_MORNING, content, cycle,
            tags=["dream", f"insights:{len(dream_log.insights)}",
                  f"advances:{len(dream_log.mission_advances)}"]
        )
        print(f"\n── Notebook: morning entry written ──")
        return content

    def write_field_notes(self, research_log, cycle: int) -> str:
        """Write field notes after a research day."""
        findings = "\n".join(
            f"- Q: {e.question}\n  Found: {', '.join(e.sources[:2])}"
            for e in research_log.entries
        ) or "none"

        resolved = "\n".join(
            f"- [{e.resolved}] {e.question}"
            for e in research_log.entries
            if e.resolved in ['partial', 'strong']
        ) or "none"

        new_qs = sum(len(getattr(e, 'node_ids', [])) for e in research_log.entries)

        content = self._llm(FIELD_NOTES_PROMPT.format(
            name        = self.name,
            mission     = self._mission(),
            findings    = findings,
            resolved    = resolved,
            new_questions = f"{new_qs} new nodes added to graph"
        ))

        entry = self._add_entry(
            ENTRY_FIELD_NOTES, content, cycle,
            tags=["research",
                  f"resolved:{sum(1 for e in research_log.entries if e.resolved in ['partial','strong'])}"]
        )
        print(f"\n── Notebook: field notes written ──")
        return content

    def write_evening_entry(self, consolidation_report, cycle: int) -> str:
        """Write an evening reflection after consolidation."""
        content = self._llm(EVENING_ENTRY_PROMPT.format(
            name         = self.name,
            mission      = self._mission(),
            merges       = consolidation_report.merges,
            syntheses    = consolidation_report.syntheses,
            abstractions = consolidation_report.abstractions,
            gaps         = consolidation_report.gaps,
            contradictions = self.brain.stats().get('contradictions', 0),
            brain_stats  = (f"{self.brain.stats()['nodes']} nodes, "
                            f"{self.brain.stats()['edges']} edges")
        ))

        entry = self._add_entry(
            ENTRY_EVENING, content, cycle,
            tags=["consolidation",
                  f"syntheses:{consolidation_report.syntheses}",
                  f"gaps:{consolidation_report.gaps}"]
        )
        print(f"\n── Notebook: evening entry written ──")
        return content

    def update_running_hypothesis(self, cycle: int) -> str:
        """
        Update the running best answer to the central question.
        Called after consolidation — when the graph is freshest.
        """
        if not self.observer:
            return ""

        # top mission advances
        progress_summary = "none yet"
        if self.observer:
            try:
                progress_summary = self.observer.get_mission_progress_summary()
            except Exception:
                progress_summary = "none yet"

        # top mission advances
        advances = sorted(
            self.observer.mission_advances,
            key=lambda a: a.strength, reverse=True
        )[:5]
        advances_text = "\n".join(
            f"- ({a.strength:.2f}) {a.explanation}"
            for a in advances
        ) or "none yet"

        # strongest structural/isomorphic insights from recent logs
        insights = []
        try:
            dream_files = [
                fname for fname in sorted(os.listdir("logs"), reverse=True)
                if fname.startswith("dream_cycle") and fname.endswith(".json")
            ]
            if not dream_files:
                print("Notebook: no dream_cycle*.json files found for insight synthesis.")
            for fname in dream_files[:10]:
                with open(f"logs/{fname}") as f:
                    d = json.load(f)
                for ins in d.get("insights", []):
                    if ins.get("depth") in ["structural", "isomorphism"]:
                        insights.append(
                            f"[{ins['depth']}] {ins['narration']}")
                if len(insights) >= 5:
                    break
        except Exception:
            pass
        insights_text = "\n".join(f"- {i}" for i in insights) or "none yet"

        # active hypotheses from graph
        hyp_nodes = self.brain.nodes_by_type(NodeType.HYPOTHESIS)
        hypotheses_text = "\n".join(
            f"- {data['statement']}"
            for _, data in hyp_nodes[:5]
        ) or "none yet"

        # active contradictions
        contradictions = []
        for u, v, data in list(self.brain.graph.edges(data=True))[:100]:
            if data.get('type') == EdgeType.CONTRADICTS.value:
                nu = self.brain.get_node(u)
                nv = self.brain.get_node(v)
                if nu and nv:
                    contradictions.append(
                        f"{nu['statement']} ↔ {nv['statement']}")
                if len(contradictions) >= 3:
                    break
        contradictions_text = "\n".join(
            f"- {c}" for c in contradictions) or "none"

        self.running_hypothesis = self._llm(RUNNING_HYPOTHESIS_PROMPT.format(
            name           = self.name,
            mission        = self._mission(),
            progress_summary = progress_summary,
            advances       = advances_text,
            insights       = insights_text,
            hypotheses     = hypotheses_text,
            contradictions = contradictions_text
        ))

        self._add_entry(
            ENTRY_HYPOTHESIS, self.running_hypothesis, cycle,
            tags=["running_hypothesis", f"cycle:{cycle}"]
        )
        print(f"\n── Notebook: running hypothesis updated ──")
        return self.running_hypothesis

    def write_breakthrough(self, detail: str, cycle: int) -> str:
        """Write a breakthrough note — called when observer flags mission_advance."""
        content = self._llm(BREAKTHROUGH_PROMPT.format(
            name    = self.name,
            mission = self._mission(),
            detail  = detail
        ))
        entry = self._add_entry(
            ENTRY_BREAKTHROUGH, content, cycle,
            tags=["breakthrough"]
        )
        print(f"\n── Notebook: BREAKTHROUGH entry written ──")
        return content

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_entries_by_type(self, entry_type: str) -> list:
        return [e for e in self.entries if e.entry_type == entry_type]

    def get_recent_entries(self, n: int = 10) -> list:
        return sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:n]

    def get_all_for_display(self) -> list:
        """Returns entries formatted for GUI display, newest first."""
        result = []
        for e in sorted(self.entries,
                        key=lambda x: x.timestamp, reverse=True):
            result.append({
                "type":      e.entry_type,
                "content":   e.content,
                "cycle":     e.cycle,
                "timestamp": e.timestamp,
                "tags":      e.tags
            })
        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(NOTEBOOK_PATH)
                    if os.path.dirname(NOTEBOOK_PATH) else ".",
                    exist_ok=True)
        data = {
            "entries":            [e.to_dict() for e in self.entries],
            "running_hypothesis": self.running_hypothesis,
            "scientist_name":     self.name
        }
        atomic_write_json(NOTEBOOK_PATH, data)

    def _load(self):
        try:
            with open(NOTEBOOK_PATH, 'r') as f:
                data = json.load(f)
            self.entries = [
                NotebookEntry(**e) for e in data.get('entries', [])
            ]
            self.running_hypothesis = data.get('running_hypothesis', '')
            self.name = data.get('scientist_name', self.name)
            print(f"Notebook loaded — {len(self.entries)} entries")
        except FileNotFoundError:
            print("Notebook: starting fresh")

    def save(self):
        self._save()
        print(f"Notebook saved — {len(self.entries)} entries")
