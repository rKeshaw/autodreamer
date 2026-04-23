import json
import time
import os
from types import SimpleNamespace
from dataclasses import dataclass, field
from graph.brain import Brain, NodeType, EdgeType
from scientist_workspace import citation_anchors_for_node
from persistence import atomic_write_json
from llm_utils import llm_call
from scientific_rigor import external_scientific_references, is_local_artifact_reference

# ── Config ────────────────────────────────────────────────────────────────────

NOTEBOOK_PATH  = "data/notebook.json"
SCIENTIST_NAME = "THE SCIENTIST"

# ── Entry types ───────────────────────────────────────────────────────────────

ENTRY_MORNING       = "morning"      # after dream cycle
ENTRY_FIELD_NOTES   = "field_notes"  # after research day
ENTRY_EVENING       = "evening"      # after consolidation
ENTRY_HYPOTHESIS    = "hypothesis"   # running best answer to mission
ENTRY_BREAKTHROUGH  = "breakthrough" # flagged manually or by strong emergence
ENTRY_DEAD_END      = "dead_end"     # hard pivot reflection
ENTRY_SYNTHESIS     = "synthesis"    # writing phase — forces clarity

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


DEAD_END_PROMPT = """
You are {name}, a scientist keeping a research journal.

Your central research question has failed and is being abandoned:
"{old_mission}"

Here are the most significant anomalies / contradicted hypotheses we found instead:
{anomalies}

And here is our radically new mission going forward:
"{new_mission}"

Write a brutal, honest "Post-Mortem" entry. Summarize why the old approach failed,
why these specific anomalies broke our models, and what the new paradigm is aiming to discover.
Do not be defensive. Science is about being wrong.

Keep it to 4-5 sentences. Sign off as: — {name}
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
        self.lab_meeting    = None
        self.name           = scientist_name
        self.entries: list[NotebookEntry] = []
        self.running_hypothesis: str = ""
        self._load()

    def _llm(self, prompt: str, temperature: float = 0.6) -> str:
        return llm_call(prompt, temperature=temperature, role="notebook")

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

    def _citation_anchor_block(self, node_ids: list[str], max_nodes: int = 6,
                               max_anchors_per_node: int = 2) -> str:
        lines = []
        seen = set()
        for node_id in list(dict.fromkeys(node_ids))[:max_nodes]:
            node = self.brain.get_node(node_id)
            if not node:
                continue
            anchors = citation_anchors_for_node(node, max_items=max_anchors_per_node)
            if not anchors:
                continue
            statement = " ".join(str(node.get("statement", "") or "").split())
            if len(statement) > 110:
                statement = statement[:107].rstrip() + "..."
            for anchor in anchors:
                if is_local_artifact_reference(anchor):
                    continue
                key = (statement, anchor)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {statement} -> {anchor}")
        return "\n".join(lines)

    def _review_blockers(self, hypothesis_id: str) -> list[dict]:
        if not self.lab_meeting or not hasattr(self.lab_meeting, "get_hypothesis_state"):
            return []
        try:
            state = self.lab_meeting.get_hypothesis_state(hypothesis_id)
        except Exception:
            return []
        return state.get("unresolved_objections", []) or []

    def _top_hypotheses(self, limit: int = 5) -> list[tuple[str, dict]]:
        hypotheses = []
        for nid, data in self.brain.nodes_by_type(NodeType.HYPOTHESIS):
            status = data.get("status", "")
            epistemic = data.get("epistemic_status", "")
            if status == "contradicted" or epistemic == "contradicted":
                continue
            hypotheses.append((nid, data))
        hypotheses.sort(
            key=lambda item: (
                len(self._review_blockers(item[0])) == 0,
                float(item[1].get("importance", 0.0) or 0.0),
            )
        )
        hypotheses.reverse()
        return hypotheses[:limit]

    # ── Entry writers ─────────────────────────────────────────────────────────

    def write_morning_entry(self, dream_log, cycle: int) -> str:
        """Write a morning entry after a dream cycle."""
        if not dream_log:
            dream_log = SimpleNamespace(
                summary="No dream activity recorded.",
                mission_advances=[],
                insights=[],
                questions=[],
            )

        mission_advances_data = getattr(dream_log, "mission_advances", []) or []
        insights_data = getattr(dream_log, "insights", []) or []
        questions_data = getattr(dream_log, "questions", []) or []

        mission_advances = "\n".join(
            f"- ({a['strength']:.2f}) {a['explanation']}"
            for a in mission_advances_data
            if isinstance(a, dict) and "strength" in a and "explanation" in a
        ) or "none"

        insights = "\n".join(
            f"- [{i['depth']}] {i['narration']}"
            for i in insights_data
            if isinstance(i, dict) and "depth" in i and "narration" in i
        ) or "none"

        questions = "\n".join(
            f"- {q}" for q in questions_data[:8]
        ) or "none"
        content = "\n".join([
            f"Morning Research Memo — Cycle {cycle}",
            f"Mission: {self._mission()}",
            "Dream summary:",
            getattr(dream_log, "summary", "No dream activity recorded."),
            "Potential advances from dreaming:",
            mission_advances if mission_advances != "none" else "- none",
            "Retained structural insights:",
            insights if insights != "none" else "- none",
            "Questions queued for follow-up:",
            questions if questions != "none" else "- none",
            "Assessment: Dream outputs remain provisional until grounded by research, experiment, or review closure.",
        ])

        entry = self._add_entry(
            ENTRY_MORNING, content, cycle,
            tags=["dream", f"insights:{len(insights_data)}",
                  f"advances:{len(mission_advances_data)}"]
        )
        print(f"\n── Notebook: morning entry written ──")
        return content

    def write_field_notes(self, research_log, cycle: int) -> str:
        """Write field notes after a research day."""
        entries = []
        if isinstance(research_log, dict):
            entries = research_log.get("entries", []) or []
        else:
            entries = getattr(research_log, "entries", []) or []

        normalized_entries = []
        for entry in entries:
            if isinstance(entry, dict):
                normalized_entries.append(SimpleNamespace(
                    question=entry.get("question", ""),
                    sources=entry.get("sources", []) or [],
                    node_ids=entry.get("node_ids", []) or [],
                    resolved=entry.get("resolved", "none"),
                ))
            else:
                normalized_entries.append(entry)

        findings = "\n".join(
            f"- [{getattr(e, 'resolved', 'none')}] {e.question} | sources: {', '.join(getattr(e, 'sources', [])[:2]) or 'none'}"
            for e in normalized_entries
            if hasattr(e, "question") and hasattr(e, "sources")
        ) or "- none"

        resolved = "\n".join(
            f"- [{e.resolved}] {e.question}"
            for e in normalized_entries
            if hasattr(e, "resolved") and hasattr(e, "question") and e.resolved in ['partial', 'strong']
        ) or "- none"

        new_qs = sum(len(getattr(e, 'node_ids', []) or []) for e in normalized_entries)

        content = "\n".join([
            f"Field Notes — Cycle {cycle}",
            f"Mission: {self._mission()}",
            "Question-by-question research outcomes:",
            findings,
            "Questions materially advanced:",
            resolved,
            f"Graph additions from research: {new_qs}",
            "Assessment: Only mechanism-specific or constraint-specific findings count as advancement.",
        ])

        cited_node_ids = []
        for entry in normalized_entries:
            cited_node_ids.extend(getattr(entry, "node_ids", []) or [])
        citation_block = self._citation_anchor_block(cited_node_ids, max_nodes=8)
        if citation_block:
            content = content + "\n\nEvidence anchors:\n" + citation_block

        entry = self._add_entry(
            ENTRY_FIELD_NOTES, content, cycle,
            tags=["research",
                  f"resolved:{sum(1 for e in normalized_entries if getattr(e, 'resolved', '') in ['partial','strong'])}"]
        )
        print(f"\n── Notebook: field notes written ──")
        return content

    def write_evening_entry(self, consolidation_report, cycle: int) -> str:
        """Write an evening reflection after consolidation."""
        if isinstance(consolidation_report, dict):
            class _Report:
                merges = int(consolidation_report.get("merges", 0) or 0)
                syntheses = int(consolidation_report.get("syntheses", 0) or 0)
                abstractions = int(consolidation_report.get("abstractions", 0) or 0)
                gaps = int(consolidation_report.get("gaps", 0) or 0)
            consolidation_report = _Report()
        if not consolidation_report:
            class _EmptyReport:
                merges = 0
                syntheses = 0
                abstractions = 0
                gaps = 0
            consolidation_report = _EmptyReport()

        blocker_count = 0
        for hyp_id, _ in self._top_hypotheses(limit=5):
            blocker_count += len(self._review_blockers(hyp_id))
        content = "\n".join([
            f"Evening Consolidation Memo — Cycle {cycle}",
            f"Mission: {self._mission()}",
            f"Merges: {getattr(consolidation_report, 'merges', 0)}",
            f"Syntheses: {getattr(consolidation_report, 'syntheses', 0)}",
            f"Abstractions: {getattr(consolidation_report, 'abstractions', 0)}",
            f"Gaps: {getattr(consolidation_report, 'gaps', 0)}",
            f"Active contradictions: {self.brain.stats().get('contradictions', 0)}",
            f"Open review blockers: {blocker_count}",
            f"Brain state: {self.brain.stats()['nodes']} nodes, {self.brain.stats()['edges']} edges",
        ])

        entry = self._add_entry(
            ENTRY_EVENING, content, cycle,
            tags=["consolidation",
                  f"syntheses:{getattr(consolidation_report, 'syntheses', 0)}",
                  f"gaps:{getattr(consolidation_report, 'gaps', 0)}"]
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
        try:
            progress_summary = self.observer.get_mission_progress_summary()
        except Exception:
            progress_summary = "none yet"

        advances = sorted(
            self.observer.mission_advances,
            key=lambda a: a.strength, reverse=True
        )[:5]
        advances_text = "\n".join(
            f"- ({a.strength:.2f}) {a.explanation}"
            for a in advances
        ) or "- none"

        grounded_nodes = self.brain.build_workspace().grounded_evidence[:5]
        grounded_text = "\n".join(node.prompt_line() for node in grounded_nodes) or "- none"

        hypothesis_lines = []
        candidate_hypotheses = self._top_hypotheses(limit=5)
        for hyp_id, data in candidate_hypotheses:
            blockers = self._review_blockers(hyp_id)
            blocker_suffix = (
                f" | blockers={len(blockers)}"
                if blockers else
                " | blockers=0"
            )
            hypothesis_lines.append(
                f"- [{data.get('status', 'unknown')}/{data.get('epistemic_status', 'open')}] "
                f"{data.get('statement', '')}{blocker_suffix}"
            )
        hypotheses_text = "\n".join(hypothesis_lines) or "- none"

        blocker_lines = []
        for hyp_id, _ in candidate_hypotheses[:3]:
            for objection in self._review_blockers(hyp_id)[:2]:
                blocker_lines.append(
                    f"- [{objection.get('issue_label', 'issue')}] {objection.get('objection', '')}"
                )
        blockers_text = "\n".join(blocker_lines) or "- none"

        leading_claim = "No publication-grade answer yet."
        for hyp_id, data in candidate_hypotheses:
            if not self._review_blockers(hyp_id):
                leading_claim = data.get("statement", leading_claim)
                break

        self.running_hypothesis = "\n".join([
            f"Running Hypothesis Memo — Cycle {cycle}",
            f"Mission: {self._mission()}",
            f"Current best answer: {leading_claim}",
            "Grounded support:",
            grounded_text,
            "Mission advances:",
            advances_text,
            "Hypotheses under consideration:",
            hypotheses_text,
            "Open review blockers:",
            blockers_text,
            "Progress summary:",
            progress_summary,
        ])

        self._add_entry(
            ENTRY_HYPOTHESIS, self.running_hypothesis, cycle,
            tags=["running_hypothesis", f"cycle:{cycle}"]
        )
        print(f"\n── Notebook: running hypothesis updated ──")
        return self.running_hypothesis

    def write_dead_end_summary(self, pivot_data: dict, cycle: int) -> str:
        """Write a reflection when the observer forces a hard pivot."""
        anomalies_str = "\n".join(f"- {a}" for a in pivot_data.get("anomalies", [])) or "none"
        
        content = self._llm(DEAD_END_PROMPT.format(
            name       = self.name,
            old_mission= pivot_data.get('old_mission', ''),
            new_mission= pivot_data.get('new_mission', ''),
            anomalies  = anomalies_str
        ))
        
        entry = self._add_entry(
            ENTRY_DEAD_END, content, cycle,
            tags=["pivot", "dead_end"]
        )
        print(f"\n── Notebook: DEAD END / PIVOT entry written ──")
        return content

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

    def write_synthesis_essay(self, cycle: int) -> dict:
        """
        Writing phase — the scientist writes a structured essay.

        Writing forces clarity. The LLM is asked to *write* about the research,
        and the act of writing produces side-effect insights that get returned
        for graph ingestion.

        Returns dict with 'essay', 'insights' (list of strings), 'questions' (list).
        """
        key_nodes = sorted(
            self.brain.all_nodes(),
            key=lambda x: x[1].get('importance', 0.5),
            reverse=True
        )[:12]

        established = []
        for node_id, data in key_nodes:
            refs = external_scientific_references(data)
            if data.get("epistemic_status") != "grounded" or not refs:
                continue
            established.append(f"- {data.get('statement', '')} | refs: {', '.join(refs[:2])}")
            if len(established) >= 6:
                break

        working = []
        open_questions = []
        for hyp_id, data in self._top_hypotheses(limit=5):
            blockers = self._review_blockers(hyp_id)
            if blockers:
                working.append(
                    f"- {data.get('statement', '')} | unresolved objections={len(blockers)}"
                )
                for objection in blockers[:2]:
                    open_questions.append(objection.get("evidence_needed", "") or objection.get("objection", ""))
            else:
                working.append(f"- {data.get('statement', '')} | review blockers=0")

        contradictions = []
        for u, v, data in list(self.brain.graph.edges(data=True))[:200]:
            if data.get('type') != EdgeType.CONTRADICTS.value:
                continue
            nu = self.brain.get_node(u)
            nv = self.brain.get_node(v)
            if nu and nv:
                contradictions.append(f"- {nu.get('statement', '')} VS {nv.get('statement', '')}")
            if len(contradictions) >= 4:
                break

        essay = "\n".join([
            f"# Synthesis Memo — Cycle {cycle}",
            "",
            f"## Mission\n{self._mission()}",
            "",
            "## Established Findings",
            "\n".join(established) if established else "- none",
            "",
            "## Working Hypotheses",
            "\n".join(working) if working else "- none",
            "",
            "## Contradictions and Weak Links",
            "\n".join(contradictions) if contradictions else "- none",
            "",
            "## Next Discriminating Questions",
            "\n".join(f"- {item}" for item in open_questions if item) or "- none",
        ])
        insights = []
        questions = [item for item in open_questions if item][:4]

        citation_node_ids = [nid for nid, _ in key_nodes]
        citation_block = self._citation_anchor_block(citation_node_ids, max_nodes=10)
        if citation_block:
            essay = essay + "\n\n## Evidence Anchors\n" + citation_block

        # Store the essay
        self._add_entry(
            ENTRY_SYNTHESIS, essay, cycle,
            tags=["synthesis_essay", f"cycle:{cycle}"]
        )

        print(f"\n── Notebook: synthesis essay written "
              f"({len(insights)} insights, {len(questions)} questions) ──")

        return {
            "essay":     essay,
            "insights":  insights if isinstance(insights, list) else [],
            "questions": questions if isinstance(questions, list) else []
        }

    # ── Reference cleanup ───────────────────────────────────────────────────

    def remove_node_references(self, removed_node_ids: list[str] | set[str],
                               removed_nodes: dict | None = None) -> dict:
        """Best-effort cleanup for notebook references to pruned nodes."""
        removed = set(removed_node_ids or [])
        if not removed:
            return {"cleaned": 0, "entries_touched": 0}

        removed_nodes = removed_nodes or {}
        statement_fragments = set()
        source_refs = set()
        for nid in removed:
            data = removed_nodes.get(nid, {}) or {}
            statement = " ".join(str(data.get("statement", "") or "").split())
            if statement:
                statement_fragments.add(statement[:72])
            for ref in data.get("source_refs", []) or []:
                if ref:
                    source_refs.add(str(ref))
            for span in data.get("provenance_spans", []) or []:
                if isinstance(span, dict):
                    span_ref = str(span.get("source_ref", "") or "").strip()
                    if span_ref:
                        source_refs.add(span_ref)

        cleaned = 0
        entries_touched = 0

        for entry in self.entries:
            original_tags = list(entry.tags)
            entry.tags = [
                tag for tag in original_tags
                if not any(tag == f"node:{nid}" for nid in removed)
            ]
            cleaned += max(0, len(original_tags) - len(entry.tags))

            lines = entry.content.splitlines()
            kept = []
            removed_lines = 0
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("-") and "->" in stripped:
                    if any(ref and ref in stripped for ref in source_refs):
                        removed_lines += 1
                        continue
                    if any(fragment and fragment in stripped for fragment in statement_fragments):
                        removed_lines += 1
                        continue
                kept.append(line)

            if removed_lines:
                entry.content = "\n".join(kept)
                entries_touched += 1
                cleaned += removed_lines

        if cleaned:
            self._save()

        return {
            "cleaned": cleaned,
            "entries_touched": entries_touched,
        }

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
