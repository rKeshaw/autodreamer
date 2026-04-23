import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, asdict, field
from math import sqrt
from typing import Optional

from graph.brain import Brain, Edge, EdgeSource, EdgeType, Node, NodeStatus, NodeType
from llm_utils import llm_call, require_json
from persistence import atomic_write_json
from scientist_workspace import ArtifactStatus


DEFAULT_STATE_PATH = "data/lab_meeting_state.json"

REVIEWER_ROSTER_PROMPT = """You are staffing a recurring scientific lab meeting.

Mission context:
{mission}

Current hypothesis:
"{hypothesis}"

Active research clusters:
{clusters}

Return exactly three persistent reviewer profiles as JSON objects. Each reviewer must
have a durable specialty that could critique many future meetings, not just this one.

Return JSON only:
[
  {{
    "name": "Reviewer name",
    "specialty": "primary specialty",
    "focus": "specific blind spot this reviewer hunts for"
  }}
]
"""

REVIEWER_CRITIQUE_PROMPT = """You are {reviewer_name}, a recurring lab-meeting reviewer.

Specialty: {specialty}
Review focus: {focus}

Hypothesis under review:
"{hypothesis}"

Open objection history already on record:
{prior_objections}

Review instructions:
1. Reuse prior objections when they remain unresolved.
2. If a prior objection has been genuinely handled, mark it resolved.
3. If a new flaw matters more than prior flaws, raise the new flaw instead.
4. Keep the objection concrete and testable.

Return JSON only:
{{
  "issue_label": "mechanism" | "evidence" | "scope" | "novelty" | "counterexample" | "resolved",
  "status": "new" | "still_open" | "resolved",
  "objection": "2-3 sentence critique or resolution note",
  "evidence_needed": "specific evidence or test needed",
  "addresses_objection_ids": ["optional prior objection ids this critique updates"]
}}
"""

TASK_SYNTHESIS_PROMPT = """You are the PI summarizing a lab meeting.

Hypothesis:
"{hypothesis}"

Reviewer critiques:
{critiques}

Open objections after discussion:
{open_objections}

Return exactly two concrete tasks, unless every objection is resolved.
Tasks must be specific enough to close named objections.

Return JSON only:
[
  {{
    "task": "Actionable verb-first task",
    "addresses": ["objection id or issue label"],
    "success_criteria": "what would count as closure"
  }}
]
"""

DEFAULT_REVIEWERS = [
    {
        "name": "The Rigorous Statistician",
        "specialty": "statistical validity",
        "focus": "sample size, controls, calibration, false positives",
    },
    {
        "name": "The Mechanistic Skeptic",
        "specialty": "causal mechanism",
        "focus": "missing mechanism, confounders, proxy explanations",
    },
    {
        "name": "The Boundary Tester",
        "specialty": "generalization and failure modes",
        "focus": "counterexamples, edge cases, external validity",
    },
]


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower())
    text = text.strip("-")
    return text or "item"


def _tokenize(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2
    }


def _token_counts(text: str) -> Counter:
    return Counter(
        token for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2
    )


def _cosine_similarity(a: str, b: str) -> float:
    counts_a = _token_counts(a)
    counts_b = _token_counts(b)
    if not counts_a or not counts_b:
        return 0.0
    shared = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[token] * counts_b[token] for token in shared)
    if numerator <= 0:
        return 0.0
    norm_a = sqrt(sum(value * value for value in counts_a.values()))
    norm_b = sqrt(sum(value * value for value in counts_b.values()))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return numerator / (norm_a * norm_b)


@dataclass
class ReviewerProfile:
    reviewer_id: str
    name: str
    specialty: str
    focus: str
    meetings_attended: int = 0
    last_meeting_cycle: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObjectionRecord:
    objection_id: str
    hypothesis_id: str
    reviewer_id: str
    issue_label: str
    objection: str
    evidence_needed: str = ""
    status: str = "open"
    created_meeting: int = 0
    last_meeting: int = 0
    revisit_count: int = 0
    task_ids: list[str] = field(default_factory=list)
    closure_score: float = 0.0
    closure_reason: str = ""
    resolution_note: str = ""
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class LabMeeting:
    def __init__(self, brain: Brain, observer=None,
                 state_path: str = DEFAULT_STATE_PATH):
        self.brain = brain
        self.observer = observer
        self.state_path = state_path
        self.state = {
            "meeting_count": 0,
            "reviewer_roster": [],
            "objections": {},
            "critique_history": {},
            "last_updated": time.time(),
        }
        self._load_state()

    def _llm(self, prompt: str, temperature: float = 0.4) -> str:
        return llm_call(prompt, temperature=temperature, role="precise")

    def _load_state(self):
        try:
            with open(self.state_path, "r") as f:
                raw = require_json(f.read(), default={}) or {}
        except FileNotFoundError:
            return
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for key in self.state:
            if key in raw:
                self.state[key] = raw[key]

    def _save_state(self):
        self.state["last_updated"] = time.time()
        atomic_write_json(self.state_path, self.state)

    def _mission_text(self) -> str:
        mission = self.brain.get_mission()
        return mission.get("question", "No active mission.") if mission else "No active mission."

    def _serialize_reviewer(self, reviewer: ReviewerProfile) -> dict:
        return reviewer.to_dict()

    def _deserialize_reviewer(self, payload: dict) -> ReviewerProfile:
        return ReviewerProfile(
            reviewer_id=str(payload.get("reviewer_id", "") or str(uuid.uuid4())),
            name=str(payload.get("name", "") or "Reviewer"),
            specialty=str(payload.get("specialty", "") or "general science"),
            focus=str(payload.get("focus", "") or "missing controls"),
            meetings_attended=int(payload.get("meetings_attended", 0) or 0),
            last_meeting_cycle=int(payload.get("last_meeting_cycle", 0) or 0),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
        )

    def _serialize_objection(self, objection: ObjectionRecord) -> dict:
        return objection.to_dict()

    def _deserialize_objection(self, payload: dict) -> ObjectionRecord:
        return ObjectionRecord(
            objection_id=str(payload.get("objection_id", "") or str(uuid.uuid4())),
            hypothesis_id=str(payload.get("hypothesis_id", "") or ""),
            reviewer_id=str(payload.get("reviewer_id", "") or ""),
            issue_label=str(payload.get("issue_label", "") or "evidence"),
            objection=str(payload.get("objection", "") or ""),
            evidence_needed=str(payload.get("evidence_needed", "") or ""),
            status=str(payload.get("status", "") or "open"),
            created_meeting=int(payload.get("created_meeting", 0) or 0),
            last_meeting=int(payload.get("last_meeting", 0) or 0),
            revisit_count=int(payload.get("revisit_count", 0) or 0),
            task_ids=list(payload.get("task_ids", []) or []),
            closure_score=float(payload.get("closure_score", 0.0) or 0.0),
            closure_reason=str(payload.get("closure_reason", "") or ""),
            resolution_note=str(payload.get("resolution_note", "") or ""),
            history=[dict(item) for item in (payload.get("history", []) or []) if isinstance(item, dict)],
        )

    def _get_roster(self) -> list[ReviewerProfile]:
        roster = []
        for item in self.state.get("reviewer_roster", []) or []:
            if isinstance(item, dict):
                roster.append(self._deserialize_reviewer(item))
        return roster

    def _set_roster(self, roster: list[ReviewerProfile]):
        self.state["reviewer_roster"] = [self._serialize_reviewer(item) for item in roster]

    def _normalize_roster_payload(self, raw_payload) -> list[ReviewerProfile]:
        profiles = []
        if isinstance(raw_payload, list):
            for item in raw_payload:
                if isinstance(item, str):
                    profiles.append(ReviewerProfile(
                        reviewer_id=str(uuid.uuid4()),
                        name=item.strip() or "Reviewer",
                        specialty="general science",
                        focus="missing evidence",
                    ))
                elif isinstance(item, dict):
                    name = str(item.get("name", "") or "").strip()
                    specialty = str(item.get("specialty", "") or "").strip()
                    focus = str(item.get("focus", "") or "").strip()
                    if not name:
                        continue
                    profiles.append(ReviewerProfile(
                        reviewer_id=str(item.get("reviewer_id", "") or str(uuid.uuid4())),
                        name=name,
                        specialty=specialty or "general science",
                        focus=focus or "missing evidence",
                    ))
        if len(profiles) < 3:
            for fallback in DEFAULT_REVIEWERS:
                profiles.append(ReviewerProfile(
                    reviewer_id=str(uuid.uuid4()),
                    name=fallback["name"],
                    specialty=fallback["specialty"],
                    focus=fallback["focus"],
                ))
                if len(profiles) >= 3:
                    break
        return profiles[:3]

    def _ensure_reviewer_roster(self, hypothesis_statement: str) -> list[ReviewerProfile]:
        roster = self._get_roster()
        if len(roster) >= 3:
            return roster

        clusters = sorted({
            data.get("cluster", "")
            for _, data in self.brain.all_nodes()
            if data.get("cluster")
        })[:6]
        raw = self._llm(REVIEWER_ROSTER_PROMPT.format(
            mission=self._mission_text(),
            hypothesis=hypothesis_statement,
            clusters=", ".join(clusters) or "general science",
        ), temperature=0.5)
        roster = self._normalize_roster_payload(require_json(raw, default=[]))
        self._set_roster(roster)
        return roster

    def _select_reviewers(self, hypothesis_statement: str, count: int = 3) -> list[ReviewerProfile]:
        roster = self._ensure_reviewer_roster(hypothesis_statement)

        def _score(reviewer: ReviewerProfile) -> tuple[float, int, int]:
            reviewer_profile = " ".join([
                reviewer.name,
                reviewer.specialty,
                reviewer.focus,
            ])
            relevance = _cosine_similarity(hypothesis_statement, reviewer_profile)
            return (
                relevance,
                -int(reviewer.meetings_attended),
                -int(reviewer.last_meeting_cycle),
            )

        selected = sorted(roster, key=_score, reverse=True)[:count]
        return selected

    def _get_objections(self, hypothesis_id: str) -> list[ObjectionRecord]:
        rows = self.state.get("objections", {}).get(hypothesis_id, []) or []
        result = []
        for row in rows:
            if isinstance(row, dict):
                result.append(self._deserialize_objection(row))
        return result

    def _set_objections(self, hypothesis_id: str, objections: list[ObjectionRecord]):
        objection_map = dict(self.state.get("objections", {}) or {})
        objection_map[hypothesis_id] = [self._serialize_objection(item) for item in objections]
        self.state["objections"] = objection_map

    def _append_history(self, hypothesis_id: str, meeting_record: dict):
        history_map = dict(self.state.get("critique_history", {}) or {})
        entries = list(history_map.get(hypothesis_id, []) or [])
        entries.append(meeting_record)
        history_map[hypothesis_id] = entries
        self.state["critique_history"] = history_map

    def _format_open_objections(self, objections: list[ObjectionRecord]) -> str:
        active = [item for item in objections if item.status != "resolved"]
        if not active:
            return "none"
        return "\n".join(
            f"- [{item.objection_id}] ({item.issue_label}/{item.status}) {item.objection}"
            for item in active
        )

    def _reviewer_prior_objections(self, reviewer_id: str,
                                   objections: list[ObjectionRecord]) -> str:
        reviewer_objections = [
            item for item in objections
            if item.reviewer_id == reviewer_id and item.status != "resolved"
        ]
        return self._format_open_objections(reviewer_objections)

    def _find_matching_objection(self, reviewer_id: str, issue_label: str,
                                 objection_text: str,
                                 objections: list[ObjectionRecord]) -> Optional[ObjectionRecord]:
        best = None
        best_score = 0.0
        for item in objections:
            if item.reviewer_id != reviewer_id:
                continue
            label_match = 1.0 if item.issue_label == issue_label else 0.0
            similarity = _cosine_similarity(
                f"{item.issue_label} {item.objection} {item.evidence_needed}",
                f"{issue_label} {objection_text}",
            )
            score = (0.6 * label_match) + (0.4 * similarity)
            if score > best_score:
                best = item
                best_score = score
        if best_score >= 0.60:
            return best
        return None

    def _resolve_objections(self, objections: list[ObjectionRecord], reviewer_id: str,
                            issue_label: str, note: str, meeting_index: int,
                            addresses: list[str] | None = None) -> list[str]:
        addresses = [str(item) for item in (addresses or []) if str(item).strip()]
        resolved = []
        for item in objections:
            if item.status == "resolved":
                continue
            if item.reviewer_id != reviewer_id:
                continue
            if addresses and item.objection_id not in addresses and item.issue_label not in addresses:
                continue
            if issue_label != "resolved" and item.issue_label != issue_label:
                continue
            item.status = "resolved"
            item.last_meeting = meeting_index
            item.resolution_note = note
            item.history.append({
                "meeting": meeting_index,
                "event": "resolved",
                "note": note,
            })
            resolved.append(item.objection_id)
        return resolved

    def _register_objection(self, objections: list[ObjectionRecord], hypothesis_id: str,
                            reviewer: ReviewerProfile, critique: dict,
                            meeting_index: int) -> ObjectionRecord:
        issue_label = str(critique.get("issue_label", "") or "evidence").strip().lower()
        objection_text = str(critique.get("objection", "") or "").strip()
        evidence_needed = str(critique.get("evidence_needed", "") or "").strip()
        matched = self._find_matching_objection(
            reviewer.reviewer_id,
            issue_label,
            objection_text,
            objections,
        )
        if matched is None:
            matched = ObjectionRecord(
                objection_id=f"obj-{_slug(issue_label)}-{uuid.uuid4().hex[:8]}",
                hypothesis_id=hypothesis_id,
                reviewer_id=reviewer.reviewer_id,
                issue_label=issue_label,
                objection=objection_text,
                evidence_needed=evidence_needed,
                status="open",
                created_meeting=meeting_index,
                last_meeting=meeting_index,
                revisit_count=1,
            )
            objections.append(matched)
        else:
            matched.objection = objection_text or matched.objection
            matched.evidence_needed = evidence_needed or matched.evidence_needed
            matched.last_meeting = meeting_index
            matched.revisit_count += 1
            matched.status = "open"

        matched.history.append({
            "meeting": meeting_index,
            "event": "criticized",
            "reviewer_id": reviewer.reviewer_id,
            "issue_label": issue_label,
            "objection": objection_text,
        })
        return matched

    def _normalize_task_payload(self, raw_payload) -> list[dict]:
        tasks = []
        if isinstance(raw_payload, list):
            for item in raw_payload:
                if isinstance(item, str):
                    statement = item.strip()
                    if statement:
                        tasks.append({
                            "task": statement,
                            "addresses": [],
                            "success_criteria": "",
                        })
                elif isinstance(item, dict):
                    statement = str(item.get("task", "") or item.get("statement", "") or "").strip()
                    if not statement:
                        continue
                    addresses = item.get("addresses", []) or item.get("addresses_objection_ids", []) or []
                    if not isinstance(addresses, list):
                        addresses = [addresses]
                    tasks.append({
                        "task": statement,
                        "addresses": [str(token).strip() for token in addresses if str(token).strip()],
                        "success_criteria": str(item.get("success_criteria", "") or "").strip(),
                    })
        return tasks[:2]

    def _find_task_node(self, statement: str) -> Optional[str]:
        for node_id, data in self.brain.nodes_by_type(NodeType.TASK):
            if str(data.get("statement", "") or "").strip() == statement.strip():
                return node_id
        return None

    def _create_task_node(self, task_payload: dict, target_id: str,
                          meeting_index: int) -> str:
        task_statement = task_payload["task"]
        existing_id = self._find_task_node(task_statement)
        if existing_id:
            if not self.brain.get_edge(existing_id, target_id):
                self.brain.add_edge(
                    existing_id,
                    target_id,
                    Edge(
                        type=EdgeType.TESTS,
                        narration="Existing lab meeting task linked to current hypothesis review.",
                        weight=0.78,
                        confidence=0.88,
                        source=EdgeSource.CONVERSATION,
                    ),
                )
            return existing_id

        task_node = Node(
            statement=task_statement,
            node_type=NodeType.TASK,
            status=NodeStatus.UNCERTAIN,
            epistemic_status=ArtifactStatus.OPEN.value,
            importance=0.82,
            created_by=f"lab_meeting_pi:{meeting_index}",
            source_excerpt=task_payload.get("success_criteria", ""),
        )
        task_id = self.brain.add_node(task_node)
        self.brain.add_edge(
            task_id,
            target_id,
            Edge(
                type=EdgeType.TESTS,
                narration="Lab meeting task generated to close a reviewer objection.",
                weight=0.82,
                confidence=0.92,
                source=EdgeSource.CONVERSATION,
            ),
        )
        if self.observer and hasattr(self.observer, "add_to_agenda"):
            try:
                self.observer.add_to_agenda(
                    text=task_statement,
                    item_type="task",
                    cycle=getattr(self.observer, "cycle_count", 0),
                    node_id=task_id,
                )
            except Exception:
                pass
        return task_id

    def _score_task_closure(self, objection: ObjectionRecord,
                            task_payloads: list[dict]) -> tuple[float, str, list[int]]:
        best_score = 0.0
        best_reason = "No task addressed this objection."
        matched_indexes = []
        for index, task in enumerate(task_payloads):
            explicit = 0.0
            addresses = [item.lower() for item in task.get("addresses", [])]
            if objection.objection_id.lower() in addresses:
                explicit = 1.0
            elif objection.issue_label.lower() in addresses:
                explicit = 0.8

            semantic = _cosine_similarity(
                f"{task.get('task', '')} {task.get('success_criteria', '')}",
                f"{objection.issue_label} {objection.objection} {objection.evidence_needed}",
            )
            score = min(1.0, max(explicit, (0.6 * explicit) + (0.4 * semantic)))
            if score > best_score:
                best_score = score
                best_reason = (
                    f"Best task='{task.get('task', '')}' explicit={explicit:.2f} semantic={semantic:.2f}"
                )
            if score >= 0.35:
                matched_indexes.append(index)
        return best_score, best_reason, sorted(set(matched_indexes))

    def _score_evidence_closure(self, hypothesis_id: str,
                                objection: ObjectionRecord) -> tuple[float, str]:
        best_score = 0.0
        best_reason = "No grounded supporting evidence linked to this objection."
        for source_id, _, edge in self.brain.graph.in_edges(hypothesis_id, data=True):
            edge_type = edge.get("type", "")
            if edge_type not in {
                EdgeType.CONFIRMED_BY.value,
                EdgeType.CORRECTED_BY.value,
                EdgeType.SUPPORTS.value,
                EdgeType.EMPIRICALLY_TESTED.value,
            }:
                continue
            node = self.brain.get_node(source_id)
            if not node:
                continue
            if node.get("epistemic_status") != ArtifactStatus.GROUNDED.value:
                continue
            if node.get("node_type") not in {
                NodeType.EVIDENCE_CLAIM.value,
                NodeType.EMPIRICAL.value,
                NodeType.ANSWER.value,
            }:
                continue
            semantic = _cosine_similarity(
                f"{node.get('statement', '')} {edge.get('narration', '')} {node.get('source_excerpt', '')}",
                f"{objection.issue_label} {objection.objection} {objection.evidence_needed}",
            )
            extraction = float(node.get("extraction_confidence", 0.0) or 0.0)
            quality = float(node.get("source_quality", 0.0) or 0.0)
            score = semantic * max(0.35, (0.55 * quality) + (0.45 * extraction))
            if score > best_score:
                best_score = score
                best_reason = (
                    f"Best evidence='{node.get('statement', '')[:72]}' edge={edge_type} "
                    f"semantic={semantic:.2f} quality={quality:.2f} extraction={extraction:.2f}"
                )
        return min(1.0, best_score), best_reason

    def _choose_target_hypothesis(self) -> tuple[Optional[str], Optional[dict]]:
        candidates = []
        for node_id, data in self.brain.nodes_by_type(NodeType.HYPOTHESIS):
            if data.get("status") not in {
                NodeStatus.UNCERTAIN.value,
                NodeStatus.HYPOTHETICAL.value,
            }:
                continue
            open_objections = sum(
                1 for item in self._get_objections(node_id)
                if item.status != "resolved"
            )
            candidates.append((
                int(open_objections),
                float(data.get("importance", 0.0) or 0.0),
                node_id,
                data,
            ))
        if not candidates:
            return None, None
        candidates.sort(reverse=True)
        _, _, target_id, target_data = candidates[0]
        return target_id, target_data

    def get_hypothesis_state(self, hypothesis_id: str) -> dict:
        objections = self._get_objections(hypothesis_id)
        unresolved = [item.to_dict() for item in objections if item.status != "resolved"]
        return {
            "objections": [item.to_dict() for item in objections],
            "unresolved_objections": unresolved,
            "blocking_objection_count": len(unresolved),
            "publication_blocked": bool(unresolved),
            "critique_history": list(
                (self.state.get("critique_history", {}) or {}).get(hypothesis_id, []) or []
            ),
            "reviewer_roster": list(self.state.get("reviewer_roster", []) or []),
        }

    def hold_meeting(self, cycle: Optional[int] = None) -> dict:
        target_id, target_data = self._choose_target_hypothesis()
        if not target_id or not target_data:
            print("  [Lab Meeting] No uncertain hypotheses to review.")
            return {"status": "skipped", "reason": "no_uncertain_hypothesis"}

        hypothesis_statement = str(target_data.get("statement", "") or "")
        meeting_index = int(self.state.get("meeting_count", 0) or 0) + 1
        self.state["meeting_count"] = meeting_index

        print(f"\n  ── Lab Meeting {meeting_index} ──")
        print(f"     Hypothesis: {hypothesis_statement[:90]}...")

        roster = self._ensure_reviewer_roster(hypothesis_statement)
        selected = self._select_reviewers(hypothesis_statement, count=3)
        for reviewer in roster:
            if reviewer.reviewer_id in {item.reviewer_id for item in selected}:
                reviewer.meetings_attended += 1
                reviewer.last_meeting_cycle = int(cycle or 0)
        self._set_roster(roster)

        objections = self._get_objections(target_id)
        critiques = []
        resolved_ids = []

        for reviewer in selected:
            prior = self._reviewer_prior_objections(reviewer.reviewer_id, objections)
            critique = require_json(self._llm(REVIEWER_CRITIQUE_PROMPT.format(
                reviewer_name=reviewer.name,
                specialty=reviewer.specialty,
                focus=reviewer.focus,
                hypothesis=hypothesis_statement,
                prior_objections=prior,
            ), temperature=0.3), default={}) or {}

            status = str(critique.get("status", "") or "new").strip().lower()
            issue_label = str(critique.get("issue_label", "") or "evidence").strip().lower()
            objection_text = str(critique.get("objection", "") or "").strip()
            addresses = critique.get("addresses_objection_ids", []) or []
            if not isinstance(addresses, list):
                addresses = [addresses]

            if status == "resolved" or issue_label == "resolved":
                resolved_ids.extend(self._resolve_objections(
                    objections,
                    reviewer_id=reviewer.reviewer_id,
                    issue_label=issue_label,
                    note=objection_text or "Reviewer marked prior objection resolved.",
                    meeting_index=meeting_index,
                    addresses=addresses,
                ))
                critiques.append({
                    "reviewer_id": reviewer.reviewer_id,
                    "reviewer_name": reviewer.name,
                    "issue_label": issue_label,
                    "status": "resolved",
                    "objection": objection_text,
                    "addresses": addresses,
                })
                continue

            objection = self._register_objection(
                objections,
                hypothesis_id=target_id,
                reviewer=reviewer,
                critique=critique,
                meeting_index=meeting_index,
            )
            critiques.append({
                "reviewer_id": reviewer.reviewer_id,
                "reviewer_name": reviewer.name,
                "issue_label": objection.issue_label,
                "status": status,
                "objection_id": objection.objection_id,
                "objection": objection.objection,
                "evidence_needed": objection.evidence_needed,
            })

        open_objections = [item for item in objections if item.status != "resolved"]
        open_objection_block = self._format_open_objections(open_objections)
        critiques_block = "\n".join(
            f"- [{item.get('reviewer_name', 'reviewer')}] "
            f"({item.get('issue_label', 'issue')}/{item.get('status', 'new')}) "
            f"{item.get('objection', '')}"
            for item in critiques
        ) or "none"

        raw_tasks = self._llm(TASK_SYNTHESIS_PROMPT.format(
            hypothesis=hypothesis_statement,
            critiques=critiques_block,
            open_objections=open_objection_block,
        ), temperature=0.2)
        task_payloads = self._normalize_task_payload(require_json(raw_tasks, default=[]))

        task_ids = []
        for task_payload in task_payloads:
            task_id = self._create_task_node(task_payload, target_id=target_id, meeting_index=meeting_index)
            task_ids.append(task_id)

        closure_scores = []
        for objection in objections:
            if objection.status == "resolved":
                objection.closure_score = 1.0
                objection.closure_reason = objection.resolution_note or "Reviewer marked objection resolved."
                closure_scores.append(1.0)
                continue
            task_score, task_reason, matched_task_indexes = self._score_task_closure(objection, task_payloads)
            evidence_score, evidence_reason = self._score_evidence_closure(target_id, objection)
            objection.closure_score = max(evidence_score, min(0.25, task_score * 0.25))
            objection.closure_reason = (
                f"evidence: {evidence_reason} | task: {task_reason}"
            )
            if task_score >= 0.45:
                objection.status = "tasked"
                linked_task_ids = [
                    task_ids[index] for index in matched_task_indexes
                    if 0 <= index < len(task_ids)
                ]
                objection.task_ids = list(dict.fromkeys(objection.task_ids + linked_task_ids))
            else:
                objection.status = "open"
            objection.history.append({
                "meeting": meeting_index,
                "event": "task_scored",
                "task_score": task_score,
                "evidence_score": evidence_score,
                "closure_score": objection.closure_score,
                "reason": objection.closure_reason,
                "matched_task_ids": list(objection.task_ids),
            })
            closure_scores.append(objection.closure_score)

        self._set_objections(target_id, objections)

        meeting_record = {
            "meeting_index": meeting_index,
            "cycle": int(cycle or 0),
            "hypothesis_id": target_id,
            "hypothesis_statement": hypothesis_statement,
            "reviewers": [
                {
                    "reviewer_id": reviewer.reviewer_id,
                    "name": reviewer.name,
                    "specialty": reviewer.specialty,
                }
                for reviewer in selected
            ],
            "critiques": critiques,
            "resolved_objection_ids": resolved_ids,
            "open_objection_ids": [
                item.objection_id for item in objections if item.status != "resolved"
            ],
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "closure_score": (
                round(sum(closure_scores) / len(closure_scores), 3)
                if closure_scores else 1.0
            ),
        }
        self._append_history(target_id, meeting_record)
        self._save_state()

        unresolved = sum(1 for item in objections if item.status != "resolved")
        print(
            "     Reviewers="
            f"{len(selected)} objections_open={unresolved} tasks={len(task_ids)} "
            f"closure={meeting_record['closure_score']:.2f}"
        )
        print("  ── Lab Meeting Concluded ──\n")

        return {
            "status": "ok",
            **meeting_record,
        }
