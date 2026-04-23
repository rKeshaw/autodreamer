import uuid
import time
import json
import networkx as nx
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from persistence import atomic_write_json
from graph.episodic import EpisodicStrip
from scientist_workspace import ArtifactStatus, ScientistWorkspace, WorkspaceNode

# ── Types ────────────────────────────────────────────────────────────────────

class BrainMode(str, Enum):
    FOCUSED      = "focused"       # mission active, everything oriented toward it
    WANDERING    = "wandering"     # no mission or suspended — free association
    TRANSITIONAL = "transitional"  # mission just set — one chaotic reorientation cycle

class NodeType(str, Enum):
    CONCEPT      = "concept"
    SOURCE       = "source"
    EVIDENCE_CLAIM = "evidence_claim"
    QUESTION     = "question"
    HYPOTHESIS   = "hypothesis"
    ANSWER       = "answer"
    SYNTHESIS    = "synthesis"
    GAP          = "gap"
    MISSION      = "mission"
    EMPIRICAL    = "empirical"
    TASK         = "task"

class NodeStatus(str, Enum):
    SETTLED        = "settled"
    UNCERTAIN      = "uncertain"
    CONTRADICTED   = "contradicted"
    HYPOTHETICAL   = "hypothetical"
    LACKS_EVIDENCE = "lacks_evidence"  # searched but found nothing either way
    TESTING        = "testing"         # currently being researched

class EdgeType(str, Enum):
    SUPPORTS           = "supports"
    CAUSES             = "causes"
    CONTRADICTS        = "contradicts"
    SURFACE_ANALOGY    = "surface_analogy"
    STRUCTURAL_ANALOGY = "structural_analogy"
    DEEP_ISOMORPHISM   = "deep_isomorphism"
    ANALOGOUS_TO       = "analogous_to"   # legacy
    ASSOCIATED         = "associated"
    ANSWERS            = "answers"
    PARTIAL            = "partial"
    TOWARD_MISSION     = "toward_mission"
    EMPIRICALLY_TESTED = "empirically_tested"
    CORRECTED_BY       = "corrected_by"     # evidence that disproved a hypothesis
    TESTS              = "tests"            # thinker query → hypothesis link
    DERIVED_FROM       = "derived_from"     # hypothesis → seed claim nodes
    CONFIRMED_BY       = "confirmed_by"     # evidence that supports hypothesis

class EdgeSource(str, Enum):
    CONVERSATION  = "conversation"
    RESEARCH      = "research"
    READING       = "reading"       # from Reader module — absorption mode
    DREAM         = "dream"
    CONSOLIDATION = "consolidation"
    SANDBOX       = "sandbox"

ANALOGY_WEIGHTS = {
    EdgeType.SURFACE_ANALOGY:    0.25,
    EdgeType.STRUCTURAL_ANALOGY: 0.55,
    EdgeType.DEEP_ISOMORPHISM:   0.85,
    EdgeType.ANALOGOUS_TO:       0.40,
}

# ── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    statement: str
    node_type: NodeType    = NodeType.CONCEPT
    cluster: str           = "unclustered"
    status: NodeStatus     = NodeStatus.UNCERTAIN
    epistemic_status: str  = ArtifactStatus.OPEN.value
    importance: float      = 0.5
    created_at: float      = field(default_factory=time.time)
    activated_at: float    = field(default_factory=time.time)
    id: str                = field(default_factory=lambda: str(uuid.uuid4()))
    predicted_answer: str  = ""
    testable_by: str       = ""
    incubation_age: int    = 0
    first_queued_cycle: int= 0
    empirical_result: str  = ""
    empirical_code: str    = ""
    empirical_metrics: dict = field(default_factory=dict)
    experiment_artifacts: dict = field(default_factory=dict)
    # ── Confidence tracking ──
    source_quality: float      = 0.5   # 1.0=primary source, 0.5=synthesis, 0.3=dream
    verification_count: int    = 0     # how many times reinforced from independent sources
    last_verified: float       = 0.0   # timestamp of last reinforcement
    source_ids: list[str]      = field(default_factory=list)
    source_refs: list[str]     = field(default_factory=list)
    provenance_spans: list[dict] = field(default_factory=list)
    extraction_confidence: float = 0.0
    created_by: str            = ""
    mission_relevance: float   = 0.0
    source_type: str           = ""
    source_excerpt: str        = ""

    def touch(self):
        self.activated_at = time.time()

    def to_dict(self):
        return asdict(self)

# ── Edge ─────────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    type: EdgeType
    narration: str
    weight: float          = 0.5
    confidence: float      = 0.5
    source: EdgeSource     = EdgeSource.CONVERSATION
    created_at: float      = field(default_factory=time.time)
    updated_at: float      = field(default_factory=time.time)
    decay_exempt: bool     = False
    analogy_depth: str     = ""
    matcher_report: dict   = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

# ── Brain ─────────────────────────────────────────────────────────────────────

class Brain:
    MAX_WORKING_MEMORY = 10        # top items under active investigation
    MAX_ACTIVE_HYPOTHESES = 15     # max unresolved hypothesis nodes in the graph

    def __init__(self, decay_rate: float = 0.01, scientificness: float = 0.7):
        self.graph           = nx.DiGraph()
        self.decay_rate      = decay_rate
        self.scientificness  = scientificness
        self.mission: Optional[dict] = None
        self._mode: BrainMode = BrainMode.WANDERING
        self._suspended_mission: Optional[dict] = None
        self.working_memory: list[str] = []  # ordered list of node IDs
        
        # Neuromodulators
        self.dopamine: float    = 0.5
        self.frustration: float = 0.0

        # Episodic Memory
        self.episodic = EpisodicStrip()

    # ── Mode ──────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> BrainMode:
        return self._mode

    def get_mode(self) -> str:
        return self._mode.value

    def set_mode(self, mode: BrainMode):
        old = self._mode
        self._mode = mode
        print(f"Brain mode: {old.value} → {mode.value}")

    def is_focused(self) -> bool:
        return self._mode == BrainMode.FOCUSED

    def is_wandering(self) -> bool:
        return self._mode == BrainMode.WANDERING

    def is_transitional(self) -> bool:
        return self._mode == BrainMode.TRANSITIONAL

    # ── Mission ───────────────────────────────────────────────────────────────

    def set_mission(self, question: str, context: str = "") -> str:
        # remove old mission node
        if self.mission:
            old_id = self.mission.get("id")
            if old_id and old_id in self.graph.nodes:
                self.graph.remove_node(old_id)

        node = Node(
            statement        = question,
            node_type        = NodeType.MISSION,
            cluster          = "mission",
            status           = NodeStatus.UNCERTAIN,
            epistemic_status = ArtifactStatus.OPEN.value,
            importance       = 1.0,
            predicted_answer = context
        )
        nid = self.add_node(node)
        self.mission = {"id": nid, "question": question, "context": context}
        # entering transitional mode — one chaotic reorientation cycle
        self.set_mode(BrainMode.TRANSITIONAL)
        print(f"Mission set: {question}...")
        return nid

    def suspend_mission(self):
        """Suspend mission — enter wandering mode. Mission is preserved."""
        if self.mission:
            self._suspended_mission = self.mission
            self.set_mode(BrainMode.WANDERING)
            print("Mission suspended — entering wandering mode.")

    def resume_mission(self):
        """Resume suspended mission — enter focused mode."""
        if self._suspended_mission:
            self.mission = self._suspended_mission
            self._suspended_mission = None
            self.set_mode(BrainMode.FOCUSED)
            print(f"Mission resumed: {self.mission['question']}...")
        elif self.mission:
            self.set_mode(BrainMode.FOCUSED)

    def complete_transition(self):
        """Called after transitional cycle — move to focused."""
        if self._mode == BrainMode.TRANSITIONAL:
            self.set_mode(BrainMode.FOCUSED)

    def get_mission(self) -> Optional[dict]:
        return self.mission

    def link_to_mission(self, node_id: str, narration: str,
                        strength: float = 0.6):
        """Only links if mission is active (focused or transitional)."""
        if not self.mission or self.is_wandering():
            return
        mission_id = self.mission.get("id")
        if not mission_id:
            return
        if (self.graph.has_edge(node_id, mission_id) or
                self.graph.has_edge(mission_id, node_id)):
            if self.graph.has_edge(node_id, mission_id):
                cur = self.graph.edges[node_id, mission_id].get('weight', 0)
                self.graph.edges[node_id, mission_id]['weight'] = min(0.95, cur + 0.05)
            return
        edge = Edge(
            type         = EdgeType.TOWARD_MISSION,
            narration    = narration,
            weight       = strength,
            confidence   = 0.6,
            source       = EdgeSource.CONSOLIDATION,
            decay_exempt = True
        )
        self.add_edge(node_id, mission_id, edge)

    # ── Node operations ──────────────────────────────────────────────────────

    def add_node(self, node: Node) -> str:
        self.graph.add_node(node.id, **node.to_dict())
        return node.id

    def get_node(self, node_id: str) -> Optional[dict]:
        return self.graph.nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs):
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id].update(kwargs)

    def all_nodes(self) -> list:
        return list(self.graph.nodes(data=True))

    def nodes_by_type(self, node_type: NodeType | str) -> list:
        if isinstance(node_type, NodeType):
            node_type_value = node_type.value
        else:
            node_type_value = str(node_type).strip().lower()
        return [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get('node_type') == node_type_value
        ]

    # ── Hypothesis throttle ──────────────────────────────────────────────────

    def count_active_hypotheses(self) -> int:
        """Count hypothesis nodes that are not yet resolved (settled or contradicted)."""
        return sum(
            1 for _, data in self.graph.nodes(data=True)
            if data.get('node_type') == NodeType.HYPOTHESIS.value
            and data.get('status') not in (
                NodeStatus.SETTLED.value,
                NodeStatus.CONTRADICTED.value,
            )
        )

    def can_spawn_hypothesis(self) -> bool:
        """Check if the graph has room for more active hypotheses."""
        return self.count_active_hypotheses() < self.MAX_ACTIVE_HYPOTHESES

    def nodes_by_epistemic_status(self, epistemic_status: str) -> list:
        return [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get('epistemic_status') == epistemic_status
        ]

    def create_source_node(self, title: str, reference: str,
                           source_type: str = "web",
                           created_by: str = "system",
                           excerpt: str = "") -> str:
        statement = (title or reference or "source").strip()
        node = Node(
            statement=statement,
            node_type=NodeType.SOURCE,
            cluster="source",
            status=NodeStatus.SETTLED,
            epistemic_status=ArtifactStatus.GROUNDED.value,
            importance=0.2,
            source_refs=[reference] if reference else [],
            created_by=created_by,
            source_type=source_type,
            source_excerpt=(excerpt or "")[:600],
        )
        return self.add_node(node)

    def _expand_source_refs(self, node_data: dict) -> list[str]:
        refs = list(node_data.get("source_refs", []) or [])
        for source_id in node_data.get("source_ids", []) or []:
            source_node = self.get_node(source_id)
            if not source_node:
                continue
            refs.extend(source_node.get("source_refs", []) or [])
            source_stmt = source_node.get("statement", "")
            if source_stmt and source_stmt not in refs:
                refs.append(source_stmt)
        return list(dict.fromkeys(refs))

    def _workspace_node(self, node_id: str, data: dict) -> WorkspaceNode:
        return WorkspaceNode(
            id=node_id,
            node_type=str(data.get("node_type", "")),
            statement=data.get("statement", ""),
            epistemic_status=data.get(
                "epistemic_status",
                ArtifactStatus.OPEN.value,
            ),
            importance=float(data.get("importance", 0.0) or 0.0),
            mission_relevance=float(data.get("mission_relevance", 0.0) or 0.0),
            source_ids=list(data.get("source_ids", []) or []),
            source_refs=self._expand_source_refs(data),
            provenance_spans=[
                dict(span) for span in (data.get("provenance_spans", []) or [])
                if isinstance(span, dict)
            ],
            extraction_confidence=float(
                data.get("extraction_confidence", 0.0) or 0.0
            ),
            created_by=data.get("created_by", ""),
        )

    def build_workspace(self, embedding_index=None, query: str = "",
                        max_evidence: int = 6, max_questions: int = 4,
                        max_hypotheses: int = 4, max_priors: int = 4,
                        max_tasks: int = 4) -> ScientistWorkspace:
        mission = self.get_mission() or {}
        scored_ids: dict[str, float] = {}

        for rank, (nid, data) in enumerate(self.get_working_memory()):
            scored_ids[nid] = max(
                scored_ids.get(nid, 0.0),
                1.2 + data.get("importance", 0.0) - (0.05 * rank),
            )

        if embedding_index and getattr(embedding_index, "size", 0) > 0 and query:
            from embedding import embed as shared_embed

            query_emb = shared_embed(query)
            matches = embedding_index.query(query_emb, threshold=0.20, top_k=14)
            for rank, (nid, score) in enumerate(matches):
                scored_ids[nid] = max(
                    scored_ids.get(nid, 0.0),
                    float(score) + max(0.0, 0.25 - (rank * 0.01)),
                )

        mission_id = mission.get("id")
        if mission_id:
            for from_id, to_id, edge in self.graph.edges(data=True):
                if to_id == mission_id:
                    scored_ids[from_id] = max(
                        scored_ids.get(from_id, 0.0),
                        float(edge.get("weight", 0.0)) + 0.35,
                    )

        for nid, data in self.all_nodes():
            if data.get("node_type") == NodeType.SOURCE.value:
                continue
            importance = float(data.get("importance", 0.0) or 0.0)
            mission_relevance = float(data.get("mission_relevance", 0.0) or 0.0)
            baseline = importance + (0.4 * mission_relevance)
            if baseline >= 0.65:
                scored_ids[nid] = max(scored_ids.get(nid, 0.0), baseline)

        ordered = sorted(scored_ids.items(), key=lambda item: item[1], reverse=True)

        grounded_evidence: list[WorkspaceNode] = []
        working_hypotheses: list[WorkspaceNode] = []
        prior_claims: list[WorkspaceNode] = []
        active_questions: list[WorkspaceNode] = []
        next_tasks: list[WorkspaceNode] = []

        for nid, _ in ordered:
            data = self.get_node(nid)
            if not data:
                continue
            node_type = data.get("node_type", "")
            epistemic = data.get("epistemic_status", ArtifactStatus.OPEN.value)
            node = self._workspace_node(nid, data)

            if (node_type in {
                NodeType.EVIDENCE_CLAIM.value,
                NodeType.ANSWER.value,
                NodeType.EMPIRICAL.value,
            } and epistemic == ArtifactStatus.GROUNDED.value):
                if len(grounded_evidence) < max_evidence:
                    grounded_evidence.append(node)
                continue

            if node_type == NodeType.HYPOTHESIS.value:
                if epistemic == ArtifactStatus.PRIOR.value:
                    if len(prior_claims) < max_priors:
                        prior_claims.append(node)
                elif epistemic in {
                    ArtifactStatus.SPECULATIVE.value,
                    ArtifactStatus.CONTRADICTED.value,
                    ArtifactStatus.LACKS_EVIDENCE.value,
                }:
                    continue
                elif len(working_hypotheses) < max_hypotheses:
                    working_hypotheses.append(node)
                continue

            if node_type in {NodeType.QUESTION.value, NodeType.GAP.value}:
                if len(active_questions) < max_questions:
                    active_questions.append(node)
                continue

            if node_type == NodeType.TASK.value:
                if len(next_tasks) < max_tasks:
                    next_tasks.append(node)

        contradictions = []
        for u, v, edge in self.graph.edges(data=True):
            if edge.get("type") != EdgeType.CONTRADICTS.value:
                continue
            node_u = self.get_node(u)
            node_v = self.get_node(v)
            if not node_u or not node_v:
                continue
            contradictions.append(
                f"{node_u.get('statement', '')} VS {node_v.get('statement', '')}"
            )
            if len(contradictions) >= 4:
                break

        return ScientistWorkspace(
            mission=mission.get("question", ""),
            mission_context=mission.get("context", ""),
            active_questions=active_questions,
            grounded_evidence=grounded_evidence,
            working_hypotheses=working_hypotheses,
            prior_claims=prior_claims,
            next_tasks=next_tasks,
            contradictions=contradictions,
        )

    # ── Working memory ────────────────────────────────────────────────────────

    def focus_on(self, node_id: str):
        """Add a node to working memory (front = highest priority)."""
        if node_id in self.working_memory:
            self.working_memory.remove(node_id)
        self.working_memory.insert(0, node_id)
        if len(self.working_memory) > self.MAX_WORKING_MEMORY:
            self.working_memory = self.working_memory[:self.MAX_WORKING_MEMORY]
        self.update_node(node_id, activated_at=time.time())

    def unfocus(self, node_id: str):
        """Remove a node from working memory."""
        if node_id in self.working_memory:
            self.working_memory.remove(node_id)

    def get_working_memory(self) -> list[tuple[str, dict]]:
        """Return working memory nodes with their data."""
        result = []
        for nid in self.working_memory:
            data = self.get_node(nid)
            if data:
                result.append((nid, data))
            else:
                self.working_memory.remove(nid)
        return result

    def is_in_focus(self, node_id: str) -> bool:
        return node_id in self.working_memory

    # ── Edge operations ──────────────────────────────────────────────────────

    def add_edge(self, from_id: str, to_id: str, edge: Edge):
        if not from_id or not to_id or from_id == to_id:
            return
        self.graph.add_edge(from_id, to_id, **edge.to_dict())

    def get_edge(self, from_id: str, to_id: str) -> Optional[dict]:
        return self.graph.edges.get((from_id, to_id))

    def update_edge(self, from_id: str, to_id: str, **kwargs):
        if self.graph.has_edge(from_id, to_id):
            self.graph.edges[from_id, to_id].update(kwargs)
            self.graph.edges[from_id, to_id]['updated_at'] = time.time()

    def neighbors(self, node_id: str) -> list:
        return list(self.graph.successors(node_id))

    # ── Pruning and synchronization ─────────────────────────────────────────

    def snapshot_nodes(self) -> list[tuple[str, dict]]:
        """Return a detached snapshot of current nodes for safe read operations."""
        return [(nid, dict(data)) for nid, data in self.graph.nodes(data=True)]

    def prune_nodes(self, node_ids: list[str] | set[str],
                    preserve_ids: set[str] | None = None) -> dict:
        """Remove nodes from the graph while maintaining core internal references."""
        preserve = set(preserve_ids or set())
        removable = []
        removed_nodes = {}

        for nid in list(dict.fromkeys(node_ids or [])):
            if not nid or nid in preserve:
                continue
            if nid not in self.graph.nodes:
                continue
            removable.append(nid)
            removed_nodes[nid] = dict(self.graph.nodes[nid])

        if not removable:
            return {
                "removed_node_ids": [],
                "removed_nodes": {},
                "removed_count": 0,
            }

        for nid in removable:
            self.graph.remove_node(nid)

        if self.mission and self.mission.get("id") in removed_nodes:
            self.mission = None
        if self._suspended_mission and self._suspended_mission.get("id") in removed_nodes:
            self._suspended_mission = None

        self.working_memory = [nid for nid in self.working_memory if nid not in removed_nodes]

        return {
            "removed_node_ids": removable,
            "removed_nodes": removed_nodes,
            "removed_count": len(removable),
        }

    def synchronized_prune(self, node_ids: list[str] | set[str],
                           embedding_index=None,
                           observer=None,
                           notebook=None,
                           insight_buffer=None,
                           preserve_ids: set[str] | None = None) -> dict:
        """Prune nodes and synchronize dependent structures to avoid dangling refs."""
        result = self.prune_nodes(node_ids=node_ids, preserve_ids=preserve_ids)
        removed_ids = set(result.get("removed_node_ids", []))

        if not removed_ids:
            result["sync"] = {
                "embedding_index_removed": 0,
                "observer": {"cleaned": 0},
                "notebook": {"cleaned": 0},
                "insight_buffer": {"cleaned": 0},
            }
            return result

        sync_report = {
            "embedding_index_removed": 0,
            "observer": {"cleaned": 0},
            "notebook": {"cleaned": 0},
            "insight_buffer": {"cleaned": 0},
        }

        if embedding_index is not None:
            removed_count = 0
            if hasattr(embedding_index, "prune_node_ids"):
                try:
                    removed_count = int(embedding_index.prune_node_ids(removed_ids) or 0)
                except Exception:
                    removed_count = 0
            else:
                for nid in removed_ids:
                    if hasattr(embedding_index, "remove"):
                        embedding_index.remove(nid)
                        removed_count += 1
                if hasattr(embedding_index, "flush"):
                    embedding_index.flush()
            sync_report["embedding_index_removed"] = removed_count

        removed_nodes = result.get("removed_nodes", {})
        if observer is not None and hasattr(observer, "remove_node_references"):
            try:
                observer_report = observer.remove_node_references(removed_ids)
                if isinstance(observer_report, dict):
                    sync_report["observer"] = observer_report
            except Exception:
                pass

        if notebook is not None and hasattr(notebook, "remove_node_references"):
            try:
                notebook_report = notebook.remove_node_references(removed_ids, removed_nodes=removed_nodes)
                if isinstance(notebook_report, dict):
                    sync_report["notebook"] = notebook_report
            except Exception:
                pass

        if insight_buffer is not None and hasattr(insight_buffer, "remove_node_references"):
            try:
                buffer_report = insight_buffer.remove_node_references(removed_ids)
                if isinstance(buffer_report, dict):
                    sync_report["insight_buffer"] = buffer_report
            except Exception:
                pass

        result["sync"] = sync_report
        return result

    # ── Analogy helper ────────────────────────────────────────────────────────

    def add_analogy_edge(self, from_id: str, to_id: str,
                         depth: str, narration: str,
                         source: EdgeSource = EdgeSource.CONVERSATION,
                         matcher_report: dict | None = None) -> Edge | None:
        type_map = {
            "surface":     EdgeType.SURFACE_ANALOGY,
            "structural":  EdgeType.STRUCTURAL_ANALOGY,
            "isomorphism": EdgeType.DEEP_ISOMORPHISM,
        }
        etype  = type_map.get(depth, EdgeType.STRUCTURAL_ANALOGY)
        if etype == EdgeType.DEEP_ISOMORPHISM:
            report = matcher_report or {}
            passed = bool(report.get("passed", report.get("isomorphic", False)))
            if not passed:
                return None
        weight = ANALOGY_WEIGHTS.get(etype, 0.4)
        edge   = Edge(
            type          = etype,
            narration     = narration,
            weight        = weight,
            confidence    = weight,
            source        = source,
            analogy_depth = depth,
            matcher_report = dict(matcher_report or {}),
        )
        self.add_edge(from_id, to_id, edge)
        return edge

    # ── NREM ─────────────────────────────────────────────────────────────────

    def proximal_reinforce(self, boost: float = 0.05, threshold: float = 0.6):
        def _activation_score(node_data, now):
            activated_at = node_data.get('activated_at', 0.0)
            if not activated_at:
                return 0.0
            age = max(0.0, now - activated_at)
            return max(0.0, 1.0 - (age / 600.0))

        def _type_bonus(edge_type):
            if edge_type in {
                EdgeType.SUPPORTS.value,
                EdgeType.CAUSES.value,
                EdgeType.CONTRADICTS.value,
                EdgeType.ANSWERS.value,
                EdgeType.TOWARD_MISSION.value,
            }:
                return 0.20
            if edge_type in {
                EdgeType.STRUCTURAL_ANALOGY.value,
                EdgeType.DEEP_ISOMORPHISM.value,
            }:
                return 0.14
            if edge_type == EdgeType.SURFACE_ANALOGY.value:
                return 0.05
            return 0.0

        now = time.time()
        candidates = []
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 0.5)
            if weight < threshold:
                continue
            node_u = self.get_node(u) or {}
            node_v = self.get_node(v) or {}
            avg_importance = (
                node_u.get('importance', 0.5) + node_v.get('importance', 0.5)
            ) / 2.0
            avg_activation = (
                _activation_score(node_u, now) + _activation_score(node_v, now)
            ) / 2.0
            priority = (
                weight +
                (0.22 * avg_importance) +
                (0.18 * avg_activation) +
                _type_bonus(data.get('type', ''))
            )
            candidates.append((priority, u, v, data))

        if not candidates:
            print("  NREM pass: reinforced 0 strong edges")
            return 0

        candidates.sort(key=lambda item: item[0], reverse=True)
        reinforce_count = max(1, int(round(len(candidates) * 0.35)))

        reinforced = 0
        for _, u, v, data in candidates[:reinforce_count]:
            weight = data.get('weight', 0.5)
            self.graph.edges[u, v]['weight'] = min(0.98, weight + boost)
            self.graph.edges[u, v]['updated_at'] = now
            reinforced += 1

        print(f"  NREM pass: reinforced {reinforced} prioritized edges")
        return reinforced

    # ── Insight restructuring ─────────────────────────────────────────────────

    def restructure_around_insight(self, node_a_id: str, node_b_id: str,
                                   narration: str, edge_type: str = "") -> dict:
        strength_map = {
            "deep_isomorphism":   1.0,
            "structural_analogy": 0.6,
            "surface_analogy":    0.3,
            "analogous_to":       0.5,
        }
        strength = strength_map.get(edge_type, 0.5)
        boost    = 0.08 * strength

        summary = {
            "node_a": node_a_id, "node_b": node_b_id,
            "narration": narration, "strength": strength,
            "nodes_updated": [], "edges_reinforced": 0,
            "contradictions_resolved": [], "mission_linked": False
        }

        for nid in [node_a_id, node_b_id]:
            node = self.get_node(nid)
            if node:
                self.update_node(nid,
                    importance = min(1.0, node.get('importance', 0.5) + 0.1 * strength),
                    status     = (NodeStatus.SETTLED.value
                                  if (node.get('status') == NodeStatus.UNCERTAIN.value
                                      and strength > 0.7)
                                  else node.get('status'))
                )
                summary["nodes_updated"].append(nid)

        for nid in [node_a_id, node_b_id]:
            for neighbor in self.neighbors(nid):
                edge = self.get_edge(nid, neighbor)
                if edge:
                    self.update_edge(nid, neighbor,
                        weight=min(0.95, edge.get('weight', 0.5) + boost))
                    summary["edges_reinforced"] += 1

        for a, b in [(node_a_id, node_b_id), (node_b_id, node_a_id)]:
            edge = self.get_edge(a, b)
            if edge and edge.get('type') == EdgeType.CONTRADICTS.value:
                if strength > 0.7:
                    self.update_edge(a, b,
                        narration  = f"[RESOLVED BY INSIGHT] {narration}",
                        weight     = 0.2, confidence = 0.6)
                    summary["contradictions_resolved"].append((a, b))

        if strength > 0.6 and not self.is_wandering():
            for nid in [node_a_id, node_b_id]:
                self.link_to_mission(nid, f"Insight: {narration}", strength * 0.6)
            summary["mission_linked"] = True

        return summary

    # ── Neuromodulation ───────────────────────────────────────────────────────

    def spike_dopamine(self, amount: float = 0.3):
        self.dopamine = min(1.0, self.dopamine + amount)
        print(f"  [Neuromodulation] Dopamine spike! Level: {self.dopamine:.2f}")

    def increase_frustration(self, amount: float = 0.2):
        self.frustration = min(1.0, self.frustration + amount)
        print(f"  [Neuromodulation] Frustration increased. Level: {self.frustration:.2f}")
        if self.frustration >= 0.8 and not self.is_wandering():
            print("  [Neuromodulation] Frustration threshold reached! Suspending mission...")
            self.suspend_mission()
            self.frustration = 0.4  # Reset partially after abandoning
            self.dopamine = min(self.dopamine, 0.4) # Kill motivation temporarily

    def apply_neuromodulator_decay(self, elapsed_days: float = 1.0):
        # Dopamine returns to baseline 0.5
        if self.dopamine > 0.5:
            self.dopamine = max(0.5, self.dopamine - (0.3 * elapsed_days))
        elif self.dopamine < 0.5:
            self.dopamine = min(0.5, self.dopamine + (0.3 * elapsed_days))
            
        # Frustration decays slowly to 0.0
        if self.frustration > 0.0:
            self.frustration = max(0.0, self.frustration - (0.2 * elapsed_days))

    # ── Decay ────────────────────────────────────────────────────────────────

    def apply_decay(self, elapsed_days: float = 1.0):
        self.apply_neuromodulator_decay(elapsed_days)
        half_life_days = 1.0 / max(self.decay_rate, 1e-9)
        decay_factor = 0.5 ** (elapsed_days / half_life_days)
        for u, v, data in self.graph.edges(data=True):
            if not data.get('decay_exempt', False):
                data['weight'] = max(
                    0.01,
                    data.get('weight', 0.5) * decay_factor
                )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "data/brain.json"):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "graph":   nx.node_link_data(self.graph),
            "mission": self.mission,
            "suspended_mission": self._suspended_mission,
            "mode":    self._mode.value,
            "working_memory": self.working_memory,
            "neuromodulators": {
                "dopamine": self.dopamine,
                "frustration": self.frustration
            },
            "config":  {
                "decay_rate":     self.decay_rate,
                "scientificness": self.scientificness
            }
        }
        atomic_write_json(path, data)
        print(f"Brain saved — {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges | mode: {self._mode.value}")

    def load(self, path: str = "data/brain.json"):
        with open(path, 'r') as f:
            raw = json.load(f)

        if "graph" in raw and "nodes" not in raw:
            self.graph              = nx.node_link_graph(raw["graph"])
            self.mission            = raw.get("mission")
            self._suspended_mission = raw.get("suspended_mission")
            cfg                     = raw.get("config", {})
            self.decay_rate         = cfg.get("decay_rate",     self.decay_rate)
            self.scientificness     = cfg.get("scientificness", self.scientificness)
            mode_str                = raw.get("mode", "wandering")
            try:
                self._mode = BrainMode(mode_str)
            except ValueError:
                self._mode = BrainMode.WANDERING
            self.working_memory = raw.get("working_memory", [])
            nemod = raw.get("neuromodulators", {})
            self.dopamine    = nemod.get("dopamine", 0.5)
            self.frustration = nemod.get("frustration", 0.0)
        else:
            # legacy format
            self.graph = nx.node_link_graph(raw)

        print(f"Brain loaded — {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges | mode: {self._mode.value}"
              + (f" | Mission: {self.mission['question']}"
                 if self.mission else ""))

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get('node_type', 'concept')
            node_types[nt] = node_types.get(nt, 0) + 1

        analogy_breakdown = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get('type', '')
            if et in ('surface_analogy', 'structural_analogy',
                      'deep_isomorphism', 'analogous_to'):
                analogy_breakdown[et] = analogy_breakdown.get(et, 0) + 1

        return {
            "nodes":              len(self.graph.nodes),
            "edges":              len(self.graph.edges),
            "clusters":           len(set(
                d.get('cluster', '?')
                for _, d in self.graph.nodes(data=True)
            )),
            "contradictions":     sum(
                1 for _, _, d in self.graph.edges(data=True)
                if d.get('type') == EdgeType.CONTRADICTS.value
            ),
            "hypotheticals":      sum(
                1 for _, d in self.graph.nodes(data=True)
                if d.get('status') == NodeStatus.HYPOTHETICAL.value
            ),
            "node_types":         node_types,
            "analogy_breakdown":  analogy_breakdown,
            "sources":            len(self.nodes_by_type(NodeType.SOURCE)),
            "evidence_claims":    len(self.nodes_by_type(NodeType.EVIDENCE_CLAIM)),
            "hypotheses":         len(self.nodes_by_type(NodeType.HYPOTHESIS)),
            "open_questions":     len(self.nodes_by_type(NodeType.QUESTION)),
            "answers":            len(self.nodes_by_type(NodeType.ANSWER)),
            "empirical":          len(self.nodes_by_type(NodeType.EMPIRICAL)),
            "tasks":              len(self.nodes_by_type(NodeType.TASK)),
            "mode":               self._mode.value,
            "mission":            self.mission['question']
                                  if self.mission else None,
            "working_memory":     len(self.working_memory),
        }
