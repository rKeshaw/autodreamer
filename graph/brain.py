import uuid
import time
import json
import networkx as nx
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from persistence import atomic_write_json

# ── Types ────────────────────────────────────────────────────────────────────

class BrainMode(str, Enum):
    FOCUSED      = "focused"       # mission active, everything oriented toward it
    WANDERING    = "wandering"     # no mission or suspended — free association
    TRANSITIONAL = "transitional"  # mission just set — one chaotic reorientation cycle

class NodeType(str, Enum):
    CONCEPT      = "concept"
    QUESTION     = "question"
    HYPOTHESIS   = "hypothesis"
    ANSWER       = "answer"
    SYNTHESIS    = "synthesis"
    GAP          = "gap"
    MISSION      = "mission"
    EMPIRICAL    = "empirical"

class NodeStatus(str, Enum):
    SETTLED      = "settled"
    UNCERTAIN    = "uncertain"
    CONTRADICTED = "contradicted"
    HYPOTHETICAL = "hypothetical"

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

    def to_dict(self):
        return asdict(self)

# ── Brain ─────────────────────────────────────────────────────────────────────

class Brain:
    def __init__(self, decay_rate: float = 0.01, scientificness: float = 0.7):
        self.graph           = nx.DiGraph()
        self.decay_rate      = decay_rate
        self.scientificness  = scientificness
        self.mission: Optional[dict] = None
        self._mode: BrainMode = BrainMode.WANDERING
        self._suspended_mission: Optional[dict] = None  # saved when suspended

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

    def nodes_by_type(self, node_type: NodeType) -> list:
        return [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get('node_type') == node_type.value
        ]

    # ── Edge operations ──────────────────────────────────────────────────────

    def add_edge(self, from_id: str, to_id: str, edge: Edge):
        self.graph.add_edge(from_id, to_id, **edge.to_dict())

    def get_edge(self, from_id: str, to_id: str) -> Optional[dict]:
        return self.graph.edges.get((from_id, to_id))

    def update_edge(self, from_id: str, to_id: str, **kwargs):
        if self.graph.has_edge(from_id, to_id):
            self.graph.edges[from_id, to_id].update(kwargs)
            self.graph.edges[from_id, to_id]['updated_at'] = time.time()

    def neighbors(self, node_id: str) -> list:
        return list(self.graph.successors(node_id))

    # ── Analogy helper ────────────────────────────────────────────────────────

    def add_analogy_edge(self, from_id: str, to_id: str,
                         depth: str, narration: str,
                         source: EdgeSource = EdgeSource.CONVERSATION) -> Edge:
        type_map = {
            "surface":     EdgeType.SURFACE_ANALOGY,
            "structural":  EdgeType.STRUCTURAL_ANALOGY,
            "isomorphism": EdgeType.DEEP_ISOMORPHISM,
        }
        etype  = type_map.get(depth, EdgeType.STRUCTURAL_ANALOGY)
        weight = ANALOGY_WEIGHTS.get(etype, 0.4)
        edge   = Edge(
            type          = etype,
            narration     = narration,
            weight        = weight,
            confidence    = weight,
            source        = source,
            analogy_depth = depth
        )
        self.add_edge(from_id, to_id, edge)
        return edge

    # ── NREM ─────────────────────────────────────────────────────────────────

    def proximal_reinforce(self, boost: float = 0.05, threshold: float = 0.6):
        reinforced = 0
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 0.5)
            if weight >= threshold:
                self.graph.edges[u, v]['weight']     = min(0.98, weight + boost)
                self.graph.edges[u, v]['updated_at'] = time.time()
                reinforced += 1
        print(f"  NREM pass: reinforced {reinforced} strong edges")
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

    # ── Decay ────────────────────────────────────────────────────────────────

    def apply_decay(self, elapsed_days: float = 1.0):
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
            "hypotheses":         len(self.nodes_by_type(NodeType.HYPOTHESIS)),
            "open_questions":     len(self.nodes_by_type(NodeType.QUESTION)),
            "answers":            len(self.nodes_by_type(NodeType.ANSWER)),
            "empirical":          len(self.nodes_by_type(NodeType.EMPIRICAL)),
            "mode":               self._mode.value,
            "mission":            self.mission['question']
                                  if self.mission else None,
        }
