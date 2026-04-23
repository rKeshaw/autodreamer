"""
Insight Buffer — Delayed insight mechanism for near-miss connections.

When the system finds two ideas that are close but below the edge threshold
(e.g., similarity 0.45-0.59 when threshold is 0.60), it saves them here.
Each consolidation cycle, the buffer is re-evaluated — if new knowledge has
arrived that strengthens the connection, the pair gets promoted to a real edge.

This mimics the "shower insight" phenomenon: you see fact A on day 1 and
fact B on day 5, and on day 7 you suddenly realize they're connected.
"""

import json
import time
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from graph.brain import Brain, Edge, EdgeType, EdgeSource
from config import THRESHOLDS
from embedding import embed as shared_embed
from llm_utils import llm_call, require_json

# ── Config ────────────────────────────────────────────────────────────────────

BUFFER_PATH        = "data/insight_buffer.json"
BUFFER_LOW         = 0.45    # minimum similarity to enter buffer
MAX_EVALUATIONS    = 10      # prune after this many re-evaluations without improvement
MAX_BUFFER_SIZE    = 200     # hard cap on buffer size
PROMOTION_BOOST    = 0.10    # how much graph context can add to similarity score
NEIGHBOR_BOOST     = 0.05    # bonus per shared neighbor

# ── Prompt for re-evaluation ──────────────────────────────────────────────────

REEVALUATE_PROMPT = """You are examining two ideas that were previously noted as potentially
related but the connection wasn't strong enough to confirm.

Idea A: {node_a}
Idea B: {node_b}
Original deferred claim: {claim}
Original deferred context: {context}

Original similarity when first seen: {original_sim:.2f}
Times re-evaluated so far: {eval_count}

New context — ideas they are each now connected to:
A's neighbors: {a_neighbors}
B's neighbors: {b_neighbors}

In light of the new context, is there now a meaningful relationship between A and B?

Respond with a JSON object:
{{
  "connected": true or false,
  "type": one of ["supports", "causes", "contradicts", "analogy", "associated"],
  "analogy_depth": if type is "analogy", one of ["surface", "structural", "isomorphism"] — else omit,
  "narration": "one sentence explaining the connection",
  "confidence": a float 0.0-1.0 (how certain are you this link exists?)
}}

Confidence rubric:
- 0.1-0.3: Speculative — might be connected but you can't clearly articulate why
- 0.4-0.6: Plausible — the new context reveals a reasonable link
- 0.7-0.9: Strong — the connection is now clearly supported by the graph context
- 1.0: Definitive — the relationship is obvious given the neighbors

If not connected: {{"connected": false}}

Respond ONLY with JSON. No preamble.
"""


REEVALUATE_NODE_PROMPT = """You are examining a previously deferred scientific hypothesis or synthesis.
When it was first proposed, there wasn't enough evidence to accept it.

Proposed Claim: "{claim}"
Original Context: {context}

Times re-evaluated so far: {eval_count}

New evidence in the knowledge graph that might be relevant:
{new_context}

In light of the new context, is there now sufficient evidence to ACCEPT this claim into the permanent knowledge graph?

Respond with a JSON object:
{{
  "accepted": true or false,
  "narration": "one sentence explaining why it is accepted or still deferred",
  "confidence": a float 0.0-1.0 (how certain are you this claim is well-supported now?)
}}

Confidence rubric:
- 0.50-0.65: Provisionally accepted
- 0.65-0.80: Solidly accepted
- 0.80-0.95: Strongly accepted

If not accepted: {{"accepted": false}}

Respond ONLY with JSON. No preamble.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PendingInsight:
    node_a_id:          str    = ""
    node_b_id:          str    = ""
    original_similarity: float = 0.0
    first_seen:         float  = field(default_factory=time.time)
    times_evaluated:    int    = 0
    last_eval_time:     float  = 0.0
    best_score:         float  = 0.0   # highest combined score seen
    promoted:           bool   = False
    
    # New fields for individual nodes
    is_node:            bool   = False
    claim:              str    = ""
    context:            str    = ""
    proposed_type:      str    = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingInsight":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Insight Buffer ────────────────────────────────────────────────────────────

class InsightBuffer:
    def __init__(self, brain: Brain, embedding_index=None,
                 buffer_path: str = BUFFER_PATH, autoload: bool = True):
        self.brain   = brain
        self.index   = embedding_index
        self.buffer_path = buffer_path
        self.pending: list[PendingInsight] = []
        if autoload:
            self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(self.buffer_path, "r") as f:
                data = json.load(f)
            self.pending = [PendingInsight.from_dict(d) for d in data]
        except (FileNotFoundError, json.JSONDecodeError):
            self.pending = []

    def save(self):
        os.makedirs(os.path.dirname(self.buffer_path) or ".", exist_ok=True)
        with open(self.buffer_path, "w") as f:
            json.dump([p.to_dict() for p in self.pending], f, indent=2)

    # ── Add near-miss pairs ───────────────────────────────────────────────────

    def add(self, node_a_id: str, node_b_id: str, similarity: float,
            claim: str = "", context: str = "", proposed_type: str = ""):
        """Add a near-miss pair to the buffer."""
        if similarity < BUFFER_LOW:
            return  # too weak even for buffer

        # Check for duplicates
        for p in self.pending:
            if not p.is_node and ((p.node_a_id == node_a_id and p.node_b_id == node_b_id) or
                (p.node_a_id == node_b_id and p.node_b_id == node_a_id)):
                # Update if this observation is stronger
                if similarity > p.original_similarity:
                    p.original_similarity = similarity
                if claim:
                    p.claim = claim
                if context:
                    p.context = context
                if proposed_type:
                    p.proposed_type = proposed_type
                return

        # Hard cap
        if len(self.pending) >= MAX_BUFFER_SIZE:
            # Drop the weakest
            self.pending.sort(key=lambda p: p.best_score or p.original_similarity)
            self.pending.pop(0)

        self.pending.append(PendingInsight(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            original_similarity=similarity,
            is_node=False,
            claim=claim,
            context=context,
            proposed_type=proposed_type
        ))

    def add_node(self, claim: str, context: str, importance: float):
        """Add a single node insight (e.g. from System 2 review) to the buffer."""
        # Check for duplicates
        for p in self.pending:
            if p.is_node and p.claim == claim:
                if importance > p.original_similarity:
                    p.original_similarity = importance
                return

        # Hard cap
        if len(self.pending) >= MAX_BUFFER_SIZE:
            self.pending.sort(key=lambda p: p.best_score or p.original_similarity)
            self.pending.pop(0)
            
        self.pending.append(PendingInsight(
            is_node=True,
            claim=claim,
            context=context,
            original_similarity=importance
        ))

    @property
    def size(self) -> int:
        return len(self.pending)

    # ── Re-evaluation ─────────────────────────────────────────────────────────

    def evaluate_all(self) -> dict:
        """
        Re-evaluate all pending pairs. Returns stats dict.
        Called once per consolidation cycle.
        """
        if not self.pending:
            return {"evaluated": 0, "promoted": 0, "pruned": 0}

        promoted = 0
        pruned   = 0
        to_remove = []

        for i, pair in enumerate(self.pending):
            if pair.promoted:
                to_remove.append(i)
                continue
                
            if pair.is_node:
                # ── Single Node Re-evaluation ──
                pair.times_evaluated += 1
                pair.last_eval_time = time.time()
                
                # Check for relevant context in index
                new_context = ""
                if self.index:
                    claim_emb = shared_embed(pair.claim)
                    if claim_emb is not None:
                        matches = self.index.query(claim_emb, threshold=0.45, top_k=5)
                        lines = []
                        for nid, score in matches:
                            node = self.brain.get_node(nid)
                            if node:
                                lines.append(f"- [sim={score:.2f}] {node['statement']}")
                        new_context = "\n".join(lines) if lines else "None yet"
                
                if not new_context or new_context == "None yet":
                    new_context = "No meaningfully new related knowledge added yet."
                    
                result = self._llm_evaluate_node(pair, new_context)
                if result and result.get("accepted"):
                    self._promote_node(pair, result)
                    to_remove.append(i)
                    promoted += 1
                    print(f"    ★ Promoted deferred node: {pair.claim[:40]}...")
                    continue
                    
                # Prune stale pairs
                if pair.times_evaluated >= MAX_EVALUATIONS:
                    to_remove.append(i)
                    pruned += 1
                continue

            # ── Edge Pairs Re-evaluation ──
            # Check if both nodes still exist
            node_a = self.brain.get_node(pair.node_a_id)
            node_b = self.brain.get_node(pair.node_b_id)
            if not node_a or not node_b:
                to_remove.append(i)
                pruned += 1
                continue

            # Check if edge already exists (added by another mechanism)
            if (self.brain.graph.has_edge(pair.node_a_id, pair.node_b_id) or
                    self.brain.graph.has_edge(pair.node_b_id, pair.node_a_id)):
                to_remove.append(i)
                continue

            # Compute current similarity
            current_sim = max(self._current_similarity(pair), pair.original_similarity)

            # Count shared graph context as boost
            shared = self._shared_neighbor_count(pair)
            similarity_gain = max(0.0, current_sim - pair.original_similarity)
            same_cluster = (
                node_a.get('cluster') and
                node_a.get('cluster') == node_b.get('cluster') and
                node_a.get('cluster') != 'unclustered'
            )
            context_score = (
                current_sim +
                (shared * NEIGHBOR_BOOST) +
                (0.40 * similarity_gain) +
                (0.03 if same_cluster else 0.0)
            )
            if pair.claim and (shared > 0 or similarity_gain >= 0.05):
                context_score += PROMOTION_BOOST

            pair.times_evaluated += 1
            pair.last_eval_time = time.time()
            pair.best_score = max(pair.best_score, context_score)

            # If context score now exceeds threshold → LLM evaluation
            # Keep this gate slightly below WEAK_EDGE so near-threshold pairs
            # can still be reviewed when graph context improves.
            promotion_gate = max(BUFFER_LOW + 0.05, THRESHOLDS.WEAK_EDGE - 0.08)
            if context_score >= promotion_gate or similarity_gain >= 0.10:
                result = self._llm_evaluate(pair, node_a, node_b)
                if result and result.get("connected"):
                    self._promote(pair, result)
                    to_remove.append(i)
                    promoted += 1
                    print(f"    ★ Promoted delayed insight: "
                          f"{node_a['statement'][:40]}... ↔ "
                          f"{node_b['statement'][:40]}...")
                    continue

            # Prune stale pairs
            if pair.times_evaluated >= MAX_EVALUATIONS:
                to_remove.append(i)
                pruned += 1

        # Remove in reverse order to maintain indices
        for i in sorted(to_remove, reverse=True):
            if i < len(self.pending):
                self.pending.pop(i)

        self.save()
        return {
            "evaluated": len(self.pending) + len(to_remove),
            "promoted":  promoted,
            "pruned":    pruned,
            "remaining": len(self.pending)
        }

    def _current_similarity(self, pair: PendingInsight) -> float:
        """Re-compute embedding similarity (may have changed if nodes were enriched)."""
        if self.index:
            emb_a = self.index.get_embedding(pair.node_a_id)
            emb_b = self.index.get_embedding(pair.node_b_id)
            if emb_a is not None and emb_b is not None:
                return float(np.dot(emb_a, emb_b))

        # Fallback: re-embed from statements
        node_a = self.brain.get_node(pair.node_a_id)
        node_b = self.brain.get_node(pair.node_b_id)
        if node_a and node_b:
            emb_a = shared_embed(node_a['statement'])
            emb_b = shared_embed(node_b['statement'])
            return float(np.dot(emb_a, emb_b))

        return pair.original_similarity

    def _undirected_neighbors(self, node_id: str) -> set[str]:
        """Return combined in/out neighborhood for structural overlap checks."""
        try:
            out_neighbors = set(self.brain.graph.successors(node_id))
            in_neighbors = set(self.brain.graph.predecessors(node_id))
            return out_neighbors | in_neighbors
        except Exception:
            return set()

    def _shared_neighbor_count(self, pair: PendingInsight) -> int:
        """Count direct and second-order shared context between A and B."""
        try:
            neighbors_a = self._undirected_neighbors(pair.node_a_id)
            neighbors_b = self._undirected_neighbors(pair.node_b_id)
            direct_overlap = len(neighbors_a & neighbors_b)

            if direct_overlap > 0:
                return direct_overlap

            # Second-order overlap captures delayed context convergence even
            # when direct shared neighbors are not yet present.
            second_a = set()
            second_b = set()
            for nid in list(neighbors_a)[:20]:
                second_a |= self._undirected_neighbors(nid)
            for nid in list(neighbors_b)[:20]:
                second_b |= self._undirected_neighbors(nid)

            second_overlap = len(second_a & second_b)
            return int(round(second_overlap * 0.5))
        except Exception:
            return 0

    def _get_neighbor_summaries(self, node_id: str, max_n: int = 5) -> str:
        """Get brief summaries of a node's neighbors for LLM context."""
        try:
            neighbors = list(self._undirected_neighbors(node_id))[:max_n]
            summaries = []
            for nid in neighbors:
                node = self.brain.get_node(nid)
                if node:
                    stmt = node['statement'][:80]
                    summaries.append(f"- {stmt}")
            return "\n".join(summaries) if summaries else "None yet"
        except Exception:
            return "None yet"

    def _llm_evaluate(self, pair: PendingInsight, node_a: dict,
                      node_b: dict) -> dict:
        """Ask LLM whether the pair is now connected given new context."""
        prompt = REEVALUATE_PROMPT.format(
            node_a=node_a['statement'],
            node_b=node_b['statement'],
            claim=pair.claim or "None recorded.",
            context=pair.context or "None recorded.",
            original_sim=pair.original_similarity,
            eval_count=pair.times_evaluated,
            a_neighbors=self._get_neighbor_summaries(pair.node_a_id),
            b_neighbors=self._get_neighbor_summaries(pair.node_b_id),
        )
        raw = llm_call(prompt, temperature=0.2, role="precise")
        return require_json(raw, default={})

    def _llm_evaluate_node(self, pair: PendingInsight, new_context: str) -> dict:
        prompt = REEVALUATE_NODE_PROMPT.format(
            claim=pair.claim,
            context=pair.context,
            eval_count=pair.times_evaluated,
            new_context=new_context
        )
        raw = llm_call(prompt, temperature=0.2, role="critic")
        return require_json(raw, default={})

    def _promote(self, pair: PendingInsight, result: dict):
        """Create a real edge from a promoted insight."""
        raw_type = result.get("type", "associated")
        confidence = result.get("confidence", 0.5)

        if raw_type == "analogy":
            depth = result.get("analogy_depth", "structural")
            self.brain.add_analogy_edge(
                pair.node_a_id, pair.node_b_id, depth,
                result.get("narration", "Delayed insight — connection emerged over time"),
                EdgeSource.CONSOLIDATION
            )
        else:
            try:
                etype = EdgeType(raw_type)
            except ValueError:
                etype = EdgeType.ASSOCIATED

            edge = Edge(
                type       = etype,
                narration  = result.get("narration",
                    "Delayed insight — connection emerged over time"),
                weight     = 0.5 + (confidence * 0.3),
                confidence = confidence,
                source     = EdgeSource.CONSOLIDATION
            )
            self.brain.add_edge(pair.node_a_id, pair.node_b_id, edge)

        pair.promoted = True

    def _promote_node(self, pair: PendingInsight, result: dict):
        from graph.brain import Node, NodeType, NodeStatus
        node = Node(
            statement      = pair.claim,
            node_type      = NodeType.HYPOTHESIS,
            cluster        = "general",
            status         = NodeStatus.UNCERTAIN,
            source_quality = 0.8,
            last_verified  = time.time(),
            importance     = 0.7
        )
        nid = self.brain.add_node(node)
        if self.index:
            emb = shared_embed(pair.claim)
            self.index.add(nid, emb)
        
        pair.promoted = True

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "pending":        len(self.pending),
            "avg_similarity": (
                sum(p.original_similarity for p in self.pending) / len(self.pending)
                if self.pending else 0
            ),
            "avg_evaluations": (
                sum(p.times_evaluated for p in self.pending) / len(self.pending)
                if self.pending else 0
            ),
            "oldest_days": (
                (time.time() - min(p.first_seen for p in self.pending)) / 86400
                if self.pending else 0
            )
        }

    # ── Reference cleanup ───────────────────────────────────────────────────

    def remove_node_references(self, removed_node_ids: list[str] | set[str]) -> dict:
        """Drop pending pair insights that reference pruned graph nodes."""
        removed = set(removed_node_ids or [])
        if not removed:
            return {"cleaned": 0, "remaining": len(self.pending)}

        before = len(self.pending)
        self.pending = [
            pair for pair in self.pending
            if pair.is_node or (pair.node_a_id not in removed and pair.node_b_id not in removed)
        ]
        cleaned = max(0, before - len(self.pending))
        if cleaned:
            self.save()

        return {
            "cleaned": cleaned,
            "remaining": len(self.pending),
        }
