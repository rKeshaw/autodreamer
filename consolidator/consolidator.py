import json
import time
import os
import numpy as np
from dataclasses import dataclass, field
from ollama import Client
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType)
from config import THRESHOLDS
from embedding import embed as shared_embed

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL          = "llama3.1:8b"
DUPLICATE_THRESHOLD   = THRESHOLDS.DUPLICATE_MERGE
SYNTHESIS_SAMPLE      = 6      # how many recent nodes to check for synthesis
ABSTRACTION_MIN_NODES = 3      # min cluster size to attempt abstraction
GAP_CONFIDENCE        = THRESHOLDS.GAP_CONFIDENCE
MAX_GAP_PER_RUN       = 10     # max gaps to infer in one consolidation run
GAP_DEDUP_THRESHOLD   = THRESHOLDS.GAP_DEDUP
LAST_CONSOLIDATION_PATH = "data/last_consolidation.txt"

# ── Prompts ───────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """
You are reflecting on a set of ideas that a scientific mind absorbed today.

These ideas all arrived from research and conversation:
{nodes}

Do these ideas, taken together, imply something that none of them state explicitly?
A synthesis is a new insight that emerges from the combination — not a summary,
but something genuinely new that the combination reveals.

If a synthesis exists, respond with a JSON object:
{{
  "synthesis": true,
  "statement": "the new synthesized idea as a rich 2-3 sentence statement",
  "cluster": "which domain this synthesis belongs to",
  "source_ids": [list of node IDs that contributed to this synthesis]
}}

If no meaningful synthesis emerges, respond with:
{{"synthesis": false}}

Respond ONLY with JSON. No preamble. No markdown.
"""

ABSTRACTION_PROMPT = """
You are looking for a higher-order pattern across a cluster of ideas.

These ideas all belong to the same domain cluster:
{nodes}

Is there a unifying principle, generalization, or abstraction that sits above
all of these — something that explains why they all belong together?

If yes, respond with a JSON object:
{{
  "abstraction": true,
  "statement": "the higher-order principle as a rich 2-3 sentence statement",
  "cluster": "{cluster}"
}}

If no meaningful abstraction exists, respond with:
{{"abstraction": false}}

Respond ONLY with JSON. No preamble.
"""

GAP_PROMPT = """
You are examining two connected ideas in a knowledge graph.

Idea A: {node_a}
Idea B: {node_b}
Their relationship: {edge_narration}

Does their connection imply a third idea that must exist between them —
something that would explain or mediate their relationship, but is not
yet stated anywhere?

If yes, respond with a JSON object:
{{
  "gap": true,
  "statement": "the missing idea as a rich 1-2 sentence statement",
  "cluster": "which domain this gap belongs to"
}}

If no gap exists, respond with:
{{"gap": false}}

Respond ONLY with JSON. No preamble.
"""

MERGE_NARRATION_PROMPT = """
Two nearly identical ideas are being merged in a knowledge graph.

Idea A: {node_a}
Idea B: {node_b}

Write a single unified statement that captures both ideas fully.
Keep it to 2-3 sentences. Rich and precise.

Respond with ONLY the unified statement. No preamble.
"""

CONSOLIDATION_SUMMARY_PROMPT = """
You are writing the evening summary for a scientific mind after a day of
research and an evening of consolidation.

Changes made to the knowledge graph today:
- New nodes added from research: {new_nodes}
- Nodes merged (near-duplicates resolved): {merges}
- Synthesis nodes created: {syntheses}
- Abstraction nodes created: {abstractions}
- Gap nodes inferred (hypothetical): {gaps}
- Contradictions updated: {contradictions}
- Edges decayed: {decayed}

Write a 3-4 sentence reflection on what changed today, what is now more
certain, what tensions remain, and what the mind seems to be building toward.
Write it like a scientist's evening notebook entry.
"""

# ── Consolidation Report ──────────────────────────────────────────────────────

@dataclass
class ConsolidationReport:
    started_at:    float = field(default_factory=time.time)
    new_nodes:     int   = 0
    merges:        int   = 0
    syntheses:     int   = 0
    abstractions:  int   = 0
    gaps:          int   = 0
    contradictions_updated: int = 0
    edges_decayed: int   = 0
    summary:       str   = ""
    new_node_ids:  list  = field(default_factory=list)
    synthesis_ids: list  = field(default_factory=list)
    gap_ids:       list  = field(default_factory=list)

    def to_dict(self):
        return self.__dict__

# ── Consolidator ──────────────────────────────────────────────────────────────

class Consolidator:
    def __init__(self, brain: Brain, observer=None):
        self.brain    = brain
        self.observer = observer
        self.llm      = Client()
        self._embedding_cache: dict = {}

    def _llm(self, prompt: str, temperature: float = 0.6) -> str:
        response = self.llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    def _embed(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        emb = shared_embed(text)
        self._embedding_cache[text] = emb
        return emb

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _get_node_embedding(self, node_id: str) -> np.ndarray:
        data = self.brain.get_node(node_id)
        if not data:
            return None
        return self._embed(data['statement'])

    # ── Step 1: Near-duplicate cleanup ────────────────────────────────────────

    def _merge_duplicates(self, report: ConsolidationReport):
        """
        Scan all node pairs for near-duplicates using cosine similarity.
        Merge them: combine statements, inherit all edges, delete the weaker node.
        """
        print("  [1/6] Near-duplicate cleanup...")
        # snapshot before loop — deletions would corrupt iteration
        nodes    = list(self.brain.all_nodes())
        node_ids = [nid for nid, _ in nodes]
        merged   = set()

        for i in range(len(node_ids)):
            if node_ids[i] in merged:
                continue
            for j in range(i + 1, len(node_ids)):
                if node_ids[j] in merged:
                    continue

                emb_i = self._get_node_embedding(node_ids[i])
                emb_j = self._get_node_embedding(node_ids[j])
                if emb_i is None or emb_j is None:
                    continue

                sim = self._cosine(emb_i, emb_j)
                if sim < DUPLICATE_THRESHOLD:
                    continue

                # merge j into i
                node_i = self.brain.get_node(node_ids[i])
                node_j = self.brain.get_node(node_ids[j])

                # generate unified statement
                unified = self._llm(MERGE_NARRATION_PROMPT.format(
                    node_a=node_i['statement'],
                    node_b=node_j['statement']
                ), temperature=0.4)
                self.brain.update_node(node_ids[i], statement=unified)

                # inherit edges from j — redirect to i
                for neighbor in list(self.brain.graph.successors(node_ids[j])):
                    if neighbor == node_ids[i]:
                        continue
                    edge_data = self.brain.get_edge(node_ids[j], neighbor)
                    if edge_data and not (
                        self.brain.graph.has_edge(node_ids[i], neighbor) or
                        self.brain.graph.has_edge(neighbor, node_ids[i])
                    ):
                        edge = Edge(
                            type       = EdgeType(edge_data['type']),
                            narration  = edge_data['narration'],
                            weight     = edge_data['weight'],
                            confidence = edge_data['confidence'],
                            source     = EdgeSource(edge_data['source']),
                            decay_exempt = edge_data.get('decay_exempt', False)
                        )
                        self.brain.add_edge(node_ids[i], neighbor, edge)

                for predecessor in list(self.brain.graph.predecessors(node_ids[j])):
                    if predecessor == node_ids[i]:
                        continue
                    edge_data = self.brain.get_edge(predecessor, node_ids[j])
                    if edge_data and not (
                        self.brain.graph.has_edge(predecessor, node_ids[i]) or
                        self.brain.graph.has_edge(node_ids[i], predecessor)
                    ):
                        edge = Edge(
                            type       = EdgeType(edge_data['type']),
                            narration  = edge_data['narration'],
                            weight     = edge_data['weight'],
                            confidence = edge_data['confidence'],
                            source     = EdgeSource(edge_data['source']),
                            decay_exempt = edge_data.get('decay_exempt', False)
                        )
                        self.brain.add_edge(predecessor, node_ids[i], edge)

                # delete j
                self.brain.graph.remove_node(node_ids[j])
                merged.add(node_ids[j])
                report.merges += 1
                print(f"    Merged {node_ids[j][:8]} → {node_ids[i][:8]} "
                      f"(sim={sim:.2f})")

        print(f"    Done — {report.merges} merges")

    # ── Step 2: Synthesis pass ────────────────────────────────────────────────

    def _synthesis_pass(self, new_node_ids: list,
                        report: ConsolidationReport):
        """
        Look at recent nodes together — do they imply something none stated?
        Creates SYNTHESIS nodes.
        """
        print("  [2/6] Synthesis pass...")
        if len(new_node_ids) < 2:
            return

        sample_ids = new_node_ids[:SYNTHESIS_SAMPLE]
        node_lines = []
        for nid in sample_ids:
            data = self.brain.get_node(nid)
            if data:
                node_lines.append(f"ID {nid[:8]}: {data['statement']}")

        if not node_lines:
            return

        raw = self._llm(SYNTHESIS_PROMPT.format(
            nodes="\n\n".join(node_lines)
        ), temperature=0.6)
        try:
            result = json.loads(raw)
            if result.get('synthesis'):
                stmt    = result['statement']
                cluster = result.get('cluster', 'synthesis')
                sources = result.get('source_ids', [])

                node = Node(
                    statement = stmt,
                    node_type = NodeType.SYNTHESIS,
                    cluster   = cluster,
                    status    = NodeStatus.UNCERTAIN,
                    importance= 0.7
                )
                nid = self.brain.add_node(node)
                report.synthesis_ids.append(nid)
                report.syntheses += 1

                # connect to source nodes
                for src_id in sample_ids:
                    if self.brain.get_node(src_id):
                        edge = Edge(
                            type      = EdgeType.SUPPORTS,
                            narration = "This node contributed to the synthesis.",
                            weight    = 0.6,
                            confidence= 0.6,
                            source    = EdgeSource.CONSOLIDATION
                        )
                        self.brain.add_edge(src_id, nid, edge)

                # check against research agenda
                if self.observer:
                    self.observer.add_to_agenda(
                        text      = f"Explore synthesis: {stmt}",
                        item_type = "question",
                        cycle     = self.observer.cycle_count
                    )

                print(f"    Synthesis node created: {stmt}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    Synthesis parse error: {e}")

    # ── Step 3: Abstraction pass ──────────────────────────────────────────────

    def _abstraction_pass(self, report: ConsolidationReport):
        """
        Per cluster — if enough nodes exist, look for a higher-order principle.
        Creates CONCEPT nodes with elevated importance.
        """
        print("  [3/6] Abstraction pass...")

        # group nodes by cluster
        clusters: dict = {}
        for nid, data in self.brain.all_nodes():
            c = data.get('cluster', 'unclustered')
            clusters.setdefault(c, []).append((nid, data))

        for cluster, members in clusters.items():
            if len(members) < ABSTRACTION_MIN_NODES:
                continue

            # sample top-importance members
            sample = sorted(
                members,
                key=lambda x: x[1].get('importance', 0.5),
                reverse=True
            )[:5]
            node_lines = [
                f"{data['statement']}" for _, data in sample
            ]

            raw = self._llm(ABSTRACTION_PROMPT.format(
                nodes   = "\n\n".join(node_lines),
                cluster = cluster
            ), temperature=0.6)
            try:
                result = json.loads(raw)
                if result.get('abstraction'):
                    stmt = result['statement']

                    # check it isn't a duplicate of something existing
                    stmt_emb = self._embed(stmt)
                    is_dup = False
                    for nid, data in self.brain.all_nodes():
                        existing_emb = self._embed(data['statement'])
                        if self._cosine(stmt_emb, existing_emb) > DUPLICATE_THRESHOLD:
                            is_dup = True
                            break

                    if is_dup:
                        continue

                    node = Node(
                        statement = stmt,
                        node_type = NodeType.CONCEPT,
                        cluster   = cluster,
                        status    = NodeStatus.UNCERTAIN,
                        importance= 0.8   # abstractions are high importance
                    )
                    nid = self.brain.add_node(node)
                    report.abstractions += 1

                    # connect to all cluster members
                    for member_id, _ in sample:
                        if self.brain.get_node(member_id):
                            edge = Edge(
                                type      = EdgeType.SUPPORTS,
                                narration = (f"This specific idea is subsumed "
                                             f"under the higher-order principle."),
                                weight    = 0.65,
                                confidence= 0.55,
                                source    = EdgeSource.CONSOLIDATION
                            )
                            self.brain.add_edge(member_id, nid, edge)

                    print(f"    Abstraction [{cluster}]: {stmt}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"    Abstraction parse error [{cluster}]: {e}")

    # ── Step 4: Gap detection ─────────────────────────────────────────────────

    def _gap_detection(self, report: ConsolidationReport):
        """
        Look at connected node pairs with high-confidence edges.
        If their connection implies a missing mediating idea, create a GAP node.
        """
        print("  [4/6] Gap detection...")
        checked = set()
        cluster_count = len(set(
            d.get('cluster', '?') for _, d in self.brain.all_nodes()
        ))

        # snapshot edges before loop — gap detection adds new nodes/edges
        edges_snapshot = list(self.brain.graph.edges(data=True))

        for u, v, data in edges_snapshot:
            if report.gaps >= MAX_GAP_PER_RUN:
                break
            if data.get('confidence', 0) < GAP_CONFIDENCE:
                continue
            if data.get('type') in [EdgeType.ASSOCIATED.value]:
                continue   # weak edges don't imply meaningful gaps
            key = tuple(sorted([u, v]))
            if key in checked:
                continue
            checked.add(key)

            node_u = self.brain.get_node(u)
            node_v = self.brain.get_node(v)
            if not node_u or not node_v:
                continue
            
            # once cluster diversity is high, prioritize cross-cluster gaps
            if (cluster_count > 3 and
                    node_u.get('cluster') == node_v.get('cluster')):
                continue

            raw = self._llm(GAP_PROMPT.format(
                node_a        = node_u['statement'],
                node_b        = node_v['statement'],
                edge_narration= data.get('narration', '')
            ), temperature=0.6)
            try:
                result = json.loads(raw)
                if result.get('gap'):
                    stmt    = result['statement']
                    cluster = result.get('cluster', node_u.get('cluster', 'general'))

                    # don't create duplicate gap nodes
                    stmt_emb = self._embed(stmt)
                    is_dup   = False
                    for nid, ndata in self.brain.all_nodes():
                        if self._cosine(stmt_emb,
                                self._embed(ndata['statement'])) > GAP_DEDUP_THRESHOLD:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                    node = Node(
                        statement = stmt,
                        node_type = NodeType.GAP,
                        cluster   = cluster,
                        status    = NodeStatus.HYPOTHETICAL,
                        importance= 0.6
                    )
                    nid = self.brain.add_node(node)
                    report.gap_ids.append(nid)
                    report.gaps += 1

                    # connect gap to both source nodes
                    for src_id in [u, v]:
                        if self.brain.get_node(src_id):
                            edge = Edge(
                                type      = EdgeType.SUPPORTS,
                                narration = "This gap was inferred from the relationship between these ideas.",
                                weight    = 0.35,
                                confidence= 0.35,
                                source    = EdgeSource.CONSOLIDATION,
                                decay_exempt = False
                            )
                            self.brain.add_edge(src_id, nid, edge)

                    # add gap to research agenda as high-priority question
                    if self.observer:
                        self.observer.add_to_agenda(
                            text      = f"Fill gap: {stmt}",
                            item_type = "question",
                            cycle     = self.observer.cycle_count,
                            node_id   = nid
                        )
                        # bump priority since gaps are inferred needs
                        for item in self.observer.agenda:
                            if item.node_id == nid:
                                item.priority = 0.75
                                break

                    print(f"    Gap [{cluster}]: {stmt}")
            except (json.JSONDecodeError, ValueError):
                continue

        print(f"    Done — {report.gaps} gaps inferred")

    # ── Step 5: Contradiction update ──────────────────────────────────────────

    def _contradiction_update(self, report: ConsolidationReport):
        """
        Revisit contradiction edges.
        If new nodes have appeared that could resolve them, annotate.
        If contradiction is very old and still unresolved, elevate importance.
        """
        print("  [5/6] Contradiction update...")
        count = 0
        edges_snapshot = list(self.brain.graph.edges(data=True))

        for u, v, data in edges_snapshot:
            if data.get('type') != EdgeType.CONTRADICTS.value:
                continue

            node_u = self.brain.get_node(u)
            node_v = self.brain.get_node(v)
            if not node_u or not node_v:
                continue

            # elevate importance of nodes in unresolved contradictions
            imp_u = min(1.0, node_u.get('importance', 0.5) + 0.05)
            imp_v = min(1.0, node_v.get('importance', 0.5) + 0.05)
            self.brain.update_node(u, importance=imp_u)
            self.brain.update_node(v, importance=imp_v)
            count += 1

        report.contradictions_updated = count
        print(f"    Done — {count} contradiction edges updated")

    # ── Step 6: Decay ─────────────────────────────────────────────────────────

    def _apply_decay(self, report: ConsolidationReport):
        """
        Decay all non-exempt edges. Contradictions are exempt.
        """
        print("  [6/6] Edge decay...")
        before = len(self.brain.graph.edges)
        try:
            with open(LAST_CONSOLIDATION_PATH, "r") as f:
                last = float(f.read().strip())
            elapsed_days = max(0.0, (time.time() - last) / 86400)
        except FileNotFoundError:
            elapsed_days = 1.0
        self.brain.apply_decay(elapsed_days=elapsed_days)
        os.makedirs("data", exist_ok=True)
        with open(LAST_CONSOLIDATION_PATH, "w") as f:
            f.write(str(time.time()))

        # prune edges that have decayed to near zero
        to_remove = [
            (u, v) for u, v, d in self.brain.graph.edges(data=True)
            if d.get('weight', 1.0) < 0.02 and
               d.get('type') != EdgeType.CONTRADICTS.value
        ]
        for u, v in to_remove:
            self.brain.graph.remove_edge(u, v)

        report.edges_decayed = before - len(self.brain.graph.edges)
        print(f"    Done — {report.edges_decayed} edges pruned")

    def _recompute_mission_relevance(self):
        mission = self.brain.get_mission()
        if not mission:
            return
        candidates = [
            (nid, data) for nid, data in self.brain.all_nodes()
            if data.get('importance', 0) > 0.6
            and data.get('node_type') != NodeType.MISSION.value
        ][:20]
        for nid, data in candidates:
            prompt = (
                f'Central research question: "{mission["question"]}"\n\n'
                f'New idea being added to the knowledge graph:\n{data["statement"]}\n\n'
                "Does this new idea meaningfully advance, inform, or relate to the central question?\n\n"
                "Respond with a JSON object:\n"
                "{\n"
                '  "relevant": true or false,\n'
                '  "strength": a float 0.0 to 1.0 indicating how strongly it relates,\n'
                '  "narration": "one sentence explaining the connection, or \'not relevant\'"\n'
                "}\n\n"
                "Respond ONLY with JSON."
            )
            raw = self._llm(prompt, temperature=0.1)
            try:
                result = json.loads(raw)
                if result.get("relevant") and result.get("strength", 0) > 0.4:
                    self.brain.link_to_mission(
                        nid,
                        result.get("narration", ""),
                        strength=result.get("strength", 0.5)
                    )
            except (json.JSONDecodeError, ValueError):
                continue

    # ── Main consolidation run ────────────────────────────────────────────────

    def consolidate(self, new_node_ids: list = None,
                    save_path: str = "data/consolidation_latest.json"
                    ) -> ConsolidationReport:
        """
        Run the full evening consolidation.
        new_node_ids: nodes added today by the Researcher (optional).
        """
        report = ConsolidationReport()
        new_node_ids = new_node_ids or []
        report.new_nodes    = len(new_node_ids)
        report.new_node_ids = new_node_ids

        print(f"\n── Evening consolidation begins ──")
        print(f"   Brain: {self.brain.stats()}")
        print(f"   New nodes from today: {len(new_node_ids)}\n")

        self._merge_duplicates(report)
        self._synthesis_pass(new_node_ids or
            [nid for nid, _ in self.brain.all_nodes()][-10:], report)
        self._abstraction_pass(report)
        self._gap_detection(report)
        self._contradiction_update(report)
        self._apply_decay(report)
        self._recompute_mission_relevance()

        # generate summary
        report.summary = self._llm(CONSOLIDATION_SUMMARY_PROMPT.format(
            new_nodes     = report.new_nodes,
            merges        = report.merges,
            syntheses     = report.syntheses,
            abstractions  = report.abstractions,
            gaps          = report.gaps,
            contradictions= report.contradictions_updated,
            decayed       = report.edges_decayed
        ), temperature=0.7)

        # save report
        import os
        os.makedirs("logs", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"\n── Consolidation complete ──")
        print(f"   Merges: {report.merges}  "
              f"Syntheses: {report.syntheses}  "
              f"Abstractions: {report.abstractions}  "
              f"Gaps: {report.gaps}")
        print(f"   Contradictions updated: {report.contradictions_updated}  "
              f"Edges decayed/pruned: {report.edges_decayed}")
        print(f"   Brain now: {self.brain.stats()}")
        print(f"\n── Evening summary ──\n{report.summary}\n")

        return report
