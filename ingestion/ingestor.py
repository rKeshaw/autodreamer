import json
import time
import numpy as np
from ollama import Client
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType, ANALOGY_WEIGHTS)
from config import THRESHOLDS
from embedding import embed as shared_embed

# ── Config ────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD  = THRESHOLDS.MERGE_NODE
WEAK_EDGE_THRESHOLD   = THRESHOLDS.WEAK_EDGE
OLLAMA_MODEL          = "llama3.1:8b"

# ── Prompts ───────────────────────────────────────────────────────────────────

NODE_EXTRACTION_PROMPT = """
You are building a knowledge graph for a scientific mind.

Read the following text carefully. Extract every distinct conceptual idea present.

Rules:
- Each node must be a self-contained conceptual statement — rich enough to stand alone
- Write each as 1-3 sentences. Not a keyword. Not a title. A thought.
- Capture the perspective, not just the topic
- If an idea contains a tension or uncertainty, include that in the statement
- Aim for 3 to 8 nodes per passage. Quality over quantity.

Respond ONLY with a JSON array of strings. No preamble. No markdown.

Text:
{text}
"""

HYPOTHESIS_EXTRACTION_PROMPT = """
You are analyzing text for scientific hypotheses.

A hypothesis is a directional claim that:
- Makes a specific prediction about how things work
- Could in principle be tested or researched
- Goes beyond just describing — it proposes a mechanism or relationship

Read the following text and extract any hypotheses present.
For each hypothesis, provide:
- statement: the hypothesis as a clear claim
- predicted_answer: what it predicts will be found or confirmed
- testable_by: how it could be investigated

Respond ONLY with a JSON array of objects. If no hypotheses are present, return [].
No preamble. No markdown.

Example:
[
  {{
    "statement": "REM sleep enables insight by loosening associative constraints",
    "predicted_answer": "People woken from REM sleep will show higher remote associate scores",
    "testable_by": "Sleep lab studies measuring creativity after REM vs NREM awakenings"
  }}
]

Text:
{text}
"""

EDGE_EXTRACTION_PROMPT = """
You are mapping relationships between ideas in a knowledge graph.

Given these two ideas, determine if they have a meaningful relationship.

Idea A: {node_a}
Idea B: {node_b}

If they are related, respond with a JSON object:
{{
  "related": true,
  "type": one of ["supports", "causes", "contradicts", "analogy", "associated"],
  "analogy_depth": if type is "analogy", one of ["surface", "structural", "isomorphism"] — else omit,
  "narration": "one or two sentences explaining exactly how and why these ideas connect",
  "weight": a float from 0.1 to 1.0,
  "confidence": a float from 0.1 to 1.0
}}

Analogy depth guide:
- surface: shared vocabulary, metaphor, or theme only. Example: "both involve networks"
- structural: same relational pattern between different entities. Example: "A relates to B the same way X relates to Y"
- isomorphism: formal mathematical or logical equivalence. Example: "the equations governing X are identical in form to those governing Y"

If not meaningfully related:
{{"related": false}}

Respond ONLY with JSON. No preamble.
"""

CLUSTER_PROMPT = """
Given this conceptual statement, assign it to a single domain cluster.
Use a short lowercase label like: neuroscience, physics, philosophy_of_science,
mathematics, linguistics, economics, biology, computer_science, psychology, general.

Statement: {statement}

Respond with ONLY the cluster label. No punctuation.
"""

CONTRADICTION_CHECK_PROMPT = """
Existing node: {existing}
New node: {new}

Do these two ideas directly contradict each other?
Respond with ONLY "yes" or "no".
"""

ANSWER_MATCH_PROMPT = """
Question/Hypothesis: {question}
New idea: {candidate}

Does the new idea answer, resolve, or significantly advance the question?

Respond with a JSON object:
{{
  "match": one of ["none", "partial", "strong"],
  "explanation": "one sentence, or 'no match'"
}}

Respond ONLY with JSON.
"""

MISSION_RELEVANCE_PROMPT = """
Central research question: {mission}

New idea being added to the knowledge graph:
{statement}

Does this new idea meaningfully advance, inform, or relate to the central question?

Respond with a JSON object:
{{
  "relevant": true or false,
  "strength": a float 0.0 to 1.0 indicating how strongly it relates,
  "narration": "one sentence explaining the connection, or 'not relevant'"
}}

Respond ONLY with JSON.
"""

# ── Ingestor ──────────────────────────────────────────────────────────────────

class Ingestor:
    def __init__(self, brain: Brain, research_agenda=None):
        self.brain            = brain
        self.llm              = Client()
        self._embedding_cache = {}
        self.research_agenda  = research_agenda

    def _llm(self, prompt: str, temperature: float = 0.7) -> str:
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

    def _get_all_embeddings(self) -> dict:
        result = {}
        for node_id, data in self.brain.all_nodes():
            if node_id in self._embedding_cache:
                result[node_id] = self._embedding_cache[node_id]
            else:
                emb = self._embed(data['statement'])
                self._embedding_cache[node_id] = emb
                result[node_id] = emb
        return result

    # ── Answer detection ──────────────────────────────────────────────────────

    def _check_against_agenda(self, node_id: str, statement: str):
        if not self.research_agenda:
            return
        open_items = self.research_agenda.get_prioritized_questions(20)
        node_emb = self._embed(statement)
        for item in open_items:
            item_emb = self._embed(item.text)
            if self._cosine(node_emb, item_emb) < THRESHOLDS.AGENDA_PREFILTER:
                continue
            raw = self._llm(ANSWER_MATCH_PROMPT.format(
                question=item.text, candidate=statement
            ), temperature=0.1)
            try:
                result      = json.loads(raw)
                match       = result.get('match', 'none')
                explanation = result.get('explanation', '')
                if match == 'strong':
                    q_node_id = getattr(item, 'node_id', None)
                    if q_node_id and self.brain.get_node(q_node_id):
                        edge = Edge(
                            type       = EdgeType.ANSWERS,
                            narration  = explanation,
                            weight     = 0.85,
                            confidence = 0.75,
                            source     = EdgeSource.RESEARCH
                        )
                        self.brain.add_edge(node_id, q_node_id, edge)
                    self.research_agenda.record_answer(
                        item.text, node_id, explanation, grade='strong'
                    )
                    print(f"  ✓ STRONG ANSWER to: {item.text}")
                elif match == 'partial':
                    self.research_agenda.record_answer(
                        item.text, node_id, explanation, grade='partial'
                    )
                    print(f"  ~ PARTIAL ANSWER to: {item.text}")
            except (json.JSONDecodeError, ValueError):
                continue

    # ── Mission relevance check ───────────────────────────────────────────────

    def _check_mission_relevance(self, node_id: str, statement: str):
        mission = self.brain.get_mission()
        if not mission:
            return
        raw = self._llm(MISSION_RELEVANCE_PROMPT.format(
            mission   = mission['question'],
            statement = statement
        ), temperature=0.1)
        try:
            result = json.loads(raw)
            if result.get('relevant') and result.get('strength', 0) > 0.4:
                self.brain.link_to_mission(
                    node_id,
                    result.get('narration', ''),
                    strength=result.get('strength', 0.5)
                )
                print(f"  ↗ Mission link (strength={result['strength']:.2f})")
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def ingest(self, text: str, source: EdgeSource = EdgeSource.CONVERSATION):
        print(f"\n── Ingesting {len(text)} chars [{source.value}] ──")

        # extract concepts
        raw = self._llm(NODE_EXTRACTION_PROMPT.format(text=text))
        try:
            statements = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  Node extraction parse error")
            return []

        print(f"  Extracted {len(statements)} candidate nodes")

        # extract hypotheses
        raw_hyp = self._llm(HYPOTHESIS_EXTRACTION_PROMPT.format(text=text))
        try:
            hypotheses = json.loads(raw_hyp)
        except json.JSONDecodeError:
            hypotheses = []

        print(f"  Extracted {len(hypotheses)} hypotheses")

        new_node_ids      = []
        existing_embeddings = self._get_all_embeddings()

        # process concepts
        for stmt in statements:
            nid = self._process_statement(
                stmt, existing_embeddings, source, NodeType.CONCEPT
            )
            if nid:
                new_node_ids.append(nid)
                existing_embeddings[nid] = self._embedding_cache.get(
                    nid, self._embed(stmt))

        # process hypotheses
        for hyp in hypotheses:
            stmt = hyp.get('statement', '')
            if not stmt:
                continue
            nid = self._process_statement(
                stmt, existing_embeddings, source, NodeType.HYPOTHESIS,
                predicted_answer=hyp.get('predicted_answer', ''),
                testable_by=hyp.get('testable_by', '')
            )
            if nid:
                new_node_ids.append(nid)
                existing_embeddings[nid] = self._embedding_cache.get(
                    nid, self._embed(stmt))

        # extract edges
        for i in range(len(new_node_ids)):
            for j in range(i + 1, len(new_node_ids)):
                id_a, id_b = new_node_ids[i], new_node_ids[j]
                if (id_a == id_b or
                        self.brain.graph.has_edge(id_a, id_b) or
                        self.brain.graph.has_edge(id_b, id_a)):
                    continue
                node_a = self.brain.get_node(id_a)
                node_b = self.brain.get_node(id_b)
                if not node_a or not node_b:
                    continue
                raw_edge = self._llm(EDGE_EXTRACTION_PROMPT.format(
                    node_a=node_a['statement'],
                    node_b=node_b['statement']
                ))
                try:
                    ed = json.loads(raw_edge)
                    if ed.get('related'):
                        raw_type = ed.get('type', 'associated')

                        if raw_type == 'analogy':
                            depth = ed.get('analogy_depth', 'structural')
                            self.brain.add_analogy_edge(
                                id_a, id_b, depth,
                                ed.get('narration', ''), source
                            )
                            print(f"  Edge [analogy:{depth}]: "
                                  f"{id_a[:8]} ↔ {id_b[:8]}")
                        else:
                            try:
                                etype  = EdgeType(raw_type)
                            except ValueError:
                                etype  = EdgeType.ASSOCIATED
                            exempt = (etype == EdgeType.CONTRADICTS)
                            edge   = Edge(
                                type         = etype,
                                narration    = ed.get('narration', ''),
                                weight       = ed.get('weight', 0.5),
                                confidence   = ed.get('confidence', 0.5),
                                source       = source,
                                decay_exempt = exempt
                            )
                            self.brain.add_edge(id_a, id_b, edge)
                            print(f"  Edge [{etype.value}]: "
                                  f"{id_a[:8]} ↔ {id_b[:8]}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"  Edge error: {e}")

        # weak associative edges
        self._add_weak_edges(new_node_ids, source)

        print(f"\n── Ingestion complete. {self.brain.stats()['nodes']} nodes, "
              f"{self.brain.stats()['edges']} edges ──\n")
        return new_node_ids

    def _process_statement(self, stmt: str, existing_embeddings: dict,
                           source: EdgeSource, node_type: NodeType,
                           predicted_answer: str = "",
                           testable_by: str = "") -> str:
        stmt_emb        = self._embed(stmt)
        best_match_id   = None
        best_similarity = 0.0

        for nid, nemb in existing_embeddings.items():
            sim = self._cosine(stmt_emb, nemb)
            if sim > best_similarity:
                best_similarity = sim
                best_match_id   = nid

        if best_similarity >= SIMILARITY_THRESHOLD:
            existing = self.brain.get_node(best_match_id)
            enriched = existing['statement'] + " | " + stmt
            self.brain.update_node(best_match_id, statement=enriched)
            self._embedding_cache[best_match_id] = self._embed(enriched)
            # upgrade type if more specific
            if (node_type == NodeType.HYPOTHESIS and
                    existing.get('node_type') == NodeType.CONCEPT.value):
                self.brain.update_node(best_match_id,
                    node_type        = NodeType.HYPOTHESIS.value,
                    predicted_answer = predicted_answer,
                    testable_by      = testable_by)
                print(f"  Upgraded to HYPOTHESIS")
            else:
                print(f"  Enriched existing (sim={best_similarity:.2f})")
            return best_match_id

        cluster = self._llm(
            CLUSTER_PROMPT.format(statement=stmt)
        ).strip().lower()

        status = NodeStatus.UNCERTAIN
        contradiction_ids = []
        for nid, nemb in existing_embeddings.items():
            sim = self._cosine(stmt_emb, nemb)
            if sim > THRESHOLDS.CONTRADICTION:
                check = self._llm(CONTRADICTION_CHECK_PROMPT.format(
                    existing=self.brain.get_node(nid)['statement'],
                    new=stmt
                ), temperature=0.1)
                if check.lower().startswith('yes'):
                    status = NodeStatus.CONTRADICTED
                    contradiction_ids.append(nid)
                    print(f"  Contradiction with {nid[:8]}")

        node = Node(
            statement        = stmt,
            node_type        = node_type,
            cluster          = cluster,
            status           = status,
            predicted_answer = predicted_answer,
            testable_by      = testable_by
        )
        nid = self.brain.add_node(node)
        self._embedding_cache[nid] = stmt_emb
        print(f"  Created {node_type.value} [{cluster}]: {stmt}")
        for contra_id in contradiction_ids:
            contra_edge = Edge(
                type         = EdgeType.CONTRADICTS,
                narration    = "Contradiction detected during ingestion.",
                weight       = 0.5,
                confidence   = 0.6,
                source       = source,
                decay_exempt = True
            )
            self.brain.add_edge(nid, contra_id, contra_edge)

        self._check_against_agenda(nid, stmt)
        self._check_mission_relevance(nid, stmt)

        return nid

    def _add_weak_edges(self, new_ids: list, source: EdgeSource):
        all_embeddings = self._get_all_embeddings()
        for nid in new_ids:
            emb = all_embeddings.get(nid)
            if emb is None:
                continue
            for other_id, other_emb in all_embeddings.items():
                if other_id == nid:
                    continue
                if (self.brain.graph.has_edge(nid, other_id) or
                        self.brain.graph.has_edge(other_id, nid)):
                    continue
                sim = self._cosine(emb, other_emb)
                if WEAK_EDGE_THRESHOLD <= sim < SIMILARITY_THRESHOLD:
                    edge = Edge(
                        type       = EdgeType.ASSOCIATED,
                        narration  = (f"Weak associative link "
                                      f"(similarity={sim:.2f})"),
                        weight     = sim * 0.4,
                        confidence = sim,
                        source     = source
                    )
                    self.brain.add_edge(nid, other_id, edge)
