import json
import time
import numpy as np
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType, ANALOGY_WEIGHTS)
from config import THRESHOLDS
from embedding import embed as shared_embed
from llm_utils import llm_call, require_json

# ── Config ────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD  = THRESHOLDS.MERGE_NODE
DEDUP_QUERY_THRESHOLD = SIMILARITY_THRESHOLD - 0.12
DEDUP_LLM_THRESHOLD   = SIMILARITY_THRESHOLD - 0.06
WEAK_EDGE_THRESHOLD   = THRESHOLDS.WEAK_EDGE

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

Example of a GOOD node:
  "REM sleep appears to loosen associative constraints, allowing ideas that
   were previously unrelated to form novel connections — this may explain why
   insights often occur upon waking."

Example of a BAD node (too vague/keyword-like):
  "REM sleep and creativity" ← this is a topic label, not a conceptual statement.

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
  "weight": a float (see rubric below),
  "confidence": a float (see rubric below)
}}

TYPE SELECTION RULES — read carefully before choosing:

Use "causes" when A is a mechanism, process, or event that directly produces B
as a physical or temporal effect. The test: does A happen BEFORE B and make B
happen? Examples:
  ✓ "High temperature increases molecular kinetic energy" → causes "increased pressure"
  ✓ "Mutations in replication introduce variation" → causes "heritable diversity"
  ✗ "Studies show X correlates with Y" → this is "supports", not "causes"

Use "supports" when A is evidence, reasoning, prior work, or context that makes
B more credible or likely — but A does not directly produce B.
  ✓ "Fossil record shows gradual change" → supports "evolution by natural selection"
  ✓ "Backpropagation computes gradients" → supports "gradient descent can train deep nets"

Use "contradicts" only when A and B make MUTUALLY EXCLUSIVE claims — if A is
true, B must be false. Do NOT use for statements that merely contrast or
emphasize different aspects of the same phenomenon.
  ✓ "All objects fall at the same rate" contradicts "heavier objects fall faster"
  ✗ "Neural nets need data" vs "neural nets can overfit" — these are compatible

Use "analogy" when the same relational pattern appears in two different domains.
Specify depth:
  surface     — shared vocabulary or theme only
  structural  — same A:B::X:Y relational pattern across domains
  isomorphism — formally identical mathematical or logical structure

Use "associated" when the ideas co-occur or share a domain but have no direct
logical, causal, or analogical link. This is the CORRECT and FREQUENT answer
for ideas from the same domain that simply appear together. Do NOT force a
stronger type just because the ideas are related — topical proximity alone
means "associated".
  ✓ "Game theory developed in 1940s" associated with "Nash equilibrium defined later"
  ✓ "DNA has four bases" associated with "proteins have 20 amino acids"

Weight rubric (how STRONG is the relationship?):
- 0.1-0.3: Tangentially related, same broad topic but no direct logical link
- 0.4-0.6: Meaningfully connected, one informs understanding of the other
- 0.7-0.9: Strongly linked, one directly supports/contradicts/implies the other
- 1.0: Definitionally equivalent or logically entailed — VERY rare

Confidence rubric (how CERTAIN are you this relationship exists?):
- 0.1-0.3: Speculative — you think there might be a connection but it's not clear
- 0.4-0.6: Reasonable — the connection is plausible and you can articulate why
- 0.7-0.9: Strong — the connection is clearly supported by the content
- 1.0: Definitive — the text explicitly states this relationship

Analogy depth guide:
- surface: shared vocabulary, metaphor, or theme only. Example: "both involve networks"
- structural: same relational pattern between different entities. Example: "A relates to B the same way X relates to Y"
- isomorphism: formal mathematical or logical equivalence. Example: "the equations governing X are identical in form to those governing Y"

IMPORTANT: Do NOT mark ideas as related just because they share a broad topic.
"The brain uses electricity" and "Lightning is electricity" warrant "associated"
at most — not "supports" or "causes". Topical proximity without a conceptual
mechanism = "associated".

If not meaningfully related:
{{"related": false}}

Respond ONLY with JSON. No preamble.
"""

CLUSTER_PROMPT = """
Given this conceptual statement, assign it to a single domain cluster.

Use a SHORT, SPECIFIC lowercase label — prefer sub-domain labels over broad ones.

Prefer SPECIFIC over BROAD:
  thermodynamics     (not physics)
  evolutionary_biology (not biology)
  deep_learning      (not computer_science, if the statement is specifically about NNs)
  game_theory        (not economics, if the statement is specifically about strategic interaction)
  molecular_biology  (not biology, if the statement is about DNA/proteins/cells)
  quantum_mechanics  (not physics, if the statement is about quantum phenomena)
  statistical_mechanics (not physics or thermodynamics, if about microstates/ensembles)

General domain labels to use when no specific sub-domain fits:
  neuroscience, physics, chemistry, biology, mathematics, computer_science,
  psychology, philosophy_of_science, linguistics, economics, sociology,
  cognitive_science, information_theory, systems_biology, ecology,
  genetics, general

Rules:
- If the statement spans two domains, choose the MOST SPECIFIC one.
- Prefer the domain the statement is ABOUT over the domain it USES.
  Example: "Neural networks learn via gradient descent" → deep_learning
  Example: "The brain's learning rule resembles backpropagation" → neuroscience
  Example: "Entropy in thermodynamic systems equals k*ln(W)" → thermodynamics
  Example: "Shannon entropy measures uncertainty in distributions" → information_theory

Statement: {statement}

Respond with ONLY the cluster label. No punctuation. No explanation.
"""

CONTRADICTION_CHECK_PROMPT = """
Existing node: {existing}
New node: {new}

Do these two ideas make MUTUALLY EXCLUSIVE claims?

The test for a genuine contradiction:
  If the existing node is TRUE, does the new node become IMPOSSIBLE or FALSE?
  If the new node is TRUE, does the existing node become IMPOSSIBLE or FALSE?

Both must be true for this to be a contradiction.

Examples of GENUINE contradictions (answer: yes):
  "All objects fall at the same rate in a vacuum"
  vs "Heavier objects fall faster than lighter ones" → YES (mutually exclusive)

  "Acquired traits can be inherited by offspring"
  vs "Only genetic mutations are heritable, not acquired traits" → YES

Examples of NOT contradictions (answer: no):
  "Neural networks need large datasets to generalize"
  vs "Regularization helps neural networks generalize with less data" → NO (compatible)

  "Natural selection favors reproductive fitness"
  vs "Genetic drift changes allele frequencies randomly" → NO (different mechanisms, not exclusive)

  "Overfitting occurs when a model memorizes training noise"
  vs "Regularization techniques reduce overfitting by penalizing complexity" → NO
  (problem + solution: both can be true simultaneously)

  "Game theory assumes perfectly rational agents"
  vs "Behavioral economics shows humans deviate from rational predictions" → NO
  (theory + empirical critique: the critique does not make the theory impossible,
  it adds boundary conditions. Both statements can be simultaneously true.)

  "X has property P"
  vs "Technique Y mitigates or reduces P" → NO, always. Mitigation is not negation.

Respond with ONLY "yes" or "no". No explanation.
"""

DEDUP_CONFIRMATION_PROMPT = """
Two knowledge graph nodes have similar embeddings. Determine whether they
express the SAME core idea and should be merged into one node.

Node A: "{node_a}"
Node B: "{node_b}"
Embedding similarity: {similarity:.3f}

Merge criteria — answer YES only if:
  - One is a paraphrase or minor rewording of the other, OR
  - One is a more detailed version of the other that adds no new distinct claim

Answer NO if:
  - They make different claims (even if about the same topic)
  - One adds a genuinely new sub-idea the other doesn't contain
  - They describe different aspects of the same phenomenon

Examples:
  YES: "Entropy measures disorder" vs "Entropy quantifies the degree of disorder in a system"
  YES: "Natural selection favors fit organisms" vs "Selection pressure preserves adaptive traits"
  NO:  "DNA stores genetic information" vs "RNA transcribes information from DNA" (different roles)
  NO:  "High temperature increases pressure" vs "Pressure depends on molecular collisions" (different claims)

Respond with ONLY "yes" or "no".
"""

ANSWER_MATCH_PROMPT = """
Question/Hypothesis: {question}
New idea: {candidate}

Does the new idea answer, resolve, or significantly advance the question?

Grading definitions:
- "none": The idea is unrelated to the question, or only shares surface-level vocabulary.
- "partial": The idea addresses PART of the question or provides indirect evidence,
  but the core question remains open.
- "strong": The idea directly answers or resolves the question, or provides definitive evidence.

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

Strength rubric:
- 0.1-0.3: Same broad domain but no direct logical connection to the question.
- 0.4-0.6: Provides useful context, background, or a building block for answering the question.
- 0.7-0.85: Directly addresses a sub-question or provides evidence that informs a possible answer.
- 0.9-1.0: Fundamentally advances or answers the central question. VERY rare.

Respond with a JSON object:
{{
  "relevant": true or false,
  "strength": a float 0.0 to 1.0 (use rubric above),
  "narration": "one sentence explaining the connection, or 'not relevant'"
}}

Respond ONLY with JSON.
"""

# ── Ingestor ──────────────────────────────────────────────────────────────────

class Ingestor:
    def __init__(self, brain: Brain, research_agenda=None, embedding_index=None,
                 insight_buffer=None):
        self.brain            = brain
        self._embedding_cache = {}
        self.research_agenda  = research_agenda
        self.index            = embedding_index
        self.insight_buffer   = insight_buffer

    def _llm(self, prompt: str, temperature: float = 0.7) -> str:
        return llm_call(prompt, temperature=temperature, role="precise")

    def _embed(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        emb = shared_embed(text)
        self._embedding_cache[text] = emb
        return emb

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _get_all_embeddings(self) -> dict:
        """Fallback when no index is available."""
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
            result = require_json(raw, default={})
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

    # ── Mission relevance check ───────────────────────────────────────────────

    def _check_mission_relevance(self, node_id: str, statement: str):
        mission = self.brain.get_mission()
        if not mission:
            return
        raw = self._llm(MISSION_RELEVANCE_PROMPT.format(
            mission   = mission['question'],
            statement = statement
        ), temperature=0.1)
        result = require_json(raw, default={})
        if result.get('relevant') and result.get('strength', 0) > 0.4:
            self.brain.link_to_mission(
                node_id,
                result.get('narration', ''),
                strength=result.get('strength', 0.5)
            )
            print(f"  ↗ Mission link (strength={result['strength']:.2f})")

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def ingest(self, text: str, source: EdgeSource = EdgeSource.CONVERSATION):
        print(f"\n── Ingesting {len(text)} chars [{source.value}] ──")

        # extract concepts
        raw = self._llm(NODE_EXTRACTION_PROMPT.format(text=text))
        statements = require_json(raw, default=[])
        if not isinstance(statements, list):
            print(f"  Node extraction parse error")
            return []

        print(f"  Extracted {len(statements)} candidate nodes")

        # extract hypotheses
        raw_hyp = self._llm(HYPOTHESIS_EXTRACTION_PROMPT.format(text=text))
        hypotheses = require_json(raw_hyp, default=[])
        if not isinstance(hypotheses, list):
            hypotheses = []

        print(f"  Extracted {len(hypotheses)} hypotheses")

        new_node_ids      = []
        existing_embeddings = None if self.index else self._get_all_embeddings()

        # process concepts
        for stmt in statements:
            if isinstance(stmt, dict):
                stmt = stmt.get('statement', '') or stmt.get('concept', '') or stmt.get('text', '') or str(stmt)
            if not isinstance(stmt, str) or not stmt.strip():
                continue
            nid = self._process_statement(
                stmt, existing_embeddings, source, NodeType.CONCEPT
            )
            if nid:
                new_node_ids.append(nid)
                if existing_embeddings is not None:
                    existing_embeddings[nid] = self._embedding_cache.get(
                        nid, self._embed(stmt))

        # process hypotheses
        for hyp in hypotheses:
            if isinstance(hyp, str):
                hyp = {'statement': hyp}
            elif not isinstance(hyp, dict):
                continue
                
            stmt = hyp.get('statement', '')
            if not isinstance(stmt, str) or not stmt.strip():
                continue
            nid = self._process_statement(
                stmt, existing_embeddings, source, NodeType.HYPOTHESIS,
                predicted_answer=hyp.get('predicted_answer', ''),
                testable_by=hyp.get('testable_by', '')
            )
            if nid:
                new_node_ids.append(nid)
                if existing_embeddings is not None:
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
                    ed = require_json(raw_edge, default={})
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

    def _process_statement(self, stmt: str, existing_embeddings,
                           source: EdgeSource, node_type: NodeType,
                           predicted_answer: str = "",
                           testable_by: str = "") -> str:
        stmt_emb        = self._embed(stmt)
        best_match_id   = None
        best_similarity = 0.0

        # ── Duplicate detection (indexed or fallback) ──
        if self.index:
            # Query wider (DEDUP_QUERY_THRESHOLD) to catch paraphrases that land
            # below the hard merge threshold due to embedding drift on re-ingest.
            matches = self.index.query(
                stmt_emb, threshold=DEDUP_QUERY_THRESHOLD, top_k=5
            )
            for candidate_id, candidate_sim in matches:
                if candidate_id == node_type:
                    continue
                if candidate_sim >= SIMILARITY_THRESHOLD:
                    best_match_id   = candidate_id
                    best_similarity = candidate_sim
                    break
                elif candidate_sim >= DEDUP_LLM_THRESHOLD:
                    candidate_data = self.brain.get_node(candidate_id)
                    if candidate_data:
                        confirm = self._llm(
                            DEDUP_CONFIRMATION_PROMPT.format(
                                node_a=candidate_data["statement"],
                                node_b=stmt,
                                similarity=candidate_sim,
                            ),
                            temperature=0.1
                        ).strip().lower()
                        if confirm.startswith("yes"):
                            best_match_id   = candidate_id
                            best_similarity = candidate_sim
                            break
        else:
            for nid, nemb in existing_embeddings.items():
                sim = self._cosine(stmt_emb, nemb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id   = nid
            if (best_match_id is not None and
                    DEDUP_LLM_THRESHOLD <= best_similarity < SIMILARITY_THRESHOLD):
                candidate_data = self.brain.get_node(best_match_id)
                if candidate_data:
                    confirm = self._llm(
                        DEDUP_CONFIRMATION_PROMPT.format(
                            node_a=candidate_data["statement"],
                            node_b=stmt,
                            similarity=best_similarity,
                        ),
                        temperature=0.1
                    ).strip().lower()
                    if not confirm.startswith("yes"):
                        best_match_id   = None
                        best_similarity = 0.0

        if best_similarity >= SIMILARITY_THRESHOLD:
            existing = self.brain.get_node(best_match_id)
            enriched = existing['statement'] + " | " + stmt
            self.brain.update_node(best_match_id, statement=enriched)
            enriched_emb = self._embed(enriched)
            self._embedding_cache[best_match_id] = enriched_emb
            if self.index:
                self.index.add(best_match_id, enriched_emb)
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

        # ── Contradiction detection (indexed or fallback) ──
        status = NodeStatus.UNCERTAIN
        contradiction_ids = []
        if self.index:
            contra_candidates = self.index.query(
                stmt_emb, threshold=0.45, top_k=15
            )
            for cand_id, cand_sim in contra_candidates:
                node_data = self.brain.get_node(cand_id)
                if not node_data:
                    continue
                check = self._llm(CONTRADICTION_CHECK_PROMPT.format(
                    existing=node_data['statement'],
                    new=stmt
                ), temperature=0.1)
                if check.lower().startswith('yes'):
                    status = NodeStatus.CONTRADICTED
                    contradiction_ids.append(cand_id)
                    print(f"  Contradiction with {cand_id[:8]}")
        else:
            for nid, nemb in existing_embeddings.items():
                if self._cosine(stmt_emb, nemb) > 0.45:
                    check = self._llm(CONTRADICTION_CHECK_PROMPT.format(
                        existing=self.brain.get_node(nid)['statement'],
                        new=stmt
                    ), temperature=0.1)
                    if check.lower().startswith('yes'):
                        status = NodeStatus.CONTRADICTED
                        contradiction_ids.append(nid)
                        print(f"  Contradiction with {nid[:8]}")

        # Source quality mapping
        source_quality_map = {
            EdgeSource.READING:       0.9,
            EdgeSource.RESEARCH:      0.8,
            EdgeSource.CONVERSATION:  0.7,
            EdgeSource.CONSOLIDATION: 0.6,
            EdgeSource.SANDBOX:       0.7,
            EdgeSource.DREAM:         0.3,
        }
        sq = source_quality_map.get(source, 0.5)

        node = Node(
            statement        = stmt,
            node_type        = node_type,
            cluster          = cluster,
            status           = status,
            predicted_answer = predicted_answer,
            testable_by      = testable_by,
            source_quality   = sq,
            last_verified    = time.time()
        )
        nid = self.brain.add_node(node)
        self._embedding_cache[nid] = stmt_emb
        if self.index:
            self.index.add(nid, stmt_emb)
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
        from insight_buffer import BUFFER_LOW
        for nid in new_ids:
            if self.index:
                emb = self.index.get_embedding(nid)
                if emb is None:
                    emb = self._embedding_cache.get(nid)
                if emb is None:
                    continue
                # Query at BUFFER_LOW to capture near-misses
                candidates = self.index.query(
                    emb, threshold=BUFFER_LOW, top_k=20
                )
                for other_id, sim in candidates:
                    if other_id == nid:
                        continue
                    if sim >= SIMILARITY_THRESHOLD:
                        continue  # too similar = duplicate, not weak edge
                    if (self.brain.graph.has_edge(nid, other_id) or
                            self.brain.graph.has_edge(other_id, nid)):
                        continue

                    if sim >= WEAK_EDGE_THRESHOLD:
                        # Strong enough for a weak edge
                        edge = Edge(
                            type       = EdgeType.ASSOCIATED,
                            narration  = (f"Weak associative link "
                                          f"(similarity={sim:.2f})"),
                            weight     = sim * 0.4,
                            confidence = sim,
                            source     = source
                        )
                        self.brain.add_edge(nid, other_id, edge)
                    elif self.insight_buffer and sim >= BUFFER_LOW:
                        # Near-miss — save for delayed re-evaluation
                        self.insight_buffer.add(nid, other_id, sim)
            else:
                all_embeddings = self._get_all_embeddings()
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
                    elif (self.insight_buffer and
                          BUFFER_LOW <= sim < WEAK_EDGE_THRESHOLD):
                        self.insight_buffer.add(nid, other_id, sim)
