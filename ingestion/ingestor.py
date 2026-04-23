import json
import re
import time
import urllib.parse
import numpy as np
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType, ANALOGY_WEIGHTS)
from config import THRESHOLDS
from embedding import embed as shared_embed
from llm_utils import llm_call, require_json
from scientist_workspace import ArtifactStatus

# ── Config ────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD  = THRESHOLDS.MERGE_NODE
DEDUP_QUERY_THRESHOLD = max(0.50, SIMILARITY_THRESHOLD - 0.16)
DEDUP_LLM_THRESHOLD   = max(0.62, SIMILARITY_THRESHOLD - 0.08)
WEAK_EDGE_THRESHOLD   = THRESHOLDS.WEAK_EDGE
MISSION_RELEVANCE_PREFILTER = max(0.24, THRESHOLDS.AGENDA_PREFILTER)
MISSION_RELEVANCE_FASTTRACK = 0.78
EDGE_LLM_MIN_SIM = max(0.22, THRESHOLDS.AGENDA_PREFILTER)
EDGE_LLM_MAX_PER_NODE = 3
EDGE_LLM_BASE_BUDGET = 8
EDGE_LLM_CALLS_PER_NODE_CAP = 2
MAX_DEDUP_LLM_CANDIDATES = 2
MAX_CONTRADICTION_LLM_CANDIDATES = 4
MAX_ANSWER_MATCH_LLM_CHECKS = 3
MAX_PROVENANCE_SPANS_PER_NODE = 3
MAX_PROVENANCE_CANDIDATES = 36
PROVENANCE_MIN_SPAN_CHARS = 80
PROVENANCE_MAX_SPAN_CHARS = 1000

# ── Prompts ───────────────────────────────────────────────────────────────────

NODE_EXTRACTION_PROMPT = """
You are building a knowledge graph for a scientific mind.

Read the following text carefully. Extract every distinct conceptual idea present.

Rules:
- Each node must be a self-contained conceptual statement — rich enough to stand alone
- Write each as 1-3 sentences. Not a keyword. Not a title. A thought.
- Capture the perspective, not just the topic
- If an idea contains a tension or uncertainty, include that in the statement
- If the source contains competing or contradictory claims, extract EACH claim
    as a separate node. Do not blend opposing claims into one reconciled summary.
- CRITICAL: Use dense, precise technical terminology. Avoid all conversational filler or introductory fluff (e.g. do NOT write "This idea suggests that...").
- Do NOT omit foundational named mechanisms, canonical examples, failure modes,
  or formal objects if they are central to understanding the text.
- For dense expository passages, prefer roughly 5 to 12 nodes rather than a
  tiny set of broad summaries.

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

COVERAGE_EXTRACTION_PROMPT = """
You are doing a second-pass extraction for a scientific knowledge graph.

The first pass often captures broad summaries but can miss anchor concepts that
future reasoning depends on.

Source text:
{text}

First-pass nodes:
{existing_nodes}

Return ONLY ADDITIONAL nodes that are important for scientific reasoning and
are missing from the first pass.

Prioritize concepts such as:
- named mechanisms or algorithms
- canonical examples, dilemmas, equilibria, or laws
- important failure modes or constraints
- formal objects that later reasoning would need explicitly

Rules:
- Return 0 to 6 additional nodes.
- Each node must still be a self-contained conceptual statement, not a label.
- Do not restate concepts already clearly present in the first-pass nodes.
- Favor concise anchor concepts over broad restatements.

Respond ONLY with a JSON array of strings. No preamble. No markdown.
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

TYPE SELECTION RULES (CRITICAL) — read carefully before choosing:

- "causes": Use ONLY if A is a mechanism, process, or event that directly and physically PRODUCES B. The test: does A happen BEFORE B and PHYSICALLY trigger B?
  ✓ "High temperature..." → causes "increased pressure"
  ✓ "Mutations in replication..." → causes "heritable diversity"
  ✗ "Studies show X correlates with Y" → this is "supports", not "causes"
  ✗ "Algorithm A implies mathematically that B is fast" → this is "supports", not "causes"

- "supports": Use when A is evidence, reasoning, prior work, or mathematical justification that makes B more credible, BUT does NOT physically trigger B.
  ✓ "Fossil record shows gradual change" → supports "evolution by natural selection"
  ✓ "Backpropagation computes gradients" → supports "gradient descent can train deep nets"

- "contradicts": Use ONLY when A and B make MUTUALLY EXCLUSIVE claims.
  ✓ "All objects fall at the same rate" contradicts "heavier objects fall faster"
  ✗ "Neural nets need data" vs "neural nets can overfit" — these are compatible
    ✗ "Game theory assumes rational agents" vs
        "Behavioral economics shows systematic deviations" — this is a critique /
        boundary condition, not a mutual exclusion
    ✗ "Energy is conserved" vs "Free energy decreases in spontaneous processes" —
        these can both be true and are not contradictory

- "analogy": Use when the same relational pattern appears in two different domains. You MUST specify depth:
  * "surface" (shared vocabulary/theme)
  * "structural" (same A:B::X:Y relational pattern across different domains)
  * "isomorphism" (formally identical mathematical equations)

- "associated": Use when ideas share a domain but lack direct causal, logical, or analogical links. This is the CORRECT AND FREQUENT answer for topical proximity.
  ✓ "Game theory developed in 1940s" associated with "Nash equilibrium"
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
  YES: "A Nash equilibrium is stable against unilateral deviation"
       vs "In Nash equilibrium, no player improves payoff by changing strategy alone"
  NO:  "DNA stores genetic information" vs "RNA transcribes information from DNA" (different roles)
  NO:  "High temperature increases pressure" vs "Pressure depends on molecular collisions" (different claims)

Respond with ONLY "yes" or "no".
"""

CANONICAL_MERGE_PROMPT = """
You are merging two duplicate scientific claims into ONE canonical claim.

Claim A: {claim_a}
Claim B: {claim_b}

Rules:
- Preserve only shared meaning that is defensible from both claims.
- Do not concatenate both claims.
- Do not add new unsupported assertions.
- Use 1-2 precise sentences.

Respond ONLY with the merged canonical claim.
"""


def _coerce_statement_list(items) -> list[str]:
    """Normalize LLM extraction output into a deduplicated list of statements."""
    if not isinstance(items, list):
        return []

    statements = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            stmt = (
                item.get('statement', '') or
                item.get('concept', '') or
                item.get('text', '') or
                str(item)
            )
        elif isinstance(item, str):
            stmt = item
        else:
            continue

        if not isinstance(stmt, str):
            continue
        stmt = stmt.strip()
        if not stmt:
            continue

        normalized = ' '.join(stmt.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        statements.append(stmt)

    return statements

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

Decide whether this idea is mission-relevant in a SCIENTIFICALLY ACTIONABLE sense.

Be conservative:
- Mark NOT relevant if the idea is only in the same broad field, only shares
  vocabulary, or is generic background that would not materially help answer
  the mission.
- Mark NOT relevant if the connection requires multiple unstated inferential
  leaps.
- Mark relevant only if the statement itself provides a direct mechanistic,
  evidential, or constraint-level bridge to answering the mission, or to a
  clear sub-question implied by it.
- Ask yourself: would a careful scientist likely cite this exact idea when
  explaining or testing an answer to the mission?

Strength rubric:
- 0.0-0.2: Unrelated, or only shares surface vocabulary / broad domain.
- 0.3-0.5: Background context or neighboring knowledge, but not something that
  should create a mission edge by itself.
- 0.6-0.85: Directly addresses a sub-question, mechanism, constraint, or
  observation that informs a possible answer.
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
                 insight_buffer=None, critic=None):
        self.brain            = brain
        self._embedding_cache = {}
        self.research_agenda  = research_agenda
        self.index            = embedding_index
        self.insight_buffer   = insight_buffer
        self.critic           = critic
        self._contradiction_cache = {}
        self._dedup_confirmation_cache = {}
        self._canonical_merge_cache = {}
        self._answer_match_cache = {}
        self._mission_relevance_cache = {}
        self._cluster_cache = {}
        self._edge_decision_cache = {}
        self._mission_embedding = None
        self._mission_embedding_text = ""
        # ── Pre-dedup state ──
        # Track raw text chunks already ingested to avoid re-extracting through
        # the LLM when the same (or very similar) source text is submitted again.
        self._ingested_text_hashes: set[int]          = set()
        self._ingested_text_embeddings: list[tuple[int, 'np.ndarray']] = []

    def _filter_novel_text(self, text: str) -> str:
        """Filter out raw text chunks that have already been ingested.

        Returns only the novel parts of the text.
        Splits by newlines to evaluate paragraphs/sections independently.
        Uses a two-tier check:
          1. Hash-based: catches exact duplicates (zero cost)
          2. Embedding-based: catches near-duplicate source text
        """
        import re
        chunks = [c.strip() for c in re.split(r'\n+', text) if c.strip()]
        
        novel_chunks = []
        for chunk in chunks:
            normalized = ' '.join(chunk.split()).lower()
            if not normalized:
                continue
                
            chunk_hash = hash(normalized)
            is_dup = False
            
            # Tier 1: exact match
            if chunk_hash in self._ingested_text_hashes:
                print(f"  [Pre-dedup] Exact text chunk already ingested — skipping chunk.")
                is_dup = True
                
            # Tier 2: embedding-based near-duplicate (only for chunks > 50 chars)
            elif len(normalized) > 50:
                chunk_emb = self._embed(normalized)
                for prev_hash, prev_emb in self._ingested_text_embeddings:
                    sim = self._cosine(chunk_emb, prev_emb)
                    if sim >= SIMILARITY_THRESHOLD:
                        print(f"  [Pre-dedup] Very similar text chunk already ingested "
                              f"(sim={sim:.3f}) — skipping chunk.")
                        is_dup = True
                        break
                        
            if not is_dup:
                novel_chunks.append(chunk)
                self._ingested_text_hashes.add(chunk_hash)
                if len(normalized) > 50:
                    chunk_emb = self._embed(normalized)
                    self._ingested_text_embeddings.append((chunk_hash, chunk_emb))
                    
        return "\n\n".join(novel_chunks)

    def _llm(self, prompt: str, temperature: float = 0.1) -> str:
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

    def _candidate_importance(self, node_type: NodeType,
                              source_quality: float,
                              contradiction_count: int) -> float:
        """Estimate initial claim importance before System 2 review."""
        base = 0.40 + (0.35 * source_quality)
        if node_type == NodeType.HYPOTHESIS:
            base += 0.10
        if contradiction_count > 0:
            base += 0.05
        return max(0.20, min(0.95, base))

    def _merge_unique_list(self, left: list[str] | None,
                           right: list[str] | None) -> list[str]:
        merged = []
        for item in (left or []) + (right or []):
            if not item or item in merged:
                continue
            merged.append(item)
        return merged

    def _merge_provenance_spans(self, left: list[dict] | None,
                                right: list[dict] | None,
                                max_items: int = MAX_PROVENANCE_SPANS_PER_NODE) -> list[dict]:
        merged = []
        seen = set()
        for span in (left or []) + (right or []):
            if not isinstance(span, dict):
                continue

            try:
                span_start = int(span.get("span_start", -1) or -1)
            except (TypeError, ValueError):
                span_start = -1
            try:
                span_end = int(span.get("span_end", -1) or -1)
            except (TypeError, ValueError):
                span_end = -1

            key = (
                str(span.get("source_id", "") or ""),
                str(span.get("source_ref", "") or ""),
                str(span.get("section_label", "") or ""),
                span_start,
                span_end,
                " ".join(str(span.get("quote", "") or "").split()).lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            normalized = dict(span)
            try:
                normalized["extraction_confidence"] = float(
                    normalized.get("extraction_confidence", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                normalized["extraction_confidence"] = 0.0
            merged.append(normalized)
            if len(merged) >= max_items:
                break
        return merged

    def _section_label_from_ref(self, source_ref: str) -> str:
        ref = str(source_ref or "")
        if "#section=" not in ref:
            return ""
        raw = ref.split("#section=", 1)[1]
        return urllib.parse.unquote(raw).strip()

    def _candidate_spans(self, text: str) -> list[dict]:
        if not text:
            return []

        spans = []
        cursor = 0
        for chunk in re.split(r"\n\s*\n", text):
            if not chunk.strip():
                continue
            start = text.find(chunk, cursor)
            if start < 0:
                start = cursor
            end = start + len(chunk)
            cursor = end

            cleaned = " ".join(chunk.split())
            if len(cleaned) < PROVENANCE_MIN_SPAN_CHARS:
                continue

            if len(cleaned) > PROVENANCE_MAX_SPAN_CHARS:
                cleaned = cleaned[:PROVENANCE_MAX_SPAN_CHARS]

            spans.append({
                "quote": cleaned,
                "span_start": start,
                "span_end": end,
            })
            if len(spans) >= MAX_PROVENANCE_CANDIDATES:
                break

        if spans:
            return spans

        cleaned = " ".join(text.split())
        if len(cleaned) >= PROVENANCE_MIN_SPAN_CHARS:
            return [{
                "quote": cleaned[:PROVENANCE_MAX_SPAN_CHARS],
                "span_start": 0,
                "span_end": min(len(text), PROVENANCE_MAX_SPAN_CHARS),
            }]
        return []

    def _statement_provenance_spans(self, statement: str,
                                    text: str,
                                    source_ids: list[str] | None,
                                    source_refs: list[str] | None,
                                    source: EdgeSource) -> list[dict]:
        if source not in {EdgeSource.READING, EdgeSource.RESEARCH}:
            return []

        refs = [
            str(ref).strip()
            for ref in (source_refs or [])
            if str(ref).strip()
        ]
        ids = [
            str(sid).strip()
            for sid in (source_ids or [])
            if str(sid).strip()
        ]
        if not refs and ids:
            for sid in ids:
                source_node = self.brain.get_node(sid)
                if not source_node:
                    continue
                source_node_refs = [
                    str(ref).strip()
                    for ref in (source_node.get("source_refs", []) or [])
                    if str(ref).strip()
                ]
                if source_node_refs:
                    refs.extend(source_node_refs)
                elif source_node.get("statement"):
                    refs.append(str(source_node.get("statement", "")).strip())
        if not refs and not ids:
            return []

        candidates = self._candidate_spans(text)
        if not candidates:
            return []

        stmt_emb = self._embed(statement)
        best_span = None
        best_sim = -1.0
        for candidate in candidates:
            sim = self._cosine(stmt_emb, self._embed(candidate["quote"]))
            if sim > best_sim:
                best_sim = sim
                best_span = candidate

        if best_span is None:
            return []

        confidence = max(0.0, min(1.0, float(best_sim)))
        pairs = max(1, len(refs), len(ids))
        spans = []
        for idx in range(pairs):
            source_ref = refs[idx] if idx < len(refs) else (refs[0] if refs else "")
            source_id = ids[idx] if idx < len(ids) else (ids[0] if ids else "")
            spans.append({
                "source_id": source_id,
                "source_ref": source_ref,
                "section_label": self._section_label_from_ref(source_ref),
                "span_start": int(best_span.get("span_start", -1)),
                "span_end": int(best_span.get("span_end", -1)),
                "quote": str(best_span.get("quote", "") or "")[:320],
                "extraction_confidence": confidence,
            })
            if len(spans) >= MAX_PROVENANCE_SPANS_PER_NODE:
                break
        return spans

    def _default_claim_node_type(self, source: EdgeSource) -> NodeType:
        if source in {EdgeSource.READING, EdgeSource.RESEARCH}:
            return NodeType.EVIDENCE_CLAIM
        return NodeType.CONCEPT

    def _epistemic_status_for(self, source: EdgeSource,
                              node_type: NodeType,
                              is_contradicted: bool = False) -> str:
        if is_contradicted:
            return ArtifactStatus.CONTRADICTED.value
        if node_type in {NodeType.QUESTION, NodeType.TASK}:
            return ArtifactStatus.OPEN.value
        if node_type == NodeType.GAP:
            return ArtifactStatus.SPECULATIVE.value
        if node_type == NodeType.HYPOTHESIS:
            if source == EdgeSource.CONVERSATION:
                return ArtifactStatus.PRIOR.value
            return ArtifactStatus.SPECULATIVE.value
        if node_type in {
            NodeType.EVIDENCE_CLAIM,
            NodeType.ANSWER,
            NodeType.EMPIRICAL,
        } and source in {EdgeSource.READING, EdgeSource.RESEARCH}:
            return ArtifactStatus.GROUNDED.value
        if source == EdgeSource.CONVERSATION:
            return ArtifactStatus.PRIOR.value
        if source in {EdgeSource.DREAM, EdgeSource.CONSOLIDATION}:
            return ArtifactStatus.SPECULATIVE.value
        return ArtifactStatus.OPEN.value

    def _normalize_text_key(self, text: str) -> str:
        return ' '.join((text or '').split()).lower()

    def _dedup_confirmed(self, claim_a: str, claim_b: str,
                         similarity: float) -> bool:
        key = self._normalize_pair_key(claim_a, claim_b)
        if key in self._dedup_confirmation_cache:
            return self._dedup_confirmation_cache[key]

        confirm = self._llm(
            DEDUP_CONFIRMATION_PROMPT.format(
                node_a=claim_a,
                node_b=claim_b,
                similarity=similarity,
            ),
            temperature=0.0
        ).strip().lower()
        is_dup = confirm.startswith("yes")
        self._dedup_confirmation_cache[key] = is_dup
        return is_dup

    def _canonical_merge_statement(self, claim_a: str, claim_b: str) -> str:
        key = self._normalize_pair_key(claim_a, claim_b)
        if key in self._canonical_merge_cache:
            return self._canonical_merge_cache[key]

        merged = self._llm(
            CANONICAL_MERGE_PROMPT.format(claim_a=claim_a, claim_b=claim_b),
            temperature=0.0
        ).strip()
        if not merged or len(merged) < 16:
            merged = claim_a
        self._canonical_merge_cache[key] = merged
        return merged

    def _cluster_for_statement(self, statement: str) -> str:
        key = self._normalize_text_key(statement)
        if key in self._cluster_cache:
            return self._cluster_cache[key]

        cluster = self._llm(
            CLUSTER_PROMPT.format(statement=statement)
        ).strip().lower()
        if not cluster:
            cluster = "general"
        self._cluster_cache[key] = cluster
        return cluster

    def _mission_similarity(self, statement: str) -> float:
        mission = self.brain.get_mission()
        if not mission:
            return 0.0
        mission_text = (mission.get('question') or '').strip()
        if not mission_text:
            return 0.0

        if mission_text != self._mission_embedding_text:
            self._mission_embedding = self._embed(mission_text)
            self._mission_embedding_text = mission_text

        stmt_emb = self._embed(statement)
        return self._cosine(stmt_emb, self._mission_embedding)

    def _node_embedding(self, node_id: str, statement: str = "") -> np.ndarray | None:
        if self.index:
            emb = self.index.get_embedding(node_id)
            if emb is not None:
                return emb
        if node_id in self._embedding_cache:
            return self._embedding_cache[node_id]
        if statement:
            return self._embed(statement)
        return None

    def _normalize_pair_key(self, a: str, b: str) -> tuple[str, str]:
        left = ' '.join((a or '').split()).lower()
        right = ' '.join((b or '').split()).lower()
        return tuple(sorted((left, right)))

    def _claims_contradict(self, existing_statement: str,
                           new_statement: str) -> bool:
        key = self._normalize_pair_key(existing_statement, new_statement)
        if key in self._contradiction_cache:
            return self._contradiction_cache[key]

        check = self._llm(
            CONTRADICTION_CHECK_PROMPT.format(
                existing=existing_statement,
                new=new_statement
            ),
            temperature=0.0
        )
        contradicts = check.lower().strip().startswith('yes')
        self._contradiction_cache[key] = contradicts
        return contradicts

    def _build_review_context(self, stmt_emb: np.ndarray, cluster: str,
                              source: EdgeSource,
                              contradiction_ids: list[str]) -> str:
        lines = [
            f"Source: {source.value}",
            f"Proposed cluster: {cluster}",
        ]
        if contradiction_ids:
            lines.append("Potential contradictions:")
            for nid in contradiction_ids[:4]:
                node = self.brain.get_node(nid)
                if node:
                    lines.append(f"- {node.get('statement', '')}")

        if self.index:
            matches = self.index.query(stmt_emb, threshold=0.58, top_k=4)
            if matches:
                lines.append("Nearest existing claims:")
                for nid, sim in matches:
                    node = self.brain.get_node(nid)
                    if node:
                        lines.append(
                            f"- [sim={sim:.2f}] {node.get('statement', '')}"
                        )
        return "\n".join(lines)

    def _review_statement(self, statement: str, node_type: NodeType,
                          source: EdgeSource, importance: float,
                          context: str, contradiction_ids: list[str],
                          source_ids: list[str] | None = None,
                          source_refs: list[str] | None = None,
                          expected_status: str = ArtifactStatus.OPEN.value,
                          ) -> tuple[str, float, bool]:
        """Send a candidate claim through System 2 review when available."""
        if not self.critic:
            return statement, importance, True

        # Skip full Critic review for grounded reading/research evidence claims.
        # These are already grounded by provenance — running the adversarial
        # dialogue is wasted compute (~3-6 LLM calls per claim saved).
        if source in (EdgeSource.READING, EdgeSource.RESEARCH) and \
           node_type in (NodeType.EVIDENCE_CLAIM, NodeType.ANSWER,
                         NodeType.EMPIRICAL, NodeType.CONCEPT):
            return statement, importance, True

        from critic.critic import CandidateThought, Verdict

        candidate = CandidateThought(
            claim=statement,
            source_module="ingestor",
            proposed_type=node_type.value,
            importance=importance,
            context=context,
            contradicts_existing=bool(contradiction_ids),
            grounded_evidence=list(source_refs or []),
            source_ids=list(source_ids or []),
            expected_status=expected_status,
        )
        critic_log = self.critic.evaluate_with_refinement(candidate)
        final_claim = (critic_log.final_claim or statement).strip()

        if critic_log.verdict == Verdict.ACCEPT:
            if final_claim != statement:
                print("  ↺ Critic refined ingested claim")
            adjusted_importance = max(importance * 0.7, critic_log.confidence)
            return final_claim, adjusted_importance, True

        if critic_log.verdict == Verdict.DEFER:
            deferred_candidate = CandidateThought(
                claim=final_claim,
                source_module=candidate.source_module,
                proposed_type=candidate.proposed_type,
                importance=candidate.importance,
                context=candidate.context,
                edge_type=candidate.edge_type,
                node_a_id=candidate.node_a_id,
                node_b_id=candidate.node_b_id,
                crosses_domains=candidate.crosses_domains,
                contradicts_existing=candidate.contradicts_existing,
                grounded_evidence=list(candidate.grounded_evidence),
                source_ids=list(candidate.source_ids),
                expected_status=candidate.expected_status,
            )
            self.critic.route_deferred(deferred_candidate)
            print("  ◇ Critic deferred ingested claim")
            return "", 0.0, False

        print("  ✗ Critic rejected ingested claim")
        return "", 0.0, False

    def _escalate_contradiction(self, new_node_id: str,
                                existing_node_id: str):
        if not self.research_agenda or not hasattr(self.research_agenda, "add_to_agenda"):
            return
        new_node = self.brain.get_node(new_node_id)
        existing = self.brain.get_node(existing_node_id)
        if not new_node or not existing:
            return

        question = (
            "Resolve contradiction: "
            f"{new_node.get('statement', '')[:120]} "
            "VS "
            f"{existing.get('statement', '')[:120]}"
        )
        item = self.research_agenda.add_to_agenda(
            text=question,
            item_type="question",
            cycle=getattr(self.research_agenda, 'cycle_count', 0),
            node_id=new_node_id,
        )
        item.priority = max(item.priority, 0.9)

    # ── Answer detection ──────────────────────────────────────────────────────

    def _check_against_agenda(self, node_id: str, statement: str):
        if not self.research_agenda:
            return
        open_items = self.research_agenda.get_prioritized_questions(20)
        if not open_items:
            return

        node_emb = self._embed(statement)
        scored_items = []
        for item in open_items:
            item_emb = self._embed(item.text)
            sim = self._cosine(node_emb, item_emb)
            if sim >= THRESHOLDS.AGENDA_PREFILTER:
                scored_items.append((sim, item))

        if not scored_items:
            return

        scored_items.sort(reverse=True, key=lambda t: t[0])
        for _, item in scored_items[:MAX_ANSWER_MATCH_LLM_CHECKS]:
            cache_key = (
                self._normalize_text_key(item.text),
                self._normalize_text_key(statement),
            )
            result = self._answer_match_cache.get(cache_key)
            if result is None:
                raw = self._llm(ANSWER_MATCH_PROMPT.format(
                    question=item.text, candidate=statement
                ), temperature=0.1)
                result = require_json(raw, default={})
                self._answer_match_cache[cache_key] = result

            if not isinstance(result, dict):
                continue

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

        cache_key = (
            self._normalize_text_key(mission['question']),
            self._normalize_text_key(statement),
        )
        mission_sim = self._mission_similarity(statement)
        self.brain.update_node(
            node_id,
            mission_relevance=max(
                mission_sim,
                float(self.brain.get_node(node_id).get("mission_relevance", 0.0) or 0.0),
            ),
        )
        if mission_sim < MISSION_RELEVANCE_PREFILTER:
            return
        if mission_sim >= MISSION_RELEVANCE_FASTTRACK:
            result = {
                "relevant": True,
                "strength": round(mission_sim, 3),
                "narration": "Direct semantic match to the active mission.",
            }
            self._mission_relevance_cache[cache_key] = result
            self.brain.link_to_mission(
                node_id,
                result["narration"],
                strength=mission_sim,
            )
            print(f"  ↗ Mission link (strength={mission_sim:.2f})")
            return

        result = self._mission_relevance_cache.get(cache_key)
        if result is None:
            raw = self._llm(MISSION_RELEVANCE_PROMPT.format(
                mission   = mission['question'],
                statement = statement
            ), temperature=0.0)
            result = require_json(raw, default={})
            self._mission_relevance_cache[cache_key] = result

        if not isinstance(result, dict):
            return

        strength = result.get('strength', 0)
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = 0.0
        if result.get('relevant') and strength >= THRESHOLDS.MISSION_LINK:
            self.brain.update_node(
                node_id,
                mission_relevance=max(
                    strength,
                    float(self.brain.get_node(node_id).get("mission_relevance", 0.0) or 0.0),
                ),
            )
            self.brain.link_to_mission(
                node_id,
                result.get('narration', ''),
                strength=strength
            )
            print(f"  ↗ Mission link (strength={strength:.2f})")

    def ingest_sections(self, sections: list[dict],
                        source: EdgeSource = EdgeSource.CONVERSATION,
                        prediction: str = "",
                        source_ids: list[str] | None = None,
                        source_refs: list[str] | None = None,
                        created_by: str = "") -> list[str]:
        """Ingest sectioned source text while preserving section anchor refs.

        Each section dict may include:
          - label: section heading
          - text: section body
          - source_ids: optional section-specific source node IDs
          - source_refs: optional section-specific refs
        """
        all_new_ids = []
        for idx, section in enumerate(sections, start=1):
            label = str(section.get("label", f"section_{idx}")).strip() or f"section_{idx}"
            section_text = str(section.get("text", "")).strip()
            if not section_text:
                continue

            section_source_ids = list(section.get("source_ids") or source_ids or [])
            section_source_refs = list(section.get("source_refs") or source_refs or [])
            if not section.get("source_refs"):
                safe_label = urllib.parse.quote(
                    label.lower().replace(" ", "_")[:80],
                    safe="",
                )
                anchored_refs = []
                for ref in section_source_refs:
                    base = str(ref or "").strip()
                    if not base:
                        continue
                    if "#" in base:
                        base = base.split("#", 1)[0]
                    anchored_refs.append(f"{base}#section={safe_label}")
                section_source_refs = anchored_refs

            section_created_by = created_by
            if section_created_by:
                section_created_by = f"{section_created_by}:{label}"

            new_ids = self.ingest(
                section_text,
                source=source,
                prediction=prediction,
                source_ids=section_source_ids,
                source_refs=section_source_refs,
                created_by=section_created_by,
            ) or []
            all_new_ids.extend(new_ids)

        return list(dict.fromkeys(all_new_ids))

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def ingest(self, text: str, source: EdgeSource = EdgeSource.CONVERSATION,
               prediction: str = "", source_ids: list[str] | None = None,
               source_refs: list[str] | None = None,
               created_by: str = ""):
        print(f"\n── Ingesting {len(text)} chars [{source.value}] ──")

        # ── Pre-dedup: filter out text that was already ingested ──
        text = self._filter_novel_text(text)
        if not text.strip():
            print("  [Pre-dedup] All text already ingested. Skipping extraction.")
            return []

        # extract concepts
        raw = self._llm(NODE_EXTRACTION_PROMPT.format(text=text), temperature=0.1)
        primary_statements = require_json(raw, default=[])
        if not isinstance(primary_statements, list):
            print(f"  Node extraction parse error")
            return []
        primary_statements = _coerce_statement_list(primary_statements)

        raw_anchor = self._llm(
            COVERAGE_EXTRACTION_PROMPT.format(
                text=text,
                existing_nodes=json.dumps(primary_statements, ensure_ascii=True),
            ),
            temperature=0.1,
        )
        anchor_statements = _coerce_statement_list(
            require_json(raw_anchor, default=[])
        )
        statements = _coerce_statement_list(
            primary_statements + anchor_statements
        )

        print(
            f"  Extracted {len(primary_statements)} primary nodes and "
            f"{len(anchor_statements)} anchor nodes "
            f"({len(statements)} unique total)"
        )

        # extract hypotheses
        raw_hyp = self._llm(
            HYPOTHESIS_EXTRACTION_PROMPT.format(text=text),
            temperature=0.1,
        )
        hypotheses = require_json(raw_hyp, default=[])
        if not isinstance(hypotheses, list):
            hypotheses = []

        print(f"  Extracted {len(hypotheses)} hypotheses")

        provenance_by_statement: dict[str, list[dict]] = {}
        if source in {EdgeSource.READING, EdgeSource.RESEARCH}:
            for stmt in statements:
                provenance_by_statement[self._normalize_text_key(stmt)] = (
                    self._statement_provenance_spans(
                        stmt,
                        text,
                        source_ids,
                        source_refs,
                        source,
                    )
                )

            for hyp in hypotheses:
                stmt = hyp if isinstance(hyp, str) else hyp.get('statement', '')
                if not isinstance(stmt, str) or not stmt.strip():
                    continue
                key = self._normalize_text_key(stmt)
                if key in provenance_by_statement:
                    continue
                provenance_by_statement[key] = self._statement_provenance_spans(
                    stmt,
                    text,
                    source_ids,
                    source_refs,
                    source,
                )

        new_node_ids      = []
        existing_embeddings = None if self.index else self._get_all_embeddings()
        default_claim_type = self._default_claim_node_type(source)

        # process concepts
        for stmt in statements:
            stmt_key = self._normalize_text_key(stmt)
            stmt_spans = provenance_by_statement.get(stmt_key, [])
            nid = self._process_statement(
                stmt, existing_embeddings, source, default_claim_type,
                source_ids=source_ids,
                source_refs=source_refs,
                provenance_spans=stmt_spans,
                extraction_confidence=max(
                    [
                        float(span.get("extraction_confidence", 0.0) or 0.0)
                        for span in stmt_spans
                    ]
                    or [0.0]
                ),
                created_by=created_by,
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
            stmt_key = self._normalize_text_key(stmt)
            stmt_spans = provenance_by_statement.get(stmt_key, [])
            nid = self._process_statement(
                stmt, existing_embeddings, source, NodeType.HYPOTHESIS,
                predicted_answer=hyp.get('predicted_answer', ''),
                testable_by=hyp.get('testable_by', ''),
                source_ids=source_ids,
                source_refs=source_refs,
                provenance_spans=stmt_spans,
                extraction_confidence=max(
                    [
                        float(span.get("extraction_confidence", 0.0) or 0.0)
                        for span in stmt_spans
                    ]
                    or [0.0]
                ),
                created_by=created_by,
            )
            if nid:
                new_node_ids.append(nid)
                if existing_embeddings is not None:
                    existing_embeddings[nid] = self._embedding_cache.get(
                        nid, self._embed(stmt))

        # Deduplicate active node ids for downstream expensive stages.
        # `_process_statement` may return existing node ids on merges/upgrades,
        # which can create duplicate pair evaluations if not normalized here.
        active_node_ids = list(dict.fromkeys(new_node_ids))

        # predictive processing (expectation engine)
        if prediction and active_node_ids:
            pred_emb  = self._embed(prediction)
            node_embs = [
                self._embedding_cache[nid]
                for nid in active_node_ids
                if nid in self._embedding_cache
            ]
            if node_embs:
                mean_emb = np.mean(node_embs, axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
                sim = self._cosine(pred_emb, mean_emb)
                surprise = 1.0 - sim
                print(f"  [Predictive Processing] Surprise (Prediction Error): {surprise:.2f}")

                # modulate importance based on surprise
                if surprise < 0.2:
                    print("  [Predictive Processing] Low surprise. Dampening importance.")
                    for nid in active_node_ids:
                        node = self.brain.get_node(nid)
                        if node:
                            self.brain.update_node(nid, importance=node.get('importance', 0.5) * 0.5)
                elif surprise > 0.6:
                    print("  [Predictive Processing] High surprise! Boosting importance.")
                    for nid in active_node_ids:
                        node = self.brain.get_node(nid)
                        if node:
                            self.brain.update_node(nid, importance=min(1.0, node.get('importance', 0.5) + 0.3))

        # extract high-value edges with semantic pre-ranking and bounded review
        edge_candidates = []
        for i in range(len(active_node_ids)):
            for j in range(i + 1, len(active_node_ids)):
                id_a, id_b = active_node_ids[i], active_node_ids[j]
                if (id_a == id_b or
                        self.brain.graph.has_edge(id_a, id_b) or
                        self.brain.graph.has_edge(id_b, id_a)):
                    continue

                node_a = self.brain.get_node(id_a)
                node_b = self.brain.get_node(id_b)
                if not node_a or not node_b:
                    continue

                emb_a = self._node_embedding(id_a, node_a.get('statement', ''))
                emb_b = self._node_embedding(id_b, node_b.get('statement', ''))
                if emb_a is None or emb_b is None:
                    continue

                sim = self._cosine(emb_a, emb_b)
                cross_cluster = (
                    node_a.get('cluster') and
                    node_b.get('cluster') and
                    node_a.get('cluster') != node_b.get('cluster')
                )

                if sim < EDGE_LLM_MIN_SIM and not (cross_cluster and sim >= 0.18):
                    continue

                priority = sim + (0.04 if cross_cluster else 0.0)
                edge_candidates.append((
                    priority,
                    id_a,
                    id_b,
                    node_a['statement'],
                    node_b['statement'],
                ))

        edge_budget = max(
            EDGE_LLM_BASE_BUDGET,
            EDGE_LLM_MAX_PER_NODE * max(1, len(active_node_ids)),
        )
        edge_candidates.sort(reverse=True)
        selected_edges = []
        per_node_counts = {}
        for candidate in edge_candidates:
            if len(selected_edges) >= edge_budget:
                break
            _, id_a, id_b, _, _ = candidate
            if (per_node_counts.get(id_a, 0) >= EDGE_LLM_CALLS_PER_NODE_CAP or
                    per_node_counts.get(id_b, 0) >= EDGE_LLM_CALLS_PER_NODE_CAP):
                continue
            selected_edges.append(candidate)
            per_node_counts[id_a] = per_node_counts.get(id_a, 0) + 1
            per_node_counts[id_b] = per_node_counts.get(id_b, 0) + 1

        skipped = max(0, len(edge_candidates) - len(selected_edges))
        if skipped:
            print(
                f"  Edge extraction budget: reviewed {len(selected_edges)} "
                f"of {len(edge_candidates)} candidate pairs"
            )

        for _, id_a, id_b, stmt_a, stmt_b in selected_edges:
            # Re-check existence because selected_edges is built before any
            # new edges are added in this pass.
            if (self.brain.graph.has_edge(id_a, id_b) or
                    self.brain.graph.has_edge(id_b, id_a)):
                continue

            pair_key = self._normalize_pair_key(stmt_a, stmt_b)
            ed = self._edge_decision_cache.get(pair_key)
            if ed is None:
                raw_edge = self._llm(EDGE_EXTRACTION_PROMPT.format(
                    node_a=stmt_a,
                    node_b=stmt_b
                ))
                ed = require_json(raw_edge, default={})
                self._edge_decision_cache[pair_key] = ed

            if not isinstance(ed, dict):
                continue

            try:
                if ed.get('related'):
                    edge_payload = dict(ed)
                    raw_type = edge_payload.get('type', 'associated')

                    # Revalidate contradiction edges with the dedicated
                    # contradiction checker so critique/exception relations are
                    # not mislabeled as mutually exclusive claims.
                    if (raw_type == 'contradicts' and
                            not self._claims_contradict(stmt_a, stmt_b)):
                        raw_type = 'associated'
                        edge_payload['type'] = 'associated'
                        if not edge_payload.get('narration'):
                            edge_payload['narration'] = (
                                "Topical relation without mutual exclusion."
                            )

                    if raw_type == 'analogy':
                        depth = edge_payload.get('analogy_depth', 'structural')
                        self.brain.add_analogy_edge(
                            id_a, id_b, depth,
                            edge_payload.get('narration', ''), source
                        )
                        print(f"  Edge [analogy:{depth}]: "
                              f"{id_a[:8]} ↔ {id_b[:8]}")
                    else:
                        try:
                            etype = EdgeType(raw_type)
                        except ValueError:
                            etype = EdgeType.ASSOCIATED

                        weight = edge_payload.get('weight', 0.5)
                        confidence = edge_payload.get('confidence', 0.5)
                        try:
                            weight = float(weight)
                        except (TypeError, ValueError):
                            weight = 0.5
                        try:
                            confidence = float(confidence)
                        except (TypeError, ValueError):
                            confidence = 0.5

                        exempt = (etype == EdgeType.CONTRADICTS)
                        edge = Edge(
                            type         = etype,
                            narration    = edge_payload.get('narration', ''),
                            weight       = max(0.0, min(1.0, weight)),
                            confidence   = max(0.0, min(1.0, confidence)),
                            source       = source,
                            decay_exempt = exempt
                        )
                        self.brain.add_edge(id_a, id_b, edge)
                        print(f"  Edge [{etype.value}]: "
                              f"{id_a[:8]} ↔ {id_b[:8]}")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"  Edge error: {e}")

        # weak associative edges
        self._add_weak_edges(active_node_ids, source)

        print(f"\n── Ingestion complete. {self.brain.stats()['nodes']} nodes, "
              f"{self.brain.stats()['edges']} edges ──\n")
        return new_node_ids

    def _process_statement(self, stmt: str, existing_embeddings,
                           source: EdgeSource, node_type: NodeType,
                           predicted_answer: str = "",
                           testable_by: str = "",
                           source_ids: list[str] | None = None,
                           source_refs: list[str] | None = None,
                           provenance_spans: list[dict] | None = None,
                           extraction_confidence: float = 0.0,
                           created_by: str = "") -> str:
        stmt_emb        = self._embed(stmt)
        best_match_id   = None
        best_similarity = 0.0

        try:
            extraction_confidence = float(extraction_confidence)
        except (TypeError, ValueError):
            extraction_confidence = 0.0
        extraction_confidence = max(0.0, min(1.0, extraction_confidence))
        provenance_spans = self._merge_provenance_spans([], provenance_spans)

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

        # ── Duplicate detection (indexed or fallback) ──
        if self.index:
            # Query wider (DEDUP_QUERY_THRESHOLD) to catch paraphrases that land
            # below the hard merge threshold due to embedding drift on re-ingest.
            matches = self.index.query(
                stmt_emb, threshold=DEDUP_QUERY_THRESHOLD, top_k=5
            )
            reviewed_candidates = 0
            for candidate_id, candidate_sim in matches:
                if candidate_sim >= SIMILARITY_THRESHOLD:
                    best_match_id   = candidate_id
                    best_similarity = candidate_sim
                    break
                elif candidate_sim >= DEDUP_LLM_THRESHOLD:
                    if reviewed_candidates >= MAX_DEDUP_LLM_CANDIDATES:
                        continue
                    candidate_data = self.brain.get_node(candidate_id)
                    if candidate_data:
                        reviewed_candidates += 1
                        if self._dedup_confirmed(
                                candidate_data["statement"],
                                stmt,
                                candidate_sim):
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
                    if not self._dedup_confirmed(
                            candidate_data["statement"],
                            stmt,
                            best_similarity):
                        best_match_id   = None
                        best_similarity = 0.0

        if best_similarity >= SIMILARITY_THRESHOLD:
            existing = self.brain.get_node(best_match_id)
            if not existing:
                best_match_id = None
                best_similarity = 0.0
            elif self._claims_contradict(existing['statement'], stmt):
                best_match_id = None
                best_similarity = 0.0
                print("  Similar but contradictory — creating separate node")
            else:
                merged_stmt = self._canonical_merge_statement(
                    existing['statement'],
                    stmt,
                )
                self.brain.update_node(
                    best_match_id,
                    statement=merged_stmt,
                    verification_count=(
                        existing.get('verification_count', 0) + 1
                    ),
                    last_verified=time.time(),
                    source_quality=max(existing.get('source_quality', 0.5), sq),
                    source_ids=self._merge_unique_list(
                        existing.get("source_ids", []),
                        source_ids,
                    ),
                    source_refs=self._merge_unique_list(
                        existing.get("source_refs", []),
                        source_refs,
                    ),
                    provenance_spans=self._merge_provenance_spans(
                        existing.get("provenance_spans", []),
                        provenance_spans,
                    ),
                    extraction_confidence=max(
                        float(existing.get("extraction_confidence", 0.0) or 0.0),
                        extraction_confidence,
                    ),
                    created_by=created_by or existing.get("created_by", ""),
                    epistemic_status=existing.get(
                        "epistemic_status",
                        self._epistemic_status_for(source, node_type),
                    ),
                )
                enriched_emb = self._embed(merged_stmt)
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
                    print(f"  Canonically merged duplicate (sim={best_similarity:.2f})")

                self._check_against_agenda(best_match_id, merged_stmt)
                self._check_mission_relevance(best_match_id, merged_stmt)
                return best_match_id

        cluster = self._cluster_for_statement(stmt)

        # ── Contradiction detection (indexed or fallback) ──
        status = NodeStatus.UNCERTAIN
        contradiction_ids = []
        if self.index:
            contra_candidates = self.index.query(
                stmt_emb, threshold=0.45, top_k=15
            )
            contradiction_checks = 0
            for cand_id, cand_sim in contra_candidates:
                if contradiction_checks >= MAX_CONTRADICTION_LLM_CANDIDATES:
                    break
                node_data = self.brain.get_node(cand_id)
                if not node_data:
                    continue
                contradiction_checks += 1
                if self._claims_contradict(node_data['statement'], stmt):
                    status = NodeStatus.CONTRADICTED
                    contradiction_ids.append(cand_id)
                    print(f"  Contradiction with {cand_id[:8]}")
        else:
            contradiction_checks = 0
            for nid, nemb in existing_embeddings.items():
                if contradiction_checks >= MAX_CONTRADICTION_LLM_CANDIDATES:
                    break
                if self._cosine(stmt_emb, nemb) > 0.45:
                    existing = self.brain.get_node(nid)
                    if not existing:
                        continue
                    contradiction_checks += 1
                    if self._claims_contradict(existing['statement'], stmt):
                        status = NodeStatus.CONTRADICTED
                        contradiction_ids.append(nid)
                        print(f"  Contradiction with {nid[:8]}")

        proposed_importance = self._candidate_importance(
            node_type,
            sq,
            len(contradiction_ids),
        )
        epistemic_status = self._epistemic_status_for(
            source,
            node_type,
            is_contradicted=bool(contradiction_ids),
        )
        review_context = self._build_review_context(
            stmt_emb,
            cluster,
            source,
            contradiction_ids,
        )
        reviewed_stmt, reviewed_importance, accepted = self._review_statement(
            stmt,
            node_type,
            source,
            proposed_importance,
            review_context,
            contradiction_ids,
            source_ids=source_ids,
            source_refs=source_refs,
            expected_status=epistemic_status,
        )
        if not accepted:
            return ""

        if reviewed_stmt != stmt:
            stmt = reviewed_stmt
            stmt_emb = self._embed(stmt)
            cluster = self._cluster_for_statement(stmt)

        node = Node(
            statement        = stmt,
            node_type        = node_type,
            cluster          = cluster,
            status           = status,
            epistemic_status = epistemic_status,
            importance       = reviewed_importance,
            predicted_answer = predicted_answer,
            testable_by      = testable_by,
            source_quality   = max(sq, reviewed_importance * 0.8),
            verification_count = 1,
            last_verified    = time.time(),
            source_ids       = list(source_ids or []),
            source_refs      = list(source_refs or []),
            provenance_spans = list(provenance_spans or []),
            extraction_confidence = extraction_confidence,
            created_by       = created_by,
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
            self._escalate_contradiction(nid, contra_id)

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
