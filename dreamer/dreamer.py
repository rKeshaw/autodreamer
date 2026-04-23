import random
import time
import json
import re
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType, BrainMode, ANALOGY_WEIGHTS)
from config import THRESHOLDS
from embedding import embed as shared_embed
from llm_utils import llm_call, require_json
from scientist_workspace import ArtifactStatus

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_STEPS       = 20
DEFAULT_TEMP        = 0.7
DEPTH_STEPS         = 3
DEPTH_COOLDOWN_STEPS = 2
DEPTH_COPY_OVERLAP  = 0.85
VISITED_PENALTY     = 0.45
RECENT_CLUSTER_WINDOW = 4
RECENT_CLUSTER_PENALTY = 0.18
EXPLORATION_JUMP_PROB = 0.28
MAX_DREAM_QUESTION_WORDS = 36
QUESTION_CONTEXT_SIM_MIN = 0.20
QUESTION_CONTEXT_TOKEN_OVERLAP_MIN = 2
MISSION_ADVANCE_SIM_MIN = 0.26
MISSION_ADVANCE_TOKEN_OVERLAP_MIN = 2
HYPOTHESIS_CONTEXT_SIM_MIN = 0.18
HYPOTHESIS_CONTEXT_TOKEN_OVERLAP_MIN = 3

# mode modifiers
MODE_TEMP_BOOST = {
    "transitional": 0.25,   # chaotic reorientation cycle
    "wandering":    0.10,   # slightly freer than focused
    "focused":      0.0,
}
MODE_STEPS_BOOST = {
    "transitional": 8,      # more steps during transitional
    "wandering":    0,
    "focused":      0,
}

class DreamMode(str, Enum):
    WANDERING = "wandering"
    PRESSURE  = "pressure"
    SEEDED    = "seeded"

@dataclass
class DreamStep:
    step:            int
    from_id:         str
    to_id:           str
    edge_type:       str
    edge_narration:  str
    narration:       str
    question:        str  = ""
    is_insight:      bool = False
    insight_depth:   str  = ""
    new_edge:        bool = False
    answer_match:    str  = "none"
    answer_detail:   str  = ""
    depth_triggered: bool = False
    mission_advance: bool = False
    mission_strength: float = 0.0

@dataclass
class DreamLog:
    mode:             str
    brain_mode:       str  = "focused"
    started_at:       float = field(default_factory=time.time)
    steps:            list  = field(default_factory=list)
    questions:        list  = field(default_factory=list)
    insights:         list  = field(default_factory=list)
    answers:          list  = field(default_factory=list)
    mission_advances: list  = field(default_factory=list)
    summary:          str   = ""

    def to_dict(self):
        return {
            "mode":             self.mode,
            "brain_mode":       self.brain_mode,
            "started_at":       self.started_at,
            "steps":            [s.__dict__ for s in self.steps],
            "questions":        self.questions,
            "insights":         self.insights,
            "answers":          self.answers,
            "mission_advances": self.mission_advances,
            "summary":          self.summary
        }

# ── Prompts ───────────────────────────────────────────────────────────────────

NARRATION_FOCUSED = """
You are a dreaming scientific mind, moving between ideas.

Central research question you are working on:
"{mission}"

From: "{from_node}"
Via [{edge_type}]: "{edge_narration}"
To: "{to_node}"

Narrate this mental journey in 2-4 sentences. Be technically precise and dense. DO NOT write qualitative fluff. State explicitly what the structural mechanism is if one exists. Let the central question color your thinking without forcing it.

If something is unresolved or a new hypothesis emerges, ask ONE empirically testable, novel, and highly specific research question starting with "Q:". A good question should propose a measurement or intervention. Avoid generic questions like "How does X affect Y?".
Keep the question anchored to the entities and mechanisms already present in the supplied concepts.
Do NOT introduce new equations, symbolic notation, named model families, or formal operators unless they already appear in the supplied concepts.
Prefer direct experimental, observational, ablation, perturbation, or comparison questions in plain scientific prose.

Classify any insight strictly:
- INSIGHT: none (Default)
- INSIGHT: surface (Shares vocabulary, topic, or abstract theme. NOT a deep insight.)
- INSIGHT: structural (Requires an explicit 1-to-1 role, mechanism, or constraint mapping. If you cannot state at least two concrete correspondences, it is merely surface.)
- INSIGHT: isomorphism (Requires the same formal or mathematical structure under renaming of variables, state updates, or objective functions.)

If you output INSIGHT: structural or INSIGHT: isomorphism, the narration MUST contain one sentence beginning exactly with "MAP:" that names at least two explicit correspondences.
For isomorphism, the MAP sentence must mention the shared variables, update rule, objective, or conserved quantity.

WARNING: Avoid superficial analogies.
BAD structural insight: "Both mitochondria and server farms produce energy." (This is merely a surface analogy).
BAD isomorphism: "Both systems seek equilibrium." (Too generic; no shared formal rule is stated.)
GOOD structural insight: "MAP: synaptic weight pruning -> parameter pruning; REM noise bursts -> annealing noise; both remove brittle local configurations while preserving global performance."
GOOD isomorphism insight: "MAP: free-energy minimization dF/dt < 0 -> gradient descent dL/dt < 0; local state update -> parameter update; both follow the same downhill optimization form under renamed variables."
"""

NARRATION_WANDERING = """
You are a dreaming scientific mind — no particular agenda, just following curiosity.

From: "{from_node}"
Via [{edge_type}]: "{edge_narration}"
To: "{to_node}"

Narrate this mental journey in 2-4 sentences. Be technically precise and dense. DO NOT write qualitative fluff. Make unexpected but scientifically rigorous connections. State explicitly what the structural mechanism is if one exists.

If something is intriguing or a new hypothesis emerges, ask ONE empirically testable, novel, and highly specific research question starting with "Q:". A good question should propose a measurement or intervention. Avoid generic questions like "How does X affect Y?".
Keep the question anchored to the entities and mechanisms already present in the supplied concepts.
Do NOT introduce new equations, symbolic notation, named model families, or formal operators unless they already appear in the supplied concepts.
Prefer direct experimental, observational, ablation, perturbation, or comparison questions in plain scientific prose.

Classify any insight strictly:
- INSIGHT: none (Default)
- INSIGHT: surface (Shares vocabulary, topic, or abstract theme. NOT a deep insight.)
- INSIGHT: structural (Requires an explicit 1-to-1 role, mechanism, or constraint mapping. If you cannot state at least two concrete correspondences, it is merely surface.)
- INSIGHT: isomorphism (Requires the same formal or mathematical structure under renaming of variables, state updates, or objective functions.)

If you output INSIGHT: structural or INSIGHT: isomorphism, the narration MUST contain one sentence beginning exactly with "MAP:" that names at least two explicit correspondences.
For isomorphism, the MAP sentence must mention the shared variables, update rule, objective, or conserved quantity.

WARNING: Avoid superficial analogies.
BAD structural insight: "Both mitochondria and server farms produce energy." (This is merely a surface analogy).
BAD isomorphism: "Both systems seek equilibrium." (Too generic; no shared formal rule is stated.)
GOOD structural insight: "MAP: synaptic weight pruning -> parameter pruning; REM noise bursts -> annealing noise; both remove brittle local configurations while preserving global performance."
GOOD isomorphism insight: "MAP: free-energy minimization dF/dt < 0 -> gradient descent dL/dt < 0; local state update -> parameter update; both follow the same downhill optimization form under renamed variables."
"""

NARRATION_TRANSITIONAL = """
You are a dreaming scientific mind in a state of reorientation.
A new central question has just arrived:
"{mission}"

The mind is reorganizing itself around this question, finding new connections
everywhere. Everything seems to relate. Be chaotic, associative, surprising.

From: "{from_node}"
Via [{edge_type}]: "{edge_narration}"
To: "{to_node}"

Narrate in 2-4 sentences. Make unexpected connections. Be technically precise and dense. DO NOT write qualitative fluff. State explicitly what the structural mechanism is if one exists.

If something sparks or a new hypothesis emerges, ask ONE empirically testable, novel, and highly specific research question starting with "Q:". A good question should propose a measurement or intervention. Avoid generic questions like "How does X affect Y?".
Keep the question anchored to the entities and mechanisms already present in the supplied concepts.
Do NOT introduce new equations, symbolic notation, named model families, or formal operators unless they already appear in the supplied concepts.
Prefer direct experimental, observational, ablation, perturbation, or comparison questions in plain scientific prose.

Classify any insight strictly:
- INSIGHT: none (Default)
- INSIGHT: surface (Shares vocabulary, topic, or abstract theme. NOT a deep insight.)
- INSIGHT: structural (Requires an explicit 1-to-1 role, mechanism, or constraint mapping. If you cannot state at least two concrete correspondences, it is merely surface.)
- INSIGHT: isomorphism (Requires the same formal or mathematical structure under renaming of variables, state updates, or objective functions.)

If you output INSIGHT: structural or INSIGHT: isomorphism, the narration MUST contain one sentence beginning exactly with "MAP:" that names at least two explicit correspondences.
For isomorphism, the MAP sentence must mention the shared variables, update rule, objective, or conserved quantity.

WARNING: Avoid superficial analogies.
BAD structural insight: "Both mitochondria and server farms produce energy." (This is merely a surface analogy).
BAD isomorphism: "Both systems seek equilibrium." (Too generic; no shared formal rule is stated.)
GOOD structural insight: "MAP: synaptic weight pruning -> parameter pruning; REM noise bursts -> annealing noise; both remove brittle local configurations while preserving global performance."
GOOD isomorphism insight: "MAP: free-energy minimization dF/dt < 0 -> gradient descent dL/dt < 0; local state update -> parameter update; both follow the same downhill optimization form under renamed variables."
"""

MISSION_ADVANCE_PROMPT = """
Central research question: "{mission}"
New idea encountered: "{node}"
Connection made: "{narration}"

Does this meaningfully advance the central question?

Strength rubric:
- 0.1-0.3: Tangential — Relates to the same topic but does not inform the question.
- 0.4-0.6: Relevant context — Provides useful background but is NOT an advance.
- 0.7-0.85: Advancing — ONLY assign if it provides a direct missing piece or evidence to resolve the mission.
- 0.9-1.0: Breakthrough

IMPORTANT: Do NOT assign a rating >= 0.5 unless the idea provides *direct* evidence or a *missing piece* to answer the mission. Shared topic alone must be < 0.5.

Respond with JSON:
{{
  "advances": true or false,
  "explanation": "one sentence",
  "strength": 0.0 to 1.0 (use rubric above)
}}

Respond ONLY with JSON. No preamble.
"""

ANSWER_CHECK_PROMPT = """
Current idea: {current_node}
Open question: {question}

Does the current idea answer or significantly advance this question?

Grading definitions:
- "none": The idea is unrelated to the question, or only shares surface-level vocabulary.
- "partial": The idea addresses PART of the question or provides indirect evidence.
  Example: Question "What causes X?" — idea describes a correlated factor but not a cause.
- "strong": The idea directly answers or resolves the question, or provides definitive evidence.
  Example: Question "What causes X?" — idea identifies the specific mechanism causing X.

Respond with JSON:
{{
  "match": "none" | "partial" | "strong",
  "explanation": "one sentence"
}}

Respond ONLY with JSON. No preamble.
"""

QUESTION_SIMILAR_PROMPT = """
Are these two questions asking the SAME THING, even if worded differently?

"Same" means: if one were fully answered, the other would also be fully answered.
"Different" means: they could have separate answers, even if they cover related topics.

Q1: {q1}
Q2: {q2}

Examples:
- "How does sleep affect memory?" vs "What is sleep's role in memory consolidation?" → YES (same question)
- "How does sleep affect memory?" vs "Does REM or NREM matter more for memory?" → NO (related but distinct)

Respond ONLY "yes" or "no".
"""

INSIGHT_VALIDATION_PROMPT = """
You are validating a claimed scientific insight depth.

Concept A: "{from_node}"
Concept B: "{to_node}"
Claimed narration: "{narration}"
Claimed depth: {claimed_depth}

Depth rules:
- none: no real insight
- surface: shared vocabulary, topic, or broad theme only
- structural: requires an explicit mechanism or role mapping between A and B
- isomorphism: requires the same formal or mathematical structure, not just similar behavior

Be conservative. If the mapping is suggestive but not explicit, downgrade it.
If the claim equates mechanism with metaphor, downgrade it.
Downgrade to surface when the relation is primarily:
- historical progression or intellectual transition
- conceptual prerequisite or enabling background
- abstract theory being realized by a concrete mechanism
- broad functional similarity with different underlying mechanisms
- causal sequence or precondition rather than mapped roles
- shared constraints, shared vocabulary, or shared domain only

For structural or isomorphic depth, require a preserved mapping of roles, variables, constraints, or update rules.

Respond with JSON:
{{
  "depth": "none" | "surface" | "structural" | "isomorphism",
  "reason": "one sentence",
  "has_explicit_mapping": true or false
}}

Respond ONLY with JSON.
"""

INSIGHT_REFINEMENT_PROMPT = """
You are sharpening a dream-derived scientific insight so that only real deep mappings survive.

Concept A: "{from_node}"
Concept B: "{to_node}"
Draft narration: "{narration}"
Draft depth: {claimed_depth}
Mission context: "{mission}"

Rules:
- surface = shared topic, shared vocabulary, historical progression, prerequisite relation, substrate-realization relation, or broad functional similarity only.
- structural = at least two explicit role, mechanism, or constraint correspondences.
- isomorphism = the same formal structure under renamed variables, update rules, objectives, equilibria, or conservation constraints.
- If the claim does NOT meet structural or isomorphism standards, downgrade it.
- If depth is structural or isomorphism, include a sentence beginning exactly with "MAP:".
- The MAP sentence must name at least two explicit correspondences.
- For isomorphism, the MAP sentence must mention the shared formal rule, variables, objective, update law, or conserved quantity.
- Avoid generic English logic such as "both optimize", "both balance", "both involve states", or "both have constraints" unless the concrete mapping is stated.

Respond EXACTLY in JSON:
{{
  "depth": "none" | "surface" | "structural" | "isomorphism",
  "narration": "1-3 technically precise sentences",
  "mapping_pairs": ["A_role -> B_role", "A_constraint -> B_constraint"],
  "has_explicit_mapping": true or false,
  "has_formal_anchor": true or false,
  "reason": "one sentence"
}}
"""

DEPTH_NARRATION_PROMPT = """
You are a scientific mind that found something significant while dreaming.
{mission_line}
You landed on: "{node}"
It connects to: "{question}"
Connection: {explanation}

Explore this in 2-3 sentences. Be technically precise. DO NOT write qualitative fluff.
Do NOT repeat the landed statement or the connection verbatim.
If you cannot add a materially new mechanistic angle, respond EXACTLY with: NO_DEPTH
End with ONE highly specific, empirically testable research question starting with "Q:". 
The question MUST propose a specific measurement, variable, or intervention. Avoid generic inquiries.
Keep the question anchored to the entities and mechanisms already present in the supplied concepts.
Do NOT introduce new equations, symbolic notation, named model families, or formal operators unless they already appear in the supplied concepts.
"""

SUMMARY_FOCUSED = """
You are summarizing a dream cycle. Brain mode: FOCUSED.
Central question: "{mission}"

Dream steps: {steps}
Answer matches: {answers}
Mission advances: {mission_advances}

Write 4-5 short scientific sentences:
1. What mechanisms or mappings were explored
2. Which points are still only speculative
3. Whether any grounded mission advance occurred
4. What question most needs follow-up

No first person. No emotional language. No sign-off.
"""

SUMMARY_WANDERING = """
You are summarizing a dream cycle. Brain mode: WANDERING (no mission — free association).

Dream steps: {steps}

Write 4-5 short scientific sentences:
1. What domains or mechanisms were traversed
2. Which connections remained weak or speculative
3. Any structural mapping worth later validation
4. What question emerged for later investigation

No first person. No playful tone. No sign-off.
"""

SUMMARY_TRANSITIONAL = """
You are summarizing a dream cycle. Brain mode: TRANSITIONAL.
A new question just arrived: "{mission}"
The mind is reorganizing itself.

Dream steps: {steps}

Write 4-5 short scientific sentences describing:
1. Which prior concepts were recruited into the new mission
2. Which tentative links appeared
3. Which parts remain unsupported
4. What should be validated next

No first person. No dramatic language. No sign-off.
"""

START_NODE_FOCUSED = """
Central research question: "{mission}"
Select the most generative starting node for a dream exploring the central question
through unexpected associations.
Nodes: {nodes}
Respond with ONLY the node ID.
"""

START_NODE_WANDERING = """
No mission — pure curiosity.
Select the most interesting node to start a free-associative dream from.
Favor nodes with unresolved tensions or underexplored connections.
Nodes: {nodes}
Respond with ONLY the node ID.
"""


DREAM_FROM_ANOMALY_PROMPT = """You are the Dreamer, a scientific anomaly analyst. A prior hypothesis was contradicted.

Hypothesis (What we expected):
{hypothesis}

Contradicting Evidence (What we actually observed):
{evidence}

Generate ONE conservative next-step alternative that is directly motivated by the contradiction.
Do not invent a new force, sector, field, geometry, symmetry, or observable unless the contradiction text already motivates it.
If the evidence only supports a narrower discriminating question, return a hypothesis that explicitly stays provisional.

Return your new hypothesis as JSON:
{{
  "new_hypothesis": "A single testable anomaly-driven claim.",
  "explanation": "Why this follows from the contradiction, in one sentence."
}}
"""

class Dreamer:


    def __init__(self, brain: Brain, research_agenda=None, critic=None,
                 observer=None, embedding_index=None):
        self.brain           = brain
        self.research_agenda = research_agenda or observer
        self.critic          = critic
        self._embedding_cache = {}


    def dream_from_anomaly(self, expected_hypothesis_id: str) -> dict | None:
        hyp_node = self.brain.get_node(expected_hypothesis_id)
        if not hyp_node:
            return None
            
        print(f"\n  ── 🌪️  Anomaly Detection: Hypothesis Contradicted ──")
        print(f"     Old: {hyp_node['statement'][:80]}...")
        
        evidence_texts = []
        edges = self.brain.graph.in_edges(expected_hypothesis_id, data=True)
        for u, v, data in edges:
            if data.get('type') == EdgeType.CORRECTED_BY.value:
                enode = self.brain.get_node(u)
                if enode:
                    evidence_texts.append(enode.get('statement', ''))
                    
        if not evidence_texts:
            print("     Could not locate contradicting evidence. Aborting anomaly dream.")
            return None
            
        evidence_str = "\n".join(f"- {t}" for t in evidence_texts)
        
        raw = self._llm(DREAM_FROM_ANOMALY_PROMPT.format(
            hypothesis=hyp_node.get('statement', ''),
            evidence=evidence_str
        ), temperature=0.35)
        
        result = self._json_object(raw, default={"new_hypothesis": "", "explanation": ""})
        new_hyp = result.get("new_hypothesis", "").strip()
        explanation = result.get("explanation", "Generated from anomaly.")
        
        if not new_hyp:
            return None

        accepted, reviewed_claim, review_reason = self._critic_review_hypothesis(
            new_hyp,
            explanation,
            evidence_str,
            expected_status=ArtifactStatus.OPEN.value,
        )
        if not accepted:
            if self.research_agenda and hasattr(self.research_agenda, "add_to_agenda"):
                self.research_agenda.add_to_agenda(
                    text=f"Test anomaly alternative: {new_hyp}",
                    item_type="question",
                    cycle=getattr(self.research_agenda, "cycle_count", 0),
                    node_id="",
                )
            print(f"  ✗ Anomaly hypothesis deferred: {review_reason or 'critic rejected unsupported alternative'}")
            return {
                "status": "deferred",
                "statement": new_hyp,
                "explanation": review_reason or explanation,
            }

        new_hyp = reviewed_claim or new_hyp
            
        print(f"  ✨ Anomaly follow-up hypothesis: {new_hyp[:80]}...")
        
        node = Node(
            statement=new_hyp,
            node_type=NodeType.HYPOTHESIS,
            status=NodeStatus.HYPOTHETICAL,
            epistemic_status=ArtifactStatus.SPECULATIVE.value,
            importance=0.9,
            source_quality=0.8,
            created_by="dreamer_anomaly",
            cluster="anomaly_drift"
        )
        new_id = self.brain.add_node(node)
        
        from graph.brain import Edge
        for u, v, data in edges:
            if data.get('type') == EdgeType.CORRECTED_BY.value:
                edge = Edge(
                    type=EdgeType.SUPPORTS,
                    narration=explanation,
                    weight=0.8,
                    confidence=0.7,
                    source=EdgeSource.DREAM
                )
                self.brain.add_edge(u, new_id, edge)
                
        return {
            "node_id": new_id,
            "statement": new_hyp,
            "explanation": explanation
        }

    def _llm(self, prompt: str, temperature: float = 0.7) -> str:


        return llm_call(prompt, temperature=temperature, role="creative")

    def _embed(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        emb = shared_embed(text)
        self._embedding_cache[text] = emb
        return emb

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _json_object(self, raw: str, default: dict | None = None) -> dict:
        parsed = require_json(raw, default=default or {})
        return parsed if isinstance(parsed, dict) else dict(default or {})

    def _critic_review_hypothesis(self, hypothesis_text: str,
                                  mechanism: str,
                                  seed_claims_text: str,
                                  expected_status: str = ArtifactStatus.OPEN.value) -> tuple[bool, str, str]:
        if not self.critic:
            return True, hypothesis_text, ""
        from critic.critic import CandidateThought, Verdict

        context = "\n".join(filter(None, [
            "Seed findings:",
            seed_claims_text,
            "Proposed mechanism:",
            mechanism,
        ]))
        candidate = CandidateThought(
            claim=hypothesis_text,
            source_module="dreamer",
            proposed_type=NodeType.HYPOTHESIS.value,
            importance=0.72,
            context=context,
            expected_status=expected_status,
        )
        review = self.critic.evaluate_with_refinement(candidate)
        accepted = review.verdict == Verdict.ACCEPT
        reviewed_claim = (
            review.final_claim or review.refinement_note or hypothesis_text
        )
        return accepted, reviewed_claim, review.verdict_reason or review.rejection_reason or ""

    def _mission_text(self) -> str:
        if self.brain.is_wandering():
            return ""
        m = self.brain.get_mission()
        return m['question'] if m else ""

    def _narration_prompt(self, from_node, edge_type, edge_narration, to_node):
        mission = self._mission_text()
        mode    = self.brain.get_mode()
        if mode == BrainMode.TRANSITIONAL.value:
            return NARRATION_TRANSITIONAL.format(
                mission=mission or "No mission yet",
                from_node=from_node, edge_type=edge_type,
                edge_narration=edge_narration, to_node=to_node)
        elif mode == BrainMode.WANDERING.value or not mission:
            return NARRATION_WANDERING.format(
                from_node=from_node, edge_type=edge_type,
                edge_narration=edge_narration, to_node=to_node)
        else:
            return NARRATION_FOCUSED.format(
                mission=mission, from_node=from_node, edge_type=edge_type,
                edge_narration=edge_narration, to_node=to_node)

    def _summary_prompt(self, step_text, answer_text, mission_text):
        mode    = self.brain.get_mode()
        mission = self._mission_text()
        if mode == BrainMode.TRANSITIONAL.value:
            return SUMMARY_TRANSITIONAL.format(
                mission=mission or "Newly set question",
                steps=step_text)
        elif mode == BrainMode.WANDERING.value or not mission:
            return SUMMARY_WANDERING.format(steps=step_text)
        else:
            return SUMMARY_FOCUSED.format(
                mission=mission, steps=step_text,
                answers=answer_text, mission_advances=mission_text)

    # ── Question deduplication ────────────────────────────────────────────────

    def _is_duplicate_question(self, new_q, existing, embeddings):
        if not existing:
            return False
        new_emb = self._embed(new_q)
        for eq, emb in zip(existing, embeddings):
            sim = self._cosine(new_emb, emb)
            if sim > THRESHOLDS.QUESTION_DEDUP_HIGH:
                return True
            if sim > THRESHOLDS.QUESTION_DEDUP_LOW:
                raw = self._llm(
                    QUESTION_SIMILAR_PROMPT.format(q1=new_q, q2=eq),
                    temperature=0.1
                )
                if raw.lower().startswith('yes'):
                    return True
        return False

    def _add_question(self, q, questions, q_embeddings):
        if not q:
            return False
        if self._is_duplicate_question(q, questions, q_embeddings):
            return False
        questions.append(q)
        q_embeddings.append(self._embed(q))
        return True

    def _statement_key(self, statement):
        if not statement:
            return ""
        return re.sub(r"\s+", " ", statement).strip().lower()

    def _token_set(self, text):
        if not text:
            return set()
        return {
            tok for tok in re.findall(r"[a-z0-9]+", text.lower())
            if len(tok) > 2
        }

    def _is_mostly_repeated_text(self, text, reference):
        text_key = self._statement_key(text)
        ref_key = self._statement_key(reference)
        if not text_key or not ref_key:
            return False
        if text_key == ref_key or text_key in ref_key or ref_key in text_key:
            return True
        text_tokens = self._token_set(text)
        ref_tokens = self._token_set(reference)
        if not text_tokens or not ref_tokens:
            return False
        overlap = len(text_tokens & ref_tokens) / max(1, min(len(text_tokens), len(ref_tokens)))
        return overlap >= DEPTH_COPY_OVERLAP

    def _depth_seed_is_novel(self, node_data, narration, question):
        statement = node_data.get('statement', '')
        if self._is_mostly_repeated_text(narration, statement):
            return False
        if question and self._is_mostly_repeated_text(question, statement):
            return False
        return True

    def _question_has_symbolic_overreach(self, question):
        text = str(question or "")
        if not text:
            return False
        if any(marker in text for marker in ("$", "\\", "\\mathbf", "\\sigma", "\\Gamma", "\\lambda")):
            return True
        if re.search(r"\b[A-Za-z]+_[A-Za-z0-9]+\b", text):
            return True
        return False

    def _question_context_metrics(self, question, context_text):
        if not question or not context_text:
            return 0, 0.0
        overlap = len(self._token_set(question) & self._token_set(context_text))
        try:
            similarity = self._cosine(
                self._embed(question),
                self._embed(context_text),
            )
        except Exception:
            similarity = 0.0
        return overlap, similarity

    def _question_snippet(self, text, max_words=10):
        clean = " ".join(str(text or "").split())
        clean = re.sub(r"^[\"'`\s]+|[\"'`\s]+$", "", clean)
        if not clean:
            return "the proposed mechanism"
        words = clean.split()
        if len(words) > max_words:
            clean = " ".join(words[:max_words]).rstrip(",;:.") + "..."
        return clean

    def _clean_generated_text(self, text):
        cleaned = str(text or "")
        cleaned = cleaned.replace("→", " maps to ")
        cleaned = cleaned.replace("↔", " corresponds to ")
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", cleaned)
        cleaned = re.sub(r"\$+", " ", cleaned)
        cleaned = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})*", " ", cleaned)
        cleaned = cleaned.replace("\\", " ")
        cleaned = re.sub(r"[_^{}]", " ", cleaned)
        return " ".join(cleaned.split())

    def _fallback_grounded_question(self, from_node, to_node, narration):
        mission = self._mission_text()
        source = self._question_snippet(from_node)
        target = self._question_snippet(to_node)
        relation_text = " ".join(str(narration or "").lower().split())
        if any(token in relation_text for token in ["contradict", "conflict", "incompatible", "versus"]):
            return f"Which observation would distinguish '{source}' from '{target}'?"
        if mission:
            mission_snippet = self._question_snippet(mission, max_words=12)
            return (
                "Which experiment or dataset would show whether "
                f"'{target}' improves our answer to '{mission_snippet}'?"
            )
        if narration:
            narration_snippet = self._question_snippet(narration, max_words=12)
            return (
                "What direct measurement would test the mechanism linking "
                f"'{source}' to '{target}' around '{narration_snippet}'?"
            )
        return f"What experiment would test whether '{source}' changes '{target}'?"

    def _mission_alignment_metrics(self, node_text, narration):
        mission = self._mission_text()
        if not mission:
            return 0, 0.0
        candidate = " ".join(filter(None, [str(node_text or ""), str(narration or "")]))
        if not candidate.strip():
            return 0, 0.0
        return self._question_context_metrics(candidate, mission)

    def _normalize_question(self, question, from_node, to_node, narration):
        question = " ".join(str(question or "").split()).strip()
        if not question:
            return ""
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        context_text = " ".join(filter(None, [
            self._mission_text(),
            str(from_node or ""),
            str(to_node or ""),
            str(narration or ""),
        ]))
        overlap, similarity = self._question_context_metrics(question, context_text)
        word_count = len(question.split())
        if (
            word_count > MAX_DREAM_QUESTION_WORDS or
            self._question_has_symbolic_overreach(question) or
            overlap < QUESTION_CONTEXT_TOKEN_OVERLAP_MIN or
            similarity < QUESTION_CONTEXT_SIM_MIN
        ):
            return self._fallback_grounded_question(from_node, to_node, narration)
        return question

    def _validate_insight_depth(self, from_node, to_node, narration, claimed_depth):
        if not claimed_depth or claimed_depth == "surface":
            return claimed_depth, narration

        try:
            raw = self._llm(INSIGHT_VALIDATION_PROMPT.format(
                from_node=from_node,
                to_node=to_node,
                narration=narration,
                claimed_depth=claimed_depth
            ), temperature=0.0)
            validation = self._json_object(raw, default={})
            depth = validation.get("depth", "surface")
            reason = str(validation.get("reason", "")).lower()
            mapping_flag = validation.get("has_explicit_mapping", False)
            if isinstance(mapping_flag, str):
                has_explicit_mapping = mapping_flag.strip().lower() in {"true", "1", "yes"}
            else:
                has_explicit_mapping = bool(mapping_flag)
            if depth not in {"none", "surface", "structural", "isomorphism"}:
                return "surface", narration
            if depth in {"structural", "isomorphism"} and not has_explicit_mapping:
                return "surface", narration
            if any(phrase in reason for phrase in (
                "historical progression",
                "intellectual transition",
                "conceptual prerequisite",
                "enabling background",
                "broad functional similarity",
                "shared vocabulary",
                "shared domain",
                "causal sequence",
                "precondition",
                "realized by a concrete mechanism",
                "realized by mechanism",
            )):
                return "surface", narration
            if depth == "none":
                return "none", narration

            mission = self._mission_text() or "No mission"
            raw_refinement = self._llm(INSIGHT_REFINEMENT_PROMPT.format(
                from_node=from_node,
                to_node=to_node,
                narration=narration,
                claimed_depth=depth,
                mission=mission,
            ), temperature=0.0)
            refined = self._json_object(raw_refinement, default={})
            refined_depth = refined.get("depth", depth)
            refined_narration = str(refined.get("narration", narration)).strip() or narration
            mapping_pairs = refined.get("mapping_pairs", [])
            if not isinstance(mapping_pairs, list):
                mapping_pairs = []
            mapping_pairs = [str(pair).strip() for pair in mapping_pairs if str(pair).strip()]
            refined_mapping_flag = refined.get("has_explicit_mapping", has_explicit_mapping)
            if isinstance(refined_mapping_flag, str):
                has_explicit_mapping = refined_mapping_flag.strip().lower() in {"true", "1", "yes"}
            else:
                has_explicit_mapping = bool(refined_mapping_flag)
            formal_anchor_flag = refined.get("has_formal_anchor", False)
            if isinstance(formal_anchor_flag, str):
                has_formal_anchor = formal_anchor_flag.strip().lower() in {"true", "1", "yes"}
            else:
                has_formal_anchor = bool(formal_anchor_flag)

            if refined_depth not in {"none", "surface", "structural", "isomorphism"}:
                refined_depth = depth
            if refined_depth in {"structural", "isomorphism"} and (
                not has_explicit_mapping or len(mapping_pairs) < 2
            ):
                return "surface", refined_narration
            if refined_depth == "isomorphism" and not has_formal_anchor:
                refined_depth = "structural"
            if refined_depth in {"structural", "isomorphism"} and "MAP:" not in refined_narration:
                map_sentence = "MAP: " + "; ".join(mapping_pairs[:3]) + "."
                refined_narration = f"{refined_narration} {map_sentence}".strip()
            return refined_depth, refined_narration
        except (json.JSONDecodeError, ValueError, TypeError):
            return "surface", narration

    # ── Node selection ────────────────────────────────────────────────────────

    def _select_start_node(self, mode, seed_id=None):
        nodes = self.brain.all_nodes()
        if not nodes:
            raise ValueError("Brain is empty")

        if mode == DreamMode.SEEDED and seed_id:
            return seed_id

        # Working memory bias: 30% chance of starting from focused item
        wm = self.brain.get_working_memory()
        if wm and random.random() < 0.3:
            wm_nid, wm_data = random.choice(wm)
            if wm_data.get('node_type') != NodeType.MISSION.value:
                print(f"   Starting from working memory: {wm_data['statement'][:60]}...")
                return wm_nid

        non_mission = [(nid, d) for nid, d in nodes
                       if d.get('node_type') != NodeType.MISSION.value]

        if mode == DreamMode.PRESSURE:
            mission_id = (self.brain.get_mission() or {}).get("id")
            candidates = [
                (nid, d) for nid, d in non_mission
                if d.get('status') in [NodeStatus.CONTRADICTED.value,
                                       NodeStatus.UNCERTAIN.value]
            ]
            if candidates:
                def score(x):
                    nid, d = x
                    s = d.get('incubation_age', 0) * 2 + self.brain.graph.degree(nid)
                    if mission_id and not self.brain.is_wandering():
                        if self.brain.graph.has_edge(nid, mission_id):
                            s += 5
                    return s
                return max(candidates, key=score)[0]

        # wandering or focused — LLM picks
        sample = random.sample(non_mission, min(8, len(non_mission)))
        node_list = "\n".join(f"{nid}: {d['statement']}"
                              for nid, d in sample)
        mission = self._mission_text()
        if mission and not self.brain.is_wandering():
            prompt = START_NODE_FOCUSED.format(mission=mission, nodes=node_list)
        else:
            prompt = START_NODE_WANDERING.format(nodes=node_list)

        chosen = self._llm(prompt, temperature=0.2).strip()
        uuid_tokens = re.findall(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            chosen.lower()
        )
        if uuid_tokens and uuid_tokens[0] in dict(nodes):
            return uuid_tokens[0]
        if chosen in dict(nodes):
            return chosen

        return random.choice(non_mission)[0] if non_mission else nodes[0][0]

    # ── Edge scoring ──────────────────────────────────────────────────────────

    def _score_edge(self, edge_data, temperature, scientificness, visited, target_id,
                    recent_clusters=None):
        weight = edge_data.get('weight', 0.5)
        etype  = edge_data.get('type', '')

        if etype in ANALOGY_WEIGHTS:
            try:
                weight = max(weight, ANALOGY_WEIGHTS.get(EdgeType(etype), 0.4))
            except ValueError:
                pass

        logical = {EdgeType.SUPPORTS.value, EdgeType.CAUSES.value,
                   EdgeType.CONTRADICTS.value}
        if etype in logical:
            weight += scientificness * 0.3
        if etype == EdgeType.ASSOCIATED.value:
            weight += (1 - scientificness) * 0.3

        # mission edges only matter in focused/transitional
        if etype == EdgeType.TOWARD_MISSION.value and not self.brain.is_wandering():
            weight += 0.3

        if target_id in visited:
            weight = (weight * 0.1) - VISITED_PENALTY

        if recent_clusters:
            target_node = self.brain.get_node(target_id) or {}
            target_cluster = target_node.get('cluster', '')
            if target_cluster and target_cluster in recent_clusters[-RECENT_CLUSTER_WINDOW:]:
                weight -= RECENT_CLUSTER_PENALTY

        noise = random.gauss(0, temperature * 0.3)
        return max(0.001, weight + noise)

    # ── Single hop ────────────────────────────────────────────────────────────

    def _hop(self, current_id, temperature, scientificness, visited,
             blocked_ids=None, recent_clusters=None):
        blocked_ids = set(blocked_ids or ())
        neighbors = self.brain.neighbors(current_id)
        if blocked_ids:
            neighbors = [nid for nid in neighbors if nid not in blocked_ids]
        if not neighbors:
            all_ids   = [nid for nid, _ in self.brain.all_nodes()]
            candidates = [n for n in all_ids if n not in blocked_ids]
            unvisited = [n for n in candidates if n not in visited]
            pool = unvisited or candidates or all_ids
            return random.choice(pool), None

        scored = []
        for nid in neighbors:
            edge = self.brain.get_edge(current_id, nid)
            if edge:
                score = self._score_edge(
                    edge, temperature, scientificness, visited, nid,
                    recent_clusters=recent_clusters)
                scored.append((nid, edge, score))

        if not scored:
            return random.choice(neighbors), None

        unvisited_scored = [item for item in scored if item[0] not in visited]
        if unvisited_scored:
            scored = unvisited_scored
        else:
            all_ids = [nid for nid, _ in self.brain.all_nodes()]
            candidates = [n for n in all_ids if n not in blocked_ids and n not in visited]
            if candidates and random.random() < EXPLORATION_JUMP_PROB:
                return random.choice(candidates), None

        total = sum(s for _, _, s in scored)
        roll  = random.uniform(0, total)
        cumulative = 0
        for nid, edge, score in scored:
            cumulative += score
            if cumulative >= roll:
                return nid, edge
        return scored[-1][0], scored[-1][1]

    # ── Answer detection ──────────────────────────────────────────────────────

    def _check_answers(self, node_id, node_data):
        if not self.research_agenda or self.brain.is_wandering():
            return "none", "", ""
        open_items = self.research_agenda.get_prioritized_questions(15)
        node_emb = self._embed(node_data['statement'])
        for item in open_items:
            item_emb = self._embed(item.text)
            if self._cosine(node_emb, item_emb) < THRESHOLDS.AGENDA_PREFILTER:
                continue
            raw = self._llm(ANSWER_CHECK_PROMPT.format(
                current_node=node_data['statement'], question=item.text
            ), temperature=0.1)
            try:
                result = self._json_object(raw, default={})
                match  = result.get('match', 'none')
                expl   = result.get('explanation', '')
                if match in ['partial', 'strong']:
                    return match, expl, item.text
            except (json.JSONDecodeError, ValueError):
                continue
        return "none", "", ""

    # ── Mission advance ───────────────────────────────────────────────────────

    def _check_mission_advance(self, node_data, narration):
        if self.brain.is_wandering():
            return False, "", 0.0
        mission = self.brain.get_mission()
        if not mission:
            return False, "", 0.0
        overlap, similarity = self._mission_alignment_metrics(
            node_data.get('statement', ''),
            narration,
        )
        if overlap < MISSION_ADVANCE_TOKEN_OVERLAP_MIN or similarity < MISSION_ADVANCE_SIM_MIN:
            return False, "", 0.0
        raw = self._llm(MISSION_ADVANCE_PROMPT.format(
            mission=mission['question'],
            node=node_data['statement'],
            narration=narration), temperature=0.1)
        try:
            result = self._json_object(raw, default={})
            strength = max(0.0, min(1.0, float(result.get('strength', 0.0))))
            is_adv = bool(result.get('advances', False)) and strength >= 0.7
            return is_adv, result.get('explanation', ''), strength
        except (json.JSONDecodeError, ValueError):
            pass
        return False, "", 0.0

    # ── Parse narration ───────────────────────────────────────────────────────

    def _parse_narration(self, raw):
        question = ""
        is_insight = False
        insight_depth = ""
        clean = []
        for line in raw.strip().split('\n'):
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.upper().startswith("INSIGHT:"):
                rest = line.split(":", 1)[1].strip().lower()
                if rest != "none":
                    is_insight = True
                    insight_depth = rest.split()[0] if rest.split() else ""
            else:
                clean.append(line)
        return " ".join(clean).strip(), question, is_insight, insight_depth

    # ── Depth exploration ─────────────────────────────────────────────────────

    def _depth_explore(self, node_id, node_data, question, explanation,
                       temperature, scientificness, visited, log,
                       questions, q_embeddings, step_offset):
        mission = self._mission_text()
        mission_line = f"Central question: \"{mission}\"" if mission else ""

        raw = self._llm(DEPTH_NARRATION_PROMPT.format(
            mission_line=mission_line,
            node=node_data['statement'],
            question=question,
            explanation=explanation), temperature=0.5)
        if raw.strip().upper().startswith("NO_DEPTH"):
            return node_id, 0
        lines  = raw.strip().split('\n')
        q_line = next((l for l in lines if l.startswith('Q:')), "")
        followup = self._normalize_question(
            q_line[2:].strip() if q_line else "",
            node_data.get('statement', ''),
            explanation,
            explanation,
        )
        self._add_question(followup, questions, q_embeddings)

        current_id, current_data = node_id, node_data
        local_visited = set(visited)
        local_visited.add(node_id)
        seen_statements = {self._statement_key(node_data.get('statement', ''))}
        explored_steps = 0
        printed_header = False
        for d in range(DEPTH_STEPS):
            next_id = None
            next_data = None
            edge = None
            for _ in range(4):
                candidate_id, candidate_edge = self._hop(
                    current_id, temperature * 0.5, scientificness,
                    local_visited, blocked_ids={current_id})
                candidate_data = self.brain.get_node(candidate_id)
                if not candidate_data:
                    local_visited.add(candidate_id)
                    continue
                statement_key = self._statement_key(
                    candidate_data.get('statement', '')
                )
                if statement_key and statement_key in seen_statements:
                    local_visited.add(candidate_id)
                    continue
                next_id, edge, next_data = candidate_id, candidate_edge, candidate_data
                if statement_key:
                    seen_statements.add(statement_key)
                break
            if not next_data:
                break
            if not printed_header:
                print(f"      ↳ Depth [{DEPTH_STEPS} steps]")
                printed_header = True
            edge_type      = edge.get('type', 'associated') if edge else 'associated'
            edge_narration = edge.get('narration', '') if edge else ''
            raw = self._llm(self._narration_prompt(
                current_data['statement'], edge_type,
                edge_narration, next_data['statement']), temperature=0.5)
            narration, _, is_insight, depth = self._parse_narration(raw)
            
            mission_advance = False
            mission_strength = 0.0
            if mission:
                mission_advance, _, mission_strength = self._check_mission_advance(next_data, narration)

            ds = DreamStep(
                step=step_offset+d, from_id=current_id, to_id=next_id,
                edge_type=edge_type, edge_narration=edge_narration,
                narration=narration, is_insight=is_insight,
                insight_depth=depth, mission_advance=mission_advance, 
                mission_strength=mission_strength)
            log.steps.append(ds)
            local_visited.add(next_id)
            visited.add(next_id)
            self.brain.update_node(next_id, activated_at=time.time())
            current_id, current_data = next_id, next_data
            print(f"      depth {d+1}: {next_data['statement']}")
            explored_steps += 1
        return current_id, explored_steps

    # ── NREM ─────────────────────────────────────────────────────────────────

    def nrem_pass(self):
        print("\n── NREM pass ──")
        
        # Hippocampal Replay
        if hasattr(self.brain, 'episodic'):
            sequence = self.brain.episodic.get_sequence(sequence_length=3)
            if sequence:
                print(f"  [Hippocampal Replay] Replaying {len(sequence)} events.")
                for event in sequence:
                    print(f"    - {event.event_type}: {event.description[:60]}")
                    for nid in getattr(event, 'nodes_involved', []):
                        if self.brain.get_node(nid):
                            self.brain.update_node(nid, activated_at=time.time())
                            
        self.brain.proximal_reinforce()
        print("── NREM complete ──\n")

    # ── Main dream loop ───────────────────────────────────────────────────────

    def dream(self, mode=DreamMode.WANDERING, steps=DEFAULT_STEPS,
              temperature=DEFAULT_TEMP, seed_id=None,
              run_nrem=True, log_path="logs/dream_latest.json",
              visited_set=None):

        brain_mode    = self.brain.get_mode()
        scientificness= self.brain.scientificness

        # mode modifiers
        temperature += MODE_TEMP_BOOST.get(brain_mode, 0)
        steps       += MODE_STEPS_BOOST.get(brain_mode, 0)

        log          = DreamLog(mode=mode.value, brain_mode=brain_mode)
        visited      = visited_set if visited_set is not None else set()
        depth_roots  = set()
        last_depth_step = -DEPTH_COOLDOWN_STEPS
        questions    = []
        q_embeddings = []
        mission      = self._mission_text()

        if run_nrem:
            self.nrem_pass()

        print(f"\n── REM [{mode.value}] [{brain_mode}] steps={steps} temp={temperature:.2f} ──")
        if mission:
            print(f"   Mission: {mission}")
        else:
            print(f"   Mode: WANDERING — free association")
        print()

        current_id = self._select_start_node(mode, seed_id)
        current    = self.brain.get_node(current_id)
        visited.add(current_id)
        print(f"   Start: {current['statement']}\n")

        step = 0
        recent_clusters = []
        while step < steps:
            source_id = current_id
            source_node = current
            next_id, edge = self._hop(current_id, temperature,
                                      scientificness, visited,
                                      recent_clusters=recent_clusters)
            next_node = self.brain.get_node(next_id)
            if not next_node:
                step += 1
                continue

            if next_node.get('node_type') == NodeType.MISSION.value:
                visited.add(next_id)
                step += 1
                continue

            edge_type      = edge.get('type', 'associated') if edge else 'associated'
            edge_narration = edge.get('narration', '') if edge else ''

            raw = self._llm(self._narration_prompt(
                source_node['statement'], edge_type,
                edge_narration, next_node['statement']), temperature=min(0.45, temperature))
            narration, question, is_insight, insight_depth = \
                self._parse_narration(raw)
            question = self._normalize_question(
                question,
                source_node.get('statement', ''),
                next_node.get('statement', ''),
                narration,
            )
            raw_insight_depth = insight_depth
            if is_insight and insight_depth in ["structural", "isomorphism"]:
                insight_depth, narration = self._validate_insight_depth(
                    source_node['statement'],
                    next_node['statement'],
                    narration,
                    insight_depth
                )
                is_insight = insight_depth != "none"

            question_added = self._add_question(question, questions, q_embeddings)

            match_grade, match_explanation, matched_q = \
                self._check_answers(next_id, next_node)

            mission_advance = False
            mission_explanation = ""
            mission_strength = 0.0
            
            if mission:
                mission_advance, mission_explanation, mission_strength = \
                    self._check_mission_advance(next_node, narration)

            # depth
            depth_triggered = False
            deep_insight = (
                is_insight and insight_depth in ['structural', 'isomorphism']
            )
            non_answer_depth_ready = (step - last_depth_step) >= DEPTH_COOLDOWN_STEPS
            depth_question = ""
            depth_explanation = ""
            if match_grade in ['partial', 'strong'] and matched_q:
                depth_question = matched_q
                depth_explanation = match_explanation
            elif (non_answer_depth_ready and
                  ((deep_insight and next_id not in depth_roots and question_added and
                    self._depth_seed_is_novel(next_node, narration, question)) or
                   next_node.get('status') == NodeStatus.CONTRADICTED.value or
                   next_node.get('incubation_age', 0) > 3 or
                   (mission_advance and mission_strength >= 0.7))):
                depth_question = question or "What mechanism here is most directly testable?"
                depth_explanation = narration if deep_insight else (
                    mission_explanation or "Contradiction, incubation, or strong mission relevance detected."
                )

            if depth_question:
                depth_roots.add(next_id)
                current_id, explored_steps = self._depth_explore(
                    next_id, next_node,
                    depth_question,
                    depth_explanation,
                    temperature, scientificness, visited, log,
                    questions, q_embeddings, 1000 + step * DEPTH_STEPS)
                depth_triggered = explored_steps > 0
                if depth_triggered:
                    last_depth_step = step
                    current = self.brain.get_node(current_id)
                    visited.add(current_id)
                else:
                    depth_roots.discard(next_id)

            # new edge on insight
            new_edge = False
            if is_insight and insight_depth:
                type_map = {
                    "surface":     EdgeType.SURFACE_ANALOGY,
                    "structural":  EdgeType.STRUCTURAL_ANALOGY,
                    "isomorphism": EdgeType.DEEP_ISOMORPHISM,
                }
                etype = type_map.get(insight_depth, EdgeType.STRUCTURAL_ANALOGY)
                
                # ── System 2 gating for high-stakes analogies ──
                requires_review = self.critic is not None and insight_depth in ["structural", "isomorphism"]
                
                if not requires_review:
                    if source_id != next_id and not (
                        self.brain.graph.has_edge(source_id, next_id) or
                        self.brain.graph.has_edge(next_id, source_id)
                    ):
                        dream_edge = Edge(
                            type=etype, narration=narration,
                            weight=ANALOGY_WEIGHTS.get(etype, 0.4),
                            confidence=0.45, source=EdgeSource.DREAM,
                            analogy_depth=insight_depth)
                        self.brain.add_edge(source_id, next_id, dream_edge)
                        new_edge = True
                    self.brain.restructure_around_insight(
                        source_id, next_id, narration, edge_type=etype.value)
                else:
                    print(f"            [System 2] High-depth insight buffered for morning review")
                    from critic.critic import CandidateThought
                    candidate = CandidateThought(
                        claim         = narration,
                        source_module = "dreamer",
                        proposed_type = "analogy",
                        importance    = float(ANALOGY_WEIGHTS.get(etype, 0.4)),
                        edge_type     = etype.value,
                        node_a_id     = source_id,
                        node_b_id     = next_id,
                        context       = f"Found during dream step from '{source_node['statement']}' to '{next_node['statement']}'"
                    )
                    self.critic.route_deferred(candidate)

                log.insights.append({
                    "step": step, "from": source_node['statement'],
                    "to": next_node['statement'],
                    "from_node_id": source_id,
                    "to_node_id": next_id,
                    "narration": narration, "depth": insight_depth,
                    "raw_depth": raw_insight_depth,
                    "mission_linked": mission_advance,
                    "requires_review": requires_review
                })

            if match_grade != 'none':
                log.answers.append({
                    "step": step, "node": next_id,
                    "question": matched_q, "grade": match_grade,
                    "explanation": match_explanation
                })

            if mission_advance:
                self.brain.link_to_mission(
                    next_id, f"Dream insight: {mission_explanation}",
                    strength=mission_strength)
                log.mission_advances.append({
                    "step": step, "node": next_id,
                    "explanation": mission_explanation,
                    "strength": mission_strength
                })

            ds = DreamStep(
                step=step, from_id=source_id, to_id=next_id,
                edge_type=edge_type, edge_narration=edge_narration,
                narration=narration, question=question,
                is_insight=is_insight, insight_depth=insight_depth,
                new_edge=new_edge, answer_match=match_grade,
                answer_detail=match_explanation,
                depth_triggered=depth_triggered,
                mission_advance=mission_advance,
                mission_strength=mission_strength)
            log.steps.append(ds)
            self.brain.update_node(next_id, activated_at=time.time())
            visited.add(next_id)
            target_cluster = next_node.get('cluster', '')
            if target_cluster:
                recent_clusters.append(target_cluster)
                if len(recent_clusters) > RECENT_CLUSTER_WINDOW:
                    recent_clusters = recent_clusters[-RECENT_CLUSTER_WINDOW:]

            ind = ""
            if is_insight:
                dep_sym = {"surface":"S","structural":"ST","isomorphism":"⊗"}.get(insight_depth,"?")
                ind = f"✦[{dep_sym}]"
            print(f"   Step {step+1:02d} [{edge_type}]: "
                  f"{next_node['statement']} {ind}")
            if question:
                print(f"            Q: {question}")
            if match_grade != 'none':
                print(f"            ◎ [{match_grade}]: {match_explanation}")
            if mission_advance:
                print(f"            ★ ({mission_strength:.2f}): {mission_explanation}")
            if depth_triggered:
                print(f"            ↳ Depth")

            if not depth_triggered:
                current_id = next_id
                current    = next_node
            step += 1

        log.questions = questions

        step_text    = "\n".join(f"- {s.narration}" for s in log.steps if s.narration)
        answer_text  = "\n".join(f"- [{a['grade']}] {a['explanation']}" for a in log.answers) or "none"
        mission_text = "\n".join(f"- ({m['strength']:.2f}) {m['explanation']}" for m in log.mission_advances) or "none"

        log.summary = self._llm(
            self._summary_prompt(step_text, answer_text, mission_text),
            temperature=0.7
        )

        # ── System 2 Post-Dream Review ──
        if self.critic:
            pending = [ins for ins in log.insights if ins.get("requires_review")]
            if pending:
                print("\n── System 2 Morning Review (Dream Insights) ──")
                from critic.critic import CandidateThought, Verdict
                for ins in pending:
                    depth = ins["depth"]
                    candidate = CandidateThought(
                        claim=ins["narration"],
                        source_module="dreamer",
                        proposed_type="structural_analogy" if depth == "structural" else "deep_isomorphism",
                        importance=0.75 if depth == "structural" else 0.85,
                        edge_type=depth,
                        node_a_id=ins["from_node_id"],
                        node_b_id=ins["to_node_id"]
                    )
                    critic_log = self.critic.evaluate_with_refinement(candidate)
                    final_claim = critic_log.final_claim or candidate.claim
                    ins["critic_verdict"] = critic_log.verdict.value
                    ins["critic_confidence"] = critic_log.confidence
                    ins["critic_final_claim"] = final_claim
                    if critic_log.verdict == Verdict.ACCEPT:
                        etype = EdgeType.DEEP_ISOMORPHISM if depth == "isomorphism" else EdgeType.STRUCTURAL_ANALOGY
                        if not (self.brain.graph.has_edge(candidate.node_a_id, candidate.node_b_id) or
                                self.brain.graph.has_edge(candidate.node_b_id, candidate.node_a_id)):
                            dream_edge = Edge(
                                type=etype, narration=final_claim,
                                weight=ANALOGY_WEIGHTS.get(etype, 0.4),
                                confidence=critic_log.confidence, source=EdgeSource.DREAM,
                                analogy_depth=depth)
                            self.brain.add_edge(candidate.node_a_id, candidate.node_b_id, dream_edge)
                        self.brain.restructure_around_insight(
                            candidate.node_a_id, candidate.node_b_id, 
                            final_claim, edge_type=etype.value
                        )
                    elif critic_log.verdict == Verdict.DEFER:
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
                        )
                        self.critic.route_deferred(deferred_candidate)

        import os
        os.makedirs("logs", exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(log.to_dict(), f, indent=2)

        print(f"\n── Dream complete [{brain_mode}] ──")
        print(f"   Steps:{len(log.steps)} Qs:{len(log.questions)} "
              f"Insights:{len(log.insights)} Answers:{len(log.answers)} "
              f"Mission advances:{len(log.mission_advances)}")
        print(f"\n── Summary ──\n{log.summary}\n")

        # after transitional cycle, move to focused
        if self.brain.is_transitional():
            self.brain.complete_transition()
            print("── Transitional cycle complete — now FOCUSED ──")

        return log

    # ── Hypothesis Generation Engine ─────────────────────────────────────────
    #
    # Unlike dream() which narrates random walks over existing edges,
    # hypothesize() is GENERATIVE — it proposes what doesn't exist yet.
    # It uses the LLM's prior knowledge + recently ingested claims to
    # produce novel, testable hypotheses ("shower thoughts").
    #
    # The full Critic is bypassed. Instead, a lightweight single-shot
    # plausibility gate filters out physically impossible proposals.
    # The Critic only engages later, when evidence arrives and the
    # hypothesis is being promoted from SPECULATIVE to WORKING.
    # ─────────────────────────────────────────────────────────────────────────

    _HYPOTHESIS_GENERATION_PROMPT = """You are a scientist who just read several new research findings. \
Using your deep expert knowledge AND these specific findings, propose ONE novel, testable hypothesis.

Recent findings:
{seed_claims}

{mission_block}

Rules:
- The hypothesis MUST propose a specific mechanism, not just a correlation.
- It MUST be testable: state what evidence would confirm or refute it.
- Stay within the entities, interactions, scales, and constraints present in the supplied findings unless the supplied findings explicitly justify a bridge.
- In mission-driven mode, prefer the most conservative mechanistic extrapolation that still makes a new testable prediction.
- Do NOT invent a new force, coupling, field, geometric effect, extra sector, or measurement target unless it is already motivated by the supplied findings.
- Do NOT use LaTeX or symbolic notation. Spell quantities out in plain scientific prose.
- State the hypothesis in 2-3 precise sentences.
- Do NOT restate the seed findings. Propose something NEW that goes beyond them.
- Use formal scientific prose. No first person.

Respond with ONLY a JSON object:
{{
  "hypothesis": "the hypothesis statement (2-3 sentences)",
  "mechanism": "the proposed mechanism in 1-2 sentences",
  "testable_by": "what search query or experiment would test this (1 sentence)",
  "confidence": 0.3
}}"""

    _PLAUSIBILITY_CHECK_PROMPT = """Is this hypothesis scientifically plausible? \
A plausible hypothesis does not violate known physical laws or established scientific consensus. \
It may be unproven or speculative, but it must not be provably impossible.

Hypothesis: "{hypothesis}"

Respond with ONLY a JSON object:
{{"plausible": true or false, "reason": "one sentence"}}"""

    _HYPOTHESIS_GROUNDING_PROMPT = """You are checking whether a proposed hypothesis stays grounded in supplied evidence.

Seed findings:
{seed_claims}

Mission:
{mission_block}

Candidate hypothesis:
{hypothesis}

Candidate mechanism:
{mechanism}

Testable by:
{testable_by}

Grounded means:
- It extrapolates one mechanistic step from the supplied entities, constraints, or observables.
- It does not invent a new interaction, sector, geometry, or observable that the seed findings do not motivate.
- Its proposed test follows from the stated mechanism.

Respond ONLY with JSON:
{{
  "grounded": true or false,
  "reason": "one sentence"
}}"""

    def __init_hypothesis_tracking(self):
        """Lazily initialize hypothesis tracking state."""
        if not hasattr(self, '_hypothesis_outcomes'):
            self._hypothesis_outcomes: dict[str, list[str]] = {}

    def record_hypothesis_outcome(self, cluster: str, outcome: str):
        """
        Record the outcome of a hypothesis for feedback learning.

        Called by the Conductor when a hypothesis is resolved.
        Tracks per-cluster success rates to modulate future creativity.

        Args:
            cluster: The cluster label of the hypothesis seed nodes.
            outcome: One of "confirmed", "contradicted", "lacks_evidence".
        """
        self.__init_hypothesis_tracking()
        self._hypothesis_outcomes.setdefault(cluster, []).append(outcome)

    def _cluster_success_rate(self, cluster: str) -> float:
        """
        Returns the fraction of hypotheses from this cluster that were confirmed.

        Returns 0.5 (neutral prior) when no outcomes have been recorded yet.
        """
        self.__init_hypothesis_tracking()
        outcomes = self._hypothesis_outcomes.get(cluster, [])
        if not outcomes:
            return 0.5  # neutral prior
        confirmed = sum(1 for o in outcomes if o == "confirmed")
        return confirmed / len(outcomes)

    def _select_hypothesis_seeds(
        self,
        seed_node_ids: list[str],
        max_seeds: int = 3,
        allow_cross_cluster: bool = False,
    ) -> list[tuple[str, dict]]:
        """
        Select the most generative seed nodes for hypothesis generation.

        Ranks by importance * (1 + surprise), then enforces cluster diversity:
        if all top seeds are from the same cluster, force-includes one node
        from a different cluster to enable cross-domain hypotheses.

        Returns list of (node_id, node_data) tuples.
        """
        # Gather node data and sort by importance
        candidates = []
        for nid in seed_node_ids:
            data = self.brain.get_node(nid)
            if not data:
                continue
            # Skip non-claim nodes (e.g., SOURCE nodes)
            if data.get('node_type') in (NodeType.SOURCE.value, NodeType.MISSION.value):
                continue
            importance = float(data.get('importance', 0.5))
            candidates.append((importance, nid, data))

        if not candidates:
            return []

        # Sort by importance descending
        candidates.sort(reverse=True)
        selected = [(nid, data) for _, nid, data in candidates[:max_seeds]]

        # ── Cluster diversity enforcement ──
        # If all selected seeds are from the same cluster, force-include
        # one high-importance node from a different cluster.
        if allow_cross_cluster and len(selected) >= 2:
            clusters = {data.get('cluster', 'unclustered') for _, data in selected}
            if len(clusters) == 1:
                selected_cluster = clusters.pop()
                # Find a high-importance node from a different cluster
                all_nodes = self.brain.all_nodes()
                cross_candidates = [
                    (float(d.get('importance', 0.5)), nid, d)
                    for nid, d in all_nodes
                    if d.get('cluster', 'unclustered') != selected_cluster
                    and d.get('cluster', 'unclustered') != 'unclustered'
                    and d.get('node_type') not in (
                        NodeType.SOURCE.value, NodeType.MISSION.value,
                    )
                    and nid not in seed_node_ids
                ]
                if cross_candidates:
                    cross_candidates.sort(reverse=True)
                    _, cross_nid, cross_data = cross_candidates[0]
                    # Replace the least important selected seed
                    selected[-1] = (cross_nid, cross_data)
                    print(f"    ⇌ Cross-cluster seed injected from "
                          f"[{cross_data.get('cluster', '?')}]")

        return selected

    def hypothesize(
        self,
        seed_node_ids: list[str],
        mode: str = "free",
        max_hypotheses: int = 3,
    ) -> list[dict]:
        """
        Generate wild hypotheses seeded by specific claim nodes.

        This is the 'shower thought' engine. It takes recently ingested
        claims and prompts the LLM to use its prior knowledge to propose
        novel, testable mechanisms.

        Unlike dream(), this method is GENERATIVE — it proposes what
        doesn't exist in the graph yet. Wild hypotheses bypass the full
        Critic and go through a lightweight plausibility gate instead.

        Args:
            seed_node_ids: Node IDs of recently ingested claims to seed from.
            mode: "free" (unconstrained) or "mission_driven" (mission-oriented).
            max_hypotheses: Maximum number of hypotheses to generate.

        Returns:
            List of dicts: {statement, mechanism, testable_by, seed_ids, node_id}
        """
        self.__init_hypothesis_tracking()

        # ── Throttle check ──
        if not self.brain.can_spawn_hypothesis():
            active = self.brain.count_active_hypotheses()
            print(f"  ⊘ Hypothesis throttle: {active}/{self.brain.MAX_ACTIVE_HYPOTHESES}"
                  f" active — skipping generation")
            return []

        if not seed_node_ids:
            print("  ⊘ No seed nodes for hypothesis generation")
            return []

        # ── Seed selection ──
        seeds = self._select_hypothesis_seeds(
            seed_node_ids,
            max_seeds=max_hypotheses,
            allow_cross_cluster=(mode != "mission_driven"),
        )
        if not seeds:
            print("  ⊘ No valid seed nodes after filtering")
            return []

        print(f"\n── Dreamer: hypothesize [{mode}] — {len(seeds)} seeds ──")

        # ── Mission block ──
        mission_block = ""
        if mode == "mission_driven":
            mission = self._mission_text()
            if mission:
                mission_block = (
                    f"Your central research question: \"{mission}\"\n"
                    f"The hypothesis should help answer this question."
                )

        results = []
        remaining_budget = self.brain.MAX_ACTIVE_HYPOTHESES - self.brain.count_active_hypotheses()

        for seed_nid, seed_data in seeds:
            if len(results) >= max_hypotheses or remaining_budget <= 0:
                break

            seed_cluster = seed_data.get('cluster', 'unclustered')
            seed_statement = seed_data.get('statement', '')

            # Gather context from seed + its neighbors
            seed_claims_lines = [f"- {seed_statement}"]
            for neighbor_id in list(self.brain.graph.predecessors(seed_nid))[:3]:
                neighbor = self.brain.get_node(neighbor_id)
                if neighbor and neighbor.get('node_type') != NodeType.SOURCE.value:
                    seed_claims_lines.append(f"- {neighbor.get('statement', '')}")
            for neighbor_id in list(self.brain.graph.successors(seed_nid))[:3]:
                neighbor = self.brain.get_node(neighbor_id)
                if neighbor and neighbor.get('node_type') != NodeType.SOURCE.value:
                    seed_claims_lines.append(f"- {neighbor.get('statement', '')}")

            seed_claims_text = "\n".join(seed_claims_lines[:6])

            # ── Modulate temperature by cluster success rate ──
            success_rate = self._cluster_success_rate(seed_cluster)
            temperature = 0.35 + (0.10 * success_rate)
            if mode == "mission_driven":
                temperature = max(0.22, temperature - 0.08)

            # ── Generate hypothesis ──
            raw = self._llm(
                self._HYPOTHESIS_GENERATION_PROMPT.format(
                    seed_claims=seed_claims_text,
                    mission_block=mission_block,
                ),
                temperature=temperature,
            )
            hyp_data = self._json_object(raw, default={})

            hypothesis_text = self._clean_generated_text(hyp_data.get('hypothesis', ''))
            mechanism = self._clean_generated_text(hyp_data.get('mechanism', ''))
            testable_by = self._clean_generated_text(hyp_data.get('testable_by', ''))

            if not hypothesis_text or len(hypothesis_text) < 20:
                print(f"    ✗ Empty/short hypothesis for seed [{seed_nid[:8]}]")
                continue

            print(f"    💭 Candidate: {hypothesis_text[:80]}...")

            candidate_payload = " ".join(filter(None, [hypothesis_text, mechanism, testable_by]))
            overlap, similarity = self._question_context_metrics(
                candidate_payload,
                " ".join(filter(None, [seed_claims_text, mission_block])),
            )
            if (
                overlap < HYPOTHESIS_CONTEXT_TOKEN_OVERLAP_MIN or
                similarity < HYPOTHESIS_CONTEXT_SIM_MIN
            ):
                print("    ✗ Rejected: candidate drifted away from supplied findings")
                continue

            # ── Plausibility gate (lightweight — 1 LLM call, NOT full Critic) ──
            plaus_raw = self._llm(
                self._PLAUSIBILITY_CHECK_PROMPT.format(hypothesis=hypothesis_text),
                temperature=0.1,
            )
            plaus = self._json_object(plaus_raw, default={"plausible": True})
            is_plausible = plaus.get("plausible", True)
            if isinstance(is_plausible, str):
                is_plausible = is_plausible.strip().lower() in {"true", "1", "yes"}

            if not is_plausible:
                reason = plaus.get("reason", "unknown")
                print(f"    ✗ Implausible: {reason}")
                continue

            grounding_raw = self._llm(
                self._HYPOTHESIS_GROUNDING_PROMPT.format(
                    seed_claims=seed_claims_text,
                    mission_block=mission_block or "none",
                    hypothesis=hypothesis_text,
                    mechanism=mechanism or "none",
                    testable_by=testable_by or "none",
                ),
                temperature=0.1,
            )
            grounding = self._json_object(grounding_raw, default={"grounded": True})
            grounded = grounding.get("grounded", True)
            if isinstance(grounded, str):
                grounded = grounded.strip().lower() in {"true", "1", "yes"}
            if not grounded:
                reason = grounding.get("reason", "candidate introduced unsupported machinery")
                print(f"    ✗ Ungrounded: {reason}")
                continue

            accepted, reviewed_claim, review_reason = self._critic_review_hypothesis(
                hypothesis_text,
                mechanism,
                seed_claims_text,
                expected_status=ArtifactStatus.OPEN.value,
            )
            if not accepted:
                print(f"    ✗ Deferred by critic: {review_reason or 'insufficient support'}")
                continue
            hypothesis_text = reviewed_claim or hypothesis_text

            # ── Create hypothesis node ──
            hyp_node = Node(
                statement=hypothesis_text,
                node_type=NodeType.HYPOTHESIS,
                cluster=seed_cluster,
                status=NodeStatus.HYPOTHETICAL,
                epistemic_status=ArtifactStatus.SPECULATIVE.value,
                importance=float(seed_data.get('importance', 0.5)),
                source_quality=0.3,  # dream-sourced
                predicted_answer=mechanism,
                testable_by=testable_by,
                created_by="dreamer_hypothesis",
            )
            hyp_nid = self.brain.add_node(hyp_node)

            # ── Create DERIVED_FROM edges to seed nodes ──
            derived_edge = Edge(
                type=EdgeType.DERIVED_FROM,
                narration=f"Hypothesis derived from claim: {seed_statement[:100]}",
                weight=0.6,
                confidence=0.5,
                source=EdgeSource.DREAM,
                decay_exempt=True,
            )
            self.brain.add_edge(hyp_nid, seed_nid, derived_edge)

            result = {
                "statement": hypothesis_text,
                "mechanism": mechanism,
                "testable_by": testable_by,
                "seed_ids": [seed_nid],
                "node_id": hyp_nid,
                "cluster": seed_cluster,
                "temperature": temperature,
            }
            results.append(result)
            remaining_budget -= 1
            print(f"    ✓ Hypothesis created [{hyp_nid[:8]}] "
                  f"(cluster={seed_cluster}, temp={temperature:.2f})")

        print(f"── Dreamer: {len(results)} hypotheses generated ──\n")
        return results
