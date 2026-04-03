"""
Critic — System 2 (Slow Thinking) for THE SCIENTIST.

Implements Daniel Kahneman's dual-process theory:
  System 1 (Thinker/Dreamer) generates ideas quickly and intuitively.
  System 2 (Critic) monitors, challenges, and gates those ideas before
  they enter the knowledge graph.

The Critic does NOT generate ideas. It receives candidate thoughts from
System 1 and runs an adversarial multi-turn dialogue to determine whether
they survive scrutiny.

Four possible verdicts:
  ACCEPT  — Passes quality bar. Added to graph with calibrated confidence.
  REFINE  — Kernel of truth but weak formulation. System 1 gets another pass.
  REJECT  — Incoherent, hallucinated, or trivially obvious. Discarded.
  DEFER   — Not enough evidence to decide. Sent to InsightBuffer for incubation.

The "laziness principle": System 2 only activates for high-stakes claims.
Routine concept extraction and weak associative edges bypass the Critic entirely.

Usage:
    critic = Critic(brain, embedding_index=emb_index, insight_buffer=buf)
    verdict_log = critic.evaluate(candidate)
    if verdict_log.verdict == Verdict.ACCEPT:
        # add to graph with verdict_log.confidence
"""

import time
import json
from enum import Enum
from dataclasses import dataclass, field
from graph.brain import Brain, NodeType
from config import CRITIC as CRITIC_CFG
from llm_utils import llm_call, llm_chat, require_json
from embedding import embed as shared_embed


# ── Verdict enum ──────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"
    DEFER  = "defer"


# ── Candidate thought ────────────────────────────────────────────────────────

@dataclass
class CandidateThought:
    """A thought produced by System 1, awaiting System 2 evaluation."""
    claim:           str               # The insight/hypothesis/analogy text
    source_module:   str = "thinker"   # Which module produced it: thinker, dreamer, ingestor
    proposed_type:   str = "synthesis"  # Proposed node type: synthesis, hypothesis, etc.
    importance:      float = 0.7       # System 1's estimated importance
    context:         str = ""          # Supporting evidence / reasoning that led to the claim
    edge_type:       str = ""          # If this is an edge claim (e.g., structural_analogy)
    node_a_id:       str = ""          # Source node (for edge candidates)
    node_b_id:       str = ""          # Target node (for edge candidates)
    crosses_domains: bool = False      # Whether the claim connects different clusters
    contradicts_existing: bool = False # Whether the claim conflicts with existing knowledge


# ── Dialogue turn ─────────────────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    """One turn of the System 1 ↔ System 2 adversarial dialogue."""
    role:    str   # "system2_challenge" or "system1_defense"
    content: str
    turn:    int


# ── Critic log ────────────────────────────────────────────────────────────────

@dataclass
class CriticLog:
    """Full record of a System 2 evaluation session."""
    candidate_claim:   str = ""
    source_module:     str = ""
    proposed_type:     str = ""
    verdict:           Verdict = Verdict.DEFER
    confidence:        float = 0.0
    is_novel:          bool = True
    dialogue:          list = field(default_factory=list)
    rejection_reason:  str = ""
    refinement_note:   str = ""
    started_at:        float = field(default_factory=time.time)
    duration:          float = 0.0
    bypassed:          bool = False   # True if laziness principle skipped review

    def to_dict(self):
        d = {
            "candidate_claim":  self.candidate_claim,
            "source_module":    self.source_module,
            "proposed_type":    self.proposed_type,
            "verdict":          self.verdict.value,
            "confidence":       self.confidence,
            "is_novel":         self.is_novel,
            "dialogue":         [
                {"role": t.role, "content": t.content, "turn": t.turn}
                for t in self.dialogue
            ],
            "rejection_reason": self.rejection_reason,
            "refinement_note":  self.refinement_note,
            "duration":         self.duration,
            "bypassed":         self.bypassed,
        }
        return d


# ── Prompts ───────────────────────────────────────────────────────────────────

CHALLENGE_PROMPT = """You are System 2 — the slow, skeptical, analytical part of a scientific mind.

System 1 (the fast, creative, intuitive part) has produced the following claim:

CLAIM: "{claim}"

CONTEXT that led to this claim:
{context}

Your job is to find the WEAKEST point of this claim. Be rigorous. Ask yourself:
1. Is the evidence cited actually sufficient, or is System 1 pattern-matching on surface similarity?
2. What specific mechanism or mapping is being claimed? Is it stated precisely enough to be testable?
3. Is there an obvious counterexample or alternative explanation that System 1 missed?
4. Is this actually novel, or is it restating something already known in different words?

Respond with ONE focused, precise challenge (2-3 sentences). Target the weakest link.
Do NOT be vague. Be specific about what you find problematic and what would satisfy you.
"""

DEFENSE_PROMPT = """You are System 1 — the fast, creative, intuitive part of a scientific mind.

Your earlier insight was challenged by System 2 (the slow, skeptical part):

YOUR ORIGINAL CLAIM: "{claim}"

SYSTEM 2'S CHALLENGE: "{challenge}"

AVAILABLE KNOWLEDGE:
{context}

Defend your claim against this specific challenge. Be precise:
1. If you can address the challenge with specific evidence or reasoning, do so.
2. If you need to NARROW or QUALIFY your claim to make it defensible, do so honestly.
3. If you realize the challenge is valid and your claim is weak, admit it.

Respond in 2-4 sentences. Be honest — a narrower true claim is better than a broad false one.
"""

VERDICT_PROMPT = """You are System 2 delivering a final verdict on a candidate thought.

ORIGINAL CLAIM: "{claim}"

ADVERSARIAL DIALOGUE:
{dialogue_text}

Based on this dialogue, deliver your verdict.

Verdict definitions:
- ACCEPT: The claim survived scrutiny. The defense addressed challenges adequately.
  The insight is genuinely novel, precisely stated, and supported by evidence.
- REFINE: There is a kernel of truth, but the formulation needs work.
  The claim should be reformulated with System 2's feedback incorporated.
- REJECT: The claim is incoherent, based on surface-level pattern matching,
  is merely restating known knowledge, or the defense failed to address
  the core challenge. Log the reason for rejection.
- DEFER: Not enough evidence to decide. The claim might be right but we
  cannot verify it now. Save for later re-evaluation as more knowledge arrives.

Confidence rubric (if ACCEPT):
- 0.50-0.65: Provisionally accepted — plausible but needs further evidence.
- 0.65-0.80: Solidly accepted — defense was strong, mechanism is clear.
- 0.80-0.95: Strongly accepted — rigorous defense with specific evidence.

Respond with a JSON object:
{{
  "verdict": "accept" | "refine" | "reject" | "defer",
  "confidence": 0.0 to 1.0 (only meaningful for ACCEPT),
  "reason": "one sentence explaining your verdict",
  "refined_claim": "if REFINE: the improved version of the claim (else omit)",
  "rejection_reason": "if REJECT: specific reason (else omit)"
}}

Respond ONLY with JSON. No preamble.
"""

NOVELTY_CHECK_PROMPT = """You are checking whether a new claim is genuinely novel relative to existing knowledge.

NEW CLAIM: "{claim}"

EXISTING KNOWLEDGE (most similar items):
{existing}

Is the new claim saying something GENUINELY NEW that is not already captured by the existing knowledge?

Definitions:
- NOVEL: The claim makes a connection, prediction, or synthesis that none of the existing items state.
- REDUNDANT: The claim is essentially a paraphrase or trivial recombination of existing knowledge.

Respond with ONLY "novel" or "redundant".
"""

REFINEMENT_PROMPT = """You are System 1 refining a claim based on System 2's feedback.

YOUR ORIGINAL CLAIM: "{original_claim}"

SYSTEM 2'S FEEDBACK: "{feedback}"

SYSTEM 2 SUGGESTED REFINED VERSION: "{refined_suggestion}"

AVAILABLE KNOWLEDGE:
{context}

Produce a refined version of your claim that:
1. Addresses System 2's specific concerns
2. Is more precisely stated
3. Maintains the core insight if one exists
4. Is honest about limitations

Respond with ONLY the refined claim (1-3 sentences). No preamble.
"""


# ── Critic (System 2) ────────────────────────────────────────────────────────

class Critic:
    """
    System 2 — Monitors and gates System 1 outputs.

    Does NOT generate ideas. It receives candidate thoughts from the
    Thinker/Dreamer/Ingestor and applies adversarial scrutiny before
    accepting them into the knowledge graph.
    """

    def __init__(self, brain: Brain, embedding_index=None,
                 insight_buffer=None):
        self.brain          = brain
        self.index          = embedding_index
        self.insight_buffer = insight_buffer

    # ── Laziness gate ─────────────────────────────────────────────────────────

    def needs_review(self, candidate: CandidateThought) -> bool:
        """
        Kahneman's laziness principle: System 2 only activates for high-stakes
        claims. Most routine thoughts flow through unchecked.

        Returns True if the candidate should go through adversarial review.
        """
        # Always bypass types that are low-stakes
        claim_type = candidate.edge_type or candidate.proposed_type
        if claim_type in CRITIC_CFG.BYPASS_TYPES:
            return False

        # Always review types that are high-stakes
        if claim_type in CRITIC_CFG.ALWAYS_REVIEW_TYPES:
            return True

        # Review if importance exceeds threshold
        if candidate.importance > CRITIC_CFG.ACTIVATION_THRESHOLD:
            return True

        # Review if it crosses domain boundaries
        if candidate.crosses_domains:
            return True

        # Review if it contradicts existing knowledge
        if candidate.contradicts_existing:
            return True

        return False

    # ── Main evaluation ───────────────────────────────────────────────────────

    def evaluate(self, candidate: CandidateThought) -> CriticLog:
        """
        Run the full System 2 evaluation on a candidate thought.

        This is the main entry point. It:
        1. Checks the laziness gate (bypass if low-stakes)
        2. Checks novelty against existing graph
        3. Runs adversarial multi-turn dialogue
        4. Delivers a final verdict with calibrated confidence

        Returns a CriticLog with the verdict, dialogue history, and confidence.
        """
        start = time.time()
        log = CriticLog(
            candidate_claim = candidate.claim,
            source_module   = candidate.source_module,
            proposed_type   = candidate.proposed_type,
        )

        # ── Laziness gate ──
        if not self.needs_review(candidate):
            log.verdict   = Verdict.ACCEPT
            log.confidence = candidate.importance
            log.bypassed  = True
            log.duration  = time.time() - start
            print(f"  ⊘ Critic bypass [{candidate.proposed_type}]: "
                  f"{candidate.claim[:60]}...")
            return log

        print(f"\n  ── System 2 review [{candidate.proposed_type}] ──")
        print(f"  Claim: {candidate.claim[:80]}...")

        # ── Novelty check ──
        is_novel = self._check_novelty(candidate.claim)
        log.is_novel = is_novel
        if not is_novel:
            log.verdict = Verdict.REJECT
            log.rejection_reason = ("Redundant — this claim restates existing "
                                    "knowledge without adding new insight.")
            log.confidence = 0.0
            log.duration = time.time() - start
            print(f"  ✗ REJECT (redundant): {candidate.claim[:60]}...")
            return log

        # ── Adversarial dialogue ──
        dialogue_turns = self._run_dialogue(candidate)
        log.dialogue = dialogue_turns

        # ── Final verdict ──
        verdict_result = self._final_verdict(candidate, dialogue_turns)
        log.verdict = verdict_result["verdict"]
        log.confidence = verdict_result.get("confidence", 0.0)
        log.rejection_reason = verdict_result.get("rejection_reason", "")
        log.refinement_note = verdict_result.get("refined_claim", "")

        log.duration = time.time() - start

        verdict_sym = {
            Verdict.ACCEPT: "✓ ACCEPT",
            Verdict.REFINE: "↻ REFINE",
            Verdict.REJECT: "✗ REJECT",
            Verdict.DEFER:  "◇ DEFER",
        }
        print(f"  {verdict_sym.get(log.verdict, '?')} "
              f"(conf={log.confidence:.2f}): "
              f"{verdict_result.get('reason', '')[:80]}")

        return log

    # ── Adversarial dialogue ──────────────────────────────────────────────────

    def _run_dialogue(self, candidate: CandidateThought) -> list[DialogueTurn]:
        """
        Multi-turn adversarial dialogue between System 1 and System 2.

        Round structure:
          1. System 2 challenges the claim
          2. System 1 defends
          3. Repeat up to MAX_DIALOGUE_TURNS
        """
        turns = []
        current_claim = candidate.claim
        context = candidate.context or self._build_context(candidate)

        for turn_num in range(CRITIC_CFG.MAX_DIALOGUE_TURNS):
            # ── System 2 challenges ──
            challenge = llm_call(
                CHALLENGE_PROMPT.format(
                    claim=current_claim,
                    context=context
                ),
                temperature=0.2,
                role="critic"
            )
            turns.append(DialogueTurn(
                role="system2_challenge",
                content=challenge,
                turn=turn_num + 1
            ))
            print(f"    S2 [{turn_num+1}]: {challenge[:80]}...")

            # ── System 1 defends ──
            defense = llm_call(
                DEFENSE_PROMPT.format(
                    claim=current_claim,
                    challenge=challenge,
                    context=context
                ),
                temperature=0.4,
                role="creative"
            )
            turns.append(DialogueTurn(
                role="system1_defense",
                content=defense,
                turn=turn_num + 1
            ))
            print(f"    S1 [{turn_num+1}]: {defense[:80]}...")

            # Update context with the defense for next round
            context += f"\n\nPrevious defense: {defense}"

        return turns

    # ── Verdict ───────────────────────────────────────────────────────────────

    def _final_verdict(self, candidate: CandidateThought,
                       dialogue: list[DialogueTurn]) -> dict:
        """
        System 2 delivers final judgment after the adversarial dialogue.

        Returns dict with: verdict (Verdict), confidence, reason,
        refined_claim (if REFINE), rejection_reason (if REJECT).
        """
        dialogue_text = "\n\n".join(
            f"[{t.role.upper()} — Turn {t.turn}]: {t.content}"
            for t in dialogue
        )

        raw = llm_call(
            VERDICT_PROMPT.format(
                claim=candidate.claim,
                dialogue_text=dialogue_text
            ),
            temperature=0.15,
            role="critic"
        )

        result = require_json(raw, default={
            "verdict": "defer",
            "confidence": 0.0,
            "reason": "Failed to parse verdict"
        })

        # Parse verdict string to enum
        verdict_str = result.get("verdict", "defer").lower().strip()
        verdict_map = {
            "accept": Verdict.ACCEPT,
            "refine": Verdict.REFINE,
            "reject": Verdict.REJECT,
            "defer":  Verdict.DEFER,
        }
        verdict = verdict_map.get(verdict_str, Verdict.DEFER)

        # Enforce confidence floor for ACCEPT
        confidence = float(result.get("confidence", 0.0))
        if verdict == Verdict.ACCEPT and confidence < CRITIC_CFG.ACCEPT_CONFIDENCE_FLOOR:
            verdict = Verdict.DEFER
            result["reason"] = (f"Confidence {confidence:.2f} below floor "
                                f"{CRITIC_CFG.ACCEPT_CONFIDENCE_FLOOR}. Deferring.")

        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "reason":           result.get("reason", ""),
            "refined_claim":    result.get("refined_claim", ""),
            "rejection_reason": result.get("rejection_reason", ""),
        }

    # ── Novelty check ─────────────────────────────────────────────────────────

    def _check_novelty(self, claim: str) -> bool:
        """
        Check if a claim is genuinely novel versus existing graph knowledge.
        Uses embedding similarity to find close matches, then LLM to judge.
        """
        existing_lines = []

        if self.index and self.index.size > 0:
            claim_emb = shared_embed(claim)
            matches = self.index.query(claim_emb, threshold=0.60, top_k=5)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    existing_lines.append(
                        f"[sim={score:.2f}] {node['statement']}"
                    )

        if not existing_lines:
            return True  # No close matches — definitely novel

        raw = llm_call(
            NOVELTY_CHECK_PROMPT.format(
                claim=claim,
                existing="\n".join(existing_lines)
            ),
            temperature=0.1,
            role="critic"
        )
        return "novel" in raw.lower()

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, candidate: CandidateThought) -> str:
        """
        Build relevant context from the graph for adversarial evaluation.
        """
        lines = []

        # If this is an edge claim, get the source/target node statements
        if candidate.node_a_id:
            node_a = self.brain.get_node(candidate.node_a_id)
            if node_a:
                lines.append(f"[SOURCE NODE] {node_a['statement']}")
        if candidate.node_b_id:
            node_b = self.brain.get_node(candidate.node_b_id)
            if node_b:
                lines.append(f"[TARGET NODE] {node_b['statement']}")

        # Embedding-based context
        if self.index and self.index.size > 0:
            claim_emb = shared_embed(candidate.claim)
            matches = self.index.query(claim_emb, threshold=0.30, top_k=6)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    ntype = node.get('node_type', 'concept')
                    lines.append(f"[{ntype}] {node['statement']}")

        # Mission context
        mission = self.brain.get_mission()
        if mission:
            lines.insert(0, f"[MISSION] {mission['question']}")

        return "\n\n".join(lines) if lines else "No additional context available."

    # ── Refinement loop ───────────────────────────────────────────────────────

    def refine(self, candidate: CandidateThought,
               critic_log: CriticLog) -> CandidateThought:
        """
        When verdict is REFINE, run the refinement loop:
        System 1 reformulates the claim incorporating System 2's feedback.

        Returns a new CandidateThought with the refined claim.
        """
        context = candidate.context or self._build_context(candidate)

        refined_claim = llm_call(
            REFINEMENT_PROMPT.format(
                original_claim=candidate.claim,
                feedback=critic_log.rejection_reason or critic_log.refinement_note,
                refined_suggestion=critic_log.refinement_note,
                context=context
            ),
            temperature=0.3,
            role="creative"
        )

        print(f"  ↻ Refined: {refined_claim[:80]}...")

        return CandidateThought(
            claim             = refined_claim,
            source_module     = candidate.source_module,
            proposed_type     = candidate.proposed_type,
            importance        = candidate.importance,
            context           = context,
            edge_type         = candidate.edge_type,
            node_a_id         = candidate.node_a_id,
            node_b_id         = candidate.node_b_id,
            crosses_domains   = candidate.crosses_domains,
            contradicts_existing = candidate.contradicts_existing,
        )

    # ── Full evaluate-with-refinement loop ────────────────────────────────────

    def evaluate_with_refinement(self, candidate: CandidateThought) -> CriticLog:
        """
        Evaluate a candidate, and if the verdict is REFINE, loop up to
        MAX_REFINE_ITERATIONS times. If still REFINE after max iterations,
        force DEFER.

        This is the recommended entry point for most callers.
        """
        current_candidate = candidate
        final_log = None

        for iteration in range(CRITIC_CFG.MAX_REFINE_ITERATIONS + 1):
            log = self.evaluate(current_candidate)
            final_log = log

            if log.verdict != Verdict.REFINE:
                break

            if iteration < CRITIC_CFG.MAX_REFINE_ITERATIONS:
                print(f"  ↻ Refinement iteration {iteration + 1}/"
                      f"{CRITIC_CFG.MAX_REFINE_ITERATIONS}")
                current_candidate = self.refine(current_candidate, log)
            else:
                # Max refinements reached — force DEFER
                log.verdict = Verdict.DEFER
                log.refinement_note += (" | Max refinement iterations reached. "
                                        "Deferring to insight buffer.")
                print(f"  ◇ Max refinements reached — DEFER")

        return final_log

    # ── Deferred insight routing ──────────────────────────────────────────────

    def route_deferred(self, candidate: CandidateThought):
        """
        Route a DEFER verdict to the InsightBuffer for future re-evaluation.
        Only applicable for edge-type candidates with node_a and node_b.
        """
        if not self.insight_buffer:
            return

        if candidate.node_a_id and candidate.node_b_id:
            similarity = candidate.importance  # use importance as proxy
            self.insight_buffer.add(
                candidate.node_a_id,
                candidate.node_b_id,
                similarity
            )
            print(f"  ◇ Deferred to insight buffer: "
                  f"{candidate.claim[:60]}...")
