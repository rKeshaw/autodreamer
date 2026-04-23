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

import re
import time
from enum import Enum
from dataclasses import dataclass, field
from graph.brain import Brain, NodeType
from config import CRITIC as CRITIC_CFG
from llm_utils import llm_call, llm_json
from embedding import embed as shared_embed
from scientist_workspace import ArtifactStatus
from scientific_rigor import hypothesis_requires_formalism


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
    grounded_evidence: list[str] = field(default_factory=list)
    source_ids:       list[str] = field(default_factory=list)
    expected_status:  str = ArtifactStatus.OPEN.value


# ── Dialogue turn ─────────────────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    """One turn of the System 1 ↔ System 2 adversarial dialogue."""
    role:    str   # "system2_challenge" or "system1_defense"
    content: str
    turn:    int
    claim:   str = ""
    issue_label: str = ""
    repeated: bool = False
    addressed: bool = False


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
    final_claim:       str = ""
    verdict_reason:    str = ""
    verdict_parse_failed: bool = False
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
                {
                    "role": t.role,
                    "content": t.content,
                    "turn": t.turn,
                    "claim": t.claim,
                    "issue_label": t.issue_label,
                    "repeated": t.repeated,
                    "addressed": t.addressed,
                }
                for t in self.dialogue
            ],
            "final_claim":      self.final_claim,
            "verdict_reason":   self.verdict_reason,
            "verdict_parse_failed": self.verdict_parse_failed,
            "rejection_reason": self.rejection_reason,
            "refinement_note":  self.refinement_note,
            "duration":         self.duration,
            "bypassed":         self.bypassed,
        }
        return d


# ── Prompts ───────────────────────────────────────────────────────────────────

CHALLENGE_PROMPT = """A scientific claim is under adversarial review.

ACTIVE CLAIM:
"{claim}"

SOURCE CONTEXT:
{context}

PRIOR REVIEW HISTORY:
{history}

Review instructions:
1. Consider the prior review history before raising a new objection.
2. Identify the most important unresolved flaw. Do not restate an objection that has already been addressed unless it remains unresolved for a specific reason.
3. Use formal scientific prose. Avoid first-person and second-person pronouns.
4. Do not introduce unsupported equations, symbols, or quantitative claims.
5. If no materially distinct unresolved flaw remains, state that explicitly.
6. A structural or isomorphic claim is only acceptable if it preserves explicit role, variable, or constraint correspondences. Generic pattern language such as "both optimize" or "both balance stability and change" is a real defect, not an acceptable shortcut.

Respond EXACTLY in JSON:
{{
  "issue_label": "mechanism" | "evidence" | "scope" | "novelty" | "counterexample" | "resolved",
  "challenge": "<2-3 sentence formal objection or resolution statement>",
  "repeat_of_prior": true or false
}}
"""

DEFENSE_PROMPT = """A scientific claim is being defended against a specific objection.

ACTIVE CLAIM:
"{claim}"

CURRENT OBJECTION:
"{challenge}"

SOURCE CONTEXT:
{context}

PRIOR REVIEW HISTORY:
{history}

Response instructions:
1. Address only the current objection.
2. If the claim must be narrowed, qualified, or partially withdrawn, do so explicitly.
3. Use formal scientific prose. Avoid first-person and second-person pronouns.
4. Do not introduce unsupported equations, symbols, or parameter names unless they already appear in the available context and are explicitly defined.
5. A valid response may preserve a structural analogy only if the revised claim still states concrete correspondences between mechanisms, constraints, variables, or update rules. If narrowing removes the mapping, the objection is NOT resolved.

Respond EXACTLY in JSON:
{{
  "defense": "<2-4 sentence formal response>",
  "revised_claim": "<best current version of the claim after this response; repeat the active claim only if no change is needed>",
  "challenge_addressed": true or false
}}
"""

VERDICT_PROMPT = """A scientific claim has completed adversarial review.

ORIGINAL CLAIM: "{claim}"
BEST-SUPPORTED CURRENT CLAIM: "{final_claim}"

ADVERSARIAL DIALOGUE:
{dialogue_text}

Based on the dialogue, deliver a final verdict.

Verdict definitions:
- ACCEPT: The claim survived scrutiny. The defense addressed challenges adequately.
  The insight is genuinely novel, precisely stated, still contains explicit correspondences, and does not retreat into generic pattern language.
- REFINE: There is a kernel of truth, but the formulation needs work.
  The claim should be reformulated with System 2's feedback incorporated. Use REFINE when the thesis may be salvageable but the current wording lacks explicit mapping, becomes too generic, or overstates the evidence.
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
  "final_claim": "best-supported current wording after review",
  "refined_claim": "if REFINE: the improved version of the claim (else omit)",
  "rejection_reason": "if REJECT: specific reason (else omit)"
}}

Use formal scientific prose in all textual fields.
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

TRIVIALITY_CHECK_PROMPT = """You are checking whether a scientific claim makes a non-trivial contribution.

CLAIM: "{claim}"

RELATED KNOWLEDGE:
{context}

Is this claim:
- SUBSTANTIVE: It proposes a specific mechanism, connection, prediction, or synthesis not obvious from basic definitions. Even a well-known idea counts as substantive if it articulates a non-obvious relationship.
- TRIVIAL: It merely restates a textbook definition, is a tautology, or states something any educated reader would already know without needing to be told.

Respond with ONLY "substantive" or "trivial".
"""

THESIS_SURVIVAL_PROMPT = """You are checking whether a claim's core thesis survived adversarial review.

ORIGINAL CLAIM: "{original}"

FINAL CLAIM (after review): "{final}"

Does the FINAL claim still affirm the ORIGINAL claim's central thesis?

- SUPPORTS: The final claim preserves the original's core idea, even if
  qualified, narrowed, or rephrased. The central assertion is intact.
- ABANDONS: The final claim contradicts, negates, or fundamentally replaces
  the original's central assertion. For example, if the original claims
  "X proves Y" but the final says "X does NOT prove Y" or omits Y entirely,
  that is ABANDONS.

Respond with ONLY "supports" or "abandons".
"""

REFINEMENT_PROMPT = """Refine the following scientific claim using the review feedback.

CURRENT CLAIM: "{original_claim}"

REVIEW FEEDBACK: "{feedback}"

SUGGESTED REFINED VERSION: "{refined_suggestion}"

AVAILABLE KNOWLEDGE:
{context}

Produce a refined claim that:
1. Addresses the specific concerns raised in review.
2. Is more precise and mechanistically grounded.
3. Preserves the core insight only if it remains defensible.
4. Keeps explicit role, variable, or constraint correspondences if the claim is still meant to be structural or isomorphic.
5. Avoids retreating into generic language such as "both involve optimization" or "both balance exploration and stability."
6. Uses formal scientific prose without first-person or second-person pronouns.

Respond with ONLY the refined claim (1-3 sentences). No preamble.
"""

ACCEPT_QUALITY_PROMPT = """You are checking whether a post-review claim is still scientifically substantive.

Original claim: "{original}"
Final claim: "{final}"
Context:
{context}

Judge the FINAL claim conservatively.

- has_explicit_mapping = true only if the final claim still states concrete role, variable, mechanism, update-rule, or constraint correspondences.
- specific_enough = true only if the final claim remains mechanistic or formally informative, rather than broad high-level English logic.
- watered_down = true if the final claim mostly retreats into safer wording, abstract common sense, or generic statements such as "both optimize", "both involve trade-offs", or "both have constraints".

Respond EXACTLY in JSON:
{{
  "has_explicit_mapping": true or false,
  "specific_enough": true or false,
  "watered_down": true or false,
  "reason": "one or two sentences"
}}
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

    _FORMAL_REPLACEMENTS = (
        (r"\bSystem 2 is correct\b", "The objection is well-founded"),
        (r"\bThe challenge is valid\b", "The objection is well-founded"),
        (r"\bI must narrow the claim\b", "The claim must be narrowed"),
        (r"\bI will narrow the claim\b", "The claim is narrowed as follows"),
        (r"\bI must qualify the claim\b", "The claim requires qualification"),
        (r"\bI will qualify the claim\b", "The claim is qualified as follows"),
        (r"\bTo strengthen this, you must\b", "To strengthen the claim, it must"),
        (r"\byou must\b", "the claim must"),
        (r"\bYour earlier insight\b", "The earlier claim"),
    )

    # ── Laziness gate ─────────────────────────────────────────────────────────

    def needs_review(self, candidate: CandidateThought) -> bool:
        """
        Kahneman's laziness principle: System 2 only activates for high-stakes
        claims. Most routine thoughts flow through unchecked.

        Returns True if the candidate should go through adversarial review.

        Override flags (crosses_domains, contradicts_existing) take priority
        over BYPASS_TYPES: an instance-level risk signal always outranks
        a category-level bypass.
        """
        # ── Instance-level risk overrides ──
        # These are semantic signals about the *content* of this specific
        # claim — they override any type-based shortcut.
        if candidate.crosses_domains:
            return True

        if candidate.contradicts_existing:
            return True

        if candidate.expected_status == ArtifactStatus.GROUNDED.value:
            return True

        # ── Type-based routing ──
        claim_type = candidate.edge_type or candidate.proposed_type

        # Bypass types: low-stakes categories (e.g. concept, associated)
        if claim_type in CRITIC_CFG.BYPASS_TYPES:
            return False

        # Always-review types: high-stakes categories
        if claim_type in CRITIC_CFG.ALWAYS_REVIEW_TYPES:
            return True

        # ── Importance threshold ──
        if candidate.importance > CRITIC_CFG.ACTIVATION_THRESHOLD:
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
            log.final_claim = candidate.claim
            log.bypassed  = True
            log.duration  = time.time() - start
            print(f"  ⊘ Critic bypass [{candidate.proposed_type}]: "
                  f"{candidate.claim[:60]}...")
            return log

        print(f"\n  ── System 2 review [{candidate.proposed_type}] ──")
        display_claim = self._formalize_dialogue_text(candidate.claim) or candidate.claim
        print(f"  Claim: {display_claim[:80]}...")

        evaluation_context = self._evaluation_context(candidate)
        if (
            candidate.expected_status == ArtifactStatus.GROUNDED.value and
            not self._has_grounding_context(candidate, evaluation_context)
        ):
            log.verdict = Verdict.DEFER
            log.final_claim = candidate.claim
            log.verdict_reason = (
                "Grounded status was requested, but the claim arrived without "
                "linked evidence or source context."
            )
            log.duration = time.time() - start
            print("  ◇ DEFER: missing grounding context for grounded claim")
            return log

        # ── Novelty check ──
        is_novel = self._check_novelty(candidate.claim)
        log.is_novel = is_novel
        if not is_novel:
            log.verdict = Verdict.REJECT
            log.final_claim = candidate.claim
            log.rejection_reason = ("Redundant — this claim restates existing "
                                    "knowledge without adding new insight.")
            log.confidence = 0.0
            log.duration = time.time() - start
            print(f"  ✗ REJECT (redundant): {candidate.claim[:60]}...")
            self.brain.increase_frustration(0.2)
            return log

        # ── Triviality check ──
        if self._check_triviality(candidate.claim, evaluation_context):
            log.verdict = Verdict.REJECT
            log.final_claim = candidate.claim
            log.rejection_reason = ("Trivial — this claim restates a well-known "
                                    "definition or tautology without adding "
                                    "non-obvious insight.")
            log.confidence = 0.0
            log.duration = time.time() - start
            print(f"  ✗ REJECT (trivial): {candidate.claim[:60]}...")
            self.brain.increase_frustration(0.1)
            return log

        # ── Adversarial dialogue ──
        dialogue_turns, defended_claim = self._run_dialogue(candidate)
        log.dialogue = dialogue_turns

        # ── Final verdict ──
        verdict_result = self._final_verdict(candidate, dialogue_turns, defended_claim)
        log.verdict = verdict_result["verdict"]
        log.confidence = verdict_result.get("confidence", 0.0)
        log.final_claim = verdict_result.get("final_claim", defended_claim or candidate.claim)
        log.verdict_reason = verdict_result.get("reason", "")
        log.verdict_parse_failed = verdict_result.get("parse_failed", False)
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

        if log.verdict in (Verdict.REJECT, Verdict.REFINE):
            self.brain.increase_frustration(0.2)
            
        return log

    # ── Adversarial dialogue ──────────────────────────────────────────────────

    def _run_dialogue(self, candidate: CandidateThought) -> tuple[list[DialogueTurn], str]:
        """
        Multi-turn adversarial dialogue between System 1 and System 2.

        Round structure:
          1. System 2 challenges the claim
          2. System 1 defends
          3. Repeat up to MAX_DIALOGUE_TURNS
        """
        turns = []
        current_claim = self._formalize_dialogue_text(candidate.claim) or candidate.claim
        context = self._evaluation_context(candidate)
        prior_challenges = []
        issue_resolution = {}

        for turn_num in range(CRITIC_CFG.MAX_DIALOGUE_TURNS):
            history = self._dialogue_history_text(turns)

            # ── System 2 challenges ──
            challenge_result = llm_json(
                CHALLENGE_PROMPT.format(
                    claim=current_claim,
                    context=context,
                    history=history
                ),
                temperature=0.15,
                default={
                    "issue_label": "evidence",
                    "challenge": "No specific unresolved flaw was identified.",
                    "repeat_of_prior": False,
                }
            )
            challenge = self._formalize_dialogue_text(
                challenge_result.get("challenge", "")
            )
            issue_label = str(
                challenge_result.get("issue_label", "evidence")
            ).strip().lower()
            repeated = self._as_bool(challenge_result.get("repeat_of_prior", False))
            if self._is_repeated_objection(challenge, prior_challenges):
                repeated = True

            prior_addressed = issue_resolution.get(issue_label, False)
            if repeated and prior_challenges and prior_addressed:
                challenge = (
                    "No materially distinct unresolved flaw remained after the "
                    "earlier objections. Final adjudication should rely on the "
                    "issues already raised."
                )
                issue_label = "resolved"
                repeated = True
            elif repeated and prior_challenges and not prior_addressed:
                repeated = False

            if not challenge:
                challenge = (
                    "No materially distinct unresolved flaw was identified "
                    "after consideration of the prior review history."
                )
                issue_label = "resolved"

            turns.append(DialogueTurn(
                role="system2_challenge",
                content=challenge,
                turn=turn_num + 1,
                claim=current_claim,
                issue_label=issue_label,
                repeated=repeated,
                addressed=(issue_label == "resolved"),
            ))
            print(f"    S2 [{turn_num+1}]: {challenge[:80]}...")
            prior_challenges.append(challenge)

            if issue_label == "resolved":
                break

            # ── System 1 defends ──
            defense_result = llm_json(
                DEFENSE_PROMPT.format(
                    claim=current_claim,
                    challenge=challenge,
                    context=context,
                    history=history
                ),
                temperature=0.3,
                default={
                    "defense": "The objection could not be resolved with additional specificity.",
                    "revised_claim": current_claim,
                    "challenge_addressed": False,
                }
            )
            defense = self._formalize_dialogue_text(
                defense_result.get("defense", "")
            )
            revised_claim = self._formalize_dialogue_text(
                defense_result.get("revised_claim", "")
            )
            challenge_addressed = self._as_bool(
                defense_result.get("challenge_addressed", False)
            )
            if not defense:
                defense = "The objection could not be resolved with additional specificity."
                challenge_addressed = False
            if not revised_claim or len(revised_claim) < 20:
                revised_claim = current_claim
            if "could not be resolved with additional specificity" in defense.lower():
                challenge_addressed = False

            turns.append(DialogueTurn(
                role="system1_defense",
                content=defense,
                turn=turn_num + 1,
                claim=revised_claim,
                issue_label=issue_label,
                addressed=challenge_addressed,
            ))
            print(f"    S1 [{turn_num+1}]: {defense[:80]}...")

            issue_resolution[issue_label] = challenge_addressed
            current_claim = revised_claim

            if self._should_end_dialogue_early(
                candidate.claim, current_claim, turns
            ):
                turns.append(DialogueTurn(
                    role="system2_challenge",
                    content=(
                        "The outstanding objection has been addressed with "
                        "sufficient qualification for final adjudication."
                    ),
                    turn=turn_num + 1,
                    claim=current_claim,
                    issue_label="resolved",
                    repeated=False,
                    addressed=True,
                ))
                break

        return turns, current_claim

    # ── Verdict ───────────────────────────────────────────────────────────────

    def _final_verdict(self, candidate: CandidateThought,
                       dialogue: list[DialogueTurn],
                       final_claim: str) -> dict:
        """
        System 2 delivers final judgment after the adversarial dialogue.

        Returns dict with: verdict (Verdict), confidence, reason,
        refined_claim (if REFINE), rejection_reason (if REJECT).
        """
        dialogue_text = "\n\n".join(
            (
                f"[{t.role.upper()} — Turn {t.turn}] "
                f"(active_claim={t.claim or candidate.claim}): {t.content}"
            )
            for t in dialogue
        )

        default_result = {
            "verdict": "defer",
            "confidence": 0.0,
            "reason": "Failed to parse verdict",
            "final_claim": final_claim or candidate.claim,
        }

        result = llm_json(
            VERDICT_PROMPT.format(
                claim=candidate.claim,
                final_claim=final_claim or candidate.claim,
                dialogue_text=dialogue_text
            ),
            temperature=0.1,
            default=default_result
        )
        parse_failed = result.get("reason") == "Failed to parse verdict"
        if parse_failed:
            result = llm_json(
                VERDICT_PROMPT.format(
                    claim=candidate.claim,
                    final_claim=final_claim or candidate.claim,
                    dialogue_text=dialogue_text
                ),
                temperature=0.0,
                default=default_result
            )
            parse_failed = result.get("reason") == "Failed to parse verdict"

        # Parse verdict string to enum
        verdict_str = result.get("verdict", "defer").lower().strip()
        verdict_map = {
            "accept": Verdict.ACCEPT,
            "refine": Verdict.REFINE,
            "reject": Verdict.REJECT,
            "defer":  Verdict.DEFER,
        }
        verdict = verdict_map.get(verdict_str, Verdict.DEFER)

        # Parse confidence early; the stricter floor is enforced after any
        # provisional-accept logic so both direct and rescued ACCEPTs use the
        # same standard.
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        verdict_final_claim = self._formalize_dialogue_text(
            result.get("final_claim", "")
        ) or final_claim or candidate.claim
        refined_claim = self._formalize_dialogue_text(
            result.get("refined_claim", "")
        )
        if verdict == Verdict.REFINE and not refined_claim:
            refined_claim = verdict_final_claim

        if (
            self._is_deep_analogy_candidate(candidate) and
            verdict in {Verdict.DEFER, Verdict.REFINE} and
            self._dialogue_supports_provisional_accept(
                candidate.claim,
                verdict_final_claim or refined_claim,
                dialogue,
            )
        ):
            verdict = Verdict.ACCEPT
            best_supported_claim = (
                refined_claim
                if self._materially_revised_claim(candidate.claim, refined_claim)
                else verdict_final_claim
            ) or candidate.claim
            verdict_final_claim = best_supported_claim
            confidence = max(
                confidence,
                0.62 if self._ends_with_resolved_objection(dialogue) else 0.58,
            )
            result["reason"] = (
                "The objections were resolved through narrowing and qualification, "
                "and the remaining claim stayed defensible as a structural analogy."
            )

        if verdict == Verdict.ACCEPT:
            floor = self._accept_confidence_floor(candidate)
            if confidence < floor:
                verdict = Verdict.DEFER
                result["reason"] = (
                    f"Confidence {confidence:.2f} below floor {floor:.2f}. Deferring."
                )
        if verdict == Verdict.ACCEPT and self._context_uses_local_artifacts(dialogue_text):
            if not self._context_has_external_support(dialogue_text):
                verdict = Verdict.DEFER
                result["reason"] = (
                    "Self-generated session artifacts are not sufficient external evidence."
                )
        if (
            verdict == Verdict.ACCEPT and
            (candidate.proposed_type or "").lower() == NodeType.HYPOTHESIS.value and
            hypothesis_requires_formalism(verdict_final_claim or candidate.claim) and
            not self._context_has_formal_support(
                candidate,
                verdict_final_claim or candidate.claim,
                dialogue_text,
            )
        ):
            verdict = Verdict.DEFER
            result["reason"] = (
                "The hypothesis introduces mechanistic or formal machinery without an explicit derivation, bound, or cited evidence."
            )

        # ── Thesis survival guard ──
        # If the accepted claim drifted significantly from the original,
        # check whether the original's core thesis was preserved or
        # abandoned. Pure word-overlap can't distinguish legitimate
        # narrowing from thesis laundering, so we ask the LLM.
        if verdict == Verdict.ACCEPT and verdict_final_claim:
            drift = self._claim_drift(candidate.claim, verdict_final_claim)
            if drift > 0.50:
                thesis_survived = self._thesis_survived(
                    candidate.claim, verdict_final_claim
                )
                if not thesis_survived:
                    # Original thesis was abandoned — downgrade
                    verdict = Verdict.REJECT
                    confidence = min(confidence, 0.25)
                    result["reason"] = (
                        f"Original thesis abandoned (drift={drift:.2f}). "
                        "The final claim contradicts or replaces the "
                        "original's central assertion rather than refining it."
                    )
                    result["rejection_reason"] = result["reason"]
            if verdict == Verdict.ACCEPT and self._is_deep_analogy_candidate(candidate):
                quality = self._accepted_claim_quality(
                    candidate.claim,
                    verdict_final_claim,
                    candidate.context or self._build_context(candidate),
                )
                if (
                    not quality["has_explicit_mapping"] or
                    not quality["specific_enough"] or
                    quality["watered_down"] or
                    self._looks_generic_pattern_language(verdict_final_claim)
                ):
                    thesis_survived = self._thesis_survived(
                        candidate.claim,
                        verdict_final_claim,
                    )
                    if thesis_survived:
                        verdict = Verdict.REFINE
                        refined_claim = verdict_final_claim
                        result["reason"] = (
                            "The core thesis may be salvageable, but the current claim "
                            "became too generic or lost explicit mapping detail."
                        )
                    else:
                        verdict = Verdict.REJECT
                        confidence = min(confidence, 0.25)
                        result["reason"] = (
                            "The accepted wording no longer preserved a defensible "
                            "mechanistic or formal mapping."
                        )
                        result["rejection_reason"] = result["reason"]

        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "reason":           result.get("reason", ""),
            "parse_failed":     parse_failed,
            "final_claim":      verdict_final_claim,
            "refined_claim":    refined_claim,
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

    def _check_triviality(self, claim: str, context: str = "") -> bool:
        """
        Check whether a claim is trivially obvious — a textbook restatement
        or tautology that adds no non-obvious insight.

        Returns True if the claim is trivial (should be rejected).
        """
        # Build context from graph if not provided
        if not context and self.index and self.index.size > 0:
            claim_emb = shared_embed(claim)
            matches = self.index.query(claim_emb, threshold=0.40, top_k=5)
            lines = []
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    lines.append(f"[sim={score:.2f}] {node['statement']}")
            context = "\n".join(lines) if lines else "No related knowledge available."

        raw = llm_call(
            TRIVIALITY_CHECK_PROMPT.format(
                claim=claim,
                context=context or "No related knowledge available."
            ),
            temperature=0.1,
            role="critic"
        )
        return "trivial" in raw.lower()

    def _claim_drift(self, original: str, final: str) -> float:
        """
        Measure how much a claim drifted during dialogue.

        Returns a float 0.0-1.0 where:
        - 0.0 = identical claims
        - 1.0 = completely different claims (no word overlap)

        Uses Jaccard distance on content words (stopwords removed).
        """
        import re

        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
            "do", "does", "for", "from", "how", "if", "in", "into", "is",
            "it", "its", "of", "on", "or", "should", "than", "that", "the",
            "to", "was", "what", "when", "where", "which", "while", "with",
            "would",
        }

        def content_words(text: str) -> set[str]:
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            return words - stopwords

        orig_words = content_words(original)
        final_words = content_words(final)
        if not orig_words and not final_words:
            return 0.0
        union = orig_words | final_words
        intersection = orig_words & final_words
        jaccard_sim = len(intersection) / max(len(union), 1)
        return 1.0 - jaccard_sim  # drift = 1 - similarity

    def _thesis_survived(self, original: str, final: str) -> bool:
        """
        Check whether the original claim's core thesis survived review.

        Uses a focused LLM call to distinguish:
        - Legitimate narrowing/qualification (thesis preserved)
        - Thesis laundering (original assertion abandoned/contradicted)

        Returns True if the thesis survived, False if it was abandoned.
        """
        raw = llm_call(
            THESIS_SURVIVAL_PROMPT.format(
                original=original,
                final=final,
            ),
            temperature=0.1,
            role="critic"
        )
        return "supports" in raw.lower()

    def _accepted_claim_quality(self, original: str, final: str, context: str) -> dict:
        result = llm_json(
            ACCEPT_QUALITY_PROMPT.format(
                original=original,
                final=final,
                context=context or "No additional context available.",
            ),
            temperature=0.0,
            default={
                "has_explicit_mapping": False,
                "specific_enough": False,
                "watered_down": True,
                "reason": "Failed to parse quality judgment.",
            }
        )
        return {
            "has_explicit_mapping": self._as_bool(result.get("has_explicit_mapping", False)),
            "specific_enough": self._as_bool(result.get("specific_enough", False)),
            "watered_down": self._as_bool(result.get("watered_down", True)),
            "reason": result.get("reason", ""),
        }

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, candidate: CandidateThought) -> str:
        """
        Build relevant context from the graph for adversarial evaluation.
        """
        lines = []

        if candidate.expected_status:
            lines.append(f"[EXPECTED EPISTEMIC STATUS] {candidate.expected_status}")

        if candidate.grounded_evidence:
            lines.append("[LINKED SOURCES]")
            for ref in candidate.grounded_evidence[:5]:
                if ref:
                    lines.append(f"- {ref}")

        if candidate.source_ids:
            lines.append("[SOURCE MATERIAL]")
            for source_id in candidate.source_ids[:3]:
                source_node = self.brain.get_node(source_id)
                if not source_node:
                    continue
                title = source_node.get("statement", "")
                refs = source_node.get("source_refs", []) or []
                ref_text = f" ({refs[0]})" if refs else ""
                excerpt = (source_node.get("source_excerpt", "") or "").strip()
                if excerpt:
                    excerpt = excerpt[:400]
                    lines.append(f"- {title}{ref_text}: {excerpt}")
                else:
                    lines.append(f"- {title}{ref_text}")

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

    def _evaluation_context(self, candidate: CandidateThought) -> str:
        parts = []
        if candidate.context and candidate.context.strip():
            parts.append(candidate.context.strip())
        built_context = self._build_context(candidate)
        if built_context and built_context != "No additional context available.":
            parts.append(built_context)
        if not parts:
            return "No additional context available."

        deduped = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return "\n\n".join(deduped)

    def _has_grounding_context(self, candidate: CandidateThought, context: str) -> bool:
        if candidate.grounded_evidence or candidate.source_ids:
            return True
        lowered = (context or "").lower()
        return any(
            marker in lowered
            for marker in ("[source material]", "[linked sources]", "source:")
        )

    def _dialogue_history_text(self, turns: list[DialogueTurn]) -> str:
        if not turns:
            return "No prior review turns."
        return "\n".join(
            (
                f"Turn {t.turn} | {t.role} | claim={t.claim or 'unchanged'} "
                f"| issue={t.issue_label or 'none'} | addressed={t.addressed} | {t.content}"
            )
            for t in turns
        )

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip().lower()
        return re.sub(r"[^a-z0-9 ]+", " ", text)

    def _formalize_dialogue_text(self, text: str) -> str:
        cleaned = str(text or "")
        cleaned = cleaned.replace("→", " maps to ")
        cleaned = cleaned.replace("↔", " corresponds to ")
        cleaned = cleaned.replace("⇒", " implies ")
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", cleaned)
        cleaned = re.sub(r"\$+", " ", cleaned)
        cleaned = re.sub(r"\\(?:rightarrow|to)\b", " maps to ", cleaned)
        cleaned = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})*", " ", cleaned)
        cleaned = re.sub(r"[_^{}]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        for pattern, replacement in self._FORMAL_REPLACEMENTS:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _as_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    def _is_repeated_objection(self, challenge: str,
                               prior_challenges: list[str]) -> bool:
        norm = self._normalize_text(challenge)
        if not norm or not prior_challenges:
            return False
        challenge_words = set(norm.split())
        if len(challenge_words) < 6:
            return False
        for prior in prior_challenges:
            prior_norm = self._normalize_text(prior)
            if not prior_norm:
                continue
            prior_words = set(prior_norm.split())
            overlap = len(challenge_words & prior_words)
            union = len(challenge_words | prior_words) or 1
            if overlap / union >= 0.65:
                return True
        return False

    def _materially_revised_claim(self, original_claim: str,
                                  revised_claim: str) -> bool:
        if not revised_claim or not original_claim:
            return False
        orig = self._normalize_text(original_claim)
        rev = self._normalize_text(revised_claim)
        if not orig or not rev or orig == rev:
            return False
        orig_words = set(orig.split())
        rev_words = set(rev.split())
        overlap = len(orig_words & rev_words)
        union = len(orig_words | rev_words) or 1
        return (overlap / union) < 0.92

    def _is_deep_analogy_candidate(self, candidate: CandidateThought) -> bool:
        claim_type = (candidate.edge_type or candidate.proposed_type or "").lower()
        return claim_type in {
            "structural",
            "isomorphism",
            "structural_analogy",
            "deep_isomorphism",
        }

    def _accept_confidence_floor(self, candidate: CandidateThought) -> float:
        floor = CRITIC_CFG.ACCEPT_CONFIDENCE_FLOOR
        if self._is_deep_analogy_candidate(candidate):
            return max(floor, 0.65)
        return floor

    def _context_has_external_support(self, context: str) -> bool:
        lowered = str(context or "").lower()
        return any(
            marker in lowered
            for marker in (
                "https://",
                "http://",
                "arxiv.org",
                "doi.org",
                "journals.aps.org",
                "nature.com",
                "science.org",
                "inspirehep.net",
                "pdg.lbl.gov",
            )
        )

    def _context_uses_local_artifacts(self, context: str) -> bool:
        lowered = str(context or "").lower()
        return any(
            marker in lowered
            for marker in (
                "file://",
                "/home/",
                "virtual_lab/",
                "session_summary",
                "stdout.txt",
                "stderr.txt",
                "logs/",
            )
        )

    def _context_has_formal_support(self, candidate: CandidateThought,
                                    claim: str,
                                    dialogue_text: str) -> bool:
        context = "\n".join(filter(None, [
            candidate.context,
            self._evaluation_context(candidate),
            dialogue_text,
            claim,
        ])).lower()
        if candidate.grounded_evidence:
            return True
        return any(
            marker in context
            for marker in (
                "equation",
                "operator",
                "lagrangian",
                "hamiltonian",
                "constraint",
                "derive",
                "derivation",
                "measured",
                "bounded by",
                "confidence interval",
                "upper limit",
            )
        )

    def _has_mapping_markers(self, claim: str) -> bool:
        text = (claim or "").lower()
        markers = (
            "map:",
            "maps to",
            "corresponds to",
            "correspondence",
            "under renaming",
            "update rule",
            "state variable",
            "objective function",
            "constraint",
            "role of",
            "acts on",
            "preserves",
        )
        return any(marker in text for marker in markers)

    def _looks_generic_pattern_language(self, claim: str) -> bool:
        text = (claim or "").lower()
        generic_markers = (
            "both involve",
            "both balance",
            "both optimize",
            "both have",
            "both show",
            "mirrors the adaptive pressure",
            "principle of localized",
            "pattern of",
            "can be understood as",
            "serves as a molecular analogue",
            "high-level",
        )
        return any(marker in text for marker in generic_markers) and not self._has_mapping_markers(text)

    def _is_qualified_claim(self, claim: str) -> bool:
        if not claim:
            return False
        claim_lower = claim.lower()
        return any(
            marker in claim_lower
            for marker in (
                "structural analogy",
                "analogous",
                "analogy",
                "limited to",
                "within the context",
                "within ",
                "rather than",
                "without implying",
                "does not imply",
                "can be modeled",
                "can be understood as",
                "provided that",
                "potential for",
                "suggests",
                "constraint",
                "pattern",
            )
        )

    def _ends_with_resolved_objection(self, dialogue: list[DialogueTurn]) -> bool:
        challenges = [t for t in dialogue if t.role == "system2_challenge"]
        return bool(challenges and challenges[-1].issue_label == "resolved")

    def _dialogue_supports_provisional_accept(self, original_claim: str,
                                              final_claim: str,
                                              dialogue: list[DialogueTurn]) -> bool:
        if not dialogue:
            return False
        challenges = [t for t in dialogue if t.role == "system2_challenge"]
        defenses = [t for t in dialogue if t.role == "system1_defense"]
        if not challenges or not defenses:
            return False
        issue_labels = [
            t.issue_label for t in challenges
            if t.issue_label and t.issue_label != "resolved"
        ]
        if not issue_labels:
            return False
        if any(label not in {"scope", "mechanism"} for label in issue_labels):
            return False
        if any(
            "could not be resolved with additional specificity" in t.content.lower()
            for t in defenses
        ):
            return False
        if not all(t.addressed for t in defenses):
            return False
        if len(defenses) > 2 and not self._ends_with_resolved_objection(dialogue):
            return False
        if not (
            self._materially_revised_claim(original_claim, final_claim) or
            self._is_qualified_claim(final_claim) or
            self._ends_with_resolved_objection(dialogue)
        ):
            return False
        if not self._has_mapping_markers(final_claim):
            return False
        if self._looks_generic_pattern_language(final_claim):
            return False
        return True

    def _should_end_dialogue_early(self, original_claim: str,
                                   current_claim: str,
                                   dialogue: list[DialogueTurn]) -> bool:
        defenses = [t for t in dialogue if t.role == "system1_defense"]
        if not defenses:
            return False
        last_defense = defenses[-1]
        if not last_defense.addressed:
            return False
        if last_defense.issue_label not in {"scope", "mechanism"}:
            return False
        if "could not be resolved with additional specificity" in last_defense.content.lower():
            return False
        if len(defenses) == 1:
            return (
                self._materially_revised_claim(original_claim, current_claim) and
                self._is_qualified_claim(current_claim)
            )
        return self._dialogue_supports_provisional_accept(
            original_claim, current_claim, dialogue
        )

    # ── Refinement loop ───────────────────────────────────────────────────────

    def refine(self, candidate: CandidateThought,
               critic_log: CriticLog) -> CandidateThought:
        """
        When verdict is REFINE, run the refinement loop:
        System 1 reformulates the claim incorporating System 2's feedback.

        Returns a new CandidateThought with the refined claim.
        """
        context = self._evaluation_context(candidate)

        base_claim = critic_log.final_claim or candidate.claim

        refined_claim = self._formalize_dialogue_text(llm_call(
            REFINEMENT_PROMPT.format(
                original_claim=base_claim,
                feedback=critic_log.rejection_reason or critic_log.refinement_note,
                refined_suggestion=critic_log.refinement_note,
                context=context
            ),
            temperature=0.3,
            role="creative"
        ))
        if not refined_claim:
            refined_claim = base_claim

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
            grounded_evidence = list(candidate.grounded_evidence),
            source_ids        = list(candidate.source_ids),
            expected_status   = candidate.expected_status,
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
                log.verdict = Verdict.DEFER
                log.refinement_note += (" | Max refinement iterations reached. "
                                        "Deferring to insight buffer.")
                print(f"  ◇ Max refinements reached — DEFER")

        return final_log

    # ── Deferred insight routing ──────────────────────────────────────────────

    def route_deferred(self, candidate: CandidateThought):
        """
        Route a DEFER verdict to the InsightBuffer for future re-evaluation.
        Applicable for both edges and single novel nodes.
        """
        if not self.insight_buffer:
            return

        if candidate.node_a_id and candidate.node_b_id:
            similarity = candidate.importance  # use importance as proxy
            self.insight_buffer.add(
                candidate.node_a_id,
                candidate.node_b_id,
                similarity,
                claim=candidate.claim,
                context=candidate.context,
                proposed_type=candidate.proposed_type
            )
            print(f"  ◇ Deferred to insight buffer: "
                  f"{candidate.claim[:60]}...")
        else:
            self.insight_buffer.add_node(
                claim=candidate.claim,
                context=candidate.context,
                importance=candidate.importance
            )
            print(f"  ◇ Deferred node to insight buffer: "
                  f"{candidate.claim[:60]}...")
