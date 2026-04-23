"""
Thinker — Deliberate, goal-directed reasoning for THE SCIENTIST.

Unlike the Dreamer (random walks, serendipity), the Thinker does structured,
convergent reasoning — working toward *answers*, not just associations.

Thinking patterns:
  1. Dialectical   — evidence for/against, then synthesis
  2. Analogical    — transfer solution from analogous domain
  3. Reductive     — break question into simpler sub-questions
  4. Experimental  — thought experiments: "If X, then we'd expect Y"
  5. Integrative   — combine ideas into a unifying principle

Usage:
    thinker = Thinker(brain, observer, embedding_index)
    log = thinker.think()              # auto-picks best question
    log = thinker.think(question="...") # think about specific topic
"""

import json
import re
import time
from dataclasses import dataclass, field
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeType, NodeStatus)
from llm_utils import llm_call, llm_json, llm_chat
from embedding import embed as shared_embed
from thinker.policy import CognitivePolicy
from scientist_workspace import ArtifactStatus, ReasoningResult, ScientistWorkspace

# ── Thinking patterns ─────────────────────────────────────────────────────────

DIALECTICAL_PROMPT = """You are a scientist reasoning carefully about a question.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

Think dialectically:
1. What evidence or reasoning SUPPORTS a positive answer?
2. What evidence or reasoning ARGUES AGAINST it?
3. Given both sides, what is the most defensible conclusion right now?
4. What SPECIFIC piece of evidence or experiment would resolve the tension?

Write your reasoning as a structured argument (2-4 paragraphs).
Be precise. Cite specific ideas from the knowledge provided.
"""

ANALOGICAL_PROMPT = """You are a scientist looking for analogies that could solve a problem.

QUESTION: {question}

KNOWLEDGE FROM VARIOUS DOMAINS:
{context}

Is there a problem in a DIFFERENT domain that has a similar structure to this one?
If so:
1. What is the analogous problem?
2. How was it solved there?
3. Can that solution transfer to our domain? What would need to change?
4. What does this analogy reveal that direct reasoning might miss?

If no useful analogy exists, say so honestly and explain what makes this problem unique.
"""

REDUCTIVE_PROMPT = """You are a scientist trying to break down a hard question.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

This question may be too complex to answer directly. Break it down:
1. What are the 2-4 simpler SUB-QUESTIONS that, if answered, would answer the main question?
2. For each sub-question, do we already have evidence in our knowledge?
3. Which sub-question is the MOST tractable right now?
4. Which sub-question, if answered, would have the HIGHEST leverage?

Respond with a JSON object:
{{
  "sub_questions": [
    {{
      "question": "the sub-question",
      "existing_evidence": "what we already know, or 'none'",
      "tractability": "high/medium/low",
      "leverage": "high/medium/low"
    }}
  ],
  "recommended_focus": "which sub-question to pursue first and why"
}}
"""

EXPERIMENTAL_PROMPT = """You are a scientist designing a thought experiment.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

Design a thought experiment to test this:
1. What HYPOTHESIS does this question imply?
2. IF the hypothesis is true, what SPECIFIC OBSERVABLE consequence would we expect?
3. IF the hypothesis is false, what would we expect instead?
4. Can we check either prediction against what we already know?
5. What is the verdict so far?

Write your thought experiment as a clear, step-by-step argument.
"""

INTEGRATIVE_PROMPT = """You are a scientist looking for a unifying principle.

These ideas all seem related but no one has articulated WHY:
{context}

QUESTION: {question}

Can you find a UNIFYING PRINCIPLE that:
1. Explains why all these ideas are connected?
2. Predicts something NEW that none of them state individually?
3. Is FALSIFIABLE — what would prove it wrong?

If you find one, state it precisely. If not, explain what's missing.
"""

HYPOTHESIS_TESTING_PROMPT = """You are a scientist evaluating a hypothesis by decomposing it into testable components.

HYPOTHESIS: {question}

RELEVANT KNOWLEDGE:
{context}

Your task is to decompose this hypothesis into testable sub-claims and produce
concrete search queries that would find evidence for or against each one.

1. What are the 2-3 SPECIFIC CLAIMS embedded in this hypothesis?
2. For each claim, what SEARCH QUERY (4-8 words) would find evidence for or against it?
3. What would CONFIRMATION look like? What would REFUTATION look like?

Respond with JSON:
{{
  "sub_claims": [
    {{
      "claim": "the specific sub-claim",
      "search_query": "concise literature search query",
      "confirmation_looks_like": "what evidence would confirm this",
      "refutation_looks_like": "what evidence would refute this"
    }}
  ],
  "overall_assessment": "one sentence: how promising is this hypothesis given current knowledge?"
}}
"""

PICK_PATTERN_PROMPT = """You are selecting a reasoning strategy for a scientific question.

QUESTION: {question}

Available strategies:
- dialectical: weigh evidence for and against (best for contested claims)
- analogical: find parallels in other domains (best for novel problems)
- reductive: break into sub-questions (best for complex, multi-part problems)
- experimental: design thought experiments (best for testable hypotheses)
- integrative: find unifying principles (best when many related facts exist)
- hypothesis_testing: decompose into testable sub-claims with search queries (best for wild hypotheses)

Which strategy is BEST for this question? Respond with ONLY the strategy name.
"""

THINKING_SUMMARY_PROMPT = """Summarize the key insight from this thinking session in 1-2 sentences.
This will be stored as a new node in the knowledge graph.

Thinking session:
{reasoning}

Respond with ONLY the insight statement. No preamble.
"""

NEXT_ROUND_PROMPT = """You are planning the next step in a scientific reasoning session.

MISSION:
{mission}

SESSION ANCHOR QUESTION:
{anchor_question}

PREVIOUS ROUND QUESTION:
{previous_question}

PREVIOUS ROUND PATTERN:
{previous_pattern}

PREVIOUS ROUND INSIGHT:
{previous_insight}

PREVIOUS ROUND SUB-QUESTIONS:
{previous_sub_questions}

EARLIER INSIGHTS:
{history}

Choose the SINGLE best next question to improve the session.

Requirements:
1. Stay tightly tied to the mission and anchor question.
2. Increase specificity, testability, or decision value.
3. Prefer a mechanism, observable, boundary condition, counterexample, or decisive comparison.
4. Do NOT merely restate the mission or previous insight at a higher level of abstraction.
5. If the previous round was integrative or analogical, prefer grounding or falsification over more synthesis.

Respond EXACTLY in JSON:
{{
  "next_question": "one precise question",
  "preferred_pattern": "dialectical|analogical|reductive|experimental|integrative",
  "goal": "one sentence describing how this question advances the session"
}}
"""

REASONING_RESULT_PROMPT = """You are structuring the output of a scientist's reasoning session.

QUESTION:
{question}

PATTERN:
{pattern}

SCIENTIST WORKSPACE:
{workspace}

REASONING:
{reasoning}

Return a JSON object with:
{{
  "grounded_claims": ["claims directly supported by grounded evidence already present in the workspace"],
  "prior_claims": ["useful background-knowledge claims used in reasoning but not directly grounded by the workspace evidence"],
  "hypotheses": ["speculative mechanisms or explanatory proposals"],
  "open_questions": ["specific unresolved scientific questions"],
  "next_actions": ["concrete next research or analysis actions"],
  "summary_claim": "the single best current statement to store"
}}

Rules:
- Put a claim in "grounded_claims" only if it is supported by the grounded evidence in the workspace.
- Put background knowledge, intuition, or memory-based reasoning in "prior_claims".
- Put speculative explanations in "hypotheses".
- The "summary_claim" must be cautious and well-labeled. If it depends on prior knowledge or speculation, say so explicitly.
- Keep each item to 1-2 sentences max.
- Use [] for empty lists and "" for an empty summary.
Respond ONLY with JSON.
"""

SUPPORTED_THINKING_PATTERNS = {
    "dialectical",
    "analogical",
    "reductive",
    "experimental",
    "integrative",
    "hypothesis_testing",
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ThinkingLog:
    question: str           = ""
    pattern: str            = ""
    node_type: str          = "question"
    cluster: str            = "unclustered"
    reasoning: str          = ""
    insight: str            = ""
    sub_questions: list     = field(default_factory=list)
    workspace: dict         = field(default_factory=dict)
    reasoning_result: dict  = field(default_factory=dict)
    node_id: str            = ""
    question_node_id: str   = ""    # ID of the hypothesis/question node being processed
    started_at: float       = field(default_factory=time.time)
    duration: float         = 0.0

    def to_dict(self):
        return self.__dict__


# ── Thinker ───────────────────────────────────────────────────────────────────

class Thinker:
    def __init__(self, brain: Brain, observer=None, embedding_index=None,
                 critic=None):
        self.brain    = brain
        self.observer = observer
        self.index    = embedding_index
        self.critic   = critic   # System 2 gating (optional)
        self.policy   = CognitivePolicy()
        self._pattern_hint_cache: dict[str, str] = {}

    def _build_workspace(self, question: str) -> ScientistWorkspace:
        return self.brain.build_workspace(
            embedding_index=self.index,
            query=question,
        )

    def _build_context(self, question: str,
                       workspace: ScientistWorkspace | None = None) -> str:
        """Build relevant context from the current scientist workspace."""
        workspace = workspace or self._build_workspace(question)
        context = workspace.to_prompt_context()
        return context if context.strip() else "No relevant knowledge found."

    def _pick_question(self) -> tuple[str, str, str]:
        """Pick the best question to think about from the agenda/graph.

        Returns:
            Tuple of (question_text, question_node_id, preferred_pattern).
            question_node_id is empty string if the question doesn't come from a node.
            preferred_pattern is empty string unless a specific pattern is warranted.
        """
        # ── HIGHEST PRIORITY: untested Dreamer hypotheses ──
        # These are wild hypotheses awaiting decomposition into search queries.
        hyp_nodes = self.brain.nodes_by_type(NodeType.HYPOTHESIS)
        untested = [
            (nid, data) for nid, data in hyp_nodes
            if data.get('status') == NodeStatus.HYPOTHETICAL.value
            and data.get('created_by') == 'dreamer_hypothesis'
        ]
        if untested:
            # Sort by importance (higher = more mission-relevant)
            untested.sort(key=lambda x: x[1].get('importance', 0), reverse=True)
            nid, data = untested[0]
            return data['statement'], nid, 'hypothesis_testing'

        # Priority: working memory hypotheses > agenda questions > high-importance gaps
        for nid, data in self.brain.get_working_memory():
            if data.get('node_type') in [NodeType.HYPOTHESIS.value,
                                          NodeType.QUESTION.value,
                                          NodeType.GAP.value,
                                          NodeType.TASK.value]:
                return data['statement'], nid, ''

        # From observer agenda
        if self.observer and hasattr(self.observer, 'agenda'):
            open_items = [
                item for item in self.observer.agenda
                if not item.resolved
            ]
            if open_items:
                # Pick highest priority
                best = max(open_items, key=lambda x: x.priority)
                return best.text, getattr(best, 'node_id', ''), ''

        # From graph — highest importance unresolved question
        questions = self.brain.nodes_by_type(NodeType.QUESTION)
        gaps      = self.brain.nodes_by_type(NodeType.GAP)
        hyps      = self.brain.nodes_by_type(NodeType.HYPOTHESIS)
        tasks     = self.brain.nodes_by_type(NodeType.TASK)

        candidates = questions + gaps + hyps + tasks
        if candidates:
            best = max(candidates,
                       key=lambda x: x[1].get('importance', 0.5))
            return best[1]['statement'], best[0], ''

        # Fallback: think about the mission
        mission = self.brain.get_mission()
        if mission:
            return mission['question'], mission.get('id', ''), ''

        return "What is the most important open question in our knowledge?", '', ''

    def _pick_pattern(self, question: str) -> tuple[str, str, str]:
        """Choose a pattern using question semantics plus procedural memory."""
        node_type = "question"
        cluster = "unclustered"
        
        q_emb = shared_embed(question)
        if self.index and self.index.size > 0:
            matches = self.index.query(q_emb, threshold=0.8, top_k=1)
            if matches:
                nid, _ = matches[0]
                node = self.brain.get_node(nid)
                if node:
                    node_type = node.get('node_type', 'question')
                    cluster = node.get('cluster', 'unclustered')

        preferred_pattern = self._semantic_pattern_hint(question)
        pattern = self.policy.choose_pattern(
            node_type, cluster, preferred_action=preferred_pattern
        )
        if pattern not in SUPPORTED_THINKING_PATTERNS:
            pattern = preferred_pattern or "dialectical"
        return node_type, cluster, pattern

    def _normalize_pattern_name(self, raw: str) -> str:
        text = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "empirical": "experimental",
            "thought_experiment": "experimental",
            "first_principles": "reductive",
            "first_principle": "reductive",
            "synthesis": "integrative",
            "hypothesis_test": "hypothesis_testing",
        }
        if text in alias_map:
            text = alias_map[text]
        return text if text in SUPPORTED_THINKING_PATTERNS else ""

    def _semantic_pattern_hint(self, question: str) -> str:
        q_key = self._normalize_text(question)
        if q_key in self._pattern_hint_cache:
            return self._pattern_hint_cache[q_key]

        raw = llm_call(
            PICK_PATTERN_PROMPT.format(question=question),
            temperature=0.1,
            role="precise"
        )
        pattern = self._normalize_pattern_name(raw)
        if not pattern:
            pattern = "dialectical"
        self._pattern_hint_cache[q_key] = pattern
        return pattern

    def _mission_text(self) -> str:
        mission = self.brain.get_mission()
        if mission and mission.get("question"):
            return mission["question"]
        return ""

    def _score_sub_question(self, sub_question: dict) -> tuple[float, float]:
        leverage_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        tractability_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        leverage = leverage_map.get(
            str(sub_question.get("leverage", "medium")).lower(),
            2.0,
        )
        tractability = tractability_map.get(
            str(sub_question.get("tractability", "medium")).lower(),
            2.0,
        )
        return leverage, tractability

    def _best_follow_up_subquestion(self, sub_questions: list[dict]) -> str:
        ranked = []
        for sq in sub_questions or []:
            question = (sq.get("question") or "").strip()
            if not question:
                continue
            leverage, tractability = self._score_sub_question(sq)
            ranked.append(((leverage, tractability, -len(question)), question))
        if not ranked:
            return ""
        ranked.sort(reverse=True)
        return ranked[0][1]

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _content_tokens(self, text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
            "does", "for", "from", "how", "if", "in", "is", "it", "of",
            "on", "or", "the", "to", "what", "when", "which", "while",
            "with", "would",
        }
        tokens = re.findall(r"[a-z0-9]+", self._normalize_text(text))
        return {token for token in tokens if token not in stopwords}

    def _question_overlap_score(self, main_question: str, sub_question: str) -> float:
        main_tokens = self._content_tokens(main_question)
        sub_tokens = self._content_tokens(sub_question)
        if not main_tokens or not sub_tokens:
            return 0.0
        return len(main_tokens & sub_tokens) / max(len(sub_tokens), 1)

    def _question_like(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        return bool(re.match(r"^(What|Which|How|Why|When|Where|Can|Could|Is|Are|Under what)\b", stripped))

    def _select_focus_subquestion(
        self,
        main_question: str,
        sub_questions: list[dict],
        recommended_focus: str,
    ) -> str:
        focus_text = self._normalize_text(recommended_focus)
        candidates = []
        for sq in sub_questions or []:
            q_text = (sq.get("question") or "").strip()
            if not q_text:
                continue
            if self._normalize_text(q_text) in focus_text:
                return q_text
            leverage, tractability = self._score_sub_question(sq)
            overlap = self._question_overlap_score(main_question, q_text)
            recommendation_overlap = self._question_overlap_score(
                recommended_focus or main_question,
                q_text,
            )
            score = (
                (leverage * 1.6) +
                (tractability * 1.2) +
                (overlap * 2.2) +
                (recommendation_overlap * 2.0)
            )
            candidates.append((score, len(q_text), q_text))

        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _format_focus_insight(self, focus_question: str, recommended_focus: str) -> str:
        focus_question = (focus_question or "").strip()
        if not focus_question:
            return (recommended_focus or "").strip()
        if recommended_focus and not self._question_like(recommended_focus):
            return f"Priority focus: {focus_question} Reason: {recommended_focus.strip()}"
        return f"Priority focus: {focus_question}"

    def _fallback_next_round(self, anchor_question: str, previous_log: ThinkingLog) -> tuple[str, str]:
        mission = self._mission_text() or anchor_question or previous_log.question
        previous_insight = previous_log.insight or previous_log.question

        if previous_log.pattern in {"integrative", "analogical"}:
            return (
                f"What concrete observation, intervention, or failure case would discriminate whether '{previous_insight}' is the right explanation for '{mission}'?",
                "experimental",
            )
        if previous_log.pattern == "dialectical":
            return (
                f"What specific experiment, dataset, or comparison would resolve the strongest remaining uncertainty in '{anchor_question or mission}'?",
                "experimental",
            )
        if previous_log.pattern == "experimental":
            return (
                f"What boundary condition or competing explanation would most strongly challenge the current prediction about '{anchor_question or mission}'?",
                "dialectical",
            )
        if previous_log.pattern == "reductive" and previous_log.sub_questions:
            best_sq = self._best_follow_up_subquestion(previous_log.sub_questions)
            if best_sq:
                return best_sq, ""
        return (
            f"What specific mechanism or decision-relevant comparison would most directly answer '{anchor_question or mission}'?",
            "dialectical",
        )

    def _plan_next_round(self, previous_log: ThinkingLog, history: list[ThinkingLog]) -> tuple[str, str]:
        if previous_log.sub_questions:
            best_sq = self._best_follow_up_subquestion(previous_log.sub_questions)
            if best_sq:
                return best_sq, ""

        anchor_question = history[0].question if history else previous_log.question
        mission = self._mission_text() or anchor_question or previous_log.question
        recent_insights = [
            log.insight.strip()
            for log in history[-3:]
            if log.insight and log.insight.strip()
        ]
        history_text = "\n".join(f"- {item}" for item in recent_insights) or "- none"
        previous_sub_questions = json.dumps(previous_log.sub_questions[:4], indent=2)

        plan = llm_json(
            NEXT_ROUND_PROMPT.format(
                mission=mission,
                anchor_question=anchor_question,
                previous_question=previous_log.question,
                previous_pattern=previous_log.pattern or "unknown",
                previous_insight=previous_log.insight or "none",
                previous_sub_questions=previous_sub_questions,
                history=history_text,
            ),
            temperature=0.2,
            default={
                "next_question": "",
                "preferred_pattern": "",
                "goal": "",
            },
        )

        next_question = (plan.get("next_question") or "").strip()
        preferred_pattern = self._normalize_pattern_name(
            plan.get("preferred_pattern", "")
        )
        if len(next_question) < 15:
            return self._fallback_next_round(anchor_question, previous_log)
        if next_question.strip().lower() == previous_log.question.strip().lower():
            return self._fallback_next_round(anchor_question, previous_log)
        return next_question, preferred_pattern

    def _reasoning_result_from_text(self, question: str, pattern: str,
                                    reasoning: str,
                                    workspace: ScientistWorkspace) -> ReasoningResult:
        result = llm_json(
            REASONING_RESULT_PROMPT.format(
                question=question,
                pattern=pattern,
                workspace=workspace.to_prompt_context(),
                reasoning=reasoning,
            ),
            temperature=0.2,
            default={},
        )
        parsed = ReasoningResult.from_dict(result if isinstance(result, dict) else {})
        if not parsed.summary_claim:
            parsed.summary_claim = llm_call(
                THINKING_SUMMARY_PROMPT.format(reasoning=reasoning),
                temperature=0.2,
                role="precise"
            ).strip()
        return parsed

    def _reasoning_result_from_reductive(self, result: dict,
                                         focus_question: str) -> ReasoningResult:
        sub_questions = []
        for item in result.get("sub_questions", [])[:4]:
            if not isinstance(item, dict):
                continue
            question = (item.get("question") or "").strip()
            if question:
                sub_questions.append(question)

        next_actions = []
        if focus_question:
            next_actions.append(f"Investigate: {focus_question}")
        recommended_focus = (result.get("recommended_focus") or "").strip()
        if recommended_focus and recommended_focus not in next_actions:
            next_actions.append(recommended_focus)

        summary = self._format_focus_insight(focus_question, recommended_focus)
        return ReasoningResult(
            open_questions=sub_questions,
            next_actions=next_actions[:3],
            summary_claim=summary,
        )

    def _source_payload_from_workspace(self, workspace: ScientistWorkspace) -> tuple[list[str], list[str]]:
        source_ids = []
        source_refs = []
        for node in workspace.grounded_evidence:
            for source_id in node.source_ids:
                if source_id not in source_ids:
                    source_ids.append(source_id)
            for ref in node.source_refs:
                if ref not in source_refs:
                    source_refs.append(ref)
        return source_ids, source_refs

    def _classify_summary_output(self, result: ReasoningResult) -> tuple[NodeType, str]:
        if result.hypotheses:
            return NodeType.HYPOTHESIS, ArtifactStatus.SPECULATIVE.value
        if result.grounded_claims and not result.prior_claims:
            return NodeType.ANSWER, ArtifactStatus.GROUNDED.value
        if result.prior_claims and not result.grounded_claims:
            return NodeType.HYPOTHESIS, ArtifactStatus.PRIOR.value
        return NodeType.SYNTHESIS, ArtifactStatus.OPEN.value

    def _add_structured_node(self, statement: str, node_type: NodeType,
                             epistemic_status: str, importance: float = 0.65,
                             cluster: str = "thinking",
                             source_ids: list[str] | None = None,
                             source_refs: list[str] | None = None,
                             created_by: str = "thinker") -> str:
        node = Node(
            statement=statement,
            node_type=node_type,
            cluster=cluster,
            status=NodeStatus.UNCERTAIN,
            epistemic_status=epistemic_status,
            importance=importance,
            source_quality=importance,
            source_ids=list(source_ids or []),
            source_refs=list(source_refs or []),
            created_by=created_by,
        )
        nid = self.brain.add_node(node)
        if self.index:
            self.index.add(nid, shared_embed(statement))
        return nid

    def _persist_reasoning_result(self, result: ReasoningResult,
                                  workspace: ScientistWorkspace):
        for question in result.open_questions[:3]:
            nid = self._add_structured_node(
                question,
                NodeType.QUESTION,
                ArtifactStatus.OPEN.value,
                importance=0.62,
            )
            if self.observer and hasattr(self.observer, "add_to_agenda"):
                self.observer.add_to_agenda(
                    text=question,
                    item_type="question",
                    cycle=getattr(self.observer, "cycle_count", 0),
                    node_id=nid,
                )

        for action in result.next_actions[:3]:
            nid = self._add_structured_node(
                action,
                NodeType.TASK,
                ArtifactStatus.OPEN.value,
                importance=0.58,
            )
            if self.observer and hasattr(self.observer, "add_to_agenda"):
                item = self.observer.add_to_agenda(
                    text=action,
                    item_type="task",
                    cycle=getattr(self.observer, "cycle_count", 0),
                    node_id=nid,
                )
                item.priority = max(item.priority, 0.6)

    def think(self, question: str = None, pattern: str = None,
              max_depth: int = 2) -> ThinkingLog:
        """
        Run a deliberate thinking session.

        Args:
            question: Topic to think about (auto-picks if None)
            pattern: Reasoning pattern to use (auto-picks if None)
            max_depth: For reductive thinking, how many levels deep

        Returns:
            ThinkingLog with the reasoning and any insights produced
        """
        start = time.time()
        log = ThinkingLog()

        # Pick question
        question_node_id = ''
        preferred_pattern = ''
        if not question:
            question, question_node_id, preferred_pattern = self._pick_question()
        log.question = question
        log.question_node_id = question_node_id
        print(f"\n── Thinking: {question[:80]}... ──")

        # Pick pattern
        if not pattern:
            if preferred_pattern and preferred_pattern in SUPPORTED_THINKING_PATTERNS:
                # Use the pattern suggested by _pick_question (e.g., hypothesis_testing)
                log.node_type, log.cluster, pattern = 'hypothesis', 'unclustered', preferred_pattern
                # Try to get cluster from the node
                if question_node_id:
                    node_data = self.brain.get_node(question_node_id)
                    if node_data:
                        log.node_type = node_data.get('node_type', 'hypothesis')
                        log.cluster = node_data.get('cluster', 'unclustered')
            else:
                log.node_type, log.cluster, pattern = self._pick_pattern(question)
        else:
            log.node_type, log.cluster = "question", "unclustered"
            
        log.pattern = pattern
        print(f"  Pattern: {pattern}")

        # Build workspace-aware context
        workspace = self._build_workspace(question)
        log.workspace = workspace.to_dict()
        context = self._build_context(question, workspace)

        # Run the appropriate reasoning pattern
        prompts = {
            "dialectical":       DIALECTICAL_PROMPT,
            "analogical":        ANALOGICAL_PROMPT,
            "reductive":         REDUCTIVE_PROMPT,
            "experimental":      EXPERIMENTAL_PROMPT,
            "integrative":       INTEGRATIVE_PROMPT,
            "hypothesis_testing": HYPOTHESIS_TESTING_PROMPT,
        }

        prompt = prompts.get(pattern, DIALECTICAL_PROMPT)

        # ── Hypothesis testing: mark the hypothesis node as TESTING ──
        if pattern == 'hypothesis_testing' and question_node_id:
            self.brain.update_node(
                question_node_id,
                status=NodeStatus.TESTING.value,
            )
            print(f"  Hypothesis [{question_node_id[:8]}] → TESTING")

        reasoning_result = ReasoningResult()

        if pattern == "hypothesis_testing":
            # Hypothesis testing returns structured JSON with search queries
            result = llm_json(
                prompt.format(question=question, context=context),
                temperature=0.3,
                default={"sub_claims": [], "overall_assessment": ""}
            )
            log.sub_questions = result.get("sub_claims", [])
            log.reasoning = json.dumps(result, indent=2)

            # Extract search queries as next_actions
            search_queries = [
                sc.get('search_query', '')
                for sc in log.sub_questions
                if sc.get('search_query')
            ]
            overall = result.get('overall_assessment', '')
            reasoning_result = ReasoningResult(
                hypotheses=[question],
                open_questions=[sc.get('claim', '') for sc in log.sub_questions if sc.get('claim')],
                next_actions=search_queries[:4],
                summary_claim=overall or f"Hypothesis under test: {question[:100]}",
            )
            log.reasoning_result = reasoning_result.to_dict()
            log.insight = overall

            print(f"  Sub-claims: {len(log.sub_questions)}")
            for sc in log.sub_questions[:4]:
                print(f"    → {sc.get('claim', '')[:60]}")
                print(f"      Search: {sc.get('search_query', '')}")

        elif pattern == "reductive":
            # Reductive returns structured JSON
            result = llm_json(
                prompt.format(question=question, context=context),
                temperature=0.4,
                default={"sub_questions": [], "recommended_focus": ""}
            )
            log.sub_questions = result.get("sub_questions", [])
            log.reasoning = json.dumps(result, indent=2)

            # Add sub-questions to the graph and agenda
            for sq in log.sub_questions[:4]:
                q_text = sq.get("question", "")
                if not q_text:
                    continue
                nid = self._add_structured_node(
                    q_text,
                    NodeType.QUESTION,
                    ArtifactStatus.OPEN.value,
                    importance=0.65,
                )

                # Add to agenda
                if self.observer and hasattr(self.observer, 'add_to_agenda'):
                    item = self.observer.add_to_agenda(
                        text=q_text,
                        item_type="question",
                        cycle=getattr(self.observer, 'cycle_count', 0),
                        node_id=nid
                    )
                    # Higher leverage = higher priority
                    leverage = sq.get("leverage", "medium")
                    if leverage == "high":
                        item.priority = 0.8

                print(f"  Sub-question: {q_text[:60]}...")

            # Convert the model's recommended focus into a stable chosen focus.
            focus = result.get("recommended_focus", "")
            focus_question = self._select_focus_subquestion(
                question,
                log.sub_questions,
                focus,
            )
            reasoning_result = self._reasoning_result_from_reductive(
                result,
                focus_question,
            )

            for action in reasoning_result.next_actions[:3]:
                nid = self._add_structured_node(
                    action,
                    NodeType.TASK,
                    ArtifactStatus.OPEN.value,
                    importance=0.58,
                )
                if self.observer and hasattr(self.observer, "add_to_agenda"):
                    item = self.observer.add_to_agenda(
                        text=action,
                        item_type="task",
                        cycle=getattr(self.observer, "cycle_count", 0),
                        node_id=nid,
                    )
                    item.priority = max(item.priority, 0.6)
        else:
            # Other patterns return free-form reasoning
            log.reasoning = llm_call(
                prompt.format(question=question, context=context),
                temperature=0.5,
                role="reasoning"
            )
            reasoning_result = self._reasoning_result_from_text(
                question,
                pattern,
                log.reasoning,
                workspace,
            )
            self._persist_reasoning_result(reasoning_result, workspace)

        log.reasoning_result = reasoning_result.to_dict()
        log.insight = reasoning_result.summary_claim

        # ── System 2 gating ──
        # Route insight through Critic before graph insertion
        if log.insight and len(log.insight) > 15:
            summary_node_type, summary_status = self._classify_summary_output(
                reasoning_result
            )
            source_ids, source_refs = self._source_payload_from_workspace(workspace)
            if self.critic:
                from critic.critic import CandidateThought, Verdict
                candidate = CandidateThought(
                    claim         = log.insight,
                    source_module = "thinker",
                    proposed_type = summary_node_type.value,
                    importance    = 0.7,
                    context       = log.reasoning,
                    grounded_evidence=source_refs,
                    source_ids    = source_ids,
                    expected_status=summary_status,
                )
                critic_log = self.critic.evaluate_with_refinement(candidate)
                final_claim = critic_log.final_claim or candidate.claim

                if critic_log.verdict == Verdict.ACCEPT:
                    reward = 1.0 * critic_log.confidence
                    # Use critic-assigned confidence instead of default
                    confidence  = critic_log.confidence
                    log.insight = final_claim
                    nid = self._add_structured_node(
                        final_claim,
                        summary_node_type,
                        summary_status,
                        importance=confidence,
                        source_ids=source_ids,
                        source_refs=source_refs,
                    )
                    log.node_id = nid
                    self.brain.focus_on(nid)
                    print(f"  ✓ Insight accepted (conf={confidence:.2f}): "
                          f"{final_claim[:80]}...")

                elif critic_log.verdict == Verdict.REJECT:
                    reward = -1.0
                    print(f"  ✗ Insight rejected: {critic_log.rejection_reason}")
                    log.insight = ""  # clear so callers know it was rejected

                elif critic_log.verdict == Verdict.DEFER:
                    reward = 0.0
                    print(f"  ◇ Insight deferred to insight buffer")
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
                    log.insight = ""  # clear — not in graph yet

                else:  # REFINE exhausted → treated as DEFER by evaluate_with_refinement
                    reward = 0.0
                    print(f"  ◇ Insight deferred after refinement")
                    log.insight = ""

                # Train RL Policy
                self.policy.update(log.node_type, log.cluster, log.pattern, reward, self.brain.dopamine)

            else:
                self.policy.update(log.node_type, log.cluster, log.pattern, 0.5, self.brain.dopamine)
                # No critic — original behavior (direct insertion)
                nid = self._add_structured_node(
                    log.insight,
                    summary_node_type,
                    summary_status,
                    importance=0.7,
                    source_ids=source_ids,
                    source_refs=source_refs,
                )
                log.node_id = nid
                self.brain.focus_on(nid)
                print(f"  Insight: {log.insight[:80]}...")

        log.duration = time.time() - start
        print(f"  Thinking complete ({log.duration:.1f}s)")

        return log

    def think_session(self, num_rounds: int = 3) -> list[ThinkingLog]:
        """
        Run multiple rounds of thinking, each building on the last.

        Round 1: Pick a question and think about it
        Round 2+: Either refine the previous insight or pivot to a related question
        """
        logs = []
        print(f"\n══ Thinking session — {num_rounds} rounds ══")

        for i in range(num_rounds):
            print(f"\n── Round {i+1}/{num_rounds} ──")

            if i == 0:
                log = self.think()
            else:
                prev = logs[-1]
                next_question, preferred_pattern = self._plan_next_round(prev, logs)
                log = self.think(
                    question=next_question or None,
                    pattern=preferred_pattern or None,
                )

            logs.append(log)

        print(f"\n══ Thinking session complete — {len(logs)} rounds ══")
        return logs
