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
import time
from dataclasses import dataclass, field
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeType, NodeStatus)
from llm_utils import llm_call, llm_json, llm_chat
from embedding import embed as shared_embed

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

PICK_PATTERN_PROMPT = """You are selecting a reasoning strategy for a scientific question.

QUESTION: {question}

Available strategies:
- dialectical: weigh evidence for and against (best for contested claims)
- analogical: find parallels in other domains (best for novel problems)
- reductive: break into sub-questions (best for complex, multi-part problems)
- experimental: design thought experiments (best for testable hypotheses)
- integrative: find unifying principles (best when many related facts exist)

Which strategy is BEST for this question? Respond with ONLY the strategy name.
"""

THINKING_SUMMARY_PROMPT = """Summarize the key insight from this thinking session in 1-2 sentences.
This will be stored as a new node in the knowledge graph.

Thinking session:
{reasoning}

Respond with ONLY the insight statement. No preamble.
"""

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ThinkingLog:
    question: str          = ""
    pattern: str           = ""
    reasoning: str         = ""
    insight: str           = ""
    sub_questions: list    = field(default_factory=list)
    node_id: str           = ""
    started_at: float      = field(default_factory=time.time)
    duration: float        = 0.0

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

    def _build_context(self, question: str, max_nodes: int = 8) -> str:
        """Build relevant context from the graph for a given question."""
        lines = []

        # From embedding index
        if self.index and self.index.size > 0:
            q_emb = shared_embed(question)
            matches = self.index.query(q_emb, threshold=0.25, top_k=max_nodes)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    ntype = node.get('node_type', 'concept')
                    status = node.get('status', 'uncertain')
                    lines.append(
                        f"[{ntype}/{status}] {node['statement']}"
                    )

        # Working memory items (always included)
        for nid, data in self.brain.get_working_memory():
            line = f"[FOCUS/{data.get('node_type','concept')}] {data['statement']}"
            if line not in lines:
                lines.append(line)

        # Mission context
        mission = self.brain.get_mission()
        if mission:
            lines.insert(0, f"[MISSION] {mission['question']}")

        return "\n\n".join(lines) if lines else "No relevant knowledge found."

    def _pick_question(self) -> str:
        """Pick the best question to think about from the agenda/graph."""
        # Priority: working memory hypotheses > agenda questions > high-importance gaps
        for nid, data in self.brain.get_working_memory():
            if data.get('node_type') in [NodeType.HYPOTHESIS.value,
                                          NodeType.QUESTION.value,
                                          NodeType.GAP.value]:
                return data['statement']

        # From observer agenda
        if self.observer and hasattr(self.observer, 'agenda'):
            open_items = [
                item for item in self.observer.agenda
                if item.status == 'open'
            ]
            if open_items:
                # Pick highest priority
                best = max(open_items, key=lambda x: x.priority)
                return best.text

        # From graph — highest importance unresolved question
        questions = self.brain.nodes_by_type(NodeType.QUESTION)
        gaps      = self.brain.nodes_by_type(NodeType.GAP)
        hyps      = self.brain.nodes_by_type(NodeType.HYPOTHESIS)

        candidates = questions + gaps + hyps
        if candidates:
            best = max(candidates,
                       key=lambda x: x[1].get('importance', 0.5))
            return best[1]['statement']

        # Fallback: think about the mission
        mission = self.brain.get_mission()
        if mission:
            return mission['question']

        return "What is the most important open question in our knowledge?"

    def _pick_pattern(self, question: str) -> str:
        """Let the LLM choose the best reasoning pattern."""
        raw = llm_call(
            PICK_PATTERN_PROMPT.format(question=question),
            temperature=0.1,
            role="reasoning"
        ).strip().lower()

        valid = ["dialectical", "analogical", "reductive",
                 "experimental", "integrative"]
        for v in valid:
            if v in raw:
                return v
        return "dialectical"  # safe default

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
        if not question:
            question = self._pick_question()
        log.question = question
        print(f"\n── Thinking: {question[:80]}... ──")

        # Pick pattern
        if not pattern:
            pattern = self._pick_pattern(question)
        log.pattern = pattern
        print(f"  Pattern: {pattern}")

        # Build context
        context = self._build_context(question)

        # Run the appropriate reasoning pattern
        prompts = {
            "dialectical":  DIALECTICAL_PROMPT,
            "analogical":   ANALOGICAL_PROMPT,
            "reductive":    REDUCTIVE_PROMPT,
            "experimental": EXPERIMENTAL_PROMPT,
            "integrative":  INTEGRATIVE_PROMPT,
        }

        prompt = prompts.get(pattern, DIALECTICAL_PROMPT)

        if pattern == "reductive":
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
                node = Node(
                    statement    = q_text,
                    node_type    = NodeType.QUESTION,
                    cluster      = "thinking",
                    status       = NodeStatus.UNCERTAIN,
                    importance   = 0.65,
                    source_quality = 0.6
                )
                nid = self.brain.add_node(node)
                if self.index:
                    self.index.add(nid, shared_embed(q_text))

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

            # Summarize the recommended focus
            focus = result.get("recommended_focus", "")
            if focus:
                log.insight = focus
        else:
            # Other patterns return free-form reasoning
            log.reasoning = llm_call(
                prompt.format(question=question, context=context),
                temperature=0.5,
                role="reasoning"
            )

            # Extract insight as a storable statement
            log.insight = llm_call(
                THINKING_SUMMARY_PROMPT.format(reasoning=log.reasoning),
                temperature=0.2,
                role="precise"
            )

        # ── System 2 gating ──
        # Route insight through Critic before graph insertion
        if log.insight and len(log.insight) > 15:
            if self.critic:
                from critic.critic import CandidateThought, Verdict
                candidate = CandidateThought(
                    claim         = log.insight,
                    source_module = "thinker",
                    proposed_type = "synthesis",
                    importance    = 0.7,
                    context       = log.reasoning,
                )
                critic_log = self.critic.evaluate_with_refinement(candidate)

                if critic_log.verdict == Verdict.ACCEPT:
                    # Use critic-assigned confidence instead of default
                    final_claim = log.insight
                    confidence  = critic_log.confidence
                    node = Node(
                        statement      = final_claim,
                        node_type      = NodeType.SYNTHESIS,
                        cluster        = "thinking",
                        status         = NodeStatus.UNCERTAIN,
                        importance     = confidence,
                        source_quality = confidence
                    )
                    nid = self.brain.add_node(node)
                    log.node_id = nid
                    if self.index:
                        self.index.add(nid, shared_embed(final_claim))
                    self.brain.focus_on(nid)
                    print(f"  ✓ Insight accepted (conf={confidence:.2f}): "
                          f"{final_claim[:80]}...")

                elif critic_log.verdict == Verdict.REJECT:
                    print(f"  ✗ Insight rejected: {critic_log.rejection_reason}")
                    log.insight = ""  # clear so callers know it was rejected

                elif critic_log.verdict == Verdict.DEFER:
                    print(f"  ◇ Insight deferred to insight buffer")
                    self.critic.route_deferred(candidate)
                    log.insight = ""  # clear — not in graph yet

                else:  # REFINE exhausted → treated as DEFER by evaluate_with_refinement
                    print(f"  ◇ Insight deferred after refinement")
                    log.insight = ""

            else:
                # No critic — original behavior (direct insertion)
                node = Node(
                    statement      = log.insight,
                    node_type      = NodeType.SYNTHESIS,
                    cluster        = "thinking",
                    status         = NodeStatus.UNCERTAIN,
                    importance     = 0.7,
                    source_quality = 0.6
                )
                nid = self.brain.add_node(node)
                log.node_id = nid
                if self.index:
                    self.index.add(nid, shared_embed(log.insight))
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
                # Use previous insight as seed for next round
                prev = logs[-1]
                if prev.sub_questions:
                    # Follow up on a sub-question
                    best_sq = prev.sub_questions[0].get("question", "")
                    if best_sq:
                        log = self.think(question=best_sq)
                    else:
                        log = self.think()
                elif prev.insight:
                    # Refine the previous insight
                    follow_up = f"Given that '{prev.insight}', " \
                                f"what does this imply for our central mission?"
                    log = self.think(question=follow_up)
                else:
                    log = self.think()

            logs.append(log)

        print(f"\n══ Thinking session complete — {len(logs)} rounds ══")
        return logs
