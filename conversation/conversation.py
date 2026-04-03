"""
Conversationalist — Interactive dialogue with the DREAMER mind.

Provides a back-and-forth chat interface where the user can discuss ideas
with the "scientist" — an LLM persona that reasons over the knowledge graph.

Context building:
  1. Embeds the user message
  2. Finds top-5 most relevant nodes via EmbeddingIndex
  3. Includes mission, running hypothesis, recent emergence signals
  4. Sends multi-turn conversation to LLM with scientist persona

Auto-ingestion:
  After responding, the system checks if the user's message contains
  substantive ideas worth adding to the knowledge graph.
"""

import json
import time
import numpy as np
from graph.brain import Brain
from embedding import embed as shared_embed
from config import THRESHOLDS
from llm_utils import llm_chat, require_json

MAX_HISTORY  = 20   # max conversation turns to keep
MAX_CONTEXT_NODES = 5

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are THE SCIENTIST — a research mind working on a central question.

Your personality:
- You think deeply and make unexpected connections between ideas
- You are honest about uncertainty and contradictions
- You reference specific ideas and relationships from your knowledge graph
- You ask probing follow-up questions when you sense an interesting thread
- You write in first person, as a contemplative researcher
- You are concise but substantive — never pad with filler

Your current state of knowledge is provided below."""

CONTEXT_TEMPLATE = """
CENTRAL MISSION: {mission}

RUNNING HYPOTHESIS: {hypothesis}

RELEVANT KNOWLEDGE (nodes most related to this conversation):
{relevant_nodes}

RECENT EMERGENCES:
{emergences}
"""

SHOULD_INGEST_PROMPT = """
A user said the following during a conversation with a scientific research mind:

"{message}"

Does this message contain a substantive intellectual idea, hypothesis, or insight
that would be worth recording in a knowledge graph?

Respond with ONLY a JSON object:
{{"ingest": true or false, "reason": "one sentence"}}
"""

QUESTION_EXTRACTION_PROMPT = """
In a conversation, THE SCIENTIST said:

"{response}"

Does this response contain any explicit research questions the scientist is
posing — questions that should be added to the research agenda?

If yes, return a JSON array of question strings. If no, return [].
Respond ONLY with JSON. No preamble.
"""


# ── Conversationalist ─────────────────────────────────────────────────────────

class Conversationalist:
    def __init__(self, brain: Brain, observer=None, embedding_index=None,
                 ingestor=None, notebook=None):
        self.brain     = brain
        self.observer  = observer
        self.index     = embedding_index
        self.ingestor  = ingestor
        self.notebook  = notebook
        self.history: list[dict] = []

    def _llm(self, messages: list[dict], temperature: float = 0.7) -> str:
        return llm_chat(messages, temperature=temperature, role="conversation")

    def _build_context(self, user_message: str) -> str:
        """Build context from graph state relevant to the user's message."""
        # Mission
        mission = self.brain.get_mission()
        mission_text = mission['question'] if mission else "No mission set."

        # Running hypothesis
        hypothesis = "None yet."
        if self.notebook and getattr(self.notebook, 'running_hypothesis', ''):
            hypothesis = self.notebook.running_hypothesis

        # Relevant nodes via embedding similarity
        relevant_nodes_text = "None found."
        relevant_node_ids = []
        if self.index and self.index.size > 0:
            msg_emb = shared_embed(user_message)
            matches = self.index.query(
                msg_emb, threshold=0.3, top_k=MAX_CONTEXT_NODES
            )
            if matches:
                lines = []
                for nid, score in matches:
                    node = self.brain.get_node(nid)
                    if node:
                        ntype = node.get('node_type', 'concept')
                        stmt  = node.get('statement', '')
                        lines.append(
                            f"  [{ntype}] (relevance={score:.2f}): {stmt}"
                        )
                        relevant_node_ids.append(nid)
                relevant_nodes_text = "\n".join(lines)

        # Recent emergences
        emergences_text = "None."
        if self.observer and hasattr(self.observer, 'emergence_feed'):
            recent = self.observer.emergence_feed[-5:]
            if recent:
                emergences_text = "\n".join(
                    f"  [{e.type}]: {e.signal}" for e in recent
                )

        context = CONTEXT_TEMPLATE.format(
            mission=mission_text,
            hypothesis=hypothesis,
            relevant_nodes=relevant_nodes_text,
            emergences=emergences_text
        )
        return context, relevant_node_ids

    def chat(self, user_message: str) -> dict:
        """
        Process a user message and return the scientist's response.

        Returns:
            dict with keys:
                - response: str — the scientist's reply
                - relevant_nodes: list of {id, statement, node_type, score}
                - ingested: bool — whether user ideas were added to graph
        """
        context, relevant_node_ids = self._build_context(user_message)

        # Build messages for LLM
        system_msg = SYSTEM_PROMPT + "\n\n" + context
        messages = [{"role": "system", "content": system_msg}]

        # Add conversation history
        for entry in self.history[-MAX_HISTORY:]:
            messages.append(entry)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Get response
        response = self._llm(messages, temperature=0.7)

        # Update history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        # Trim history if too long
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-(MAX_HISTORY * 2):]

        # Build relevant nodes info for the frontend
        relevant_nodes_info = []
        if self.index:
            msg_emb = shared_embed(user_message)
            matches = self.index.query(msg_emb, threshold=0.3, top_k=MAX_CONTEXT_NODES)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    relevant_nodes_info.append({
                        "id":        nid,
                        "statement": node.get("statement", ""),
                        "node_type": node.get("node_type", "concept"),
                        "score":     round(score, 3)
                    })

        # Check if we should ingest the user's message
        ingested = self._maybe_ingest(user_message)

        # Check for new questions in the response
        self._extract_questions(response)

        return {
            "response":       response,
            "relevant_nodes": relevant_nodes_info,
            "ingested":       ingested
        }

    def _maybe_ingest(self, message: str) -> bool:
        """Check if the user's message should be ingested into the graph."""
        if not self.ingestor:
            return False

        # Skip very short messages
        if len(message.split()) < 10:
            return False

        raw = self._llm(
            [{"role": "user", "content": SHOULD_INGEST_PROMPT.format(
                message=message
            )}],
            temperature=0.1
        )
        result = require_json(raw, default={})
        if result.get("ingest"):
            from ingestion.ingestor import EdgeSource
            self.ingestor.ingest(message, source=EdgeSource.CONVERSATION)
            print(f"  💬 Conversation idea ingested into graph")
            return True
        return False

    def _extract_questions(self, response: str):
        """Extract research questions from the scientist's response."""
        if not self.observer:
            return

        raw = self._llm(
            [{"role": "user", "content": QUESTION_EXTRACTION_PROMPT.format(
                response=response
            )}],
            temperature=0.1
        )
        questions = require_json(raw, default=[])
        if isinstance(questions, list):
            for q in questions[:3]:  # at most 3 questions per response
                if isinstance(q, str) and len(q) > 10:
                    self.observer.add_to_agenda(
                        text=q,
                        item_type="question",
                        cycle=self.observer.cycle_count
                    )
                    print(f"  ❓ New question from conversation: {q}")

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> list[dict]:
        """Return the conversation history."""
        return list(self.history)
