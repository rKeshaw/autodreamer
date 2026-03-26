import random
import time
import json
import re
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from ollama import Client
from graph.brain import (Brain, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType, BrainMode, ANALOGY_WEIGHTS)
from config import THRESHOLDS
from embedding import embed as shared_embed

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL        = "llama3.1:8b"
DEFAULT_STEPS       = 20
DEFAULT_TEMP        = 0.7
DEPTH_STEPS         = 3
VISITED_PENALTY     = 0.25

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

Narrate this mental journey in 2-4 sentences. Think like a scientist half-asleep —
associative, curious. Let the central question color your thinking without forcing it.

If something is unresolved, ask ONE question starting with "Q:"

Classify any insight:
- INSIGHT: surface — shared vocabulary or theme only
- INSIGHT: structural — same relational pattern between domains
- INSIGHT: isomorphism — formal mathematical or logical equivalence
- INSIGHT: none
"""

NARRATION_WANDERING = """
You are a dreaming scientific mind — no particular agenda, just following curiosity.

From: "{from_node}"
Via [{edge_type}]: "{edge_narration}"
To: "{to_node}"

Narrate this mental journey in 2-4 sentences. Let your mind wander freely —
no destination, just association. Be playful, unexpected, open.

If something is intriguing, ask ONE question starting with "Q:"

Classify any insight:
- INSIGHT: surface
- INSIGHT: structural
- INSIGHT: isomorphism
- INSIGHT: none
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

Narrate in 2-4 sentences. Make unexpected connections. Be wild but coherent.

If something sparks, ask ONE question starting with "Q:"

Classify any insight:
- INSIGHT: surface
- INSIGHT: structural
- INSIGHT: isomorphism
- INSIGHT: none
"""

MISSION_ADVANCE_PROMPT = """
Central research question: "{mission}"
New idea encountered: "{node}"
Connection made: "{narration}"

Does this meaningfully advance the central question?

Respond with JSON:
{{
  "advances": true or false,
  "explanation": "one sentence",
  "strength": 0.0 to 1.0
}}
"""

ANSWER_CHECK_PROMPT = """
Current idea: {current_node}
Open question: {question}

Does the current idea answer or significantly advance this question?

Respond with JSON:
{{
  "match": "none" | "partial" | "strong",
  "explanation": "one sentence"
}}
"""

QUESTION_SIMILAR_PROMPT = """
Are these two questions essentially the same?
Q1: {q1}
Q2: {q2}
Respond ONLY "yes" or "no".
"""

DEPTH_NARRATION_PROMPT = """
You are a scientific mind that found something significant while dreaming.
{mission_line}
You landed on: "{node}"
It connects to: "{question}"
Connection: {explanation}

Explore this in 2-3 sentences. End with one follow-up question starting with "Q:"
"""

SUMMARY_FOCUSED = """
You are summarizing a dream cycle. Brain mode: FOCUSED.
Central question: "{mission}"

Dream steps: {steps}
Answer matches: {answers}
Mission advances: {mission_advances}

Write a 4-6 sentence morning notebook entry addressing:
1. What the dream explored
2. Key connections made
3. Whether the central question was advanced
4. Most pressing open questions

First person. Sign off: — THE SCIENTIST
"""

SUMMARY_WANDERING = """
You are summarizing a dream cycle. Brain mode: WANDERING (no mission — free association).

Dream steps: {steps}

Write a 4-6 sentence morning notebook entry:
1. What the dream explored
2. What surprised you
3. Any unexpected connections
4. What you find yourself curious about

First person. Playful tone. Sign off: — THE SCIENTIST
"""

SUMMARY_TRANSITIONAL = """
You are summarizing a dream cycle. Brain mode: TRANSITIONAL.
A new question just arrived: "{mission}"
The mind is reorganizing itself.

Dream steps: {steps}

Write a 4-6 sentence entry capturing the chaotic reorientation — ideas flying together,
new connections forming rapidly, the mind finding its new gravitational center.
Sign off: — THE SCIENTIST
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

class Dreamer:
    def __init__(self, brain: Brain, research_agenda=None):
        self.brain           = brain
        self.llm             = Client()
        self.research_agenda = research_agenda
        self._embedding_cache = {}

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

    # ── Node selection ────────────────────────────────────────────────────────

    def _select_start_node(self, mode, seed_id=None):
        nodes = self.brain.all_nodes()
        if not nodes:
            raise ValueError("Brain is empty")

        if mode == DreamMode.SEEDED and seed_id:
            return seed_id

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

    def _score_edge(self, edge_data, temperature, scientificness, visited, target_id):
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
            weight -= VISITED_PENALTY

        noise = random.gauss(0, temperature * 0.3)
        return max(0.001, weight + noise)

    # ── Single hop ────────────────────────────────────────────────────────────

    def _hop(self, current_id, temperature, scientificness, visited):
        neighbors = self.brain.neighbors(current_id)
        if not neighbors:
            all_ids   = [nid for nid, _ in self.brain.all_nodes()]
            unvisited = [n for n in all_ids if n not in visited]
            return random.choice(unvisited if unvisited else all_ids), None

        scored = []
        for nid in neighbors:
            edge = self.brain.get_edge(current_id, nid)
            if edge:
                score = self._score_edge(
                    edge, temperature, scientificness, visited, nid)
                scored.append((nid, edge, score))

        if not scored:
            return random.choice(neighbors), None

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
                result = json.loads(raw)
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
        raw = self._llm(MISSION_ADVANCE_PROMPT.format(
            mission=mission['question'],
            node=node_data['statement'],
            narration=narration), temperature=0.4)
        try:
            result = json.loads(raw)
            if result.get('advances') and result.get('strength', 0) > 0.5:
                return True, result.get('explanation', ''), result.get('strength', 0.5)
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
        print(f"      ↳ Depth [{DEPTH_STEPS} steps]")
        mission = self._mission_text()
        mission_line = f"Central question: \"{mission}\"" if mission else ""

        raw = self._llm(DEPTH_NARRATION_PROMPT.format(
            mission_line=mission_line,
            node=node_data['statement'],
            question=question,
            explanation=explanation), temperature=0.85)
        lines  = raw.strip().split('\n')
        q_line = next((l for l in lines if l.startswith('Q:')), "")
        followup = q_line[2:].strip() if q_line else ""
        self._add_question(followup, questions, q_embeddings)

        current_id, current_data = node_id, node_data
        for d in range(DEPTH_STEPS):
            next_id, edge = self._hop(
                current_id, temperature * 0.5, scientificness, visited)
            next_data = self.brain.get_node(next_id)
            if not next_data:
                continue
            edge_type      = edge.get('type', 'associated') if edge else 'associated'
            edge_narration = edge.get('narration', '') if edge else ''
            raw = self._llm(self._narration_prompt(
                current_data['statement'], edge_type,
                edge_narration, next_data['statement']), temperature=0.85)
            narration, _, is_insight, depth = self._parse_narration(raw)
            ds = DreamStep(
                step=step_offset+d, from_id=current_id, to_id=next_id,
                edge_type=edge_type, edge_narration=edge_narration,
                narration=narration, is_insight=is_insight,
                insight_depth=depth)
            log.steps.append(ds)
            visited.add(next_id)
            self.brain.update_node(next_id, activated_at=time.time())
            current_id, current_data = next_id, next_data
            print(f"      depth {d+1}: {next_data['statement']}")
        return current_id

    # ── NREM ─────────────────────────────────────────────────────────────────

    def nrem_pass(self):
        print("\n── NREM pass ──")
        self.brain.proximal_reinforce()
        print("── NREM complete ──\n")

    # ── Main dream loop ───────────────────────────────────────────────────────

    def dream(self, mode=DreamMode.WANDERING, steps=DEFAULT_STEPS,
              temperature=DEFAULT_TEMP, seed_id=None,
              run_nrem=True, log_path="logs/dream_latest.json"):

        brain_mode    = self.brain.get_mode()
        scientificness= self.brain.scientificness

        # mode modifiers
        temperature += MODE_TEMP_BOOST.get(brain_mode, 0)
        steps       += MODE_STEPS_BOOST.get(brain_mode, 0)

        log          = DreamLog(mode=mode.value, brain_mode=brain_mode)
        visited      = set()
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
        while step < steps:
            next_id, edge = self._hop(current_id, temperature,
                                      scientificness, visited)
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
                current['statement'], edge_type,
                edge_narration, next_node['statement']), temperature=0.85)
            narration, question, is_insight, insight_depth = \
                self._parse_narration(raw)

            self._add_question(question, questions, q_embeddings)

            match_grade, match_explanation, matched_q = \
                self._check_answers(next_id, next_node)

            mission_advance = False
            mission_explanation = ""
            mission_strength = 0.0
            if is_insight or match_grade == 'strong':
                mission_advance, mission_explanation, mission_strength = \
                    self._check_mission_advance(next_node, narration)

            # depth
            depth_triggered = False
            if match_grade in ['partial', 'strong'] and matched_q:
                depth_triggered = True
                current_id = self._depth_explore(
                    next_id, next_node, matched_q, match_explanation,
                    temperature, scientificness, visited, log,
                    questions, q_embeddings, step + 1000)
                current = self.brain.get_node(current_id)
                visited.add(current_id)
            elif (is_insight or
                  next_node.get('status') == NodeStatus.CONTRADICTED.value or
                  next_node.get('incubation_age', 0) > 3 or
                  mission_advance):
                depth_triggered = True
                current_id = self._depth_explore(
                    next_id, next_node,
                    "An interesting connection worth exploring.",
                    mission_explanation or "Insight or tension detected.",
                    temperature, scientificness, visited, log,
                    questions, q_embeddings, step + 1000)
                current = self.brain.get_node(current_id)
                visited.add(current_id)

            # new edge on insight
            new_edge = False
            if is_insight and insight_depth:
                type_map = {
                    "surface":     EdgeType.SURFACE_ANALOGY,
                    "structural":  EdgeType.STRUCTURAL_ANALOGY,
                    "isomorphism": EdgeType.DEEP_ISOMORPHISM,
                }
                etype = type_map.get(insight_depth, EdgeType.STRUCTURAL_ANALOGY)
                if not (self.brain.graph.has_edge(current_id, next_id) or
                        self.brain.graph.has_edge(next_id, current_id)):
                    dream_edge = Edge(
                        type=etype, narration=narration,
                        weight=ANALOGY_WEIGHTS.get(etype, 0.4),
                        confidence=0.45, source=EdgeSource.DREAM,
                        analogy_depth=insight_depth)
                    self.brain.add_edge(current_id, next_id, dream_edge)
                    new_edge = True
                self.brain.restructure_around_insight(
                    current_id, next_id, narration, edge_type=etype.value)
                log.insights.append({
                    "step": step, "from": current['statement'],
                    "to": next_node['statement'],
                    "narration": narration, "depth": insight_depth,
                    "mission_linked": mission_advance
                })

            if match_grade != 'none':
                log.answers.append({
                    "step": step, "node": next_id,
                    "question": matched_q, "grade": match_grade,
                    "explanation": match_explanation
                })
                if self.research_agenda:
                    self.research_agenda.record_answer(
                        matched_q, next_id, match_explanation,
                        grade=match_grade)

            if mission_advance:
                self.brain.link_to_mission(
                    next_id, f"Dream insight: {mission_explanation}",
                    strength=mission_strength)
                log.mission_advances.append({
                    "step": step, "node": next_id,
                    "explanation": mission_explanation,
                    "strength": mission_strength
                })
                if self.research_agenda:
                    self.research_agenda.record_mission_advance(
                        next_id, mission_explanation, mission_strength)

            ds = DreamStep(
                step=step, from_id=current_id, to_id=next_id,
                edge_type=edge_type, edge_narration=edge_narration,
                narration=narration, question=question,
                is_insight=is_insight, insight_depth=insight_depth,
                new_edge=new_edge, answer_match=match_grade,
                answer_detail=match_explanation,
                depth_triggered=depth_triggered,
                mission_advance=mission_advance)
            log.steps.append(ds)
            self.brain.update_node(next_id, activated_at=time.time())
            visited.add(next_id)

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
