import json
import time
import requests
from dataclasses import dataclass, field
import re
import numpy as np
from urllib.parse import urlparse
from graph.brain import Brain, EdgeSource, Edge, EdgeType, NodeStatus, NodeType
from scientist_workspace import ArtifactStatus
from scientific_rigor import HIGH_RIGOR_DOMAINS, LOW_RIGOR_DOMAINS, highest_reference_rigor
from ingestion.ingestor import Ingestor
from persistence import atomic_write_json
from llm_utils import llm_call, require_json

# ── Config ────────────────────────────────────────────────────────────────────
 
MAX_QUESTIONS_PER_DAY = 5      # how many agenda items to research per cycle
MAX_RESULTS_PER_QUERY = 3      # web results per search query
MAX_ARXIV_RESULTS    = 2       # arXiv papers per query
RESEARCH_DEPTH       = "standard"  # "shallow", "standard", "deep"
MIN_TEXT_LENGTH      = 200     # ignore snippets shorter than this
MIN_SOURCE_QUALITY   = 0.45
MAX_FINDINGS_PER_QUESTION = 6

# ── Research depth profiles ───────────────────────────────────────────────────
 
DEPTH_PROFILES = {
    "shallow":  {"web": True, "arxiv": False, "queries_per_q": 1},
    "standard": {"web": True, "arxiv": True,  "queries_per_q": 2},
    "deep":     {"web": True, "arxiv": True,  "queries_per_q": 3},
}

# ── Prompts ───────────────────────────────────────────────────────────────────
 
QUERY_GENERATION_PROMPT = """
You are a scientific researcher generating search queries for a question.
 
Question to research: {question}
Mission: {mission}

Current grounded evidence and open state:
{workspace}
 
Generate {n} distinct search queries that would help answer this question.
Each query should approach the question from a different angle.
Keep each query concise — 4 to 8 words.

You MAY use scientific background knowledge to propose smart search angles,
but treat that background as planning prior, not as evidence.

CRITICAL: If the question spans TWO domains (e.g. biology and machine learning),
at least one query MUST explicitly mention BOTH domains or use bridging language
that connects them. Do NOT write queries that cover only one side.

Example for "How does synaptic consolidation relate to experience replay in RL?":
  GOOD: ["synaptic consolidation experience replay comparison", "hippocampal replay reinforcement learning"]
  BAD:  ["synaptic consolidation during sleep", "experience replay DQN algorithm"]
  (BAD because each query covers only one domain — they won't find cross-domain papers)
 
Respond ONLY with a JSON array of strings.
Example: ["query one here", "query two here"]
"""

RELEVANCE_PROMPT = """
Is this text at least partially related to this research question?

Be generous — if the text provides ANY factual content that could inform the question,
say yes. Only say no if the text is completely off-topic.

Examples:
- Question: "How does neuroplasticity work?" + Text about brain anatomy → yes (related domain)
- Question: "How does neuroplasticity work?" + Text about cooking recipes → no (unrelated)

Question: {question}
Text: {text}

Respond with ONLY "yes" or "no".
"""

EXTRACTION_QUALITY_PROMPT = """
Given this research question and retrieved text, extract only the parts
that are directly relevant to answering the question.
 
Question: {question}
Text: {text}
 
Return a cleaned, relevant excerpt of 2-5 sentences.
If nothing is relevant, return "IRRELEVANT".
Respond with ONLY the excerpt or "IRRELEVANT". No preamble.
"""

RESOLUTION_CHECK_PROMPT = """
A researcher has been investigating this question:
"{question}"
 
After today's research, these findings were added to the knowledge graph:
{findings}
 
Has the question been meaningfully answered or significantly advanced?
 
Grading definitions:
- "none": The findings are unrelated or too tangential to advance the question.
- "partial": The findings provide useful context or answer a sub-aspect, but the core question
  remains open. Test: could you now write a better literature review section, but NOT a conclusion?
- "strong": The findings directly answer the question or provide conclusive evidence.
  Test: could you now write a confident conclusion section?

Respond with a JSON object:
{{
  "resolved": true or false,
  "grade": one of ["none", "partial", "strong"],
  "explanation": "one sentence"
}}
 
Respond ONLY with JSON.
"""

HYPOTHESIS_EVALUATION_PROMPT = """
HYPOTHESIS: "{hypothesis}"

NEW EVIDENCE FOUND:
{evidence}

Based on the evidence, what is the verdict on the hypothesis?

- "confirms": The evidence directly supports the hypothesis mechanism.
- "contradicts": The evidence explicitly disproves or is incompatible with
  the hypothesis mechanism.
- "irrelevant": The evidence is about a different topic or doesn't speak
  to this hypothesis either way.
- "partial": The evidence supports part of the hypothesis but leaves the
  core claim unresolved.

Strict rules:
- Topic overlap alone is NOT confirmation.
- Broad background review counts as "partial" unless it supports the specific mechanism.
- If the evidence constrains a prerequisite but not the mechanism itself, return "partial".
- If nothing in the evidence addresses the stated mechanism or discriminating prediction, return "irrelevant".

Respond with JSON:
{{
  "verdict": "confirms" | "contradicts" | "partial" | "irrelevant",
  "explanation": "one sentence explaining why",
  "specific_claim": "the specific evidence claim that is most relevant"
}}
"""

# ── Research Log ──────────────────────────────────────────────────────────────
 
@dataclass
class ResearchEntry:
    question:    str
    queries:     list = field(default_factory=list)
    sources:     list = field(default_factory=list)   # URLs / arXiv IDs
    node_ids:    list = field(default_factory=list)   # nodes created
    resolved:    str  = "none"                        # none / partial / strong
    timestamp:   float = field(default_factory=time.time)

@dataclass
class ResearchLog:
    date:    float = field(default_factory=time.time)
    entries: list  = field(default_factory=list)
 
    def to_dict(self):
        return {
            "date":    self.date,
            "entries": [e.__dict__ for e in self.entries]
        }

# ── Researcher ────────────────────────────────────────────────────────────────
 
class Researcher:
    def __init__(self, brain: Brain, observer=None,
                 depth: str = RESEARCH_DEPTH, ingestor=None,
                 embedding_index=None, insight_buffer=None, critic=None):
        self.brain    = brain
        self.observer = observer
        self.ingestor = ingestor or Ingestor(
            brain,
            research_agenda=observer,
            embedding_index=embedding_index,
            insight_buffer=insight_buffer,
            critic=critic
        )
        self.depth    = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["standard"])
        self.log      = ResearchLog()
 
    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        return llm_call(prompt, temperature=temperature, role="precise")

    def _source_type_from_reference(self, reference: str) -> str:
        ref = (reference or "").lower()
        if "arxiv.org/abs/" in ref or "arxiv.org/pdf/" in ref:
            return "arxiv_paper"
        return "research_result"

    def _domain_matches(self, domain: str, candidates: set[str]) -> bool:
        return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in candidates)

    def _source_domain(self, source: str) -> str:
        try:
            domain = urlparse(source or "").netloc.lower().strip()
        except Exception:
            domain = ""
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _source_quality_score(self, source: str) -> float:
        ref = (source or "").strip().lower()
        if "arxiv.org/abs/" in ref or "arxiv.org/pdf/" in ref:
            return 1.0
        domain = self._source_domain(ref)
        if not domain:
            return 0.5
        if self._domain_matches(domain, HIGH_RIGOR_DOMAINS):
            return 0.95
        if self._domain_matches(domain, LOW_RIGOR_DOMAINS):
            return 0.1
        if domain.endswith(".edu") or domain.endswith(".ac.uk"):
            return 0.72
        if domain.endswith(".gov"):
            return 0.85
        if domain.endswith(".org"):
            return 0.62
        return 0.5

    def _question_tokens(self, text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
            "do", "does", "for", "from", "how", "if", "in", "into", "is",
            "it", "its", "of", "on", "or", "that", "the", "their", "them",
            "these", "this", "those", "to", "using", "what", "when", "where",
            "which", "while", "with", "would",
        }
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {token for token in tokens if len(token) > 2 and token not in stopwords}

    def _finding_priority(self, question: str, title: str, text: str, source: str) -> float:
        source_quality = self._source_quality_score(source)
        question_tokens = self._question_tokens(question)
        evidence_tokens = self._question_tokens(f"{title} {text}")
        overlap = len(question_tokens & evidence_tokens) / max(len(question_tokens), 1)
        length_bonus = min(len(text or "") / 1200.0, 1.0) * 0.05
        return (0.65 * source_quality) + (0.30 * overlap) + length_bonus

    def _rank_findings(self, question: str, findings: list) -> list:
        deduped = {}
        for title, text, source in findings:
            clean_title = " ".join((title or "").split())
            clean_text = " ".join((text or "").split())
            clean_source = (source or "").strip()
            if not clean_text:
                continue
            key = (clean_source or clean_title.lower(), clean_title.lower())
            priority = self._finding_priority(question, clean_title, clean_text, clean_source)
            current = deduped.get(key)
            if current is None or priority > current[0]:
                deduped[key] = (priority, (clean_title, clean_text, clean_source))
        ranked = sorted(deduped.values(), key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked[:MAX_FINDINGS_PER_QUESTION]]

    # ── Query generation ──────────────────────────────────────────────────────
 
    def _generate_queries(self, question: str, n: int) -> list:
        workspace = self.brain.build_workspace(
            embedding_index=getattr(self.ingestor, "index", None),
            query=question,
        )
        mission = self.brain.get_mission() or {}
        raw = self._llm(QUERY_GENERATION_PROMPT.format(
            question=question,
            mission=mission.get("question", "none"),
            workspace=workspace.to_prompt_context(),
            n=n,
        ), temperature=0.5)
        queries = require_json(raw, default=[])
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)][:n]
        return [question]

    # ── Web search ────────────────────────────────────────────────────────────
 
    def _web_search(self, query: str) -> list:
        from ddgs import DDGS
        import time
        for attempt in range(3):
            try:
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=MAX_RESULTS_PER_QUERY):
                        text = r.get('body', '')
                        if len(text) > MIN_TEXT_LENGTH:
                            results.append((
                                r.get('title', query),
                                text,
                                r.get('href', '')
                            ))
                return results
            except Exception as e:
                print(f"      Web search error on attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))  # 2s, 4s backoff before retry
                else:
                    return []
        return []

    # ── arXiv search ──────────────────────────────────────────────────────────
 
    def _arxiv_search(self, query: str) -> list:
        """
        arXiv API — free, no key needed.
        Returns list of (title, abstract, arxiv_url).
        """
        try:
            base = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start":        0,
                "max_results":  MAX_ARXIV_RESULTS,
                "sortBy":       "relevance",
                "sortOrder":    "descending",
            }
            resp = requests.get(base, params=params, timeout=15)
            resp.raise_for_status()
 
            # parse Atom XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
 
            results = []
            for entry in root.findall("atom:entry", ns):
                title    = entry.find("atom:title", ns)
                summary  = entry.find("atom:summary", ns)
                link_el  = entry.find("atom:id", ns)
 
                if title is None or summary is None:
                    continue
 
                title_text   = title.text.strip().replace("\n", " ")
                summary_text = summary.text.strip().replace("\n", " ")
                arxiv_url    = link_el.text.strip() if link_el is not None else ""

                # Normalize to canonical abs URL when possible.
                match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_url)
                if match:
                    arxiv_url = f"https://arxiv.org/abs/{match.group(1)}"
 
                if len(summary_text) > MIN_TEXT_LENGTH:
                    results.append((title_text, summary_text, arxiv_url))
 
            return results[:MAX_ARXIV_RESULTS]
 
        except Exception as e:
            print(f"      arXiv search error: {e}")
            return []

    # ── Relevance filtering ───────────────────────────────────────────────────
 
    def _filter_relevant(self, question: str,
                         results: list) -> list:
        """
        Filter search results to only those relevant to the question.
        Returns list of cleaned (title, text, source) tuples.
        """
        filtered = []
        for title, text, source in results:
            if self._source_quality_score(source) < MIN_SOURCE_QUALITY:
                continue
            # quick relevance check
            check = self._llm(RELEVANCE_PROMPT.format(
                question=question,
                text=text
            ), temperature=0.2)
            if not check.lower().startswith('yes'):
                continue
 
            # extract only the relevant parts
            cleaned = self._llm(EXTRACTION_QUALITY_PROMPT.format(
                question=question,
                text=text
            ), temperature=0.3)
            if cleaned.strip().upper() == "IRRELEVANT":
                continue

            filtered.append((title, cleaned, source))
 
        return self._rank_findings(question, filtered)

    # ── Hypothesis Evaluation ─────────────────────────────────────────────────

    def _evaluate_hypothesis(self, hypothesis: str, findings: str) -> dict:
        raw = self._llm(HYPOTHESIS_EVALUATION_PROMPT.format(
            hypothesis=hypothesis,
            evidence=findings
        ), temperature=0.2)
        from llm_utils import require_json
        return require_json(raw, default={"verdict": "irrelevant", "explanation": "Failed to parse verdict", "specific_claim": ""})

    def _log_hypothesis_research(self, payload: dict, path: str = "logs/research_latest.json"):
        import os
        os.makedirs("logs", exist_ok=True)
        atomic_write_json(path, payload)

    def _apply_hypothesis_verdict(self, hypothesis_id: str, verdict: str, evidence_node_ids: list, explanation: str):
        hyp_node = self.brain.get_node(hypothesis_id)
        if not hyp_node:
            return

        if verdict == "contradicts":
            # Mark hypothesis as CONTRADICTED
            self.brain.update_node(hypothesis_id,
                status=NodeStatus.CONTRADICTED.value,
                epistemic_status=ArtifactStatus.CONTRADICTED.value)
            # Draw CORRECTED_BY edges from evidence to hypothesis
            for eid in evidence_node_ids:
                edge = Edge(
                    type=EdgeType.CORRECTED_BY,
                    narration=explanation,
                    weight=0.85,
                    confidence=0.8,
                    source=EdgeSource.RESEARCH,
                    decay_exempt=True,  # corrections are permanent
                )
                self.brain.add_edge(eid, hypothesis_id, edge)
            
        elif verdict == "confirms":
            # Promote hypothesis to WORKING / UNCERTAIN status
            self.brain.update_node(hypothesis_id,
                status=NodeStatus.UNCERTAIN.value,  # promoted from HYPOTHETICAL
                epistemic_status=ArtifactStatus.OPEN.value,
                importance=min(1.0, hyp_node.get('importance', 0.5) + 0.2))
            # Draw CONFIRMED_BY edges
            for eid in evidence_node_ids:
                edge = Edge(
                    type=EdgeType.CONFIRMED_BY,
                    narration=explanation,
                    weight=0.75,
                    confidence=0.7,
                    source=EdgeSource.RESEARCH,
                )
                self.brain.add_edge(eid, hypothesis_id, edge)
            # Send to Critic for rigorous review (promotion gate)
            if self.ingestor.critic:
                from critic.critic import CandidateThought
                candidate = CandidateThought(
                    claim=hyp_node.get('statement', ''),
                    source_module="researcher",
                    proposed_type=NodeType.HYPOTHESIS.value,
                    importance=hyp_node.get('importance', 0.5),
                    context=explanation,
                    expected_status=NodeStatus.UNCERTAIN.value,
                )
                self.ingestor.critic.route_deferred(candidate)
        
        elif verdict in ["irrelevant", "partial"]:
            if not evidence_node_ids:
                # No results at all — LACKS_EVIDENCE
                self.brain.update_node(hypothesis_id,
                    status=NodeStatus.LACKS_EVIDENCE.value,
                    epistemic_status=ArtifactStatus.LACKS_EVIDENCE.value)
            else:
                # Partial — keep as HYPOTHETICAL
                self.brain.update_node(hypothesis_id,
                    status=NodeStatus.HYPOTHETICAL.value)

    def _reinforce_matching_edges(self, new_claim_text: str):
        """Boost confidence of existing edges whose narration matches new evidence."""
        try:
            from embedding import embed
            claim_emb = embed(new_claim_text)
            for u, v, data in self.brain.graph.edges(data=True):
                narration = data.get('narration', '')
                if not narration or len(narration) < 20:
                    continue
                narr_emb = embed(narration)
                import numpy as np
                sim = float(np.dot(claim_emb, narr_emb))
                if sim >= 0.70:
                    old_conf = data.get('confidence', 0.5)
                    new_conf = min(0.95, old_conf + 0.08)
                    self.brain.update_edge(u, v, confidence=new_conf)
                    vcount = data.get('verification_count', 0) + 1
                    self.brain.graph.edges[u, v]['verification_count'] = vcount
        except ImportError:
            pass

    # ── Research one question ─────────────────────────────────────────────────
 
    def _research_question(self, question_text: str, node_id: str = "") -> ResearchEntry:
        entry = ResearchEntry(question=question_text)
 
        print(f"\n  ── Researching: {question_text}")
 
        n_queries = self.depth["queries_per_q"]
        context = ""
        if node_id and self.brain.get_node(node_id):
            neighbors = self.brain.neighbors(node_id)
            neighbor_stmts = [
                self.brain.get_node(n)['statement']
                for n in neighbors[:4] if self.brain.get_node(n)
            ]
            if neighbor_stmts:
                context = ("\n\nRelated concepts already in the knowledge graph:\n" +
                           "\n".join(f"- {s}" for s in neighbor_stmts))
        queries = self._generate_queries(question_text + context, n_queries)
        print(f"     Queries: {queries}")
 
        all_findings = []
 
        for query in queries:
            entry.queries.append(query)
 
            # web search
            if self.depth["web"]:
                web_results = self._web_search(query)
                relevant    = self._filter_relevant(question_text, web_results)
                all_findings.extend(relevant)
                for title, text, source in relevant:
                    print(f"     [web] {title}...")
                    entry.sources.append(source)
                time.sleep(2)
 
            # arXiv
            if self.depth["arxiv"]:
                arxiv_results = self._arxiv_search(query)
                relevant      = self._filter_relevant(question_text, arxiv_results)
                all_findings.extend(relevant)
                for title, text, source in relevant:
                    print(f"     [arxiv] {title}...")
                    entry.sources.append(source)
 
        if not all_findings:
            print(f"     No relevant findings.")
            return entry

        all_findings = self._rank_findings(question_text, all_findings)

        new_ids = []
        for title, text, source in all_findings:
            source_node_id = self.brain.create_source_node(
                title=title,
                reference=source,
                source_type=self._source_type_from_reference(source),
                created_by="researcher",
                excerpt=text[:800],
            )
            ids = self.ingestor.ingest(
                text,
                source=EdgeSource.RESEARCH,
                source_ids=[source_node_id],
                source_refs=[source] if source else [],
                created_by="researcher",
            ) or []
            new_ids.extend(ids)
        entry.node_ids = list(dict.fromkeys(new_ids))
 
        # reinforce matching edges
        for _, text, _ in all_findings:
            self._reinforce_matching_edges(text)

        # check if question was resolved by these findings
        findings_summary = "\n".join(
            f"- {text}" for _, text, _ in all_findings
        )
        raw = self._llm(RESOLUTION_CHECK_PROMPT.format(
            question=question_text,
            findings=findings_summary
        ), temperature=0.2)
        result = require_json(raw, default={})
        entry.resolved = result.get('grade', 'none')
        explanation    = result.get('explanation', '')
 
        if entry.resolved in ['partial', 'strong'] and self.observer:
            self.observer.record_answer(
                question_text=question_text,
                answer_node_id=new_ids[0] if new_ids else "",
                explanation=explanation,
                grade=entry.resolved
            )
            print(f"     Resolution: [{entry.resolved}] {explanation}")
 
        return entry

    # ── Hypothesis Research ───────────────────────────────────────────────────

    def research_hypothesis(self, hypothesis_node_id: str, search_queries: list) -> "ResearchEntry":
        """
        Research a specific hypothesis using decomposed search queries.
        Unlike research_day() which pulls from the agenda, this takes
        explicit queries from the Thinker's hypothesis_testing output.
        """
        import time
        # Get hypothesis statement
        hyp_node = self.brain.get_node(hypothesis_node_id)
        question_text = hyp_node['statement'] if hyp_node else "Unknown Hypothesis"

        entry = ResearchEntry(question=question_text)
        print(f"\n  ── Researching Hypothesis: {question_text[:80]}...")
        
        all_findings = []

        for query in search_queries[:4]:
            entry.queries.append(query)
            print(f"     Query: {query}")

            # web search
            if self.depth["web"]:
                web_results = self._web_search(query)
                relevant    = self._filter_relevant(question_text, web_results)
                all_findings.extend(relevant)
                for title, text, source in relevant:
                    print(f"     [web] {title}...")
                    entry.sources.append(source)
                time.sleep(2)

            # arXiv
            if self.depth["arxiv"]:
                arxiv_results = self._arxiv_search(query)
                relevant      = self._filter_relevant(question_text, arxiv_results)
                all_findings.extend(relevant)
                for title, text, source in relevant:
                    print(f"     [arxiv] {title}...")
                    entry.sources.append(source)

        if not all_findings:
            print(f"     No relevant findings.")
            self._apply_hypothesis_verdict(hypothesis_node_id, "irrelevant", [], "No findings.")
            self._log_hypothesis_research({
                "timestamp": time.time(),
                "hypothesis_node_id": hypothesis_node_id,
                "question": question_text,
                "queries": list(entry.queries),
                "sources": [],
                "node_ids": [],
                "resolved": "none",
                "verdict": "irrelevant",
                "explanation": "No relevant findings.",
            })
            return entry

        all_findings = self._rank_findings(question_text, all_findings)

        new_ids = []
        for title, text, source in all_findings:
            source_node_id = self.brain.create_source_node(
                title=title,
                reference=source,
                source_type=self._source_type_from_reference(source),
                created_by="researcher_hypothesis",
                excerpt=text[:800],
            )
            ids = self.ingestor.ingest(
                text,
                source=EdgeSource.RESEARCH,
                source_ids=[source_node_id],
                source_refs=[source] if source else [],
                created_by="researcher_hypothesis",
            ) or []
            new_ids.extend(ids)
        entry.node_ids = list(dict.fromkeys(new_ids))

        # reinforce edges
        for _, text, _ in all_findings:
            self._reinforce_matching_edges(text)

        findings_summary = "\n".join(
            f"- {text}" for _, text, _ in all_findings
        )
        eval_result = self._evaluate_hypothesis(question_text, findings_summary)
        verdict = eval_result.get("verdict", "irrelevant")
        explanation = eval_result.get("explanation", "")
        
        print(f"     Verdict: [{verdict}] {explanation}")
        self._apply_hypothesis_verdict(hypothesis_node_id, verdict, entry.node_ids, explanation)
        entry.resolved = "strong" if verdict in ["confirms", "contradicts"] else ("partial" if entry.node_ids else "none")
        self._log_hypothesis_research({
            "timestamp": time.time(),
            "hypothesis_node_id": hypothesis_node_id,
            "question": question_text,
            "queries": list(entry.queries),
            "sources": list(dict.fromkeys(entry.sources)),
            "node_ids": list(entry.node_ids),
            "resolved": entry.resolved,
            "verdict": verdict,
            "explanation": explanation,
            "source_rigor_floor": max(
                [highest_reference_rigor(self.brain.get_node(node_id) or {}) for node_id in entry.node_ids] or [0.0]
            ),
        })
        return entry

    # ── Main research day ─────────────────────────────────────────────────────
 
    def research_day(self,
                     max_questions: int = MAX_QUESTIONS_PER_DAY,
                     log_path: str = "logs/research_latest.json"
                     ) -> ResearchLog:
        """
        Run the full day research cycle.
        Pulls top-priority questions from the agenda,
        researches each, ingests findings, marks resolutions.
        """
        import os
        os.makedirs("logs", exist_ok=True)
 
        self.log = ResearchLog()
 
        if not self.observer:
            print("── Researcher: no observer connected, skipping ──")
            return self.log
 
        questions = self.observer.get_prioritized_questions(max_questions)
 
        if not questions:
            print("── Researcher: no open questions in agenda ──")
            return self.log
 
        print(f"\n── Research day begins ──")
        print(f"   Depth: {RESEARCH_DEPTH}")
        print(f"   Questions to research: {len(questions)}\n")
 
        for item in questions:
            if item.resolved:
                continue
            entry = self._research_question(item.text, node_id=item.node_id)
            self.log.entries.append(entry)
            # small pause between questions
            time.sleep(1)
 
        # save log
        atomic_write_json(log_path, self.log.to_dict())
 
        resolved = sum(1 for e in self.log.entries
                       if e.resolved in ['partial', 'strong'])
        print(f"\n── Research day complete ──")
        print(f"   Questions researched: {len(self.log.entries)}")
        print(f"   Resolved/advanced:    {resolved}")
        print(f"   Brain: {self.brain.stats()}")
 
        return self.log
