import json
import time
import requests
from dataclasses import dataclass, field
from ollama import Client
from graph.brain import Brain, EdgeSource
from ingestion.ingestor import Ingestor
from persistence import atomic_write_json

# ── Config ────────────────────────────────────────────────────────────────────
 
OLLAMA_MODEL         = "llama3.1:8b"
MAX_QUESTIONS_PER_DAY = 5      # how many agenda items to research per cycle
MAX_RESULTS_PER_QUERY = 3      # web results per search query
MAX_ARXIV_RESULTS    = 2       # arXiv papers per query
RESEARCH_DEPTH       = "standard"  # "shallow", "standard", "deep"
MIN_TEXT_LENGTH      = 200     # ignore snippets shorter than this

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
 
Generate {n} distinct search queries that would help answer this question.
Each query should approach the question from a different angle.
Keep each query concise — 4 to 8 words.
 
Respond ONLY with a JSON array of strings.
Example: ["query one here", "query two here"]
"""

RELEVANCE_PROMPT = """
Is this text at least partially related to this research question?
Be generous — if it touches the topic at all, say yes.

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
 
Respond with a JSON object:
{{
  "resolved": true or false,
  "grade": one of ["none", "partial", "strong"],
  "explanation": "one sentence"
}}
 
Respond ONLY with JSON.
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
                 depth: str = RESEARCH_DEPTH):
        self.brain    = brain
        self.observer = observer
        self.ingestor = Ingestor(brain, research_agenda=observer)
        self.llm      = Client()
        self.depth    = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["standard"])
        self.log      = ResearchLog()
 
    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        response = self.llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    # ── Query generation ──────────────────────────────────────────────────────
 
    def _generate_queries(self, question: str, n: int) -> list:
        raw = self._llm(QUERY_GENERATION_PROMPT.format(
            question=question, n=n
        ), temperature=0.5)
        try:
            queries = json.loads(raw)
            return [q for q in queries if isinstance(q, str)][:n]
        except (json.JSONDecodeError, ValueError):
            # fallback — use question directly
            return [question]

    # ── Web search ────────────────────────────────────────────────────────────
 
    def _web_search(self, query: str) -> list:
        try:
            from ddgs import DDGS
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
            print(f"      Web search error: {e}")
            return []

    # ── arXiv search ──────────────────────────────────────────────────────────
 
    def _arxiv_search(self, query: str) -> list:
        """
        arXiv API — free, no key needed.
        Returns list of (title, abstract, arxiv_url).
        """
        try:
            import urllib.parse
            base = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{urllib.parse.quote(query)}",
                "start":        0,
                "max_results":  MAX_ARXIV_RESULTS,
            }
            resp = requests.get(base, params=params, timeout=15)
 
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
 
        return filtered

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
 
        # ingest all findings into the brain
        combined_text = "\n\n".join(
            f"{title}:\n{text}" for title, text, _ in all_findings
        )
 
        new_ids = self.ingestor.ingest(
            combined_text, source=EdgeSource.RESEARCH
        ) or []
        entry.node_ids = new_ids
 
        # check if question was resolved by these findings
        findings_summary = "\n".join(
            f"- {text}" for _, text, _ in all_findings
        )
        raw = self._llm(RESOLUTION_CHECK_PROMPT.format(
            question=question_text,
            findings=findings_summary
        ), temperature=0.2)
        try:
            result = json.loads(raw)
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
 
        except (json.JSONDecodeError, ValueError):
            pass
 
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