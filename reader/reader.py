import json
import os
import time
import uuid
import requests
from dataclasses import dataclass, field
from graph.brain import Brain, EdgeSource
from ingestion.ingestor import Ingestor
from persistence import atomic_write_json
from llm_utils import llm_call

# ── Config ────────────────────────────────────────────────────────────────────

READING_LIST_PATH = "data/reading_list.json"
MAX_TEXT_CHARS    = 8000   # truncate very long pages
MIN_TEXT_CHARS    = 200    # ignore tiny pages

# ── Prompts ───────────────────────────────────────────────────────────────────

ABSORPTION_SUMMARY_PROMPT = """
You are a scientist keeping a reading log.

You just finished absorbing this source:
Title: {title}
URL/Source: {source}

Key ideas extracted ({node_count} concepts added to the knowledge graph):
{node_summaries}

Write a brief reading log entry (3-4 sentences) noting:
- What the source was about
- The most interesting ideas encountered
- Any surprising connections to other things you know
- One question this reading opened up

Write in first person. No markdown headers.
"""

READING_LIST_GENERATION_PROMPT = """
You are a curious scientific mind building a reading list.

Current research mission: "{mission}"
Current knowledge clusters: {clusters}
Recent dream questions that need deeper understanding:
{questions}

Suggest 5 specific Wikipedia articles or arXiv search terms that would
meaningfully expand the knowledge graph. Focus on:
- Domains not yet well-represented in the graph
- Foundational concepts that seem to be missing
- Cross-domain bridges that could be interesting

Respond ONLY with a JSON array of objects:
[
  {{
    "title": "Article or topic title",
    "url": "https://en.wikipedia.org/wiki/... or arxiv search term",
    "source_type": "wikipedia or arxiv",
    "reason": "one sentence explaining why this would be valuable"
  }}
]
"""

WANDERING_READING_PROMPT = """
You are a curious mind on intellectual vacation — no specific mission,
just following what seems interesting.

Current knowledge clusters: {clusters}
Recent questions from dreaming: {questions}

Suggest 3 Wikipedia articles that seem intellectually interesting and
would create unexpected new connections in the knowledge graph.
Favor surprising, cross-domain, or underexplored topics.

Respond ONLY with a JSON array:
[
  {{
    "title": "Article title",
    "url": "https://en.wikipedia.org/wiki/...",
    "source_type": "wikipedia",
    "reason": "one sentence"
  }}
]
"""

# ── Reading list entry ────────────────────────────────────────────────────────

@dataclass
class ReadingEntry:
    id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    url:         str   = ""
    title:       str   = ""
    source_type: str   = "web"      # wikipedia | arxiv | web | pdf | text
    priority:    float = 0.5
    status:      str   = "unread"   # unread | reading | read | failed
    added_by:    str   = "user"     # user | system | dream | auto
    added_reason:str   = ""
    raw_text:    str   = ""
    absorbed_at: float = 0.0
    node_count:  int   = 0          # nodes added when absorbed

    def to_dict(self):
        return self.__dict__

# ── Absorption result ─────────────────────────────────────────────────────────

@dataclass
class AbsorptionResult:
    entry:       ReadingEntry
    title:       str
    text_length: int
    node_count:  int
    node_ids:    list
    summary:     str
    success:     bool
    error:       str  = ""

# ── Reader ────────────────────────────────────────────────────────────────────

class Reader:
    def __init__(self, brain: Brain, observer=None, notebook=None,
                 ingestor=None, embedding_index=None, insight_buffer=None):
        self.brain    = brain
        self.observer = observer
        self.notebook = notebook
        self.ingestor = ingestor or Ingestor(
            brain,
            research_agenda=observer,
            embedding_index=embedding_index,
            insight_buffer=insight_buffer
        )
        self.reading_list: list[ReadingEntry] = []
        self._load_list()

    def _llm(self, prompt: str, temperature: float = 0.6) -> str:
        return llm_call(prompt, temperature=temperature, role="precise")

    # ── Reading list management ───────────────────────────────────────────────

    def _load_list(self):
        try:
            with open(READING_LIST_PATH, 'r') as f:
                data = json.load(f)
            for entry in data:
                if entry.get("source_type") == "text" and not entry.get("raw_text"):
                    entry["raw_text"] = entry.get("added_reason", "")
            self.reading_list = [ReadingEntry(**e) for e in data]
            print(f"Reading list loaded — {len(self.reading_list)} entries "
                  f"({sum(1 for e in self.reading_list if e.status=='unread')} unread)")
        except FileNotFoundError:
            self.reading_list = []

    def _save_list(self):
        atomic_write_json(
            READING_LIST_PATH,
            [e.to_dict() for e in self.reading_list]
        )

    def add_to_list(self, url: str, title: str = "", source_type: str = "web",
                    priority: float = 0.5, added_by: str = "user",
                    reason: str = "") -> ReadingEntry:
        # check for duplicate URL
        for existing in self.reading_list:
            if existing.url == url:
                print(f"Already in reading list: {url}")
                return existing

        entry = ReadingEntry(
            url=url, title=title or url, source_type=source_type,
            priority=priority, added_by=added_by, added_reason=reason
        )
        self.reading_list.append(entry)
        self._save_list()
        print(f"Added to reading list: {title or url} (by {added_by})")
        return entry

    def add_text(self, text: str, title: str = "Manual text",
                 priority: float = 0.7) -> ReadingEntry:
        """Add raw text directly to be absorbed."""
        entry = ReadingEntry(
            url=f"text://{uuid.uuid4()}", title=title,
            source_type="text", priority=priority, added_by="user"
        )
        entry.raw_text = text
        self.reading_list.append(entry)
        self._save_list()
        # absorb immediately
        return self._absorb_text(text, entry)

    def get_unread(self, n: int = 5) -> list:
        unread = [e for e in self.reading_list if e.status == 'unread']
        return sorted(unread, key=lambda e: e.priority, reverse=True)[:n]

    def list_all(self) -> list:
        return [e.to_dict() for e in self.reading_list]

    # ── Fetching ──────────────────────────────────────────────────────────────

    def _fetch_wikipedia(self, url: str) -> tuple:
        try:
            # extract title from URL
            title = url.split('/wiki/')[-1]
            import urllib.parse
            title = urllib.parse.unquote(title).replace('_', ' ')

            # use the extract API directly — most reliable
            api = "https://en.wikipedia.org/w/api.php"
            params = {
                "action":  "query",
                "titles":  title,
                "prop":    "extracts",
                "format":  "json",
                "explaintext": 1,   # plain text, no HTML
                "exsectionformat": "plain",
            }
            resp = requests.get(api, params=params, timeout=20,
                                headers={'User-Agent': 'DREAMER/1.0'})
            pages = resp.json().get('query', {}).get('pages', {})
            for page in pages.values():
                page_title = page.get('title', title)
                extract    = page.get('extract', '')
                if extract:
                    return page_title, extract[:MAX_TEXT_CHARS]
            return title, ""
        except Exception as e:
            return "", f"Error: {e}"

    def _fetch_web(self, url: str) -> tuple:
        """Fetch and extract text from any web page."""
        try:
            resp = requests.get(url, timeout=15,
                                headers={'User-Agent': 'DREAMER/1.0'})
            resp.raise_for_status()
            text = self._clean_html(resp.text)
            # try to extract title
            import re
            title_match = re.search(r'<title>(.*?)</title>', resp.text, re.IGNORECASE)
            title = title_match.group(1) if title_match else url
            return title, text[:MAX_TEXT_CHARS]
        except Exception as e:
            return "", f"Error fetching URL: {e}"

    def _fetch_arxiv(self, arxiv_id_or_url: str) -> tuple:
        """Fetch arXiv abstract."""
        try:
            import re
            arxiv_id = re.findall(r'\d{4}\.\d{4,5}', arxiv_id_or_url)
            if not arxiv_id:
                return "", "Could not extract arXiv ID"
            aid = arxiv_id[0]
            import xml.etree.ElementTree as ET
            resp = requests.get(
                f"http://export.arxiv.org/api/query?id_list={aid}",
                timeout=15)
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title   = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                t = title.text.strip() if title is not None else aid
                s = summary.text.strip() if summary is not None else ""
                return t, s[:MAX_TEXT_CHARS]
            return aid, ""
        except Exception as e:
            return "", f"Error fetching arXiv: {e}"

    def _clean_html(self, html: str) -> str:
        """Strip HTML tags."""
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ── Absorption ────────────────────────────────────────────────────────────

    def _absorb_text(self, text: str, entry: ReadingEntry) -> AbsorptionResult:
        """Core absorption — runs ingestor in reading mode, no agenda."""
        if len(text) < MIN_TEXT_CHARS:
            entry.status = 'failed'
            self._save_list()
            return AbsorptionResult(
                entry=entry, title=entry.title, text_length=len(text),
                node_count=0, node_ids=[], summary="Text too short to absorb.",
                success=False, error="Text too short"
            )

        nodes_before = len(self.brain.graph.nodes)

        # absorb with READING source — no agenda checking
        created_node_ids = self.ingestor.ingest(text, source=EdgeSource.READING) or []

        nodes_after  = len(self.brain.graph.nodes)
        node_count   = len(created_node_ids)
        entry.node_count  = node_count
        entry.status      = 'read'
        entry.absorbed_at = time.time()
        self._save_list()

        # get summaries of new nodes for the reading log
        new_node_ids = created_node_ids
        node_summaries = "\n".join(
            f"- {self.brain.get_node(nid)['statement']}"
            for nid in new_node_ids[:8]
            if self.brain.get_node(nid)
        ) or "No new nodes extracted."

        # write reading log entry
        summary = self._llm(ABSORPTION_SUMMARY_PROMPT.format(
            title        = entry.title,
            source       = entry.url,
            node_count   = node_count,
            node_summaries = node_summaries
        ))

        # write to notebook if available
        if self.notebook:
            self.notebook._add_entry(
                "reading", summary, 0,
                tags=[f"source:{entry.source_type}",
                      f"nodes:{node_count}",
                      entry.title]
            )

        print(f"  Absorbed: {entry.title} — {node_count} new nodes")
        return AbsorptionResult(
            entry=entry, title=entry.title,
            text_length=len(text), node_count=node_count, node_ids=new_node_ids,
            summary=summary, success=True
        )

    def absorb_entry(self, entry: ReadingEntry) -> AbsorptionResult:
        """Fetch and absorb a reading list entry."""
        entry.status = 'reading'
        self._save_list()
        print(f"\n── Reader: absorbing '{entry.title}' ──")

        text  = ""
        title = entry.title

        if entry.source_type == "text":
            text = entry.raw_text or entry.added_reason
        elif entry.source_type == "wikipedia":
            title, text = self._fetch_wikipedia(entry.url)
        elif entry.source_type == "arxiv":
            title, text = self._fetch_arxiv(entry.url)
        else:
            title, text = self._fetch_web(entry.url)

        if entry.title == entry.url and title:
            entry.title = title

        if not text or text.startswith("Error"):
            entry.status = 'failed'
            self._save_list()
            return AbsorptionResult(
                entry=entry, title=title, text_length=0,
                node_count=0, node_ids=[], summary="", success=False,
                error=text or "Empty response"
            )

        return self._absorb_text(text, entry)

    def absorb_url(self, url: str, title: str = "",
                   source_type: str = "web") -> AbsorptionResult:
        """Convenience: add and immediately absorb a URL."""
        entry = self.add_to_list(url, title, source_type, priority=0.9,
                                 added_by="user")
        if entry.status == 'read':
            # already absorbed — re-absorb if explicitly requested
            entry.status = 'unread'
        return self.absorb_entry(entry)

    # ── Autonomous reading day ────────────────────────────────────────────────

    def reading_day(self, max_items: int = 2) -> list:
        """
        Called during the day cycle after the Researcher.
        Absorbs top unread items from the reading list.
        """
        unread = self.get_unread(max_items)
        if not unread:
            print("── Reader: reading list empty ──")
            return []

        print(f"\n── Reader: absorbing {len(unread)} items ──")
        results = []
        for entry in unread:
            result = self.absorb_entry(entry)
            results.append(result)
            time.sleep(1)

        return results

    # ── Autonomous list generation ────────────────────────────────────────────

    def generate_reading_list(self, n_questions: int = 8) -> list:
        """
        LLM generates new reading list entries based on current brain state.
        Called periodically — after consolidation, or when list runs low.
        """
        print("\n── Reader: generating reading list ──")

        clusters = list(set(
            d.get('cluster', 'unknown')
            for _, d in self.brain.all_nodes()
            if d.get('cluster') and d['cluster'] != 'unclustered'
        ))[:10]

        # get recent dream questions from observer
        questions = []
        if self.observer:
            items = self.observer.get_prioritized_questions(n_questions)
            questions = [i.text for i in items]

        mission  = self.brain.get_mission()
        is_wandering = self.brain.is_wandering()

        if is_wandering or not mission:
            prompt = WANDERING_READING_PROMPT.format(
                clusters  = ", ".join(clusters),
                questions = "\n".join(f"- {q}" for q in questions[:5]) or "none"
            )
        else:
            prompt = READING_LIST_GENERATION_PROMPT.format(
                mission   = mission['question'],
                clusters  = ", ".join(clusters),
                questions = "\n".join(f"- {q}" for q in questions[:8]) or "none"
            )

        raw = self._llm(prompt)
        try:
            suggestions = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            print(f"  Parse error in reading list generation")
            return []

        added = []
        for s in suggestions:
            url    = s.get('url', '')
            title  = s.get('title', '')
            stype  = s.get('source_type', 'web')
            reason = s.get('reason', '')
            if url:
                entry = self.add_to_list(
                    url=url, title=title, source_type=stype,
                    priority=0.6, added_by="system", reason=reason
                )
                added.append(entry)
                print(f"  → {title} ({stype})")

        return added

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total   = len(self.reading_list)
        unread  = sum(1 for e in self.reading_list if e.status == 'unread')
        read    = sum(1 for e in self.reading_list if e.status == 'read')
        failed  = sum(1 for e in self.reading_list if e.status == 'failed')
        by_src  = {}
        for e in self.reading_list:
            by_src[e.source_type] = by_src.get(e.source_type, 0) + 1
        return {
            "total": total, "unread": unread,
            "read": read, "failed": failed,
            "by_source": by_src
        }
