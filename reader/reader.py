import io
import json
import os
import re
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from graph.brain import Brain, EdgeSource
from ingestion.ingestor import Ingestor
from llm_utils import llm_call
from persistence import atomic_write_json

# ── Config ────────────────────────────────────────────────────────────────────

READING_LIST_PATH = "data/reading_list.json"
MAX_TEXT_CHARS    = 32000  # ~8-10 pages of content
MAX_PDF_TEXT_CHARS = 140000
MIN_TEXT_CHARS    = 200    # ignore tiny pages
MIN_SECTION_CHARS = 120
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?"
    r"(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,2}\s+)?"
    r"(abstract|introduction|background|related work|methods?|methodology|"
    r"materials|results?|discussion|conclusion|limitations|references)\b",
    re.IGNORECASE,
)

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

EXPECTATION_PROMPT = """
You are generating a top-down prediction of what a text will contain BEFORE reading it.
This is the Free Energy Principle (Active Inference) in action.

Title: {title}
URL/Context: {url}

Current Mission: {mission}
Current Knowledge Clusters: {clusters}

Predict what specific concepts, hypotheses, and answers this text will contain.
Keep it to 2-3 sentences. No formatting.
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
        "source_type": "wikipedia or arxiv_query",
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
    source_type: str   = "web"      # wikipedia | arxiv_query | arxiv_paper | web | pdf | text
    priority:    float = 0.5
    status:      str   = "unread"   # unread | reading | read | failed
    added_by:    str   = "user"     # user | system | dream | auto
    added_reason:str   = ""
    raw_text:    str   = ""
    absorbed_at: float = 0.0
    node_count:  int   = 0          # nodes added when absorbed
    resolved_arxiv_id: str = ""
    resolved_pdf_url:  str = ""
    resolved_from_query: str = ""
    section_count: int = 0
    section_labels: list[str] = field(default_factory=list)

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
                 ingestor=None, embedding_index=None,
                 insight_buffer=None, critic=None):
        self.brain    = brain
        self.observer = observer
        self.notebook = notebook
        self.ingestor = ingestor or Ingestor(
            brain,
            research_agenda=observer,
            embedding_index=embedding_index,
            insight_buffer=insight_buffer,
            critic=critic
        )
        self.reading_list: list[ReadingEntry] = []
        self._load_list()

    def _llm(self, prompt: str, temperature: float = 0.6) -> str:
        return llm_call(prompt, temperature=temperature, role="precise")

    def _normalize_source_type(self, source_type: str) -> str:
        stype = (source_type or "web").strip().lower()
        legacy_aliases = {
            "arxiv": "arxiv_query",
        }
        return legacy_aliases.get(stype, stype)

    def _extract_arxiv_id(self, value: str) -> str:
        match = ARXIV_ID_RE.search(value or "")
        return match.group("id") if match else ""

    def _arxiv_abs_url(self, arxiv_id: str) -> str:
        return f"https://arxiv.org/abs/{arxiv_id}"

    def _arxiv_pdf_url(self, arxiv_id_or_url: str) -> str:
        arxiv_id = self._extract_arxiv_id(arxiv_id_or_url)
        if not arxiv_id:
            return ""
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    def _parse_arxiv_entry(self, entry_el: ET.Element) -> dict:
        title_el = entry_el.find("atom:title", ARXIV_NS)
        summary_el = entry_el.find("atom:summary", ARXIV_NS)
        id_el = entry_el.find("atom:id", ARXIV_NS)

        title = " ".join((title_el.text or "").split()) if title_el is not None else ""
        summary = " ".join((summary_el.text or "").split()) if summary_el is not None else ""

        abs_url = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = self._extract_arxiv_id(abs_url)
        if not abs_url and arxiv_id:
            abs_url = self._arxiv_abs_url(arxiv_id)

        pdf_url = ""
        for link_el in entry_el.findall("atom:link", ARXIV_NS):
            href = (link_el.get("href") or "").strip()
            title_attr = (link_el.get("title") or "").strip().lower()
            if title_attr == "pdf" or href.endswith(".pdf"):
                pdf_url = href
                break
        if not pdf_url and arxiv_id:
            pdf_url = self._arxiv_pdf_url(arxiv_id)

        return {
            "title": title,
            "abstract": summary,
            "arxiv_id": arxiv_id,
            "abs_url": abs_url,
            "pdf_url": pdf_url,
        }

    def _fetch_arxiv_metadata(self, arxiv_id_or_url: str) -> tuple[dict | None, str]:
        arxiv_id = self._extract_arxiv_id(arxiv_id_or_url)
        if not arxiv_id:
            return None, "Could not extract arXiv ID"

        try:
            resp = requests.get(
                ARXIV_API_URL,
                params={"id_list": arxiv_id},
                timeout=15,
                headers={"User-Agent": "DREAMER/1.0"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            entry_el = root.find("atom:entry", ARXIV_NS)
            if entry_el is None:
                return None, f"No arXiv record found for ID {arxiv_id}"
            return self._parse_arxiv_entry(entry_el), ""
        except Exception as e:
            return None, f"Error fetching arXiv metadata: {e}"

    def _resolve_arxiv_query(self, query: str) -> tuple[dict | None, str]:
        query = (query or "").strip()
        if not query:
            return None, "Empty arXiv query"

        if self._extract_arxiv_id(query):
            metadata, error = self._fetch_arxiv_metadata(query)
            if metadata is not None:
                metadata["query"] = query
            return metadata, error

        try:
            resp = requests.get(
                ARXIV_API_URL,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": 1,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=15,
                headers={"User-Agent": "DREAMER/1.0"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            entry_el = root.find("atom:entry", ARXIV_NS)
            if entry_el is None:
                return None, f"No arXiv results found for query '{query}'"
            metadata = self._parse_arxiv_entry(entry_el)
            metadata["query"] = query
            return metadata, ""
        except Exception as e:
            return None, f"Error resolving arXiv query: {e}"

    # ── Reading list management ───────────────────────────────────────────────

    def _load_list(self):
        try:
            with open(READING_LIST_PATH, 'r') as f:
                data = json.load(f)
            migrated = False
            for entry in data:
                if entry.get("source_type") == "text" and not entry.get("raw_text"):
                    entry["raw_text"] = entry.get("added_reason", "")
                normalized = self._normalize_source_type(entry.get("source_type", "web"))
                if normalized != entry.get("source_type", "web"):
                    entry["source_type"] = normalized
                    migrated = True
            self.reading_list = [ReadingEntry(**e) for e in data]
            if migrated:
                self._save_list()
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
        source_type = self._normalize_source_type(source_type)
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
                 priority: float = 0.7) -> AbsorptionResult:
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
        """Fetch arXiv abstract text for a specific paper."""
        metadata, error = self._fetch_arxiv_metadata(arxiv_id_or_url)
        if metadata is None:
            return "", error
        return metadata.get("title", ""), metadata.get("abstract", "")[:MAX_TEXT_CHARS]

    def _fetch_pdf_text(self, pdf_path_or_url: str) -> tuple:
        """Fetch and parse PDF text from URL or local path."""
        if not pdf_path_or_url:
            return "", "Error: Empty PDF URL/path"

        try:
            if pdf_path_or_url.startswith(("http://", "https://")):
                resp = requests.get(
                    pdf_path_or_url,
                    timeout=25,
                    headers={"User-Agent": "DREAMER/1.0"},
                )
                resp.raise_for_status()
                pdf_bytes = resp.content
                name_hint = pdf_path_or_url.split("?")[0].rstrip("/").split("/")[-1]
            else:
                local_path = pdf_path_or_url
                if local_path.startswith("file://"):
                    local_path = local_path[7:]
                with open(local_path, "rb") as f:
                    pdf_bytes = f.read()
                name_hint = os.path.basename(local_path)

            try:
                from pypdf import PdfReader
            except Exception:
                return "", "Error: pypdf is required for PDF ingestion"

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages.append(page_text)

            full_text = "\n\n".join(pages).strip()
            if not full_text:
                return "", "Error: PDF parsed but no extractable text was found"

            title = ""
            metadata = getattr(reader, "metadata", None)
            if metadata is not None:
                try:
                    title = (getattr(metadata, "title", None) or "").strip()
                except Exception:
                    title = ""
                if not title and isinstance(metadata, dict):
                    title = str(metadata.get("/Title", "") or "").strip()

            if not title:
                title = os.path.splitext(name_hint or "PDF document")[0] or "PDF document"

            return title, full_text[:MAX_PDF_TEXT_CHARS]
        except Exception as e:
            return "", f"Error parsing PDF: {e}"

    def _segment_document_sections(self, text: str) -> list[dict]:
        """Split long paper-like text into section chunks for ingestion."""
        if not text or not text.strip():
            return []

        sections = []
        current_label = "document_overview"
        current_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if SECTION_HEADING_RE.match(line):
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append({"label": current_label, "text": section_text})
                current_label = line[:120]
                current_lines = []
                continue

            current_lines.append(line)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append({"label": current_label, "text": section_text})

        if len(sections) <= 1:
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", text)
                if len(p.strip()) >= MIN_SECTION_CHARS
            ]
            sections = [
                {"label": f"section_{i + 1}", "text": p}
                for i, p in enumerate(paragraphs[:24])
            ]

        return sections

    def _clean_html(self, html: str) -> str:
        """Strip HTML tags."""
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _build_expectation(self, entry: ReadingEntry) -> str:
        mission = self.brain.get_mission()
        mission_text = mission['question'] if mission else "None"
        clusters = list(set(
            d.get('cluster', 'unknown')
            for _, d in self.brain.all_nodes()
            if d.get('cluster') != 'unclustered'
        ))[:10]
        return self._llm(EXPECTATION_PROMPT.format(
            title=entry.title,
            url=entry.url,
            mission=mission_text,
            clusters=", ".join(clusters)
        ), temperature=0.6)

    def _make_section_ref(self, source_ref: str, label: str, idx: int) -> str:
        base = (source_ref or "section://unknown").strip()
        safe_label = urllib.parse.quote(
            (label or f"section_{idx}").strip().lower().replace(" ", "_")[:80],
            safe="",
        )
        if "#" in base:
            base = base.split("#", 1)[0]
        return f"{base}#section={safe_label or f'section_{idx}'}"

    def _node_summaries(self, node_ids: list[str], limit: int = 8) -> str:
        return "\n".join(
            f"- {self.brain.get_node(nid)['statement']}"
            for nid in node_ids[:limit]
            if self.brain.get_node(nid)
        ) or "No new nodes extracted."

    def _absorb_sections(self, sections: list[dict], entry: ReadingEntry,
                         title_override: str = "") -> AbsorptionResult:
        """Absorb sectioned text by ingesting each section independently."""
        valid_sections = []
        for idx, section in enumerate(sections, start=1):
            label = str(section.get("label", f"section_{idx}")).strip() or f"section_{idx}"
            section_text = str(section.get("text", "")).strip()
            if len(section_text) < MIN_SECTION_CHARS:
                continue
            valid_sections.append((idx, label, section_text))

        if not valid_sections:
            fallback_text = "\n\n".join(
                str(section.get("text", ""))
                for section in sections
                if section.get("text")
            )
            return self._absorb_text(fallback_text[:MAX_PDF_TEXT_CHARS], entry)

        if title_override:
            entry.title = title_override

        prediction = self._build_expectation(entry)
        print(f"  [Predictive Processing] Expectation: {prediction[:80]}...")

        section_payloads = []
        for idx, label, section_text in valid_sections:
            section_ref = self._make_section_ref(entry.url, label, idx)
            source_node_id = self.brain.create_source_node(
                title=f"{entry.title} — {label}",
                reference=section_ref,
                source_type=f"{entry.source_type}_section",
                created_by="reader",
                excerpt=section_text[:800],
            )

            section_payloads.append({
                "label": label,
                "text": section_text,
                "source_ids": [source_node_id],
                "source_refs": [section_ref],
            })

        if hasattr(self.ingestor, "ingest_sections"):
            created_node_ids = self.ingestor.ingest_sections(
                section_payloads,
                source=EdgeSource.READING,
                prediction=prediction,
                created_by="reader",
            ) or []
        else:
            created_node_ids = []
            for payload in section_payloads:
                section_node_ids = self.ingestor.ingest(
                    payload["text"],
                    source=EdgeSource.READING,
                    prediction=prediction,
                    source_ids=payload["source_ids"],
                    source_refs=payload["source_refs"],
                    created_by="reader",
                ) or []
                created_node_ids.extend(section_node_ids)

        unique_node_ids = list(dict.fromkeys(created_node_ids))
        node_count = len(unique_node_ids)

        entry.node_count = node_count
        entry.section_count = len(valid_sections)
        entry.section_labels = [label for _, label, _ in valid_sections]
        entry.status = 'read'
        entry.absorbed_at = time.time()
        self._save_list()

        summary = self._llm(ABSORPTION_SUMMARY_PROMPT.format(
            title=entry.title,
            source=entry.url,
            node_count=node_count,
            node_summaries=self._node_summaries(unique_node_ids),
        ))

        if self.notebook:
            self.notebook._add_entry(
                "reading", summary, 0,
                tags=[
                    f"source:{entry.source_type}",
                    f"nodes:{node_count}",
                    f"sections:{entry.section_count}",
                    entry.title,
                ]
            )

        text_length = sum(len(text) for _, _, text in valid_sections)
        print(
            f"  Absorbed by section: {entry.title} — "
            f"{entry.section_count} sections, {node_count} new nodes"
        )
        return AbsorptionResult(
            entry=entry,
            title=entry.title,
            text_length=text_length,
            node_count=node_count,
            node_ids=unique_node_ids,
            summary=summary,
            success=True,
        )

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

        prediction = self._build_expectation(entry)

        print(f"  [Predictive Processing] Expectation: {prediction[:80]}...")

        source_node_id = self.brain.create_source_node(
            title=entry.title,
            reference=entry.url,
            source_type=entry.source_type,
            created_by="reader",
            excerpt=text[:800],
        )

        # absorb with READING source — pass prediction down
        created_node_ids = self.ingestor.ingest(
            text,
            source=EdgeSource.READING,
            prediction=prediction,
            source_ids=[source_node_id],
            source_refs=[entry.url],
            created_by="reader",
        ) or []

        node_count   = len(created_node_ids)
        entry.node_count  = node_count
        entry.status      = 'read'
        entry.absorbed_at = time.time()
        self._save_list()

        # get summaries of new nodes for the reading log
        new_node_ids = created_node_ids
        node_summaries = self._node_summaries(new_node_ids)

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
        entry.source_type = self._normalize_source_type(entry.source_type)
        self._save_list()
        print(f"\n── Reader: absorbing '{entry.title}' ──")

        text  = ""
        title = entry.title

        if entry.source_type == "arxiv_query":
            resolved, resolve_error = self._resolve_arxiv_query(entry.url or entry.title)
            if not resolved:
                entry.status = 'failed'
                self._save_list()
                return AbsorptionResult(
                    entry=entry,
                    title=title,
                    text_length=0,
                    node_count=0,
                    node_ids=[],
                    summary="",
                    success=False,
                    error=resolve_error,
                )

            entry.resolved_from_query = resolved.get("query", entry.url)
            entry.resolved_arxiv_id = resolved.get("arxiv_id", "")
            entry.resolved_pdf_url = resolved.get("pdf_url", "")
            entry.source_type = "arxiv_paper"
            entry.url = resolved.get("abs_url", entry.url)
            if resolved.get("title"):
                entry.title = resolved["title"]
                title = entry.title
            print(f"  Resolved arXiv query to paper: {entry.resolved_arxiv_id or entry.url}")
            self._save_list()

        if entry.source_type == "text":
            text = entry.raw_text or entry.added_reason
        elif entry.source_type == "wikipedia":
            title, text = self._fetch_wikipedia(entry.url)
        elif entry.source_type == "arxiv_paper":
            title, text = self._fetch_arxiv(entry.url)
            pdf_url = entry.resolved_pdf_url or self._arxiv_pdf_url(entry.url)
            if pdf_url:
                pdf_title, pdf_text = self._fetch_pdf_text(pdf_url)
                if pdf_text and not pdf_text.startswith("Error"):
                    entry.resolved_pdf_url = pdf_url
                    if pdf_title:
                        title = pdf_title
                    sections = self._segment_document_sections(pdf_text)
                    if sections:
                        if title:
                            entry.title = title
                        return self._absorb_sections(sections, entry, title_override=title)
                    text = pdf_text
                elif not text:
                    text = pdf_text
        elif entry.source_type == "pdf":
            title, pdf_text = self._fetch_pdf_text(entry.url)
            if pdf_text and not pdf_text.startswith("Error"):
                sections = self._segment_document_sections(pdf_text)
                if sections:
                    if title:
                        entry.title = title
                    return self._absorb_sections(sections, entry, title_override=title)
                text = pdf_text
            else:
                text = pdf_text
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

        entry.section_count = 0
        entry.section_labels = []

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
            stype  = self._normalize_source_type(s.get('source_type', 'web'))
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
