"""
DREAMER Bootstrap Script — Dynamic Version

Builds the initial knowledge graph dynamically based on ANY mission:
1. Sets the central mission question
2. Decomposes the mission into relevant domains via LLM
3. Generates foundational search queries per domain
4. Seeds the reading list by searching Wikipedia + arXiv
5. Seeds the observer agenda with auto-generated questions
6. Absorbs first batch of articles
7. Runs initial dream cycle and consolidation

Optionally forks from a pre-built template brain.

Run: python3 bootstrap.py "Your research question here"
  or: python3 bootstrap.py --template general_scientist "Your question"
"""

import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from graph.brain import Brain, BrainMode
from observer.observer import Observer
from ingestion.ingestor import Ingestor, EdgeSource
from dreamer.dreamer import Dreamer, DreamMode
from consolidator.consolidator import Consolidator
from notebook.notebook import Notebook
from reader.reader import Reader
from thinker.thinker import Thinker
from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed
from llm_utils import llm_call, llm_json

# ── Template paths ────────────────────────────────────────────────────────────

TEMPLATE_DIR = "brain_templates"

# ── Dynamic prompts ───────────────────────────────────────────────────────────

DOMAIN_DECOMPOSITION_PROMPT = """You are helping a researcher begin work on a new scientific question.

RESEARCH QUESTION: {mission}

What are the 5-8 knowledge domains that are MOST relevant to answering this question?
For each domain, provide:
- A domain name (2-4 words)
- Why it's relevant (1 sentence)
- 3 foundational concepts that any researcher in this area must understand

Respond with a JSON array:
[
  {{
    "domain": "domain name",
    "relevance": "why it matters",
    "concepts": ["concept 1", "concept 2", "concept 3"]
  }}
]
"""

SEED_QUESTIONS_PROMPT = """You are a research advisor helping a scientist begin work on:

RESEARCH QUESTION: {mission}

RELEVANT DOMAINS: {domains}

Generate 12-15 foundational questions that would guide early-stage research.
Mix these types:
- 4-5 "What is X?" definitional questions (establish the basics)
- 4-5 "How does X relate to Y?" relational questions (find connections)
- 3-4 "What if X?" speculative questions (generate hypotheses)
- 1-2 "What evidence would..." methodological questions

Each question should be specific enough to be answerable via research.

Respond with a JSON array of question strings.
"""

SEARCH_QUERIES_PROMPT = """Generate 3 specific Wikipedia search queries for researching this concept:

CONCEPT: {concept}
DOMAIN: {domain}

The queries should find the most informative Wikipedia articles.
Respond with a JSON array of 3 search query strings.
"""


def find_template(name: str) -> str | None:
    """Find a template brain by name."""
    path = os.path.join(TEMPLATE_DIR, name)
    if os.path.exists(path + ".brain.json"):
        return path
    return None


def bootstrap(mission: str, mission_context: str = "",
              template: str = None, max_articles: int = 15):
    """
    Dynamically bootstrap a DREAMER brain for any research question.

    Args:
        mission: The central research question
        mission_context: Optional additional context
        template: Optional template name to fork from
        max_articles: Maximum articles to absorb in initial batch
    """
    print("\n" + "="*60)
    print("DREAMER — Dynamic Bootstrap")
    print("="*60)
    print(f"\nMission: {mission}")

    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ── Load or create brain ──────────────────────────────────────────────────

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)

    if template:
        template_path = find_template(template)
        if template_path:
            print(f"\n[0] Forking from template: {template}")
            brain.load(template_path + ".brain.json")
            emb_index = EmbeddingIndex.load(template_path + ".index")
            print(f"  Template loaded: {brain.stats()['nodes']} nodes")
        else:
            print(f"  Template '{template}' not found — starting fresh")

    observer     = Observer(brain)
    notebook     = Notebook(brain, observer=observer)
    ingestor     = Ingestor(brain, embedding_index=emb_index)
    reader       = Reader(brain, observer=observer, notebook=notebook,
                          ingestor=ingestor)
    thinker      = Thinker(brain, observer=observer, embedding_index=emb_index)

    # ── Step 1: Set mission ───────────────────────────────────────────────────

    print("\n[1/7] Setting mission...")
    brain.set_mission(mission, mission_context)
    print(f"  Mode: {brain.get_mode()}")

    # ── Step 2: Decompose into domains ────────────────────────────────────────

    print("\n[2/7] Decomposing mission into knowledge domains...")
    domains = llm_json(
        DOMAIN_DECOMPOSITION_PROMPT.format(mission=mission),
        temperature=0.4,
        default=[]
    )

    if not domains:
        print("  Domain decomposition failed — using fallback")
        domains = [{"domain": "general science", "relevance": "broad coverage",
                     "concepts": ["scientific method", "epistemology", "research methodology"]}]

    print(f"  Identified {len(domains)} domains:")
    for d in domains:
        print(f"    • {d['domain']}: {d.get('relevance', '')}")

    # ── Step 3: Generate seed questions ───────────────────────────────────────

    domain_names = ", ".join(d['domain'] for d in domains)
    print(f"\n[3/7] Generating seed questions from domains...")

    questions = llm_json(
        SEED_QUESTIONS_PROMPT.format(mission=mission, domains=domain_names),
        temperature=0.5,
        default=[]
    )

    if isinstance(questions, list):
        for i, q in enumerate(questions):
            if isinstance(q, str):
                item = observer.add_to_agenda(
                    text      = q,
                    item_type = "question",
                    cycle     = 0,
                    step      = i
                )
                item.priority = 0.7
        print(f"  Generated {len(questions)} seed questions")
    else:
        print("  Question generation failed")

    # ── Step 4: Build reading list from domain concepts ───────────────────────

    print(f"\n[4/7] Building reading list from {len(domains)} domains...")
    total_articles = 0

    for domain_info in domains:
        domain_name = domain_info['domain']
        concepts = domain_info.get('concepts', [])

        for concept in concepts:
            # Generate search queries for this concept
            queries = llm_json(
                SEARCH_QUERIES_PROMPT.format(concept=concept, domain=domain_name),
                temperature=0.3,
                default=[concept]
            )

            if not isinstance(queries, list):
                queries = [concept]

            for query in queries[:2]:  # at most 2 queries per concept
                try:
                    reader.add_to_list(
                        url         = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                        title       = query,
                        source_type = "wikipedia",
                        priority    = 0.8,
                        added_by    = "bootstrap",
                        reason      = f"{domain_name}: {concept}"
                    )
                    total_articles += 1
                except Exception:
                    pass

            if total_articles >= max_articles * 2:
                break

    print(f"  Reading list: {total_articles} articles seeded")

    # ── Step 5: Absorb first batch ────────────────────────────────────────────

    absorb_count = min(max_articles, total_articles)
    print(f"\n[5/7] Absorbing first {absorb_count} articles...")
    print("  (This will take several minutes — each article runs LLM extraction)\n")

    absorbed = 0
    for entry in reader.get_unread(absorb_count):
        try:
            result = reader.absorb_entry(entry)
            if result.success:
                absorbed += 1
                print(f"  ✓ {result.title} — {result.node_count} nodes")
            else:
                print(f"  ✗ {entry.title} — {result.error}")
        except Exception as e:
            print(f"  ✗ {entry.title} — ERROR: {e}")
        time.sleep(1)

    print(f"\n  Absorbed {absorbed} articles")
    print(f"  Brain: {brain.stats()['nodes']} nodes, "
          f"{brain.stats()['edges']} edges, "
          f"{brain.stats()['clusters']} clusters")

    # ── Step 6: Initial thinking session ──────────────────────────────────────

    print("\n[6/7] Running initial thinking session...")
    brain.complete_transition()

    # One round of deliberate thinking about the mission
    try:
        thinking_log = thinker.think(question=mission, pattern="reductive")
        if thinking_log.sub_questions:
            print(f"  Decomposed into {len(thinking_log.sub_questions)} sub-questions")
    except Exception as e:
        print(f"  Thinking session error: {e}")

    # ── Step 7: Dream + consolidation ─────────────────────────────────────────

    print("\n[7/7] Running initial dream cycle + consolidation...")
    dreamer      = Dreamer(brain, research_agenda=observer)
    consolidator = Consolidator(brain, observer=observer, embedding_index=emb_index)

    try:
        log = dreamer.dream(
            mode        = DreamMode.WANDERING,
            steps       = 12,
            temperature = 0.8,
            run_nrem    = True,
            log_path    = "logs/dream_bootstrap.json"
        )
        observer.observe(log)
        notebook.write_morning_entry(log, cycle=0)
    except Exception as e:
        print(f"  Dream error: {e}")

    try:
        report = consolidator.consolidate(save_path="logs/consolidation_bootstrap.json")
        notebook.write_evening_entry(report, cycle=0)
        notebook.update_running_hypothesis(cycle=0)
    except Exception as e:
        print(f"  Consolidation error: {e}")

    # ── Save everything ───────────────────────────────────────────────────────

    brain.save("data/brain.json")
    observer.save("data/observer.json")
    emb_index.save("data/embedding_index")
    notebook.save()

    print("\n" + "="*60)
    print("Bootstrap complete.")
    print(f"Brain: {brain.stats()['nodes']} nodes, "
          f"{brain.stats()['edges']} edges, "
          f"{brain.stats()['clusters']} clusters")
    print(f"Agenda: {len(observer.agenda)} questions")
    print(f"Reading list: {reader.stats()['unread']} articles still unread")
    print(f"Mode: {brain.get_mode()}")
    print(f"\nMission: {mission}")
    print("\nReady. Start the GUI with: python3 gui/app.py")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DREAMER Bootstrap")
    parser.add_argument("mission", nargs="?",
                        default="Are there universal patterns in how scientific "
                                "revolutions happen — and if so, what does the "
                                "current state of knowledge suggest about when "
                                "and where the next major revolution will occur?",
                        help="Central research question")
    parser.add_argument("--context", default="", help="Additional mission context")
    parser.add_argument("--template", default=None,
                        help="Template brain to fork from (e.g. 'general_scientist')")
    parser.add_argument("--max-articles", type=int, default=15,
                        help="Max articles to absorb in initial batch")

    args = parser.parse_args()
    bootstrap(args.mission, args.context, args.template, args.max_articles)
