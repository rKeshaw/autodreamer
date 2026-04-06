<p align="center">
  <h1 align="center">🧠 AutoScientist</h1>
  <p align="center">
    <em>An autonomous research agent that thinks, reads, dreams, and writes — modeled after the cognitive rhythms of a human scientist.</em>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#modules">Modules</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#license">License</a>
  </p>
</p>

---

AutoScientist is an **autonomous scientific research system** that builds and maintains a knowledge graph by reading papers, forming hypotheses, dreaming about connections between ideas, and consolidating knowledge — all on a continuous daily schedule, without human intervention.

It uses a **dual-process cognitive architecture** (Kahneman's System 1 / System 2): fast intuitive reasoning (Thinker, Dreamer) is adversarially checked by a slow, skeptical Critic before insights enter the knowledge graph.

## Features

| | |
|---|---|
| 🧠 **Dual-Process Cognition** | System 1 (Thinker/Dreamer) generates ideas; System 2 (Critic) gates them via adversarial dialogue |
| 🌐 **Knowledge Graph** | NetworkX-backed graph with typed nodes and edges |
| 🔍 **Autonomous Research** | Searches Wikipedia & arXiv, extracts concepts, links them into the graph |
| 🌙 **Dream Cycles** | Nightly graph walks finding unexpected analogies across domains |
| 🧪 **Computational Sandbox** | Auto-generates and runs Python experiments to test hypotheses |
| 💡 **Insight Buffer** | Near-miss ideas are saved and re-evaluated as knowledge grows |
| 🔄 **Self-Regulating Knowledge** | Confidence decay on stale nodes; working memory biases active threads |
| ⚡ **FAISS Embeddings** | Scalable vector similarity search via FAISS |
| 💬 **Web UI** | Flask + SocketIO interface for graph, notebook, and chat |

## Quickstart

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** running locally with a model pulled (default: `mixtral:latest`)

### Installation

```bash
git clone https://github.com/yourusername/autoscientist.git
cd autoscientist
pip install -r requirements.txt
```

### Bootstrap a Research Brain

```bash
python bootstrap.py "How does sleep contribute to creative problem-solving?"
```

### Launch the Web UI

```bash
python gui/app.py
# Open http://localhost:5000
```

### Start the Autonomous Scheduler

```bash
python -m scheduler.scheduler
```

Run a single phase manually:

```bash
python -m scheduler.scheduler --mode dream
python -m scheduler.scheduler --mode thinking
python -m scheduler.scheduler --mode cycle   # full cycle immediately
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    AutoScientist                      │
│                                                       │
│  Reader → Ingestor → Brain (NetworkX + FAISS)         │
│                          │                            │
│        ┌─────────────────┼──────────────────┐         │
│        │                 │                  │         │
│    Dreamer           Thinker           Researcher     │
│    (System 1)        (System 1)                       │
│        │                 │                            │
│        └────────┬─────────┘                           │
│                 │                                     │
│              Critic  ◄── System 2 gating              │
│         (adversarial dialogue)                        │
│                 │                                     │
│        ┌────────┴──────────┐                          │
│        │                   │                          │
│   InsightBuffer       Knowledge Graph                 │
│   (deferred ideas)    (accepted claims)               │
│                                                       │
│  Consolidator · Observer · Notebook · Sandbox         │
│  Scheduler (cron) · GUI (Flask)                       │
│  llm_utils (unified LLM layer)                        │
└──────────────────────────────────────────────────────┘
```

### Daily Cycle

| Time | Phase | What Happens |
|------|-------|--------------|
| **09:00** | 🔬 Research | Searches web/arXiv, extracts concepts into the graph |
| **11:00** | 🤔 Thinking | Applies a reasoning strategy (dialectical, analogical, reductive, experimental, integrative); insights gated by Critic |
| **14:00** | 📖 Reading | Absorbs articles from the reading list |
| **16:00** | ✍️ Writing | Synthesizes accumulated knowledge into structured essays |
| **20:00** | 🔧 Consolidation | Merges duplicates, decays stale nodes, re-evaluates insight buffer |
| **23:00** | 🌙 Dreaming | Nightly graph walks finding unexpected connections; insights gated by Critic |

## Modules

### Core

| Module | Purpose |
|--------|---------|
| `graph/brain.py` | Knowledge graph — nodes, edges, mission, working memory |
| `embedding_index.py` | FAISS-backed vector index |
| `llm_utils.py` | Unified LLM interface — role-based model selection, JSON parsing |
| `config.py` | Thresholds, per-role model config, Critic config |
| `insight_buffer.py` | Delayed insight mechanism — buffers near-miss pairs for re-evaluation |
| `persistence.py` | Atomic JSON writes |

### Cognitive

| Module | Purpose |
|--------|---------|
| `thinker/` | System 1 — structured reasoning with 5 patterns |
| `dreamer/` | System 1 — nightly graph walks (wandering, focused, pressure modes) |
| `critic/` | System 2 — adversarial multi-turn dialogue gating high-stakes claims |
| `consolidator/` | Evening knowledge maintenance — merge, synthesize, decay, gap detection |
| `observer/` | Meta-cognitive monitoring — mission progress, emergence signals, agenda |

### Knowledge Acquisition

| Module | Purpose |
|--------|---------|
| `reader/` | Wikipedia/arXiv reader with prioritized reading list |
| `researcher/` | Active research — generates queries, extracts findings |
| `ingestion/` | Converts raw text → typed graph nodes and edges |

### Interface & Scheduling

| Module | Purpose |
|--------|---------|
| `gui/app.py` | Flask + SocketIO web UI |
| `notebook/` | Persistent research journal |
| `conversation/` | Chat interface that feeds responses into the graph |
| `scheduler/` | APScheduler-based daily cycle |
| `bootstrap.py` | Brain initialization from any research question |
| `sandbox/` | Computational hypothesis testing |

## Configuration

### Model Selection (`config.py`)

Role-based model routing — different tasks can use different models:

```python
class ModelConfig:
    CREATIVE     = "mixtral:latest"   # Dreaming, synthesis
    PRECISE      = "mixtral:latest"   # JSON extraction
    CODE         = "mixtral:latest"   # Sandbox experiments
    REASONING    = "mixtral:latest"   # Thinker
    CONVERSATION = "mixtral:latest"   # Chat
    CRITIC       = "mixtral:latest"   # System 2 adversarial review
```

Swap models per role as needed:
```python
MODELS.CRITIC    = "llama3.1:70b"   # more rigorous System 2
MODELS.CREATIVE  = "llama3.1:70b"   # better dreaming
```

### Thresholds (`config.py`)

```python
class ThresholdConfig:
    MERGE_NODE      = 0.72   # Cosine similarity to merge near-duplicate nodes
    DUPLICATE_MERGE = 0.88   # Strict duplicate detection
    WEAK_EDGE       = 0.58   # Minimum similarity for associative edges
    COHERENCE       = 0.65   # Cross-domain insight quality threshold
    GAP_CONFIDENCE  = 0.75   # Confidence needed to infer gap nodes
```

### Critic / System 2 (`config.py`)

```python
class CriticConfig:
    ACTIVATION_THRESHOLD    = 0.65   # Minimum importance to trigger review
    MAX_DIALOGUE_TURNS      = 3      # Adversarial rounds
    ACCEPT_CONFIDENCE_FLOOR = 0.50   # Minimum confidence to ACCEPT
    ALWAYS_REVIEW_TYPES     = ["hypothesis", "synthesis",
                               "structural_analogy", "deep_isomorphism"]
    BYPASS_TYPES            = ["concept", "associated", "surface_analogy"]
    MAX_REFINE_ITERATIONS   = 2
```

Critic verdicts: **ACCEPT** · **REFINE** · **REJECT** · **DEFER** (→ InsightBuffer)

## Knowledge Graph

### Node Types

`concept` · `hypothesis` · `question` · `answer` · `synthesis` · `gap` · `mission` · `empirical`

### Edge Types

`supports` · `causes` · `contradicts` · `surface_analogy` · `structural_analogy` · `deep_isomorphism` · `associated`

## Data & Logs

```
data/
├── brain.json              # Knowledge graph state
├── observer.json           # Observer state and agenda
├── embedding_index/        # FAISS index
├── insight_buffer.json     # Pending near-miss pairs
└── daily_new_nodes.json    # Daily node ledger

logs/
├── cycle_log.json
├── research_log.json
├── notebook.json
└── sandbox_results.json
```

## Running Tests

```bash
pytest tests/ -v
```

## License

[GNU General Public License v3.0](LICENSE)

---

<p align="center">
  <em>"The mind, once stretched by a new idea, never returns to its original dimensions."</em><br/>
  — Oliver Wendell Holmes Sr.
</p>
