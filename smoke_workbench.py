import argparse
import json
import sys

from critic.critic import Critic
from embedding_index import EmbeddingIndex
from graph.brain import Brain, EdgeSource, Node, NodeStatus, NodeType
from ingestion.ingestor import Ingestor
from observer.observer import Observer
from scientist_workspace import ArtifactStatus
from thinker.thinker import Thinker
from researcher.researcher import Researcher
from dreamer.dreamer import Dreamer
from reader.reader import Reader
from insight_buffer import InsightBuffer
from notebook.notebook import Notebook
from conductor import Conductor
from experimenter.experimenter import Experimenter

STRUCTURAL_MISSION = (
    "Use prior knowledge to guide scientific research without laundering it "
    "into settled fact."
)

STRUCTURAL_QUERY = (
    "How should prior knowledge and external evidence interact during "
    "scientific reasoning?"
)

SAMPLE_EVIDENCE_TEXT = """
Hippocampal replay is increasingly understood as a selective process rather than
a passive memory echo. Replay tends to prioritize trajectories that are useful
for future planning, especially under uncertainty or changing reward structure.
This implies that prior internal models can guide what is replayed, but replay
still remains constrained by contact with experienced trajectories and observed
outcomes. A scientific reasoning system could use the same division of labor:
background knowledge should propose search directions, while externally sourced
evidence should determine what becomes established knowledge.
""".strip()

SAMPLE_PRIOR_TEXT = """
Scientists almost never begin from an empty state. The productive role of prior
knowledge is to narrow the search space, suggest mechanisms, and propose better
questions. That prior should remain explicitly provisional until evidence or
directly cited sources support it.
""".strip()

THINKER_QUESTION = (
    "What operating rule should the scientist follow when combining prior "
    "knowledge with new evidence?"
)

LIVE_SMOKE_CYCLES = 2


def _print_header(title: str):
    print(f"\n== {title} ==")


def _print_workspace(workspace):
    print(workspace.to_prompt_context())


def _print_checks(checks: list[tuple[str, bool]]):
    passed = True
    _print_header("Checks")
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        passed = passed and ok
    return passed


def _add_node(brain: Brain, statement: str, node_type: NodeType,
              epistemic_status: str, importance: float = 0.7,
              source_ids: list[str] | None = None,
              source_refs: list[str] | None = None,
              created_by: str = "smoke") -> str:
    node = Node(
        statement=statement,
        node_type=node_type,
        status=NodeStatus.UNCERTAIN,
        epistemic_status=epistemic_status,
        importance=importance,
        source_quality=importance,
        source_ids=list(source_ids or []),
        source_refs=list(source_refs or []),
        created_by=created_by,
        cluster="smoke",
    )
    return brain.add_node(node)


def run_structural_demo() -> int:
    brain = Brain()
    brain.set_mission(STRUCTURAL_MISSION)

    source_id = brain.create_source_node(
        title="Example paper on replay and planning",
        reference="https://example.org/replay-planning",
        source_type="paper",
        created_by="smoke",
        excerpt="Replay can guide planning while remaining constrained by experience.",
    )

    _add_node(
        brain,
        "Replay can support planning by revisiting experienced trajectories that "
        "matter for future decisions.",
        NodeType.EVIDENCE_CLAIM,
        ArtifactStatus.GROUNDED.value,
        importance=0.82,
        source_ids=[source_id],
        source_refs=["https://example.org/replay-planning"],
    )
    _add_node(
        brain,
        "Prior theoretical knowledge should narrow search before evidence arrives.",
        NodeType.HYPOTHESIS,
        ArtifactStatus.PRIOR.value,
        importance=0.73,
    )
    _add_node(
        brain,
        "How should the system separate grounded conclusions from prior-based conjectures?",
        NodeType.QUESTION,
        ArtifactStatus.OPEN.value,
        importance=0.74,
    )
    _add_node(
        brain,
        "Design prompts that force grounded claims and prior claims into separate fields.",
        NodeType.TASK,
        ArtifactStatus.OPEN.value,
        importance=0.68,
    )

    workspace = brain.build_workspace(query=STRUCTURAL_QUERY)

    _print_header("Structural Workspace")
    _print_workspace(workspace)

    checks = [
        ("workspace contains grounded evidence", len(workspace.grounded_evidence) >= 1),
        ("workspace contains prior claims", len(workspace.prior_claims) >= 1),
        ("workspace contains active questions", len(workspace.active_questions) >= 1),
        ("workspace contains next tasks", len(workspace.next_tasks) >= 1),
    ]
    return 0 if _print_checks(checks) else 1


def run_live_model_demo() -> int:
    brain = Brain()
    brain.set_mission(STRUCTURAL_MISSION)

    index = EmbeddingIndex()
    observer = Observer(brain)
    critic = Critic(brain, embedding_index=index)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=index, critic=critic)
    thinker = Thinker(brain, observer=observer, embedding_index=index, critic=critic)
    dreamer = Dreamer(brain, research_agenda=observer, critic=critic)
    researcher = Researcher(brain, observer=observer, depth="shallow", ingestor=ingestor, embedding_index=index, critic=critic)
    reader = Reader(brain, observer=observer, ingestor=ingestor, embedding_index=index, critic=critic)
    experimenter = Experimenter(brain)
    insight_buffer = InsightBuffer(brain, embedding_index=index)
    notebook = Notebook(brain, observer=observer)

    conductor = Conductor(
        brain=brain,
        dreamer=dreamer,
        thinker=thinker,
        researcher=researcher,
        reader=reader,
        observer=observer,
        notebook=notebook,
        ingestor=ingestor,
        embedding_index=index,
        insight_buffer=insight_buffer,
        critic=critic,
        experimenter=experimenter
    )

    ingestor.ingest(
        SAMPLE_PRIOR_TEXT,
        source=EdgeSource.CONVERSATION,
        created_by="smoke",
    )
    reader.add_text(SAMPLE_EVIDENCE_TEXT, title="Replay and planning note")

    _print_header("Conductor: Starting Autonomous Loop")
    for _ in range(LIVE_SMOKE_CYCLES):
        conductor.run_cycle(mode="mission_driven")

    _print_header("Brain Stats")
    import json
    print(json.dumps(brain.stats(), indent=2))

    checks = [
        (
            f"Conductor ran {LIVE_SMOKE_CYCLES} cycles",
            conductor.cycle == LIVE_SMOKE_CYCLES,
        ),
        ("Brain has nodes", brain.stats()['nodes'] > 0),
    ]

    return 0 if _print_checks(checks) else 1


def run_llm_demo() -> int:
    """Backward-compatible alias for older callers."""
    return run_live_model_demo()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the evidence-grounded scientist workbench."
    )
    parser.add_argument(
        "--mode",
        choices=["structural", "live", "llm"],
        default="structural",
        help=(
            "Use 'structural' for deterministic architecture checks. "
            "Use 'live' (or legacy alias 'llm') for optional end-to-end model smoke."
        ),
    )
    args = parser.parse_args()

    if args.mode == "structural":
        return run_structural_demo()
    return run_live_model_demo()


if __name__ == "__main__":
    raise SystemExit(main())
