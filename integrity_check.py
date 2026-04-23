import argparse
import json
import os
import re
from typing import Any

from graph.brain import Brain


NODE_TAG_RE = re.compile(r"^node:(.+)$")


def _get(obj: Any, key: str, default: Any):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _field(item: Any, key: str, default: Any):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _add_issue(issues: list[dict], section: str, code: str,
               detail: str, reference: str = ""):
    issues.append({
        "section": section,
        "code": code,
        "detail": detail,
        "reference": reference,
    })


def run_integrity_check(brain: Brain,
                        embedding_index=None,
                        observer=None,
                        notebook=None,
                        insight_buffer=None) -> dict:
    """Validate cross-component references and detect dangling IDs."""
    issues: list[dict] = []
    node_ids = set(brain.graph.nodes)

    checked = {
        "brain_nodes": len(node_ids),
        "brain_edges": len(list(brain.graph.edges)),
        "working_memory_refs": 0,
        "embedding_refs": 0,
        "observer_refs": 0,
        "notebook_refs": 0,
        "insight_buffer_refs": 0,
    }

    # Brain-level integrity
    mission = brain.get_mission() or {}
    mission_id = mission.get("id")
    if mission_id and mission_id not in node_ids:
        _add_issue(
            issues,
            section="brain",
            code="mission_missing_node",
            detail=f"Mission node '{mission_id}' is missing from graph.",
            reference="mission.id",
        )

    suspended = getattr(brain, "_suspended_mission", None) or {}
    suspended_id = suspended.get("id") if isinstance(suspended, dict) else None
    if suspended_id and suspended_id not in node_ids:
        _add_issue(
            issues,
            section="brain",
            code="suspended_mission_missing_node",
            detail=(
                f"Suspended mission node '{suspended_id}' is missing from graph."
            ),
            reference="_suspended_mission.id",
        )

    for idx, node_id in enumerate(list(getattr(brain, "working_memory", []))):
        checked["working_memory_refs"] += 1
        if node_id not in node_ids:
            _add_issue(
                issues,
                section="brain",
                code="working_memory_dangling",
                detail=f"Working memory references missing node '{node_id}'.",
                reference=f"working_memory[{idx}]",
            )

    for u, v in brain.graph.edges:
        if u not in node_ids or v not in node_ids:
            _add_issue(
                issues,
                section="brain",
                code="edge_dangling_endpoint",
                detail=f"Edge endpoint missing for ({u} -> {v}).",
                reference="graph.edges",
            )

    # Embedding index integrity
    if embedding_index is not None:
        embeddings = getattr(embedding_index, "_embeddings", {}) or {}
        for node_id in embeddings.keys():
            checked["embedding_refs"] += 1
            if node_id not in node_ids:
                _add_issue(
                    issues,
                    section="embedding_index",
                    code="embedding_dangling_node",
                    detail=f"Embedding stored for removed node '{node_id}'.",
                    reference="_embeddings",
                )

        id_to_int = getattr(embedding_index, "_id_to_int", {}) or {}
        int_to_id = getattr(embedding_index, "_int_to_id", {}) or {}
        for node_id in id_to_int.keys():
            if node_id not in embeddings:
                _add_issue(
                    issues,
                    section="embedding_index",
                    code="id_map_missing_embedding",
                    detail=(
                        f"id_to_int references '{node_id}' but embedding is missing."
                    ),
                    reference="_id_to_int",
                )
        for int_id, node_id in int_to_id.items():
            if node_id not in id_to_int:
                _add_issue(
                    issues,
                    section="embedding_index",
                    code="reverse_map_missing_forward",
                    detail=(
                        f"int_to_id key '{int_id}' maps to '{node_id}' absent from id_to_int."
                    ),
                    reference="_int_to_id",
                )

    # Observer integrity
    if observer is not None:
        agenda = _get(observer, "agenda", []) or []
        for idx, item in enumerate(agenda):
            node_id = str(_field(item, "node_id", "") or "")
            answer_node_id = str(_field(item, "answer_node_id", "") or "")
            partial_leads = _field(item, "partial_leads", []) or []

            if node_id:
                checked["observer_refs"] += 1
                if node_id not in node_ids:
                    _add_issue(
                        issues,
                        section="observer",
                        code="agenda_node_dangling",
                        detail=f"Agenda node '{node_id}' does not exist.",
                        reference=f"agenda[{idx}].node_id",
                    )

            if answer_node_id:
                checked["observer_refs"] += 1
                if answer_node_id not in node_ids:
                    _add_issue(
                        issues,
                        section="observer",
                        code="agenda_answer_dangling",
                        detail=f"Agenda answer node '{answer_node_id}' does not exist.",
                        reference=f"agenda[{idx}].answer_node_id",
                    )

            for lid, lead_id in enumerate(partial_leads):
                lead_id = str(lead_id or "")
                if not lead_id:
                    continue
                checked["observer_refs"] += 1
                if lead_id not in node_ids:
                    _add_issue(
                        issues,
                        section="observer",
                        code="agenda_partial_lead_dangling",
                        detail=f"Partial lead node '{lead_id}' does not exist.",
                        reference=f"agenda[{idx}].partial_leads[{lid}]",
                    )

        mission_advances = _get(observer, "mission_advances", []) or []
        for idx, advance in enumerate(mission_advances):
            node_id = str(_field(advance, "node_id", "") or "")
            if not node_id:
                continue
            checked["observer_refs"] += 1
            if node_id not in node_ids:
                _add_issue(
                    issues,
                    section="observer",
                    code="mission_advance_dangling",
                    detail=f"Mission advance node '{node_id}' does not exist.",
                    reference=f"mission_advances[{idx}].node_id",
                )

        emergence_feed = _get(observer, "emergence_feed", None)
        if emergence_feed is None:
            emergence_feed = _get(observer, "emergences", []) or []
        for sidx, signal in enumerate(emergence_feed):
            signal_node_ids = _field(signal, "node_ids", []) or []
            for nidx, node_id in enumerate(signal_node_ids):
                node_id = str(node_id or "")
                if not node_id:
                    continue
                checked["observer_refs"] += 1
                if node_id not in node_ids:
                    _add_issue(
                        issues,
                        section="observer",
                        code="emergence_node_dangling",
                        detail=f"Emergence signal references missing node '{node_id}'.",
                        reference=f"emergence_feed[{sidx}].node_ids[{nidx}]",
                    )

        edge_traversal_counts = _get(observer, "edge_traversal_counts", {}) or {}
        for key in edge_traversal_counts.keys():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                from_id, to_id = str(key[0]), str(key[1])
            elif isinstance(key, str) and "|" in key:
                from_id, to_id = key.split("|", 1)
            else:
                continue

            if from_id:
                checked["observer_refs"] += 1
                if from_id not in node_ids:
                    _add_issue(
                        issues,
                        section="observer",
                        code="edge_traversal_from_dangling",
                        detail=f"Edge traversal references missing node '{from_id}'.",
                        reference="edge_traversal_counts",
                    )
            if to_id:
                checked["observer_refs"] += 1
                if to_id not in node_ids:
                    _add_issue(
                        issues,
                        section="observer",
                        code="edge_traversal_to_dangling",
                        detail=f"Edge traversal references missing node '{to_id}'.",
                        reference="edge_traversal_counts",
                    )

    # Notebook integrity
    if notebook is not None:
        entries = _get(notebook, "entries", []) or []
        for eidx, entry in enumerate(entries):
            tags = _field(entry, "tags", []) or []
            for tidx, tag in enumerate(tags):
                match = NODE_TAG_RE.match(str(tag or ""))
                if not match:
                    continue
                node_id = match.group(1).strip()
                if not node_id:
                    continue
                checked["notebook_refs"] += 1
                if node_id not in node_ids:
                    _add_issue(
                        issues,
                        section="notebook",
                        code="node_tag_dangling",
                        detail=f"Notebook tag references missing node '{node_id}'.",
                        reference=f"entries[{eidx}].tags[{tidx}]",
                    )

    # Insight buffer integrity
    if insight_buffer is not None:
        pending = insight_buffer
        if not isinstance(pending, list):
            pending = _get(insight_buffer, "pending", []) or []

        for pidx, pair in enumerate(pending):
            is_node = bool(_field(pair, "is_node", False))
            if is_node:
                continue
            node_a_id = str(_field(pair, "node_a_id", "") or "")
            node_b_id = str(_field(pair, "node_b_id", "") or "")
            if node_a_id:
                checked["insight_buffer_refs"] += 1
                if node_a_id not in node_ids:
                    _add_issue(
                        issues,
                        section="insight_buffer",
                        code="pending_pair_a_dangling",
                        detail=f"Pending pair references missing node '{node_a_id}'.",
                        reference=f"pending[{pidx}].node_a_id",
                    )
            if node_b_id:
                checked["insight_buffer_refs"] += 1
                if node_b_id not in node_ids:
                    _add_issue(
                        issues,
                        section="insight_buffer",
                        code="pending_pair_b_dangling",
                        detail=f"Pending pair references missing node '{node_b_id}'.",
                        reference=f"pending[{pidx}].node_b_id",
                    )

    report = {
        "ok": len(issues) == 0,
        "error_count": len(issues),
        "checked": checked,
        "issues": issues,
    }
    return report


def _load_json_if_exists(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def run_integrity_check_from_files(brain_path: str = "data/brain.json",
                                   embedding_index_path: str | None = "data/embedding_index",
                                   observer_path: str | None = "data/observer.json",
                                   notebook_path: str | None = "data/notebook.json",
                                   insight_buffer_path: str | None = "data/insight_buffer.json") -> dict:
    brain = Brain()
    brain.load(brain_path)

    embedding_index = None
    if embedding_index_path:
        faiss_path = embedding_index_path + ".faiss"
        meta_path = embedding_index_path + ".json"
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            from embedding_index import EmbeddingIndex
            embedding_index = EmbeddingIndex.load(embedding_index_path)

    observer = _load_json_if_exists(observer_path) if observer_path else None
    notebook = _load_json_if_exists(notebook_path) if notebook_path else None
    insight_buffer = _load_json_if_exists(insight_buffer_path) if insight_buffer_path else None

    return run_integrity_check(
        brain=brain,
        embedding_index=embedding_index,
        observer=observer,
        notebook=notebook,
        insight_buffer=insight_buffer,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cross-component integrity checks for the scientist state."
    )
    parser.add_argument("--brain", default="data/brain.json", help="Path to brain JSON state")
    parser.add_argument("--embedding-index", default="data/embedding_index",
                        help="Path prefix for embedding index (.faiss/.json)")
    parser.add_argument("--observer", default="data/observer.json", help="Path to observer JSON state")
    parser.add_argument("--notebook", default="data/notebook.json", help="Path to notebook JSON state")
    parser.add_argument("--insight-buffer", default="data/insight_buffer.json",
                        help="Path to insight buffer JSON state")
    parser.add_argument("--output", default="",
                        help="Optional path to save integrity report JSON")
    args = parser.parse_args()

    report = run_integrity_check_from_files(
        brain_path=args.brain,
        embedding_index_path=args.embedding_index,
        observer_path=args.observer,
        notebook_path=args.notebook,
        insight_buffer_path=args.insight_buffer,
    )

    print(json.dumps(report, indent=2))

    if args.output:
        from persistence import atomic_write_json
        atomic_write_json(args.output, report)

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
