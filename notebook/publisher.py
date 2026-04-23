import os
import json
import time
from typing import Dict, Any
from graph.brain import Brain, NodeType, NodeStatus
from scientist_workspace import citation_anchors_for_node
from persistence import atomic_write_json
from integrity_check import run_integrity_check
from scientific_rigor import external_scientific_references, is_local_artifact_reference

class Publisher:
    def __init__(self, brain: Brain,
                 output_dir: str = "virtual_lab/publications/",
                 embedding_index=None,
                 observer=None,
                 notebook=None,
                 insight_buffer=None):
        self.brain = brain
        self.output_dir = output_dir
        self.embedding_index = embedding_index
        self.observer = observer
        self.notebook = notebook
        self.insight_buffer = insight_buffer
        self.lab_meeting = None
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        from llm_utils import llm_call
        return llm_call(prompt, temperature=temperature, role="publication")

    def _citation_anchor_block(self, nodes: list[tuple[str, dict]],
                               max_nodes: int = 10,
                               max_anchors_per_node: int = 2) -> str:
        lines = []
        seen = set()
        for _, data in nodes[:max_nodes]:
            statement = " ".join(str(data.get('statement', '') or '').split())
            if len(statement) > 110:
                statement = statement[:107].rstrip() + "..."
            anchors = citation_anchors_for_node(data, max_items=max_anchors_per_node)
            for anchor in anchors:
                if is_local_artifact_reference(anchor):
                    continue
                key = (statement, anchor)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {statement} -> {anchor}")
        return "\n".join(lines)

    def _review_blockers(self, hypothesis_id: str) -> list[dict]:
        if not self.lab_meeting or not hasattr(self.lab_meeting, "get_hypothesis_state"):
            return []
        try:
            state = self.lab_meeting.get_hypothesis_state(hypothesis_id)
        except Exception:
            return []
        return state.get("unresolved_objections", []) or []

    def draft_publication(self, cycle: int) -> str:
        """Generate a synthesis paper from a frozen snapshot and prune safely."""
        print(f"\\n  🎓 ── THE SYNTHESIS EVENT (Epoch {cycle//30}) ──")

        # Freeze a read-only snapshot so manuscript content is stable even if
        # live graph mutations happen during publication.
        if hasattr(self.brain, "snapshot_nodes"):
            snapshot_nodes = self.brain.snapshot_nodes()
        else:
            snapshot_nodes = [(n, dict(d)) for n, d in self.brain.all_nodes()]
        
        top_nodes = sorted(snapshot_nodes, key=lambda x: x[1].get('importance', 0), reverse=True)[:24]
        mission = self.brain.mission.get("question", "No mission") if self.brain.mission else "No mission"

        established = []
        supported_hypotheses = []
        blocked_hypotheses = []
        falsified_paths = []

        for node_id, data in top_nodes:
            node_type = data.get("node_type", "")
            status = data.get("status", "")
            epistemic = data.get("epistemic_status", "")
            refs = external_scientific_references(data)
            if (
                node_type in {
                    NodeType.EVIDENCE_CLAIM.value,
                    NodeType.EMPIRICAL.value,
                    NodeType.ANSWER.value,
                } and
                epistemic == "grounded" and
                refs
            ):
                established.append(
                    f"- {data.get('statement', '')} | refs: {', '.join(refs[:2])}"
                )
                continue
            if node_type == NodeType.HYPOTHESIS.value:
                blockers = self._review_blockers(node_id)
                line = f"- {data.get('statement', '')}"
                if status == NodeStatus.CONTRADICTED.value or epistemic == "contradicted":
                    falsified_paths.append(line)
                elif status == NodeStatus.LACKS_EVIDENCE.value or epistemic == "lacks_evidence":
                    blocked_hypotheses.append(line + " | lacks evidence")
                elif blockers:
                    blocker_labels = ", ".join(
                        sorted({item.get("issue_label", "issue") for item in blockers if item.get("issue_label")})
                    )
                    blocked_hypotheses.append(
                        line + f" | unresolved objections={len(blockers)} ({blocker_labels or 'review'})"
                    )
                else:
                    supported_hypotheses.append(line + " | review blockers=0")

        citation_block = self._citation_anchor_block(top_nodes)
        raw_title = self._llm(
            f'Generate a conservative markdown title line for a scientific research memo about: "{mission}"\n'
            "Return one line only. Prefer the form '# Title: ...'.",
            temperature=0.05,
        ).strip()
        title_line = raw_title.splitlines()[0].strip() if raw_title else ""
        if not title_line.startswith("#"):
            title_line = f"# Epoch {cycle//30} Research Memo"
        article = "\n".join([
            title_line,
            "",
            "## Mission",
            mission,
            "",
            "## Publication Verdict",
            (
                "No resolution claim is justified yet. This memo records established findings, surviving hypotheses, and remaining blockers."
                if not supported_hypotheses else
                "At least one working hypothesis currently survives available review blockers, but this memo remains a research summary rather than a resolved theory claim."
            ),
            "",
            "## Established Findings",
            "\n".join(established[:8]) if established else "- none",
            "",
            "## Surviving Working Hypotheses",
            "\n".join(supported_hypotheses[:6]) if supported_hypotheses else "- none",
            "",
            "## Open Hypotheses and Review Blockers",
            "\n".join(blocked_hypotheses[:8]) if blocked_hypotheses else "- none",
            "",
            "## Falsified or Weakened Paths",
            "\n".join(falsified_paths[:6]) if falsified_paths else "- none",
            "",
            "## Next Scientific Requirements",
            "- derive explicit mechanism where missing",
            "- close unresolved lab-meeting objections before promotion",
            "- separate external evidence from self-generated virtual-lab artifacts",
            "- treat failed or crashed experiments as blockers, not support",
        ])

        if citation_block:
            article = article + "\n\n## Citation Anchors\n" + citation_block
        
        filename = os.path.join(self.output_dir, f"synthesis_epoch_{cycle//30}.md")
        with open(filename, "w") as f:
            f.write(article)
            
        print(f"  📝 Publication saved to {filename}")
        
        # 2. Pruning
        self._prune_graph(snapshot_nodes=snapshot_nodes, cycle=cycle)
        return article
        
    def _prune_graph(self, snapshot_nodes: list[tuple[str, dict]] | None = None,
                     cycle: int = 0) -> dict:
        """Prune low-importance nodes via synchronized API to prevent dangling refs."""
        if snapshot_nodes is None:
            if hasattr(self.brain, "snapshot_nodes"):
                snapshot_nodes = self.brain.snapshot_nodes()
            else:
                snapshot_nodes = [(n, dict(d)) for n, d in self.brain.all_nodes()]

        mission_id = (self.brain.mission or {}).get("id") if self.brain.mission else None

        candidates_to_prune = [
            n for n, d in snapshot_nodes
            if d.get('node_type') not in [
                NodeType.EVIDENCE_CLAIM.value,
                NodeType.SOURCE.value,
                NodeType.MISSION.value,
            ]
            and d.get('importance', 0) < 0.4
            and n != mission_id
        ]
        
        # Sort by lowest importance
        importance_by_id = {
            nid: float(data.get("importance", 0) or 0)
            for nid, data in snapshot_nodes
        }
        candidates_to_prune.sort(key=lambda n: importance_by_id.get(n, 0.0))
        
        # Prune 70% of candidates
        num_prune = int(len(candidates_to_prune) * 0.7)
        to_remove = candidates_to_prune[:num_prune]

        preserve_ids = {mission_id} if mission_id else set()
        prune_result = self.brain.synchronized_prune(
            node_ids=to_remove,
            embedding_index=self.embedding_index,
            observer=self.observer,
            notebook=self.notebook,
            insight_buffer=self.insight_buffer,
            preserve_ids=preserve_ids,
        )

        removed_count = int(prune_result.get("removed_count", 0) or 0)
        print(f"  🧹 Graph Pruning: Removed {removed_count} low-importance nodes to clear context.")

        prune_report = {
            "cycle": cycle,
            "requested_prune_count": num_prune,
            "requested_node_ids": to_remove,
            "removed_node_ids": prune_result.get("removed_node_ids", []),
            "removed_count": removed_count,
            "sync": prune_result.get("sync", {}),
        }

        integrity_report = run_integrity_check(
            brain=self.brain,
            embedding_index=self.embedding_index,
            observer=self.observer,
            notebook=self.notebook,
            insight_buffer=self.insight_buffer,
        )
        prune_report["integrity_check"] = integrity_report
        if not integrity_report.get("ok", False):
            print(
                "  ⚠️ Integrity check failed after prune: "
                f"{integrity_report.get('error_count', 0)} issue(s)"
            )

        epoch = cycle // 30 if cycle else 0
        report_path = os.path.join(self.output_dir, f"synthesis_epoch_{epoch}_prune_report.json")
        atomic_write_json(report_path, prune_report)
        return prune_report
