from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArtifactStatus(str, Enum):
    GROUNDED = "grounded"
    PRIOR = "prior"
    SPECULATIVE = "speculative"
    CONTRADICTED = "contradicted"
    OPEN = "open"
    LACKS_EVIDENCE = "lacks_evidence"  # searched but found nothing either way


def format_citation_anchor(span: dict, quote_chars: int = 96) -> str:
    """Render a compact, citation-ready provenance anchor."""
    if not isinstance(span, dict):
        return ""

    source_ref = str(span.get("source_ref", "") or "").strip()
    section_label = str(span.get("section_label", "") or "").strip()
    quote = " ".join(str(span.get("quote", "") or "").split())
    if len(quote) > quote_chars:
        quote = quote[: quote_chars - 3].rstrip() + "..."

    try:
        confidence = float(span.get("extraction_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    parts = []
    if source_ref:
        parts.append(source_ref)
    if section_label:
        parts.append(f"section={section_label}")
    if quote:
        parts.append(f'quote="{quote}"')
    parts.append(f"conf={confidence:.2f}")
    return " | ".join(parts)


def citation_anchors_for_node(node_data: dict | None, max_items: int = 2) -> list[str]:
    if not isinstance(node_data, dict):
        return []

    anchors = []
    spans = node_data.get("provenance_spans", []) or []
    for span in spans:
        anchor = format_citation_anchor(span)
        if anchor and anchor not in anchors:
            anchors.append(anchor)
        if len(anchors) >= max_items:
            return anchors

    for ref in node_data.get("source_refs", []) or []:
        ref_text = str(ref or "").strip()
        if ref_text and ref_text not in anchors:
            anchors.append(ref_text)
        if len(anchors) >= max_items:
            break
    return anchors


@dataclass
class WorkspaceNode:
    id: str
    node_type: str
    statement: str
    epistemic_status: str = ArtifactStatus.OPEN.value
    importance: float = 0.0
    mission_relevance: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    provenance_spans: list[dict] = field(default_factory=list)
    extraction_confidence: float = 0.0
    created_by: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "statement": self.statement,
            "epistemic_status": self.epistemic_status,
            "importance": self.importance,
            "mission_relevance": self.mission_relevance,
            "source_ids": list(self.source_ids),
            "source_refs": list(self.source_refs),
            "provenance_spans": [dict(span) for span in self.provenance_spans],
            "extraction_confidence": self.extraction_confidence,
            "created_by": self.created_by,
        }

    def prompt_line(self) -> str:
        refs = ""
        anchors = citation_anchors_for_node(
            {
                "provenance_spans": self.provenance_spans,
                "source_refs": self.source_refs,
            },
            max_items=2,
        )
        if anchors:
            refs = f" | citations: {' ; '.join(anchors)}"
        return (
            f"- [{self.node_type}/{self.epistemic_status}] "
            f"(importance={self.importance:.2f}, mission={self.mission_relevance:.2f}) "
            f"{self.statement}{refs}"
        )


@dataclass
class ScientistWorkspace:
    mission: str = ""
    mission_context: str = ""
    active_questions: list[WorkspaceNode] = field(default_factory=list)
    grounded_evidence: list[WorkspaceNode] = field(default_factory=list)
    working_hypotheses: list[WorkspaceNode] = field(default_factory=list)
    prior_claims: list[WorkspaceNode] = field(default_factory=list)
    next_tasks: list[WorkspaceNode] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mission": self.mission,
            "mission_context": self.mission_context,
            "active_questions": [node.to_dict() for node in self.active_questions],
            "grounded_evidence": [node.to_dict() for node in self.grounded_evidence],
            "working_hypotheses": [node.to_dict() for node in self.working_hypotheses],
            "prior_claims": [node.to_dict() for node in self.prior_claims],
            "next_tasks": [node.to_dict() for node in self.next_tasks],
            "contradictions": list(self.contradictions),
        }

    def to_prompt_context(self) -> str:
        def _section(title: str, nodes: list[WorkspaceNode]) -> str:
            if not nodes:
                return f"{title}:\n- none"
            return f"{title}:\n" + "\n".join(node.prompt_line() for node in nodes)

        contradiction_block = (
            "Contradictions:\n" + "\n".join(f"- {item}" for item in self.contradictions)
            if self.contradictions else
            "Contradictions:\n- none"
        )

        return "\n\n".join([
            f"Mission:\n- {self.mission or 'none'}",
            (
                f"Mission context:\n- {self.mission_context}"
                if self.mission_context else
                "Mission context:\n- none"
            ),
            _section("Grounded evidence", self.grounded_evidence),
            _section("Working hypotheses", self.working_hypotheses),
            _section("Prior claims", self.prior_claims),
            _section("Active questions", self.active_questions),
            _section("Next tasks", self.next_tasks),
            contradiction_block,
        ])


@dataclass
class ReasoningResult:
    grounded_claims: list[str] = field(default_factory=list)
    prior_claims: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    summary_claim: str = ""

    def to_dict(self) -> dict:
        return {
            "grounded_claims": list(self.grounded_claims),
            "prior_claims": list(self.prior_claims),
            "hypotheses": list(self.hypotheses),
            "open_questions": list(self.open_questions),
            "next_actions": list(self.next_actions),
            "summary_claim": self.summary_claim,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ReasoningResult":
        data = data or {}
        return cls(
            grounded_claims=[
                item for item in data.get("grounded_claims", [])
                if isinstance(item, str) and item.strip()
            ],
            prior_claims=[
                item for item in data.get("prior_claims", [])
                if isinstance(item, str) and item.strip()
            ],
            hypotheses=[
                item for item in data.get("hypotheses", [])
                if isinstance(item, str) and item.strip()
            ],
            open_questions=[
                item for item in data.get("open_questions", [])
                if isinstance(item, str) and item.strip()
            ],
            next_actions=[
                item for item in data.get("next_actions", [])
                if isinstance(item, str) and item.strip()
            ],
            summary_claim=(data.get("summary_claim") or "").strip(),
        )
