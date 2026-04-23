import re
from urllib.parse import urlparse


HIGH_RIGOR_DOMAINS = {
    "arxiv.org",
    "doi.org",
    "journals.aps.org",
    "physics.aps.org",
    "aps.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "link.springer.com",
    "springer.com",
    "iopscience.iop.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "openreview.net",
    "acm.org",
    "dl.acm.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "inspirehep.net",
    "pdg.lbl.gov",
    "cern.ch",
    "fermilab.gov",
}

LOW_RIGOR_DOMAINS = {
    "lesswrong.com",
    "medium.com",
    "substack.com",
    "reddit.com",
    "quora.com",
    "youtube.com",
    "academia.edu",
    "researchgate.net",
    "wikipedia.org",
    "towardsdatascience.com",
}

FORMALISM_MARKERS = {
    "lagrangian",
    "hamiltonian",
    "operator",
    "coupling",
    "symmetry",
    "sector",
    "field",
    "axion",
    "curvature",
    "torsion",
    "geometry",
    "gravitational",
    "qcd",
    "theta angle",
    "dipole moment",
    "mirror sector",
}


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def text_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2
    }


def reference_domain(reference: str) -> str:
    try:
        domain = urlparse(str(reference or "")).netloc.lower().strip()
    except Exception:
        domain = ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _domain_matches(domain: str, candidates: set[str]) -> bool:
    return any(domain == item or domain.endswith(f".{item}") for item in candidates)


def is_local_artifact_reference(reference: str) -> bool:
    ref = str(reference or "").strip()
    if not ref:
        return False
    lowered = ref.lower()
    if lowered.startswith("file://"):
        return True
    if lowered.startswith(("logs/", "data/", "virtual_lab/", "./", "../", "/")):
        return True
    parsed = urlparse(ref)
    if parsed.scheme in {"http", "https", "doi"}:
        return False
    return bool(parsed.scheme) and parsed.scheme not in {"arxiv", "virtual_lab"}


def source_rigor_score(reference: str) -> float:
    ref = str(reference or "").strip().lower()
    if not ref:
        return 0.0
    if is_local_artifact_reference(ref):
        return 0.0
    if ref.startswith("virtual_lab://"):
        return 0.15
    if "arxiv.org/abs/" in ref or "arxiv.org/pdf/" in ref:
        return 1.0
    domain = reference_domain(ref)
    if not domain:
        return 0.4
    if _domain_matches(domain, HIGH_RIGOR_DOMAINS):
        return 0.95
    if _domain_matches(domain, LOW_RIGOR_DOMAINS):
        return 0.1
    if domain.endswith(".gov"):
        return 0.85
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return 0.72
    if domain.endswith(".org"):
        return 0.6
    return 0.45


def external_scientific_references(node_data: dict | None) -> list[str]:
    if not isinstance(node_data, dict):
        return []
    refs = []
    for ref in node_data.get("source_refs", []) or []:
        ref_text = normalize_text(ref)
        if not ref_text or is_local_artifact_reference(ref_text):
            continue
        if ref_text not in refs:
            refs.append(ref_text)
    for span in node_data.get("provenance_spans", []) or []:
        if not isinstance(span, dict):
            continue
        ref_text = normalize_text(span.get("source_ref", ""))
        if not ref_text or is_local_artifact_reference(ref_text):
            continue
        if ref_text not in refs:
            refs.append(ref_text)
    return refs


def local_artifact_references(node_data: dict | None) -> list[str]:
    if not isinstance(node_data, dict):
        return []
    refs = []
    for ref in node_data.get("source_refs", []) or []:
        ref_text = normalize_text(ref)
        if ref_text and is_local_artifact_reference(ref_text) and ref_text not in refs:
            refs.append(ref_text)
    for span in node_data.get("provenance_spans", []) or []:
        if not isinstance(span, dict):
            continue
        ref_text = normalize_text(span.get("source_ref", ""))
        if ref_text and is_local_artifact_reference(ref_text) and ref_text not in refs:
            refs.append(ref_text)
    return refs


def highest_reference_rigor(node_data: dict | None) -> float:
    refs = external_scientific_references(node_data)
    if not refs:
        return 0.0
    return max(source_rigor_score(ref) for ref in refs)


def hypothesis_requires_formalism(text: str) -> bool:
    lowered = normalize_text(text).lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in FORMALISM_MARKERS)


def deterministic_progress_stage(grounded_count: int, supported_hypotheses: int,
                                 blocker_count: int) -> str:
    if grounded_count <= 1 and supported_hypotheses == 0:
        return "early"
    if blocker_count >= 3:
        return "blocked"
    if grounded_count >= 4 and supported_hypotheses >= 1 and blocker_count <= 1:
        return "intermediate"
    if grounded_count >= 6 and supported_hypotheses >= 2 and blocker_count == 0:
        return "advanced"
    return "partial"
