import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Optional

import networkx as nx
from networkx.algorithms import isomorphism as iso

from graph.brain import Brain, EdgeSource, NodeType
from llm_utils import llm_call, require_json


STRUCTURE_PROMPT = """
You are a structural abstraction engine.

Translate the scientific concept below into an explicit, domain-agnostic structure.
Use compact abstract roles such as A, B, C. Keep relation labels generic.

Concept:
"{concept}"

Context:
{context}

Return JSON only:
{{
  "roles": ["A", "B", "C"],
  "relations": [
    {{"from": "A", "type": "promotes", "to": "B"}},
    {{"from": "B", "type": "inhibits", "to": "C"}}
  ],
  "constraints": ["bounded resource", "thresholded response"],
  "update_rules": ["A increases B each step", "B suppresses C when active"],
  "objective": "stabilize output while preserving responsiveness"
}}
"""

EXPLANATION_PROMPT = """
Two scientific ideas passed a deterministic structural isomorphism matcher.

Structure 1:
{struct1}

Structure 2:
{struct2}

Matcher report:
{report}

Return JSON only:
{{
  "unified_pattern": "short structural pattern name",
  "justification": "2-3 sentence explanation grounded in the supplied mapping"
}}
"""

RELATION_ALIASES = {
    "activate": "positive",
    "activates": "positive",
    "promote": "positive",
    "promotes": "positive",
    "increase": "positive",
    "increases": "positive",
    "amplify": "positive",
    "amplifies": "positive",
    "cause": "positive",
    "causes": "positive",
    "drive": "positive",
    "drives": "positive",
    "inhibit": "negative",
    "inhibits": "negative",
    "suppress": "negative",
    "suppresses": "negative",
    "decrease": "negative",
    "decreases": "negative",
    "reduce": "negative",
    "reduces": "negative",
    "balance": "balance",
    "balances": "balance",
    "compete": "competition",
    "competes": "competition",
    "constrain": "constraint",
    "constrains": "constraint",
    "feedback": "feedback",
    "self": "self",
}

TEXT_ALIASES = {
    "activate": "increase",
    "activates": "increase",
    "promote": "increase",
    "promotes": "increase",
    "increase": "increase",
    "increases": "increase",
    "amplify": "increase",
    "amplifies": "increase",
    "inhibit": "decrease",
    "inhibits": "decrease",
    "suppress": "decrease",
    "suppresses": "decrease",
    "reduce": "decrease",
    "reduces": "decrease",
    "decrease": "decrease",
    "decreases": "decrease",
    "maximise": "maximize",
    "maximizes": "maximize",
    "maximize": "maximize",
    "minimise": "minimize",
    "minimizes": "minimize",
    "minimize": "minimize",
    "stabilise": "stabilize",
    "stabilizes": "stabilize",
    "stabilize": "stabilize",
    "bounded": "bound",
    "bounds": "bound",
    "thresholded": "threshold",
    "thresholds": "threshold",
    "feedback": "feedback",
    "conserved": "conserve",
    "conservation": "conserve",
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "while",
    "when", "where", "than", "then", "each", "step", "active", "state",
    "under", "over", "through", "toward", "towards", "maintain", "maintains",
}

MAX_ISOMORPHISM_SHORTLIST = 16
STRUCTURE_PREFILTER_MIN = 0.20


def _normalize_relation_type(label: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
    return RELATION_ALIASES.get(token, token or "related")


def _normalize_text(text: str) -> str:
    words = []
    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        token = TEXT_ALIASES.get(token, token)
        if len(token) <= 2 or token in STOPWORDS:
            continue
        words.append(token)
    return " ".join(words)


def _token_set(text: str) -> set[str]:
    return set(_normalize_text(text).split())


def _text_counts(text: str) -> Counter:
    return Counter(_normalize_text(text).split())


def _text_similarity(text_a: str, text_b: str) -> float:
    counts_a = _text_counts(text_a)
    counts_b = _text_counts(text_b)
    if not counts_a and not counts_b:
        return 1.0
    if not counts_a or not counts_b:
        return 0.0
    shared = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[token] * counts_b[token] for token in shared)
    if numerator <= 0:
        return 0.0
    norm_a = sqrt(sum(value * value for value in counts_a.values()))
    norm_b = sqrt(sum(value * value for value in counts_b.values()))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _list_similarity(items_a: list[str], items_b: list[str]) -> float:
    if not items_a and not items_b:
        return 1.0
    if not items_a or not items_b:
        return 0.0
    remaining = list(items_b)
    scores = []
    for item_a in items_a:
        best_index = -1
        best_score = -1.0
        for index, item_b in enumerate(remaining):
            score = _text_similarity(item_a, item_b)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0:
            remaining.pop(best_index)
        scores.append(max(best_score, 0.0))
    return sum(scores) / max(len(items_a), len(items_b))


def _parse_roles(raw_roles) -> list[str]:
    roles = []
    if isinstance(raw_roles, list):
        for item in raw_roles:
            if isinstance(item, str):
                role = item.strip()
            elif isinstance(item, dict):
                role = str(
                    item.get("id", "") or
                    item.get("name", "") or
                    item.get("role", "")
                ).strip()
            else:
                role = ""
            if role:
                roles.append(role)
    return list(dict.fromkeys(roles))


def _parse_relation_string(raw_relation: str) -> Optional[tuple[str, str, str]]:
    text = str(raw_relation or "").strip()
    if not text:
        return None
    arrow_match = re.findall(r"([A-Za-z0-9_]+)\s*(?:-+>|→)\s*([A-Za-z0-9_]+)", text)
    if arrow_match:
        source, target = arrow_match[0]
        return source, "positive", target
    triple = re.match(r"([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)", text)
    if triple:
        source, rel_type, target = triple.groups()
        return source, _normalize_relation_type(rel_type), target
    return None


def _parse_relations(raw_relations) -> list[tuple[str, str, str]]:
    relations = []
    if isinstance(raw_relations, list):
        for item in raw_relations:
            parsed = None
            if isinstance(item, dict):
                source = str(item.get("from", "") or item.get("source", "") or "").strip()
                target = str(item.get("to", "") or item.get("target", "") or "").strip()
                rel_type = str(item.get("type", "") or item.get("relation", "") or "").strip()
                if source and target:
                    parsed = (source, _normalize_relation_type(rel_type), target)
            elif isinstance(item, str):
                parsed = _parse_relation_string(item)
            if parsed:
                relations.append(parsed)
    return relations


def _coerce_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass
class StructuralRepresentation:
    roles: list[str]
    relations: list[tuple[str, str, str]]
    constraints: list[str]
    update_rules: list[str]
    objective: str = ""

    def to_dict(self) -> dict:
        return {
            "roles": list(self.roles),
            "relations": [
                {"from": source, "type": rel_type, "to": target}
                for source, rel_type, target in self.relations
            ],
            "constraints": list(self.constraints),
            "update_rules": list(self.update_rules),
            "objective": self.objective,
        }


class IsomorphismEngine:
    def __init__(self, brain: Brain):
        self.brain = brain
        self._structure_cache: dict[tuple[str, str], Optional[dict]] = {}

    def _llm(self, prompt: str, temperature: float = 0.3) -> str:
        return llm_call(prompt, temperature=temperature, role="reasoning")

    def _coerce_structure(self, structure) -> Optional[StructuralRepresentation]:
        if isinstance(structure, StructuralRepresentation):
            return structure

        if isinstance(structure, str):
            raw = structure.strip()
            if not raw:
                return None
            parsed = _parse_relation_string(raw)
            if parsed:
                roles = list(dict.fromkeys([parsed[0], parsed[2]]))
                return StructuralRepresentation(
                    roles=roles,
                    relations=[parsed],
                    constraints=[],
                    update_rules=[raw],
                    objective="",
                )
            return StructuralRepresentation(
                roles=[],
                relations=[],
                constraints=[],
                update_rules=[raw],
                objective="",
            )

        if not isinstance(structure, dict):
            return None

        roles = _parse_roles(
            structure.get("roles") or
            structure.get("entities") or
            structure.get("nodes") or
            []
        )
        relations = _parse_relations(structure.get("relations", []))
        if not roles and relations:
            roles = list(dict.fromkeys(
                [source for source, _, _ in relations] +
                [target for _, _, target in relations]
            ))

        return StructuralRepresentation(
            roles=roles,
            relations=relations,
            constraints=[_normalize_text(item) for item in _coerce_list(structure.get("constraints", []))],
            update_rules=[_normalize_text(item) for item in _coerce_list(structure.get("update_rules", []))],
            objective=_normalize_text(str(structure.get("objective", "") or "").strip()),
        )

    def abstract_node(self, statement: str, context: str = "") -> Optional[dict]:
        cache_key = (str(statement or ""), str(context or ""))
        if cache_key in self._structure_cache:
            cached = self._structure_cache[cache_key]
            return dict(cached) if isinstance(cached, dict) else cached
        raw = self._llm(STRUCTURE_PROMPT.format(concept=statement, context=context), temperature=0.2)
        data = require_json(raw, default={})
        structure = self._coerce_structure(data)
        result = structure.to_dict() if structure else None
        self._structure_cache[cache_key] = dict(result) if isinstance(result, dict) else result
        return result

    def _structure_graph(self, structure: StructuralRepresentation) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for role in structure.roles:
            graph.add_node(role)
        for source, rel_type, target in structure.relations:
            graph.add_node(source)
            graph.add_node(target)
            graph.add_edge(source, target, relation_type=rel_type)
        return graph

    def deterministic_match(self, structure_a, structure_b) -> dict:
        struct_a = self._coerce_structure(structure_a)
        struct_b = self._coerce_structure(structure_b)
        if not struct_a or not struct_b:
            return {
                "passed": False,
                "isomorphic": False,
                "reason": "Invalid structural representation.",
            }

        if len(struct_a.roles) != len(struct_b.roles):
            return {
                "passed": False,
                "isomorphic": False,
                "reason": "Role count mismatch.",
                "matcher_report": {
                    "role_count_a": len(struct_a.roles),
                    "role_count_b": len(struct_b.roles),
                },
            }

        if len(struct_a.relations) != len(struct_b.relations):
            return {
                "passed": False,
                "isomorphic": False,
                "reason": "Relation count mismatch.",
                "matcher_report": {
                    "relation_count_a": len(struct_a.relations),
                    "relation_count_b": len(struct_b.relations),
                },
            }

        if len(struct_a.roles) > 7:
            return {
                "passed": False,
                "isomorphic": False,
                "reason": "Structure too large for deterministic matcher.",
            }

        graph_a = self._structure_graph(struct_a)
        graph_b = self._structure_graph(struct_b)
        matcher = iso.MultiDiGraphMatcher(
            graph_a,
            graph_b,
            edge_match=iso.categorical_multiedge_match("relation_type", None),
        )
        if not matcher.is_isomorphic():
            return {
                "passed": False,
                "isomorphic": False,
                "reason": "No role mapping preserved the relation graph.",
                "matcher_report": {
                    "relation_score": 0.0,
                },
            }

        mapping = dict(matcher.mapping)
        constraint_score = _list_similarity(struct_a.constraints, struct_b.constraints)
        update_rule_score = _list_similarity(struct_a.update_rules, struct_b.update_rules)
        objective_score = _text_similarity(struct_a.objective, struct_b.objective)

        relation_score = 1.0
        total_score = (
            (0.55 * relation_score) +
            (0.15 * constraint_score) +
            (0.15 * update_rule_score) +
            (0.15 * objective_score)
        )
        passed = (
            relation_score == 1.0 and
            constraint_score >= 0.25 and
            update_rule_score >= 0.25 and
            objective_score >= 0.20
        )

        return {
            "passed": passed,
            "isomorphic": passed,
            "reason": (
                "Deterministic relation graph and higher-order descriptors align."
                if passed else
                "Relation graph matched, but constraints/update rules/objective diverged."
            ),
            "matcher_report": {
                "role_mapping": mapping,
                "relation_score": relation_score,
                "constraint_score": round(constraint_score, 3),
                "update_rule_score": round(update_rule_score, 3),
                "objective_score": round(objective_score, 3),
                "total_score": round(total_score, 3),
            },
        }

    def _explain_match(self, struct_a: dict, struct_b: dict, match: dict) -> dict:
        raw = self._llm(EXPLANATION_PROMPT.format(
            struct1=struct_a,
            struct2=struct_b,
            report=match,
        ), temperature=0.2)
        data = require_json(raw, default={}) or {}
        return {
            "unified_pattern": str(data.get("unified_pattern", "") or "deterministic structural isomorphism"),
            "justification": str(data.get("justification", "") or match.get("reason", "")),
        }

    def check_isomorphism(self, struct1, struct2) -> dict:
        match = self.deterministic_match(struct1, struct2)
        if not match.get("passed"):
            return {
                "isomorphic": False,
                "reason": match.get("reason", "Matcher failed."),
                "matcher_report": match.get("matcher_report", {}),
            }

        struct_a = self._coerce_structure(struct1)
        struct_b = self._coerce_structure(struct2)
        explanation = self._explain_match(
            struct_a.to_dict() if struct_a else {},
            struct_b.to_dict() if struct_b else {},
            match,
        )
        return {
            "isomorphic": True,
            "matcher_report": match.get("matcher_report", {}),
            "unified_pattern": explanation.get("unified_pattern", ""),
            "justification": explanation.get("justification", match.get("reason", "")),
        }

    def _structure_descriptor(self, structure) -> str:
        coerced = self._coerce_structure(structure)
        if not coerced:
            return ""
        relation_terms = [
            f"{rel_type} {source} {target}"
            for source, rel_type, target in coerced.relations
        ]
        parts = relation_terms + coerced.constraints + coerced.update_rules
        if coerced.objective:
            parts.append(coerced.objective)
        return " | ".join(parts)

    def _prefilter_score(self, structure_a, structure_b) -> float:
        coerced_a = self._coerce_structure(structure_a)
        coerced_b = self._coerce_structure(structure_b)
        if not coerced_a or not coerced_b:
            return 0.0
        if len(coerced_a.roles) != len(coerced_b.roles):
            return 0.0
        if len(coerced_a.relations) != len(coerced_b.relations):
            return 0.0
        descriptor_score = _text_similarity(
            self._structure_descriptor(coerced_a),
            self._structure_descriptor(coerced_b),
        )
        relation_overlap = _list_similarity(
            [rel_type for _, rel_type, _ in coerced_a.relations],
            [rel_type for _, rel_type, _ in coerced_b.relations],
        )
        return (0.6 * descriptor_score) + (0.4 * relation_overlap)

    def run_radical_isomorphism(self) -> dict:
        nodes_data = [
            (node_id, data) for node_id, data in self.brain.all_nodes()
            if data.get("importance", 0) >= 0.5
            and data.get("node_type") in {
                NodeType.EVIDENCE_CLAIM.value,
                NodeType.HYPOTHESIS.value,
                NodeType.CONCEPT.value,
            }
        ]

        if len(nodes_data) < 2:
            return {"status": "skipped", "reason": "Not enough nodes"}

        structured_nodes = []
        for node_id, data in nodes_data:
            structure = self.abstract_node(data.get("statement", ""), data.get("context", ""))
            if not structure:
                continue
            structured_nodes.append((node_id, data, structure))

        if len(structured_nodes) < 2:
            return {"status": "failed", "reason": "Failed to build enough structures"}

        scored_pairs = []
        for (node_a_id, node_a, struct_a), (node_b_id, node_b, struct_b) in combinations(structured_nodes, 2):
            if self.brain.graph.has_edge(node_a_id, node_b_id) or self.brain.graph.has_edge(node_b_id, node_a_id):
                continue
            prefilter = self._prefilter_score(struct_a, struct_b)
            if prefilter < STRUCTURE_PREFILTER_MIN:
                continue
            cross_cluster = int(
                node_a.get("cluster", "") not in {"", "unclustered"} and
                node_b.get("cluster", "") not in {"", "unclustered"} and
                node_a.get("cluster", "") != node_b.get("cluster", "")
            )
            mission_relevance = float(node_a.get("mission_relevance", 0.0) or 0.0) + float(
                node_b.get("mission_relevance", 0.0) or 0.0
            )
            importance = float(node_a.get("importance", 0.0) or 0.0) + float(
                node_b.get("importance", 0.0) or 0.0
            )
            scored_pairs.append((
                round(
                    (0.45 * prefilter) +
                    (0.20 * cross_cluster) +
                    (0.20 * importance) +
                    (0.15 * mission_relevance),
                    6,
                ),
                node_a_id,
                node_b_id,
                struct_a,
                struct_b,
            ))

        if not scored_pairs:
            return {"status": "skipped", "reason": "No structurally compatible node pairs"}

        scored_pairs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        shortlisted = scored_pairs[:MAX_ISOMORPHISM_SHORTLIST]
        best_failure = None

        for _, node_a_id, node_b_id, struct_a, struct_b in shortlisted:
            node_a = self.brain.get_node(node_a_id) or {}
            node_b = self.brain.get_node(node_b_id) or {}
            print(
                "  Checking structural isomorphism between:\n"
                f"     A: {node_a.get('statement', '')[:60]}...\n"
                f"     B: {node_b.get('statement', '')[:60]}..."
            )

            result = self.check_isomorphism(struct_a, struct_b)
            if not result.get("isomorphic"):
                best_failure = {
                    "status": "failed",
                    "reason": result.get("reason", "Matcher failed"),
                    "matcher_report": result.get("matcher_report", {}),
                }
                continue

            pattern = result.get("unified_pattern", "deterministic structural isomorphism")
            justification = result.get("justification", "")
            narration = f"Isomorphic pattern: {pattern}. {justification}".strip()
            edge = self.brain.add_analogy_edge(
                node_a_id,
                node_b_id,
                "isomorphism",
                narration,
                source=EdgeSource.CONSOLIDATION,
                matcher_report=result.get("matcher_report", {}),
            )
            if edge is None:
                best_failure = {
                    "status": "failed",
                    "reason": "Graph gate rejected isomorphism edge.",
                    "matcher_report": result.get("matcher_report", {}),
                }
                continue

            self.brain.restructure_around_insight(
                node_a_id,
                node_b_id,
                narration,
                edge_type="deep_isomorphism",
            )

            node_a["importance"] = min(1.0, node_a.get("importance", 0.5) + 0.2)
            node_b["importance"] = min(1.0, node_b.get("importance", 0.5) + 0.2)
            self.brain.graph.nodes[node_a_id].update(node_a)
            self.brain.graph.nodes[node_b_id].update(node_b)

            print(
                "  Radical isomorphism confirmed:\n"
                f"     Pattern: {pattern}\n"
                f"     Mapping: {result.get('matcher_report', {}).get('role_mapping', {})}"
            )
            return {
                "status": "success",
                "pattern": pattern,
                "node1": node_a_id,
                "node2": node_b_id,
                "matcher_report": result.get("matcher_report", {}),
            }

        return best_failure or {"status": "failed", "reason": "No candidate pair matched"}
