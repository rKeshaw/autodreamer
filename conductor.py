"""
Conductor — The Autonomous Scientist's main loop.

Drives the scientific method as a state machine:
  READ → HYPOTHESIZE → TEST → RESEARCH → EVALUATE → repeat

Each cycle corresponds to one "day" of scientific work.
The Conductor decides what to do next based on the current state
of the graph, the agenda, and the hypothesis queue.
"""

import json
import os
import time
from types import SimpleNamespace

from critic.lab_meeting import LabMeeting
from thinker.isomorphism import IsomorphismEngine
from notebook.publisher import Publisher
from persistence import atomic_write_json


DEFAULT_LAB_MEETING_CADENCE = 3
DEFAULT_WEEKLY_REFLECTION_CADENCE = 7
DEFAULT_SYNTHESIS_CADENCE = 10
DEFAULT_PUBLICATION_CADENCE = 30
DEFAULT_STATE_PATH = "data/conductor_state.json"

class Conductor:
    def __init__(self, brain, dreamer, thinker, researcher,
                 reader, observer, notebook, ingestor,
                 embedding_index, insight_buffer, critic,
                 experimenter=None,
                 lab_meeting_cadence: int = DEFAULT_LAB_MEETING_CADENCE,
                 weekly_reflection_cadence: int = DEFAULT_WEEKLY_REFLECTION_CADENCE,
                 synthesis_cadence: int = DEFAULT_SYNTHESIS_CADENCE,
                 publication_cadence: int = DEFAULT_PUBLICATION_CADENCE,
                 state_path: str = DEFAULT_STATE_PATH,
                 resume_state: bool = False):
        self.brain     = brain
        self.dreamer   = dreamer
        self.thinker   = thinker
        self.researcher = researcher
        self.reader    = reader
        self.observer  = observer
        self.notebook  = notebook
        self.ingestor  = ingestor
        self.index     = embedding_index
        self.insight_buffer = insight_buffer
        self.critic    = critic
        self.experimenter = experimenter
        self.lab_meeting = LabMeeting(self.brain, observer=self.observer)
        self.isomorphism_engine = IsomorphismEngine(self.brain)
        self.publisher = Publisher(
            self.brain,
            embedding_index=self.index,
            observer=self.observer,
            notebook=self.notebook,
            insight_buffer=self.insight_buffer,
        )
        if self.observer is not None:
            self.observer.lab_meeting = self.lab_meeting
        if self.notebook is not None:
            self.notebook.lab_meeting = self.lab_meeting
        self.publisher.lab_meeting = self.lab_meeting
        self.cycle     = 0
        self.last_cycle_started_at = 0.0
        self.last_cycle_completed_at = 0.0
        self.last_cycle_error = ""

        self.state_path = state_path
        self.schedule = {
            "lab_meeting_cadence": max(1, int(lab_meeting_cadence or 1)),
            "weekly_reflection_cadence": max(1, int(weekly_reflection_cadence or 1)),
            "synthesis_cadence": max(1, int(synthesis_cadence or 1)),
            "publication_cadence": max(1, int(publication_cadence or 1)),
        }
        self.cadence_state = self._default_cadence_state()
        if resume_state:
            self._load_state()

    def _default_cadence_state(self) -> dict:
        return {
            "cycle": self.cycle,
            "schedule": dict(self.schedule),
            "daily": {
                "morning_entries": 0,
                "field_notes": 0,
                "evening_entries": 0,
                "running_hypothesis_updates": 0,
            },
            "weekly": {
                "reflections": 0,
                "pivot_triggers": 0,
                "dead_end_entries": 0,
            },
            "periodic": {
                "synthesis_events": 0,
                "publication_events": 0,
                "lab_meetings": 0,
            },
            "last_cycle_mode": "",
            "last_updated": time.time(),
        }

    def _load_state(self):
        try:
            with open(self.state_path, "r") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        loaded_cycle = int(payload.get("cycle", 0) or 0)
        self.cycle = max(self.cycle, loaded_cycle)

        loaded_schedule = payload.get("schedule", {}) or {}
        if isinstance(loaded_schedule, dict):
            for key in self.schedule:
                value = loaded_schedule.get(key)
                if isinstance(value, int) and value > 0:
                    self.schedule[key] = value

        state = self._default_cadence_state()
        for section in ["daily", "weekly", "periodic"]:
            loaded = payload.get(section, {}) or {}
            if isinstance(loaded, dict):
                for key in state[section]:
                    try:
                        state[section][key] = int(loaded.get(key, state[section][key]) or 0)
                    except Exception:
                        pass

        state["cycle"] = self.cycle
        state["schedule"] = dict(self.schedule)
        state["last_cycle_mode"] = str(payload.get("last_cycle_mode", "") or "")
        state["last_updated"] = float(payload.get("last_updated", time.time()) or time.time())
        self.cadence_state = state

    def _save_state(self, mode: str):
        self.cadence_state["cycle"] = self.cycle
        self.cadence_state["schedule"] = dict(self.schedule)
        self.cadence_state["last_cycle_mode"] = mode
        self.cadence_state["last_updated"] = time.time()
        atomic_write_json(self.state_path, self.cadence_state)

    def _empty_dream_log(self):
        return SimpleNamespace(
            summary="No dream activity recorded.",
            mission_advances=[],
            insights=[],
            questions=[],
            answers=[],
            steps=[],
        )

    def _coerce_research_log(self, research_obj):
        if research_obj is None:
            return SimpleNamespace(entries=[])

        if hasattr(research_obj, "entries"):
            entries = getattr(research_obj, "entries", []) or []
            return SimpleNamespace(entries=list(entries))

        if isinstance(research_obj, dict):
            entries = research_obj.get("entries")
            if isinstance(entries, list):
                coerced = []
                for item in entries:
                    if isinstance(item, dict):
                        coerced.append(SimpleNamespace(
                            question=item.get("question", ""),
                            sources=item.get("sources", []) or [],
                            node_ids=item.get("node_ids", []) or [],
                            resolved=item.get("resolved", "none"),
                        ))
                return SimpleNamespace(entries=coerced)

            return SimpleNamespace(entries=[SimpleNamespace(
                question=research_obj.get("question", ""),
                sources=research_obj.get("sources", []) or [],
                node_ids=research_obj.get("node_ids", []) or [],
                resolved=research_obj.get("resolved", "none"),
            )])

        return SimpleNamespace(entries=[research_obj])

    def _consolidation_report(self, insight_stats: dict | None):
        stats = insight_stats or {}
        return SimpleNamespace(
            merges=int(stats.get("pruned", 0) or 0),
            syntheses=int(stats.get("promoted", 0) or 0),
            abstractions=0,
            gaps=int(stats.get("remaining", 0) or 0),
        )

    def _research_log_payload(self, research_log) -> dict:
        entries = []
        for entry in getattr(research_log, "entries", []) or []:
            if isinstance(entry, dict):
                entries.append(dict(entry))
            else:
                entries.append({
                    "question": getattr(entry, "question", ""),
                    "sources": list(getattr(entry, "sources", []) or []),
                    "node_ids": list(getattr(entry, "node_ids", []) or []),
                    "resolved": getattr(entry, "resolved", "none"),
                })
        return {"entries": entries}

    def _dream_log_payload(self, dream_log) -> dict:
        if hasattr(dream_log, "to_dict"):
            return dream_log.to_dict()
        return {
            "mode": getattr(dream_log, "mode", ""),
            "brain_mode": getattr(dream_log, "brain_mode", ""),
            "started_at": getattr(dream_log, "started_at", 0.0),
            "steps": list(getattr(dream_log, "steps", []) or []),
            "questions": list(getattr(dream_log, "questions", []) or []),
            "insights": list(getattr(dream_log, "insights", []) or []),
            "answers": list(getattr(dream_log, "answers", []) or []),
            "mission_advances": list(getattr(dream_log, "mission_advances", []) or []),
            "summary": getattr(dream_log, "summary", ""),
        }

    def run_cycles(self, count: int = 1, mode: str = "mission_driven") -> list[dict]:
        results = []
        total = max(1, int(count or 1))
        for _ in range(total):
            result = self.run_cycle(mode=mode)
            if isinstance(result, dict):
                results.append(result)
        return results

    def runtime_status(self) -> dict:
        mission = self.brain.get_mission() or {}
        return {
            "cycle": self.cycle,
            "mission": {
                "id": mission.get("id", ""),
                "question": mission.get("question", ""),
                "context": mission.get("context", ""),
            },
            "graph": self.brain.stats(),
            "schedule": dict(self.schedule),
            "cadence_state": json.loads(json.dumps(self.cadence_state)),
            "last_cycle_started_at": self.last_cycle_started_at,
            "last_cycle_completed_at": self.last_cycle_completed_at,
            "last_cycle_error": self.last_cycle_error,
        }

    def run_cycle(self, mode: str = "mission_driven"):
        """Run one complete scientific cycle."""
        self.last_cycle_started_at = time.time()
        self.last_cycle_error = ""
        self.cycle += 1
        print(f"\n{'='*60}")
        print(f"  CONDUCTOR — Cycle {self.cycle} [{mode}]")
        print(f"{'='*60}\n")

        # ── Step 1: READ ──
        # Absorb unread items from the reading list.
        # The Reader creates SOURCE nodes and EVIDENCE_CLAIM nodes.
        reading_results = self.reader.reading_day(max_items=2) if self.reader and hasattr(self.reader, 'reading_day') else []
        new_claim_ids = []
        for result in reading_results:
            if getattr(result, 'success', False):
                new_claim_ids.extend(getattr(result, 'node_ids', []))

        # Flush deferred FAISS rebuilds after ingestion batch
        if self.index:
            self.index.flush()

        # ── Step 2: HYPOTHESIZE ──
        # Feed fresh claims to the Dreamer for 'shower thoughts'.
        # This is the Reader → Dreamer bridge (Issue 14).
        hypothesis_results = []
        if new_claim_ids and self.brain.can_spawn_hypothesis() and self.dreamer:
            hypothesis_results = self.dreamer.hypothesize(
                seed_node_ids=new_claim_ids,
                mode=mode,
                max_hypotheses=3,
            )
            for h in hypothesis_results:
                print(f"  💡 Hypothesis spawned: {h['statement'][:80]}...")

        # ── Step 3: TEST (Thinker) ──
        # Pick the highest-priority untested hypothesis and decompose it.
        test_queries = []
        tested_hyp_id = None
        think_log = self.thinker.think() if self.thinker else None
        
        if think_log and getattr(think_log, "pattern", "") == "hypothesis_testing":
            tested_hyp_id = getattr(think_log, "question_node_id", None)
            
            # Extract search queries from the reasoning result
            result = getattr(think_log, "reasoning_result", {})
            if hasattr(result, "to_dict"):
                result = result.to_dict()
                
            if isinstance(result, dict) and 'next_actions' in result:
                test_queries = result['next_actions']
                
            # Also extract from sub_claims if present
            if hasattr(think_log, "sub_questions"):
                sub_qs = getattr(think_log, "sub_questions", [])
                if isinstance(sub_qs, list):
                    test_queries.extend(
                        [sc.get("search_query", "") for sc in sub_qs
                         if isinstance(sc, dict) and "search_query" in sc]
                    )

            # Clean and filter empty queries
            test_queries = [q for q in test_queries if q and isinstance(q, str) and q.strip()]

        # ── Step 4: RESEARCH ──
        # If the Thinker produced search queries for a hypothesis, research it.
        research_entry = None
        research_log = SimpleNamespace(entries=[])
        if tested_hyp_id and test_queries and self.researcher:
            research_entry = self.researcher.research_hypothesis(
                hypothesis_node_id=tested_hyp_id,
                search_queries=test_queries[:4],
            )
            research_log = self._coerce_research_log(research_entry)
            atomic_write_json(
                os.path.join("logs", f"research_cycle_{self.cycle:04d}.json"),
                self._research_log_payload(research_log),
            )
        elif self.researcher:
            # Fallback: standard research day from the agenda
            research_log = self.researcher.research_day(max_questions=3)
            research_log = self._coerce_research_log(research_log)
            atomic_write_json(
                os.path.join("logs", f"research_cycle_{self.cycle:04d}.json"),
                self._research_log_payload(research_log),
            )

        # ── Step 4.5: VIRTUAL LAB (Experimenter) ──
        # If the hypothesis lacked literature evidence, run empirical experiments
        if tested_hyp_id and research_entry:
            hyp_node = self.brain.get_node(tested_hyp_id)
            if hyp_node and hyp_node.get('status') == 'lacks_evidence':
                if getattr(self, 'experimenter', None):
                    exp_report = self.experimenter.run_experiment(tested_hyp_id)
                    if isinstance(exp_report, dict):
                        print(
                            "  🔬 Virtual Lab report: "
                            f"verdict={exp_report.get('verdict', 'unknown')} "
                            f"runs={exp_report.get('run_count', 0)}"
                        )

        # Flush again after research ingestion
        if self.index:
            self.index.flush()

        # ── Step 4.8: ANOMALY EXPLOITATION ──
        # If hypothesis was CONTRADICTED by research or virtual lab, exploit it.
        if tested_hyp_id and self.dreamer:
            hyp_node = self.brain.get_node(tested_hyp_id)
            if hyp_node and hyp_node.get('status') == 'contradicted':
                # Trigger Phase 8 anomaly exploitation
                self.dreamer.dream_from_anomaly(tested_hyp_id)

        # ── Step 5: DREAM (traditional) ──
        # Run the existing random-walk dreamer for serendipity.
        dream_log = self.dreamer.dream() if self.dreamer else None
        if not dream_log:
            dream_log = self._empty_dream_log()
        atomic_write_json(
            os.path.join("logs", f"dream_cycle_{self.cycle:04d}.json"),
            self._dream_log_payload(dream_log),
        )

        # ── Step 6: OBSERVE ──
        if self.observer:
            self.observer.observe(dream_log)

        # ── Step 6.5: LAB MEETING ──
        if self.cycle % self.schedule["lab_meeting_cadence"] == 0:
            try:
                self.lab_meeting.hold_meeting(cycle=self.cycle)
            except TypeError:
                self.lab_meeting.hold_meeting()
            self.cadence_state["periodic"]["lab_meetings"] += 1
            
        # ── Step 7: CONSOLIDATE ──
        # Apply decay, reinforce strong edges, run insight buffer.
        self.brain.apply_decay(elapsed_days=1.0)
        self.brain.proximal_reinforce()
        
        # ── 7.2 Radical Isomorphism (Phase 11) ──
        # Attempt to find deep structural similarities across domains
        self.isomorphism_engine.run_radical_isomorphism()
        
        if self.insight_buffer:
            insight_stats = self.insight_buffer.evaluate_all()
            print(
                "  Insight buffer: "
                f"{insight_stats.get('promoted', 0)} promoted, "
                f"{insight_stats.get('remaining', 0)} remaining"
            )
        else:
            insight_stats = {"promoted": 0, "pruned": 0, "remaining": 0}

        consolidation_report = self._consolidation_report(insight_stats)

        # ── Step 8: REFLECT ──
        # Daily cadence notebook entries.
        if self.notebook:
            self.notebook.write_morning_entry(dream_log, self.cycle)
            self.cadence_state["daily"]["morning_entries"] += 1

            self.notebook.write_field_notes(research_log, self.cycle)
            self.cadence_state["daily"]["field_notes"] += 1

            self.notebook.write_evening_entry(consolidation_report, self.cycle)
            self.cadence_state["daily"]["evening_entries"] += 1

            self.notebook.update_running_hypothesis(self.cycle)
            self.cadence_state["daily"]["running_hypothesis_updates"] += 1

        # ── Step 8.5: WEEKLY REFLECTION / PIVOT ──
        if self.observer and self.cycle % self.schedule["weekly_reflection_cadence"] == 0:
            self.cadence_state["weekly"]["reflections"] += 1
            pivot_data = self.observer.reflection_week()
            if isinstance(pivot_data, dict) and pivot_data.get("pivot_triggered"):
                self.cadence_state["weekly"]["pivot_triggers"] += 1
                if self.notebook:
                    self.notebook.write_dead_end_summary(pivot_data, self.cycle)
                    self.cadence_state["weekly"]["dead_end_entries"] += 1

        # ── Step 8.8: PERIODIC SYNTHESIS/PUBLICATION ──
        if self.notebook and self.cycle % self.schedule["synthesis_cadence"] == 0:
            self.notebook.write_synthesis_essay(self.cycle)
            self.cadence_state["periodic"]["synthesis_events"] += 1

        if self.cycle % self.schedule["publication_cadence"] == 0:
            self.publisher.draft_publication(cycle=self.cycle)
            self.cadence_state["periodic"]["publication_events"] += 1

        # ── Step 9: SAVE ──
        self.brain.save()
        if self.observer:
            self.observer.save()
        if self.index:
            self.index.save()
        if self.notebook:
            self.notebook.save()
        self._save_state(mode)

        print(f"\n  Cycle {self.cycle} complete. "
              f"Graph: {self.brain.stats()['nodes']} nodes, "
              f"{self.brain.stats()['edges']} edges")
        print(f"  Active hypotheses: {self.brain.count_active_hypotheses()}")
        self.last_cycle_completed_at = time.time()
        return {
            "cycle": self.cycle,
            "mode": mode,
            "graph": self.brain.stats(),
            "active_hypotheses": self.brain.count_active_hypotheses(),
        }
