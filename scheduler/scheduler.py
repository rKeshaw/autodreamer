import time
import json
import os
import signal
import sys
from datetime import datetime
from dataclasses import dataclass, field
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from graph.brain import Brain, BrainMode
from ingestion.ingestor import Ingestor, EdgeSource
from dreamer.dreamer import Dreamer, DreamMode
from observer.observer import Observer
from consolidator.consolidator import Consolidator
from researcher.researcher import Researcher
from reader.reader import Reader
from notebook.notebook import Notebook
from sandbox.sandbox import Sandbox
from thinker.thinker import Thinker
from insight_buffer import InsightBuffer
from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed

# ── Config ────────────────────────────────────────────────────────────────────

SCHEDULE = {
    "nrem_rem":    "0 23 * * *",
    "research":    "0 9  * * *",
    "thinking":    "0 11 * * *",   # morning thinking session
    "reading":     "0 14 * * *",   # afternoon reading
    "writing":     "0 16 * * *",   # afternoon writing phase
    "consolidate": "0 20 * * *",
}

DREAM_STEPS          = 20
DREAM_TEMPERATURE    = 0.7
RESEARCH_QUESTIONS   = 5
RESEARCH_DEPTH       = "standard"
READING_ITEMS        = 2
REGEN_LIST_THRESHOLD = 3   # regenerate reading list when fewer than this unread

BRAIN_PATH    = "data/brain.json"
OBSERVER_PATH = "data/observer.json"
INDEX_PATH    = "data/embedding_index"
DAILY_LEDGER_PATH = "data/daily_new_nodes.json"

# ── Cycle log ─────────────────────────────────────────────────────────────────

@dataclass
class CycleEntry:
    cycle:      int
    phase:      str
    brain_mode: str
    started_at: float = field(default_factory=time.time)
    ended_at:   float = 0.0
    summary:    str   = ""
    interrupted:bool  = False

    def to_dict(self):
        return self.__dict__

class CycleLog:
    def __init__(self, path="logs/cycle_log.json"):
        self.path    = path
        self.entries = []
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                self.entries = json.load(f).get('entries', [])
        except FileNotFoundError:
            self.entries = []

    def add(self, entry: CycleEntry):
        self.entries.append(entry.to_dict())
        import os
        os.makedirs("logs", exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({"entries": self.entries}, f, indent=2)

    def current_cycle(self):
        return max((e.get('cycle', 0) for e in self.entries), default=0)

# ── Scheduler ─────────────────────────────────────────────────────────────────

class DreamerScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.cycle_log = CycleLog()
        self._running  = False

        self.brain    = Brain()
        self.observer = Observer(self.brain)
        self._load_state()

        # Load or build embedding index
        try:
            self.emb_index = EmbeddingIndex.load(INDEX_PATH)
        except (FileNotFoundError, Exception):
            self.emb_index = EmbeddingIndex.build_from_brain(
                self.brain, shared_embed)

        # Shared insight buffer
        self.insight_buffer = InsightBuffer(self.brain,
                                            embedding_index=self.emb_index)

        # System 2 gatekeeper
        from critic.critic import Critic
        self.critic = Critic(self.brain, embedding_index=self.emb_index,
                             insight_buffer=self.insight_buffer)

        self.ingestor    = Ingestor(self.brain, research_agenda=self.observer,
                                    embedding_index=self.emb_index,
                                    insight_buffer=self.insight_buffer)
        self.dreamer     = Dreamer(self.brain, research_agenda=self.observer,
                                   critic=self.critic)
        self.consolidator= Consolidator(self.brain, observer=self.observer,
                                         embedding_index=self.emb_index,
                                         insight_buffer=self.insight_buffer)
        self.researcher  = Researcher(self.brain, observer=self.observer,
                                      depth=RESEARCH_DEPTH)
        self.notebook    = Notebook(self.brain, observer=self.observer)
        self.reader      = Reader(self.brain, observer=self.observer,
                                  notebook=self.notebook)
        self.sandbox     = Sandbox(self.brain, observer=self.observer)
        self.thinker     = Thinker(self.brain, observer=self.observer,
                                   embedding_index=self.emb_index,
                                   critic=self.critic)

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _load_state(self):
        try:
            self.brain.load(BRAIN_PATH)
        except FileNotFoundError:
            print("No brain state — starting fresh")
        try:
            self.observer.load(OBSERVER_PATH)
        except Exception:
            print("No observer state — starting fresh")

    def _save_state(self):
        self.brain.save(BRAIN_PATH)
        self.observer.save(OBSERVER_PATH)
        self.emb_index.save(INDEX_PATH)

    def _shutdown(self, signum, frame):
        print("\n── Shutdown signal — saving state ──")
        self._save_state()
        self.stop()
        sys.exit(0)

    # ── Phase runners ─────────────────────────────────────────────────────────

    def _last_phase_interrupted(self, phase: str) -> bool:
        for entry in reversed(self.cycle_log.entries):
            if entry.get("phase") == phase:
                return entry.get("interrupted", False)
        return False

    def _append_daily_nodes(self, node_ids: list[str]):
        if not node_ids:
            return
        os.makedirs("data", exist_ok=True)
        try:
            with open(DAILY_LEDGER_PATH, "r") as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = []
        merged = list(dict.fromkeys(existing + node_ids))
        with open(DAILY_LEDGER_PATH, "w") as f:
            json.dump(merged, f)

    def run_dream_phase(self):
        cycle      = self.cycle_log.current_cycle() + 1
        brain_mode = self.brain.get_mode()
        entry      = CycleEntry(cycle=cycle, phase="dream", brain_mode=brain_mode)
        print(f"\n{'='*60}")
        print(f"DREAMER — Night Cycle {cycle} [{brain_mode.upper()}]")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        try:
            log1 = self.dreamer.dream(
                mode=DreamMode.WANDERING, steps=DREAM_STEPS,
                temperature=DREAM_TEMPERATURE, run_nrem=True,
                log_path=f"logs/dream_cycle{cycle}_wandering.json")
            self.observer.observe(log1)
            self.notebook.write_morning_entry(log1, cycle)
            for signal in self.observer.emergence_feed[-5:]:
                if (signal.type == "mission_advance" and
                        signal.cycle == self.observer.cycle_count):
                    self.notebook.write_breakthrough(signal.signal, cycle)
                    break

            # pressure dream only in focused/transitional
            if not self.brain.is_wandering():
                log2 = self.dreamer.dream(
                    mode=DreamMode.PRESSURE, steps=DREAM_STEPS//2,
                    temperature=DREAM_TEMPERATURE-0.1, run_nrem=False,
                    log_path=f"logs/dream_cycle{cycle}_pressure.json")
                self.observer.observe_supplemental(log2)

            entry.summary = log1.summary
        except Exception as e:
            print(f"Dream error: {e}")
            entry.interrupted = True
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_research_phase(self):
        """Research + reading in the day cycle."""
        if self._last_phase_interrupted("dream"):
            print("Skipping research — previous dream cycle was interrupted.")
            return
        cycle      = self.cycle_log.current_cycle()
        brain_mode = self.brain.get_mode()
        entry      = CycleEntry(cycle=cycle, phase="research", brain_mode=brain_mode)
        print(f"\n{'='*60}")
        print(f"DREAMER — Research Day {cycle} [{brain_mode.upper()}]")
        print(f"{'='*60}")
        try:
            # targeted research (skip in pure wandering — follow curiosity freely)
            if not self.brain.is_wandering():
                log = self.researcher.research_day(
                    max_questions=RESEARCH_QUESTIONS,
                    log_path=f"logs/research_cycle{cycle}.json")
                self.notebook.write_field_notes(log, cycle)
                all_new = [nid for r in log.entries for nid in r.node_ids]
                self._append_daily_nodes(all_new)
            else:
                print("Wandering mode — skipping targeted research")
        except Exception as e:
            print(f"Research error: {e}")
            entry.interrupted = True
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_reading_phase(self):
        """Afternoon reading — autonomous absorption from reading list."""
        cycle      = self.cycle_log.current_cycle()
        brain_mode = self.brain.get_mode()
        entry      = CycleEntry(cycle=cycle, phase="reading", brain_mode=brain_mode)
        print(f"\n{'='*60}")
        print(f"DREAMER — Reading [{brain_mode.upper()}]")
        print(f"{'='*60}")
        try:
            # regenerate reading list if running low
            unread = len(self.reader.get_unread(20))
            if unread < REGEN_LIST_THRESHOLD:
                print(f"Reading list low ({unread} items) — generating more...")
                self.reader.generate_reading_list()

            results = self.reader.reading_day(max_items=READING_ITEMS)
            absorbed = sum(1 for r in results if r.success)
            reading_new = [
                nid for result in results if result.success
                for nid in getattr(result, "node_ids", [])
            ]
            self._append_daily_nodes(reading_new)
            print(f"Reading day: {absorbed}/{len(results)} absorbed")
            entry.summary = f"Absorbed {absorbed} texts."
        except Exception as e:
            print(f"Reading error: {e}")
            entry.interrupted = True
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_consolidation_phase(self):
        if self._last_phase_interrupted("research"):
            print("Warning: previous research was interrupted — consolidating anyway.")
        cycle      = self.cycle_log.current_cycle()
        brain_mode = self.brain.get_mode()
        entry      = CycleEntry(cycle=cycle, phase="consolidation", brain_mode=brain_mode)
        print(f"\n{'='*60}")
        print(f"DREAMER — Consolidation {cycle} [{brain_mode.upper()}]")
        print(f"{'='*60}")
        try:
            try:
                with open(DAILY_LEDGER_PATH, "r") as f:
                    new_node_ids = json.load(f)
                os.remove(DAILY_LEDGER_PATH)
            except FileNotFoundError:
                new_node_ids = []
            report = self.consolidator.consolidate(
                new_node_ids=new_node_ids,
                save_path=f"logs/consolidation_cycle{cycle}.json")
            try:
                self.sandbox.scan_and_test(max_tests=2)
            except Exception as e:
                print(f"Sandbox error: {e}")
            self.notebook.write_evening_entry(report, cycle)
            self.notebook.update_running_hypothesis(cycle)
            entry.summary = report.summary
        except Exception as e:
            print(f"Consolidation error: {e}")
            entry.interrupted = True
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_thinking_phase(self):
        """Active deliberate thinking — structured reasoning sessions."""
        cycle      = self.cycle_log.current_cycle()
        brain_mode = self.brain.get_mode()
        entry      = CycleEntry(cycle=cycle, phase="thinking", brain_mode=brain_mode)
        print(f"\n{'='*60}")
        print(f"DREAMER — Thinking Session {cycle} [{brain_mode.upper()}]")
        print(f"{'='*60}")
        try:
            logs = self.thinker.think_session(num_rounds=3)
            insights = [l.insight for l in logs if l.insight]
            entry.summary = f"{len(logs)} rounds, {len(insights)} insights"
            print(f"  Thinking complete: {len(insights)} insights")
        except Exception as e:
            print(f"Thinking error: {e}")
            entry.interrupted = True
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_writing_phase(self):
        """Writing phase — synthesis essay that forces clarity."""
        cycle      = self.cycle_log.current_cycle()
        brain_mode = self.brain.get_mode()
        entry      = CycleEntry(cycle=cycle, phase="writing", brain_mode=brain_mode)
        print(f"\n{'='*60}")
        print(f"DREAMER — Writing Phase {cycle} [{brain_mode.upper()}]")
        print(f"{'='*60}")
        try:
            result = self.notebook.write_synthesis_essay(cycle)
            # Ingest any side-effect insights back into the graph
            for insight in result.get('insights', []):
                if isinstance(insight, str) and len(insight) > 15:
                    self.ingestor.ingest(insight, source=EdgeSource.CONSOLIDATION)
            # Add any new questions to the agenda
            for question in result.get('questions', []):
                if isinstance(question, str) and len(question) > 10:
                    self.observer.add_to_agenda(
                        text=question, item_type="question",
                        cycle=cycle
                    )
            entry.summary = f"Essay written, {len(result.get('insights',[]))} insights"
        except Exception as e:
            print(f"Writing error: {e}")
            entry.interrupted = True
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    # ── Public interface ──────────────────────────────────────────────────────

    def ingest(self, text, source=EdgeSource.CONVERSATION):
        print(f"\n── Manual ingestion ──")
        self.ingestor.ingest(text, source=source)
        self._save_state()

    def read_url(self, url, title="", source_type="web"):
        print(f"\n── Reading URL: {url} ──")
        result = self.reader.absorb_url(url, title, source_type)
        self._save_state()
        return result

    def add_to_reading_list(self, url, title="", source_type="web", priority=0.7):
        entry = self.reader.add_to_list(
            url=url, title=title, source_type=source_type,
            priority=priority, added_by="user")
        self._save_state()
        return entry

    def set_mission(self, question, context=""):
        self.brain.set_mission(question, context)
        self._save_state()

    def suspend_mission(self):
        self.brain.suspend_mission()
        self._save_state()

    def resume_mission(self):
        self.brain.resume_mission()
        self._save_state()

    def seed_dream(self, concept):
        item = self.observer.add_to_agenda(concept, cycle=self.cycle_log.current_cycle())
        item.priority = 0.95
        self._save_state()

    def run_now(self, phase):
        phases = {
            "dream":         self.run_dream_phase,
            "research":      self.run_research_phase,
            "thinking":      self.run_thinking_phase,
            "reading":       self.run_reading_phase,
            "writing":       self.run_writing_phase,
            "consolidation": self.run_consolidation_phase,
        }
        if phase not in phases:
            print(f"Unknown phase: {phase}")
            return
        phases[phase]()

    def run_full_cycle_now(self):
        print("\n── Running full cycle now ──")
        self.run_dream_phase()
        self.run_research_phase()
        self.run_thinking_phase()
        self.run_reading_phase()
        self.run_writing_phase()
        self.run_consolidation_phase()
        print("\n── Full cycle complete ──")

    def start(self, schedule=None):
        s = schedule or SCHEDULE
        self.scheduler.add_job(self.run_dream_phase,
            CronTrigger.from_crontab(s["nrem_rem"]), id="dream",
            replace_existing=True)
        self.scheduler.add_job(self.run_research_phase,
            CronTrigger.from_crontab(s["research"]), id="research",
            replace_existing=True)
        self.scheduler.add_job(self.run_thinking_phase,
            CronTrigger.from_crontab(s["thinking"]), id="thinking",
            replace_existing=True)
        self.scheduler.add_job(self.run_reading_phase,
            CronTrigger.from_crontab(s["reading"]), id="reading",
            replace_existing=True)
        self.scheduler.add_job(self.run_writing_phase,
            CronTrigger.from_crontab(s["writing"]), id="writing",
            replace_existing=True)
        self.scheduler.add_job(self.run_consolidation_phase,
            CronTrigger.from_crontab(s["consolidate"]), id="consolidation",
            replace_existing=True)
        self.scheduler.start()
        self._running = True

        print(f"\n{'='*60}")
        print(f"DREAMER is running | mode: {self.brain.get_mode().upper()}")
        print(f"Brain: {self.brain.stats()['nodes']} nodes | "
              f"{self.brain.stats()['edges']} edges")
        if self.brain.get_mission():
            print(f"Mission: {self.brain.get_mission()['question']}")
        print(f"Reading list: {self.reader.stats()['unread']} unread")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop.\n")

    def stop(self):
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False

    def status(self):
        return {
            "running":      self._running,
            "cycle":        self.cycle_log.current_cycle(),
            "brain_mode":   self.brain.get_mode(),
            "brain":        self.brain.stats(),
            "agenda":       len(self.observer.agenda),
            "resolved":     sum(1 for i in self.observer.agenda if i.resolved),
            "emergences":   len(self.observer.emergence_feed),
            "reading_list": self.reader.stats(),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DREAMER Scheduler")
    parser.add_argument("--mode", choices=[
        "auto","dream","research","thinking","reading","writing",
        "consolidation","cycle","status"
    ], default="auto")
    parser.add_argument("--ingest",  type=str, default=None)
    parser.add_argument("--url",     type=str, default=None)
    parser.add_argument("--seed",    type=str, default=None)
    parser.add_argument("--depth",   type=str, default="standard",
                        choices=["shallow","standard","deep"])
    parser.add_argument("--suspend-mission", action="store_true")
    parser.add_argument("--resume-mission",  action="store_true")
    args = parser.parse_args()

    RESEARCH_DEPTH = args.depth
    ds = DreamerScheduler()

    if args.ingest:       ds.ingest(args.ingest)
    if args.url:          ds.read_url(args.url)
    if args.seed:         ds.seed_dream(args.seed)
    if args.suspend_mission: ds.suspend_mission()
    if args.resume_mission:  ds.resume_mission()

    if args.mode == "auto":
        ds.start()
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            pass
    elif args.mode == "status":
        import json
        print(json.dumps(ds.status(), indent=2))
    elif args.mode == "cycle":
        ds.run_full_cycle_now()
    else:
        ds.run_now(args.mode)
