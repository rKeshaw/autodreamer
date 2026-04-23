import argparse
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

from conductor import Conductor
from conversation.conversation import ConversationSessionManager, Conversationalist
from critic.critic import Critic
from dreamer.dreamer import Dreamer
from embedding_index import EmbeddingIndex
from experimenter.experimenter import Experimenter
from graph.brain import Brain
from ingestion.ingestor import Ingestor
from insight_buffer import InsightBuffer
from notebook.notebook import Notebook
from observer.observer import Observer
from reader.reader import Reader
from researcher.researcher import Researcher
from thinker.thinker import Thinker


def build_default_runtime() -> Conductor:
    brain = Brain()
    index = EmbeddingIndex()
    observer = Observer(brain)
    critic = Critic(brain, embedding_index=index)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=index, critic=critic)
    thinker = Thinker(brain, observer=observer, embedding_index=index, critic=critic)
    dreamer = Dreamer(brain, research_agenda=observer, critic=critic)
    researcher = Researcher(
        brain,
        observer=observer,
        depth="shallow",
        ingestor=ingestor,
        embedding_index=index,
        critic=critic,
    )
    reader = Reader(
        brain,
        observer=observer,
        ingestor=ingestor,
        embedding_index=index,
        critic=critic,
    )
    experimenter = Experimenter(brain)
    insight_buffer = InsightBuffer(brain, embedding_index=index)
    notebook = Notebook(brain, observer=observer)
    return Conductor(
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
        experimenter=experimenter,
    )


class ScientistServiceShell:
    def __init__(self, conductor: Conductor,
                 conversation_manager: ConversationSessionManager | None = None):
        self.conductor = conductor
        self._cycle_lock = threading.RLock()
        self.scheduler = BackgroundScheduler(daemon=True)
        self.scheduler_job_id = "autoscientist-cycle"
        self.scheduler_interval_seconds = 0.0
        self.scheduler_mode = "mission_driven"
        self.scheduler_runs = 0
        self.scheduler_last_triggered_at = 0.0
        self.scheduler_last_error = ""
        self.conversation_manager = conversation_manager or ConversationSessionManager(
            self._build_conversationalist
        )

    def _build_conversationalist(self) -> Conversationalist:
        return Conversationalist(
            self.conductor.brain,
            observer=self.conductor.observer,
            embedding_index=self.conductor.index,
            ingestor=self.conductor.ingestor,
            notebook=self.conductor.notebook,
        )

    def _scheduler_tick(self):
        if not self._cycle_lock.acquire(blocking=False):
            return
        try:
            self.scheduler_last_triggered_at = time.time()
            self.scheduler_last_error = ""
            self.conductor.run_cycle(mode=self.scheduler_mode)
            self.scheduler_runs += 1
        except Exception as exc:
            self.scheduler_last_error = str(exc)
        finally:
            self._cycle_lock.release()

    def scheduler_status(self) -> dict:
        job = self.scheduler.get_job(self.scheduler_job_id) if self.scheduler.running else None
        return {
            "enabled": job is not None,
            "running": self.scheduler.running,
            "interval_seconds": self.scheduler_interval_seconds,
            "mode": self.scheduler_mode,
            "runs": self.scheduler_runs,
            "last_triggered_at": self.scheduler_last_triggered_at,
            "last_error": self.scheduler_last_error,
        }

    def start_scheduler(self, interval_seconds: float = 60.0,
                        mode: str = "mission_driven") -> dict:
        interval = max(0.05, float(interval_seconds or 60.0))
        self.scheduler_interval_seconds = interval
        self.scheduler_mode = mode or "mission_driven"
        if not self.scheduler.running:
            self.scheduler.start()
        if self.scheduler.get_job(self.scheduler_job_id):
            self.scheduler.remove_job(self.scheduler_job_id)
        self.scheduler.add_job(
            self._scheduler_tick,
            trigger="interval",
            seconds=interval,
            id=self.scheduler_job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        return self.scheduler_status()

    def stop_scheduler(self) -> dict:
        if self.scheduler.running and self.scheduler.get_job(self.scheduler_job_id):
            self.scheduler.remove_job(self.scheduler_job_id)
        self.scheduler_interval_seconds = 0.0
        return self.scheduler_status()

    def shutdown(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def run_cycles(self, count: int = 1, mode: str = "mission_driven") -> list[dict]:
        with self._cycle_lock:
            return self.conductor.run_cycles(count=max(1, int(count or 1)), mode=mode)

    def get_status(self) -> dict:
        notebook = self.conductor.notebook
        return {
            "service": "autoscientist_workbench",
            "runtime": self.conductor.runtime_status(),
            "scheduler": self.scheduler_status(),
            "conversation": self.conversation_manager.stats(),
            "notebook": {
                "entry_count": len(getattr(notebook, "entries", []) or []),
                "running_hypothesis": getattr(notebook, "running_hypothesis", ""),
            },
        }

    def set_mission(self, question: str, context: str = "") -> dict:
        mission_id = self.conductor.brain.set_mission(question, context=context)
        return {
            "id": mission_id,
            "question": question,
            "context": context,
        }

    def notebook_entries(self, limit: int = 10, entry_type: str = "") -> dict:
        notebook = self.conductor.notebook
        if notebook is None:
            return {"entries": [], "running_hypothesis": ""}
        if entry_type:
            entries = notebook.get_entries_by_type(entry_type)
            entries = sorted(entries, key=lambda item: item.timestamp, reverse=True)[:limit]
        else:
            entries = notebook.get_recent_entries(limit)
        return {
            "entries": [entry.to_dict() for entry in entries],
            "running_hypothesis": notebook.running_hypothesis,
        }

    def create_app(self) -> Flask:
        app = Flask(__name__)

        @app.get("/health")
        def health():
            return jsonify({
                "ok": True,
                "cycle": self.conductor.cycle,
                "scheduler": self.scheduler_status(),
            })

        @app.get("/status")
        def status():
            return jsonify(self.get_status())

        @app.get("/mission")
        def mission_get():
            return jsonify(self.conductor.brain.get_mission() or {})

        @app.post("/mission")
        def mission_set():
            payload = request.get_json(silent=True) or {}
            question = str(payload.get("question", "") or "").strip()
            if not question:
                return jsonify({"ok": False, "error": "question is required"}), 400
            context = str(payload.get("context", "") or "")
            mission = self.set_mission(question, context=context)
            return jsonify({"ok": True, "mission": mission})

        @app.post("/cycles/run")
        def cycles_run():
            payload = request.get_json(silent=True) or {}
            count = int(payload.get("count", 1) or 1)
            mode = str(payload.get("mode", "mission_driven") or "mission_driven")
            results = self.run_cycles(count=count, mode=mode)
            return jsonify({
                "ok": True,
                "results": results,
                "status": self.conductor.runtime_status(),
            })

        @app.get("/notebook")
        def notebook_get():
            limit = int(request.args.get("limit", 10) or 10)
            entry_type = str(request.args.get("type", "") or "").strip()
            return jsonify({
                "ok": True,
                **self.notebook_entries(limit=max(1, limit), entry_type=entry_type),
            })

        @app.post("/conversation")
        def conversation_chat():
            payload = request.get_json(silent=True) or {}
            message = str(payload.get("message", "") or "").strip()
            if not message:
                return jsonify({"ok": False, "error": "message is required"}), 400
            session_id = str(payload.get("session_id", "") or "").strip() or None
            reset = bool(payload.get("reset", False))
            result = self.conversation_manager.chat(
                message=message,
                session_id=session_id,
                reset=reset,
            )
            return jsonify({"ok": True, **result})

        @app.get("/conversation/<session_id>")
        def conversation_history(session_id: str):
            return jsonify({
                "ok": True,
                "session_id": session_id,
                "history": self.conversation_manager.get_history(session_id),
            })

        @app.delete("/conversation/<session_id>")
        def conversation_delete(session_id: str):
            deleted = self.conversation_manager.close_session(session_id)
            return jsonify({"ok": deleted, "session_id": session_id})

        @app.post("/scheduler/start")
        def scheduler_start():
            payload = request.get_json(silent=True) or {}
            interval = float(payload.get("interval_seconds", 60.0) or 60.0)
            mode = str(payload.get("mode", "mission_driven") or "mission_driven")
            return jsonify({
                "ok": True,
                "scheduler": self.start_scheduler(interval_seconds=interval, mode=mode),
            })

        @app.post("/scheduler/stop")
        def scheduler_stop():
            return jsonify({
                "ok": True,
                "scheduler": self.stop_scheduler(),
            })

        return app


def create_app(shell: ScientistServiceShell | None = None,
               conductor: Conductor | None = None) -> Flask:
    active_shell = shell or ScientistServiceShell(conductor or build_default_runtime())
    app = active_shell.create_app()
    app.config["SERVICE_SHELL"] = active_shell
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Autonomous Scientist service shell.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scheduler-interval", type=float, default=0.0)
    args = parser.parse_args()

    shell = ScientistServiceShell(build_default_runtime())
    if args.scheduler_interval and args.scheduler_interval > 0:
        shell.start_scheduler(interval_seconds=args.scheduler_interval)

    app = shell.create_app()
    try:
        app.run(host=args.host, port=args.port)
    finally:
        shell.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
