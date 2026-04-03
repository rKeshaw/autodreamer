import json
import os
import sys
import time
import queue
import hashlib
import threading
from flask import Flask, jsonify, request, render_template, Response
from persistence import atomic_write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from graph.brain import Brain, NodeType, EdgeType, BrainMode
from observer.observer import Observer
from ingestion.ingestor import Ingestor, EdgeSource
from dreamer.dreamer import Dreamer, DreamMode
from consolidator.consolidator import Consolidator
from researcher.researcher import Researcher
from reader.reader import Reader
from notebook.notebook import Notebook
from sandbox.sandbox import Sandbox
from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed
from conversation.conversation import Conversationalist
from thinker.thinker import Thinker
from insight_buffer import InsightBuffer
from critic.critic import Critic

BRAIN_PATH    = "data/brain.json"
OBSERVER_PATH = "data/observer.json"
INDEX_PATH    = "data/embedding_index"
DAILY_LEDGER_PATH = "data/daily_new_nodes.json"

app = Flask(__name__, template_folder='templates')
state = {
    "phase":          "idle",
    "message":        "Standing by.",
    "cycle":          0,
    "active_node_id": None,
}

# ── Event stream ──────────────────────────────────────────────────────────────

_event_queue = queue.Queue(maxsize=400)
_dropped_events = 0
state_lock = threading.RLock()

def emit(event_type, data):
    global _dropped_events
    try:
        _event_queue.put_nowait({
            "type": event_type, "data": data, "timestamp": time.time()
        })
    except queue.Full:
        _dropped_events += 1


def _restore_cycle() -> int:
    cycle = observer.cycle_count
    try:
        with open("logs/cycle_log.json", "r") as f:
            entries = json.load(f).get("entries", [])
        cycle = max(cycle, max((e.get("cycle", 0) for e in entries), default=0))
    except FileNotFoundError:
        pass
    return cycle

@app.route("/api/stream")
def api_stream():
    def generate():
        while True:
            try:
                event = _event_queue.get(timeout=20)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type':'heartbeat','data':{}})}\n\n"
    return Response(generate(), mimetype='text/event-stream',
        headers={'Cache-Control':'no-cache,no-transform',
                 'X-Accel-Buffering':'no','Connection':'keep-alive'})

# ── Load components ───────────────────────────────────────────────────────────

brain    = Brain()
observer = Observer(brain)

try:    brain.load(BRAIN_PATH)
except FileNotFoundError: pass
try:    observer.load(OBSERVER_PATH)
except Exception: pass
state["cycle"] = _restore_cycle()

# ── Load or build embedding index ─────────────────────────────────────────────
try:
    emb_index = EmbeddingIndex.load(INDEX_PATH)
except (FileNotFoundError, Exception):
    emb_index = EmbeddingIndex.build_from_brain(brain, shared_embed)

notebook     = Notebook(brain, observer=observer)
insight_buffer = InsightBuffer(brain, embedding_index=emb_index)
critic = Critic(brain, embedding_index=emb_index, insight_buffer=insight_buffer)
ingestor     = Ingestor(
    brain,
    research_agenda=observer,
    embedding_index=emb_index,
    insight_buffer=insight_buffer
)
dreamer      = Dreamer(brain, research_agenda=observer, critic=critic)
consolidator = Consolidator(
    brain,
    observer=observer,
    embedding_index=emb_index,
    insight_buffer=insight_buffer
)
researcher   = Researcher(
    brain,
    observer=observer,
    depth="standard",
    ingestor=ingestor
)
sandbox      = Sandbox(brain, observer=observer)
reader       = Reader(
    brain,
    observer=observer,
    notebook=notebook,
    ingestor=ingestor
)
conversation = Conversationalist(
    brain,
    observer=observer,
    embedding_index=emb_index,
    ingestor=ingestor,
    notebook=notebook
)
thinker = Thinker(
    brain,
    observer=observer,
    embedding_index=emb_index,
    critic=critic
)

def save_state():
    with state_lock:
        brain.save(BRAIN_PATH)
        observer.save(OBSERVER_PATH)
        emb_index.save(INDEX_PATH)

def append_daily_nodes(node_ids):
    if not node_ids:
        return
    os.makedirs("data", exist_ok=True)
    try:
        with open(DAILY_LEDGER_PATH, "r") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []
    merged = list(dict.fromkeys(existing + node_ids))
    atomic_write_json(DAILY_LEDGER_PATH, merged)

# ── Status ────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "phase":          state["phase"],
        "message":        state["message"],
        "cycle":          state["cycle"],
        "brain":          brain.stats(),
        "brain_mode":     brain.get_mode(),
        "agenda":         len(observer.agenda),
        "resolved":       sum(1 for i in observer.agenda if i.resolved),
        "emergences":     len(observer.emergence_feed),
        "mission_advances": len(observer.mission_advances),
        "active_node_id": state["active_node_id"],
        "reading_list":   reader.stats(),
        "dropped_events": _dropped_events,
    })

# ── Graph ─────────────────────────────────────────────────────────────────────

@app.route("/api/graph")
def api_graph():
    sig = f"{len(brain.graph.nodes)}-{len(brain.graph.edges)}-{brain.stats().get('edges',0)}"
    etag = hashlib.md5(sig.encode()).hexdigest()[:8]
    if request.headers.get("If-None-Match") == etag:
        return Response(status=304)
    nodes, edges = [], []
    for nid, data in brain.all_nodes():
        nodes.append({
            "id":         nid,
            "statement":  data.get("statement",""),
            "cluster":    data.get("cluster","unclustered"),
            "node_type":  data.get("node_type","concept"),
            "status":     data.get("status","uncertain"),
            "importance": data.get("importance",0.5),
            "incubation": data.get("incubation_age",0),
        })
    for u,v,data in brain.graph.edges(data=True):
        edges.append({
            "source":       u, "target": v,
            "type":         data.get("type","associated"),
            "weight":       data.get("weight",0.5),
            "narration":    data.get("narration",""),
            "analogy_depth":data.get("analogy_depth",""),
        })
    resp = jsonify({"nodes":nodes,"edges":edges})
    resp.headers["ETag"] = etag
    return resp

@app.route("/api/node/<node_id>")
def api_node(node_id):
    node = brain.get_node(node_id)
    if not node: return jsonify({"error":"not found"}),404
    edges_out = []
    for nb in brain.neighbors(node_id):
        e  = brain.get_edge(node_id,nb)
        nd = brain.get_node(nb)
        if e and nd:
            edges_out.append({
                "target_id":     nb,
                "target_label":  nd.get("statement",""),
                "type":          e.get("type",""),
                "weight":        e.get("weight",0),
                "narration":     e.get("narration",""),
                "analogy_depth": e.get("analogy_depth",""),
            })
    return jsonify({**node, "edges_out":edges_out})

# ── Agenda ────────────────────────────────────────────────────────────────────

@app.route("/api/agenda")
def api_agenda():
    return jsonify([{
        "text":           i.text,
        "type":           i.item_type,
        "priority":       round(i.priority,2),
        "incubation_age": i.incubation_age,
        "resolved":       i.resolved,
        "grade":          i.resolution_grade,
        "count":          i.count,
    } for i in sorted(observer.agenda, key=lambda x:x.priority, reverse=True)])

# ── Emergences ────────────────────────────────────────────────────────────────

@app.route("/api/emergences")
def api_emergences():
    return jsonify([{
        "signal":    e.signal, "type": e.type,
        "cycle":     e.cycle,  "timestamp": e.timestamp,
    } for e in reversed(observer.emergence_feed[-80:])])

# ── Mission ───────────────────────────────────────────────────────────────────

@app.route("/api/mission")
def api_mission():
    m = brain.get_mission()
    advances = sorted(observer.mission_advances,
                      key=lambda a:a.strength, reverse=True)[:10]
    return jsonify({
        "set":        bool(m),
        "question":   m["question"] if m else "",
        "context":    m.get("context","") if m else "",
        "id":         m["id"] if m else "",
        "mode":       brain.get_mode(),
        "advances":   [{"explanation":a.explanation,"strength":a.strength,
                        "cycle":a.cycle} for a in advances],
    })

@app.route("/api/mission/progress")
def api_mission_progress():
    return jsonify({
        "summary": observer.get_mission_progress_summary()
    })

@app.route("/api/mission", methods=["POST"])
def api_set_mission():
    data = request.json
    q    = data.get("question","").strip()
    ctx  = data.get("context","").strip()
    if not q: return jsonify({"error":"no question"}),400
    with state_lock:
        brain.set_mission(q,ctx)
    save_state()
    emit("mission_set",{"question":q,"mode":brain.get_mode()})
    return jsonify({"status":"set","question":q,"mode":brain.get_mode()})

@app.route("/api/mission/suspend", methods=["POST"])
def api_suspend_mission():
    with state_lock:
        brain.suspend_mission()
    save_state()
    emit("mode_change",{"mode":brain.get_mode(),"message":"Mission suspended — wandering mode"})
    return jsonify({"status":"suspended","mode":brain.get_mode()})

@app.route("/api/mission/resume", methods=["POST"])
def api_resume_mission():
    with state_lock:
        brain.resume_mission()
    save_state()
    emit("mode_change",{"mode":brain.get_mode(),"message":"Mission resumed — focused mode"})
    return jsonify({"status":"resumed","mode":brain.get_mode()})

# ── Notebook ──────────────────────────────────────────────────────────────────

@app.route("/api/notebook")
def api_notebook():
    return jsonify({
        "entries":            notebook.get_all_for_display(),
        "running_hypothesis": notebook.running_hypothesis,
    })

# ── Sandbox ───────────────────────────────────────────────────────────────────

@app.route("/api/sandbox")
def api_sandbox():
    return jsonify([{
        "hypothesis":     r.hypothesis,
        "verdict":        r.verdict,
        "confidence":     r.confidence,
        "interpretation": r.interpretation,
        "implications":   r.implications,
        "approach":       r.approach,
        "plot_path":      r.plot_path,
        "timestamp":      r.timestamp,
    } for r in reversed(sandbox.results[-20:])])

@app.route("/api/sandbox/run", methods=["POST"])
def api_sandbox_run():
    with state_lock:
        if state["phase"] != "idle":
            return jsonify({"error":"already running"}),409
        state["phase"]="sandbox"; state["message"]="Computing..."
    hyp = request.json.get("hypothesis","").strip()
    if not hyp: return jsonify({"error":"no hypothesis"}),400
    def run():
        emit("phase_change",{"phase":"sandbox","message":"Sandbox running..."})
        try:
            r = sandbox.test_hypothesis(hyp)
            emit("sandbox_result",{"verdict":r.verdict,"confidence":r.confidence,
                "interpretation":r.interpretation,"implications":r.implications,
                "hypothesis":r.hypothesis,"plot_path":r.plot_path})
        except Exception as e:
            emit("error",{"message":str(e)})
        finally:
            save_state()
            state["phase"]="idle"; state["message"]="Computation complete."
            emit("phase_change",{"phase":"idle","message":"Computation complete."})
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"})

# ── Reading list ──────────────────────────────────────────────────────────────

@app.route("/api/reading-list")
def api_reading_list():
    return jsonify(reader.list_all())

@app.route("/api/reading-list/add", methods=["POST"])
def api_add_reading():
    data   = request.json
    url    = data.get("url","").strip()
    title  = data.get("title","").strip()
    stype  = data.get("source_type","web")
    priority = float(data.get("priority",0.7))
    if not url: return jsonify({"error":"no url"}),400
    entry = reader.add_to_list(url=url,title=title,source_type=stype,
                               priority=priority,added_by="user")
    save_state()
    return jsonify({"status":"added","entry":entry.to_dict()})

@app.route("/api/reading-list/absorb", methods=["POST"])
def api_absorb_now():
    with state_lock:
        if state["phase"] != "idle":
            return jsonify({"error":"already running"}),409
        state["phase"]="ingesting"; state["message"]="Reading..."
    data  = request.json
    entry_id = data.get("id","")
    url   = data.get("url","").strip()

    def run():
        emit("phase_change",{"phase":"ingesting","message":"Absorbing text..."})
        try:
            if entry_id:
                entry = next((e for e in reader.reading_list if e.id==entry_id),None)
                if entry:
                    result = reader.absorb_entry(entry)
                else:
                    result = None
            elif url:
                result = reader.absorb_url(url)
            else:
                result = None

            if result and result.success:
                emit("absorption_complete",{
                    "title":      result.title,
                    "node_count": result.node_count,
                    "summary":    result.summary,
                })
        except Exception as e:
            emit("error",{"message":str(e)})
        finally:
            save_state()
            state["phase"]="idle"; state["message"]="Reading complete."
            emit("phase_change",{"phase":"idle","message":"Reading complete."})

    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/reading-list/generate", methods=["POST"])
def api_generate_reading_list():
    with state_lock:
        if state["phase"] != "idle":
            return jsonify({"error":"already running"}),409
        state["phase"]="ingesting"; state["message"]="Generating reading list..."
    def run():
        try:
            added = reader.generate_reading_list()
            save_state()
            emit("reading_list_updated",{"added":len(added)})
        except Exception as e:
            emit("error",{"message":str(e)})
        finally:
            state["phase"]="idle"; state["message"]="Reading list updated."
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/reading-list/absorb-text", methods=["POST"])
def api_absorb_text():
    with state_lock:
        if state["phase"] != "idle":
            return jsonify({"error":"already running"}),409
        state["phase"]="ingesting"; state["message"]="Absorbing text..."
    data  = request.json
    text  = data.get("text","").strip()
    title = data.get("title","Manual text").strip()
    if not text: return jsonify({"error":"no text"}),400

    def run():
        emit("phase_change",{"phase":"ingesting","message":"Absorbing text..."})
        try:
            result = reader.add_text(text, title=title)
            save_state()
            emit("absorption_complete",{
                "title":      title,
                "node_count": getattr(result,"node_count",0),
                "summary":    getattr(result,"summary",""),
            })
        except Exception as e:
            emit("error",{"message":str(e)})
        finally:
            state["phase"]="idle"; state["message"]="Absorption complete."
            emit("phase_change",{"phase":"idle","message":"Absorption complete."})

    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"})

# ── Dream log ─────────────────────────────────────────────────────────────────

@app.route("/api/dreamlog")
def api_dreamlog():
    logs = []
    if os.path.exists("logs"):
        for fname in sorted(os.listdir("logs"),reverse=True):
            if fname.startswith("dream_cycle") and fname.endswith(".json"):
                try:
                    with open(os.path.join("logs",fname)) as f:
                        d = json.load(f)
                    logs.append({
                        "file":     fname,
                        "mode":     d.get("mode",""),
                        "brain_mode": d.get("brain_mode",""),
                        "summary":  d.get("summary",""),
                        "questions":len(d.get("questions",[])),
                        "insights": len(d.get("insights",[])),
                        "answers":  len(d.get("answers",[])),
                        "steps":    len(d.get("steps",[])),
                        "mission_advances":len(d.get("mission_advances",[])),
                        "started":  d.get("started_at",0),
                    })
                except Exception: continue
    if not logs:
        app.logger.warning("No dream_cycle*.json files found for /api/dreamlog")
    return jsonify(logs[:20])

# ── Controls ──────────────────────────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    with state_lock:
        if state["phase"] != "idle":
            return jsonify({"error":"already running"}),409
    text = request.json.get("text","").strip()
    if not text: return jsonify({"error":"no text"}),400
    with state_lock:
        state["phase"]="ingesting"; state["message"]="Ingesting..."
    def run():
        emit("phase_change",{"phase":"ingesting","message":"Ingesting..."})
        ingestor.ingest(text,source=EdgeSource.CONVERSATION)
        save_state()
        state["phase"]="idle"; state["message"]="Ingestion complete."
        emit("phase_change",{"phase":"idle","message":"Ingestion complete."})
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/seed", methods=["POST"])
def api_seed():
    concept = request.json.get("concept","").strip()
    if not concept: return jsonify({"error":"no concept"}),400
    item = observer.add_to_agenda(concept,cycle=observer.cycle_count)
    item.priority = 0.95
    save_state()
    emit("seed_planted",{"concept":concept})
    return jsonify({"status":"seeded"})

@app.route("/api/run/<phase>", methods=["POST"])
def api_run_phase(phase):
    with state_lock:
        if state["phase"] != "idle":
            return jsonify({"error":"already running"}),409
        state["phase"] = "starting"
        state["message"] = f"Starting {phase}..."
    runners = {
        "dream":         _run_dream,
        "research":      _run_research,
        "thinking":      _run_thinking,
        "consolidation": _run_consolidation,
        "writing":       _run_writing,
        "sandbox":       _run_sandbox,
        "reading":       _run_reading,
    }
    if phase not in runners:
        return jsonify({"error":"unknown phase"}),400
    threading.Thread(target=runners[phase],daemon=True).start()
    return jsonify({"status":"started","phase":phase})

@app.route("/api/scientificness", methods=["POST"])
def api_scientificness():
    val = float(request.json.get("value",0.7))
    brain.scientificness = max(0.0,min(1.0,val))
    save_state()
    return jsonify({"scientificness":brain.scientificness})

# ── Phase runners ─────────────────────────────────────────────────────────────

def _run_dream():
    state["cycle"] += 1
    state["phase"]  = "dreaming"
    brain_mode      = brain.get_mode()
    state["message"]= f"Dreaming [{brain_mode}]..."
    emit("phase_change",{"phase":"dreaming","message":f"Night cycle [{brain_mode}]..."})
    try:
        log1 = dreamer.dream(
            mode=DreamMode.WANDERING, steps=15, temperature=0.7,
            run_nrem=True,
            log_path=f"logs/dream_cycle{state['cycle']}_wandering.json")
        for step in log1.steps:
            state["active_node_id"] = step.to_id
            emit("dream_step",{
                "step":      step.step, "from_id": step.from_id,
                "to_id":     step.to_id, "edge_type": step.edge_type,
                "narration": step.narration, "is_insight": step.is_insight,
                "insight_depth": getattr(step,"insight_depth",""),
                "mission_advance": getattr(step,"mission_advance",False),
                "question":  step.question, "brain_mode": brain_mode,
            })
        observer.observe(log1)
        notebook.write_morning_entry(log1, state["cycle"])
        for signal in observer.emergence_feed[-5:]:
            if (signal.type == "mission_advance" and
                    signal.cycle == observer.cycle_count):
                notebook.write_breakthrough(signal.signal, state["cycle"])
                break

        if not brain.is_wandering():
            log2 = dreamer.dream(
                mode=DreamMode.PRESSURE, steps=8, temperature=0.6,
                run_nrem=False,
                log_path=f"logs/dream_cycle{state['cycle']}_pressure.json")
            for step in log2.steps:
                state["active_node_id"] = step.to_id
                emit("dream_step",{
                    "step": step.step, "from_id": step.from_id,
                    "to_id": step.to_id, "edge_type": step.edge_type,
                    "narration": step.narration, "is_insight": step.is_insight,
                    "insight_depth": getattr(step,"insight_depth",""),
                    "mission_advance": getattr(step,"mission_advance",False),
                    "question": step.question, "brain_mode": brain_mode,
                })
            observer.observe_supplemental(log2)

        emit("dream_complete",{
            "cycle":      state["cycle"],
            "brain_mode": brain.get_mode(),
            "insights":   len(log1.insights),
            "advances":   len(log1.mission_advances),
            "summary":    log1.summary,
        })
        state["message"] = f"Dream cycle {state['cycle']} complete [{brain.get_mode()}]."
    except Exception as e:
        state["message"] = f"Dream error: {e}"
        emit("error",{"message":str(e)})
    finally:
        state["active_node_id"] = None
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"],
                              "mode":brain.get_mode()})

def _run_research():
    state["phase"]  = "researching"
    state["message"]= "Research day..."
    emit("phase_change",{"phase":"researching","message":"Research day..."})
    try:
        if brain.is_wandering():
            state["message"] = "Wandering mode — no targeted research."
            return
        log = researcher.research_day(
            max_questions=5,
            log_path=f"logs/research_cycle{state['cycle']}.json")
        append_daily_nodes([nid for entry in log.entries for nid in entry.node_ids])
        notebook.write_field_notes(log, state["cycle"])
        resolved = sum(1 for e in log.entries if e.resolved in ["partial","strong"])
        state["message"] = f"Research complete. {resolved} questions advanced."
        emit("research_complete",{"questions_researched":len(log.entries),
                                   "resolved":resolved})
    except Exception as e:
        state["message"] = f"Research error: {e}"
        emit("error",{"message":str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"]})

def _run_reading():
    state["phase"]  = "ingesting"
    state["message"]= "Reading..."
    emit("phase_change",{"phase":"ingesting","message":"Afternoon reading..."})
    try:
        unread = len(reader.get_unread(20))
        if unread < 3:
            reader.generate_reading_list()
        results  = reader.reading_day(max_items=2)
        absorbed = sum(1 for r in results if r.success)
        reading_new = [
            nid for result in results if result.success
            for nid in getattr(result, "node_ids", [])
        ]
        append_daily_nodes(reading_new)
        state["message"] = f"Reading complete. {absorbed} texts absorbed."
        emit("reading_complete",{"absorbed":absorbed,
            "summaries":[r.summary for r in results if r.success]})
    except Exception as e:
        state["message"] = f"Reading error: {e}"
        emit("error",{"message":str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"]})

def _run_consolidation():
    state["phase"]  = "consolidating"
    state["message"]= "Consolidation..."
    emit("phase_change",{"phase":"consolidating","message":"Evening consolidation..."})
    try:
        try:
            with open(DAILY_LEDGER_PATH, "r") as f:
                new_node_ids = json.load(f)
            os.remove(DAILY_LEDGER_PATH)
        except FileNotFoundError:
            new_node_ids = []
        report = consolidator.consolidate(
            new_node_ids=new_node_ids,
            save_path=f"logs/consolidation_cycle{state['cycle']}.json")
        notebook.write_evening_entry(report, state["cycle"])
        notebook.update_running_hypothesis(state["cycle"])
        state["message"] = "Consolidation complete."
        emit("consolidation_complete",{
            "merges":report.merges, "syntheses":report.syntheses,
            "abstractions":report.abstractions, "gaps":report.gaps,
            "summary":report.summary})
    except Exception as e:
        state["message"] = f"Consolidation error: {e}"
        emit("error",{"message":str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"]})

def _run_sandbox():
    state["phase"]  = "sandbox"
    state["message"]= "Running sandbox..."
    emit("phase_change",{"phase":"sandbox","message":"Sandbox scanning..."})
    try:
        results = sandbox.scan_and_test(max_tests=3)
        for r in results:
            emit("sandbox_result",{
                "verdict":r.verdict, "confidence":r.confidence,
                "interpretation":r.interpretation, "implications":r.implications,
                "hypothesis":r.hypothesis, "plot_path":r.plot_path})
        state["message"] = f"Sandbox: {len(results)} tests complete."
    except Exception as e:
        state["message"] = f"Sandbox error: {e}"
        emit("error",{"message":str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"]})

# ── Thinking & Writing ────────────────────────────────────────────────────────

def _run_thinking():
    state["phase"]  = "thinking"
    state["message"]= "Thinking session..."
    emit("phase_change",{"phase":"thinking","message":"Deliberate reasoning..."})
    try:
        logs = thinker.think_session(num_rounds=3)
        insights = [l.insight for l in logs if l.insight]
        state["message"] = f"Thinking: {len(insights)} insights produced."
        for log in logs:
            emit("thinking_step",{
                "pattern":log.pattern,"question":log.question,
                "insight":log.insight,"duration":log.duration})
    except Exception as e:
        state["message"] = f"Thinking error: {e}"
        emit("error",{"message":str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"]})

def _run_writing():
    state["phase"]  = "writing"
    state["message"]= "Writing synthesis..."
    emit("phase_change",{"phase":"writing","message":"Writing synthesis essay..."})
    try:
        result = notebook.write_synthesis_essay(state.get("cycle",0))
        # Ingest insights back into graph
        from ingestion.ingestor import EdgeSource
        for insight in result.get('insights', []):
            if isinstance(insight, str) and len(insight) > 15:
                ingestor.ingest(insight, source=EdgeSource.CONSOLIDATION)
        for question in result.get('questions', []):
            if isinstance(question, str) and len(question) > 10:
                observer.add_to_agenda(text=question, item_type="question",
                                       cycle=state.get("cycle",0))
        state["message"] = f"Essay written, {len(result.get('insights',[]))} insights."
    except Exception as e:
        state["message"] = f"Writing error: {e}"
        emit("error",{"message":str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change",{"phase":"idle","message":state["message"]})

# ── Conversation ──────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    message = request.json.get("message", "").strip()
    if not message:
        return jsonify({"error": "no message"}), 400
    try:
        result = conversation.chat(message)
        save_state()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    conversation.reset()
    return jsonify({"status": "cleared"})

@app.route("/api/chat/history")
def api_chat_history():
    return jsonify(conversation.get_history())

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("logs",exist_ok=True)
    os.makedirs("data",exist_ok=True)
    print(f"DREAMER GUI → http://0.0.0.0:5000")
    print(f"Brain: {brain.stats()}")
    print(f"Mode: {brain.get_mode()}")
    if brain.get_mission():
        print(f"Mission: {brain.get_mission()['question']}")
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True)
