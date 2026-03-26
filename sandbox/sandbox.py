import json
import time
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from ollama import Client
from graph.brain import Brain, Node, Edge, EdgeType, EdgeSource, NodeType, NodeStatus
from persistence import atomic_write_json

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL      = "llama3.1:8b"
SANDBOX_TIMEOUT   = 30
MAX_OUTPUT_CHARS  = 2000
SANDBOX_LOG_PATH  = "logs/sandbox_log.json"

# ── Prompts ───────────────────────────────────────────────────────────────────

TESTABILITY_PROMPT = """
You are evaluating whether a scientific hypothesis can be tested computationally.

Hypothesis: {hypothesis}

A hypothesis is computationally testable if it:
- Makes a quantitative prediction that can be modeled numerically
- Proposes a relationship between variables that can be simulated
- Suggests a mathematical structure that can be checked formally
- Can be partially validated through data analysis or statistical modeling

Respond with a JSON object:
{{
  "testable": true or false,
  "reason": "one sentence",
  "approach": "if testable: describe what kind of calculation would test it"
}}

Respond ONLY with JSON.
"""

CODE_GENERATION_PROMPT = """
You are a scientific programmer. Write Python code to test this hypothesis:

Hypothesis: {hypothesis}
Testing approach: {approach}
Central research question context: {mission}

Requirements:
- Use only standard library + numpy + scipy + matplotlib
- The code must run standalone with no user input
- Always use np.clip() to prevent overflow in any iterative calculations
- Always normalize arrays before matrix operations using array / (np.linalg.norm(array) + 1e-10)
- Use small values for learning rates (< 0.01) and time steps (< 0.1)
- Print clear, interpretable results at each stage
- If generating a plot, save it as 'sandbox_output.png' in the current directory
- Keep it under 80 lines
- Include a final print statement summarizing what the result means for the hypothesis

Write ONLY the Python code. No preamble, no markdown fences, no explanation.
"""

RESULT_INTERPRETATION_PROMPT = """
You are a scientist interpreting a computational result.

Hypothesis tested: {hypothesis}
Central research question: {mission}
Code that was run: {code}
Output produced: {output}
Any errors: {errors}

Important distinction:
- "error" means the code crashed and the test could NOT be performed — not a result about the hypothesis
- "inconclusive" means the test ran but the results don't clearly support or contradict
- "supports" means the output provides positive evidence
- "contradicts" means the output provides negative evidence

Interpret this result honestly.

Respond with a JSON object:
{{
  "verdict": one of ["supports", "contradicts", "inconclusive", "error"],
  "confidence": a float 0.0 to 1.0,
  "interpretation": "2-3 sentences interpreting the result",
  "implications": "1-2 sentences on what this means for the central question"
}}

Respond ONLY with JSON.
"""

# ── Sandbox result ────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    hypothesis_node_id: str
    hypothesis:         str
    approach:           str
    code:               str
    stdout:             str
    stderr:             str
    verdict:            str
    confidence:         float
    interpretation:     str
    implications:       str
    plot_path:          str   = ""
    timestamp:          float = field(default_factory=time.time)
    duration_seconds:   float = 0.0

    def to_dict(self):
        return self.__dict__

# ── Sandbox ───────────────────────────────────────────────────────────────────

class Sandbox:
    def __init__(self, brain: Brain, observer=None):
        self.brain    = brain
        self.observer = observer
        self.llm      = Client()
        self.results: list[SandboxResult] = []
        self._load()

    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        response = self.llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    def _mission(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

    # ── Testability check ─────────────────────────────────────────────────────

    def is_testable(self, hypothesis: str) -> tuple:
        raw = self._llm(TESTABILITY_PROMPT.format(hypothesis=hypothesis), temperature=0.2)
        try:
            result = json.loads(raw)
            return (
                result.get('testable', False),
                result.get('reason', ''),
                result.get('approach', '')
            )
        except (json.JSONDecodeError, ValueError):
            return False, "Parse error", ""

    # ── Code execution ────────────────────────────────────────────────────────

    def _run_code(self, code: str) -> tuple:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT,
                cwd=os.getcwd()
            )
            stdout = proc.stdout[:MAX_OUTPUT_CHARS]
            stderr = proc.stderr[:MAX_OUTPUT_CHARS]
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Timeout: execution exceeded {SANDBOX_TIMEOUT}s"
        except Exception as e:
            stdout = ""
            stderr = str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return stdout, stderr, time.time() - start

    # ── Full test pipeline ────────────────────────────────────────────────────

    def test_hypothesis(self, hypothesis: str,
                        node_id: str = "") -> SandboxResult:
        print(f"\n── Sandbox: testing hypothesis ──")
        print(f"   {hypothesis}")

        # step 1: testability
        testable, reason, approach = self.is_testable(hypothesis)
        if not testable:
            print(f"   Not computationally testable: {reason}")
            result = SandboxResult(
                hypothesis_node_id = node_id,
                hypothesis         = hypothesis,
                approach           = "not testable",
                code               = "",
                stdout             = "",
                stderr             = reason,
                verdict            = "inconclusive",
                confidence         = 0.0,
                interpretation     = f"Not computationally testable: {reason}",
                implications       = "Further conceptual or empirical work needed."
            )
            self.results.append(result)
            self._save()
            return result

        print(f"   Approach: {approach}")

        # step 2: generate code
        code = self._llm(CODE_GENERATION_PROMPT.format(
            hypothesis = hypothesis,
            approach   = approach,
            mission    = self._mission()
        ), temperature=0.2)

        # strip markdown fences
        if '```' in code:
            lines = code.split('\n')
            code  = '\n'.join(
                l for l in lines if not l.strip().startswith('```')
            )

        print(f"   Running code ({len(code)} chars)...")

        # step 3: run
        stdout, stderr, duration = self._run_code(code)
        print(f"   Completed in {duration:.1f}s")
        if stdout:
            print(f"   Output: {stdout}")
        if stderr:
            print(f"   Errors: {stderr}")

        # step 4: interpret
        raw = self._llm(RESULT_INTERPRETATION_PROMPT.format(
            hypothesis = hypothesis,
            mission    = self._mission(),
            code       = code,
            output     = stdout or "no output",
            errors     = stderr or "none"
        ), temperature=0.2)
        try:
            interp = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            interp = {
                "verdict":        "inconclusive",
                "confidence":     0.3,
                "interpretation": raw,
                "implications":   ""
            }

        verdict        = interp.get('verdict', 'inconclusive')
        confidence     = interp.get('confidence', 0.3)
        interpretation = interp.get('interpretation', '')
        implications   = interp.get('implications', '')

        print(f"   Verdict: {verdict} (confidence={confidence:.2f})")
        print(f"   {interpretation}")

        # check for plot
        plot_path = ""
        if os.path.exists("sandbox_output.png"):
            import shutil
            dest = f"logs/sandbox_{int(time.time())}.png"
            shutil.move("sandbox_output.png", dest)
            plot_path = dest
            print(f"   Plot saved: {dest}")

        result = SandboxResult(
            hypothesis_node_id = node_id,
            hypothesis         = hypothesis,
            approach           = approach,
            code               = code,
            stdout             = stdout,
            stderr             = stderr,
            verdict            = verdict,
            confidence         = confidence,
            interpretation     = interpretation,
            implications       = implications,
            plot_path          = plot_path,
            duration_seconds   = duration
        )
        self.results.append(result)
        self._integrate_result(result, node_id)
        self._save()
        return result

    # ── Graph integration ─────────────────────────────────────────────────────

    def _integrate_result(self, result: SandboxResult,
                          hypothesis_node_id: str):
        statement = (
            f"Computational test of: {result.hypothesis}. "
            f"Verdict: {result.verdict} (confidence={result.confidence:.2f}). "
            f"{result.interpretation}"
        )

        node = Node(
            statement        = statement,
            node_type        = NodeType.EMPIRICAL,
            cluster          = "empirical",
            status           = (NodeStatus.SETTLED
                                if result.confidence > 0.7
                                else NodeStatus.UNCERTAIN),
            importance       = result.confidence,
            empirical_result = result.interpretation,
            empirical_code   = result.code
        )
        nid = self.brain.add_node(node)

        # link to hypothesis node
        if hypothesis_node_id and self.brain.get_node(hypothesis_node_id):
            edge = Edge(
                type         = EdgeType.EMPIRICALLY_TESTED,
                narration    = (f"Computational test: {result.verdict} "
                                f"(confidence={result.confidence:.2f}). "
                                f"{result.implications}"),
                weight       = result.confidence,
                confidence   = result.confidence,
                source       = EdgeSource.SANDBOX,
                decay_exempt = result.confidence > 0.7
            )
            self.brain.add_edge(hypothesis_node_id, nid, edge)

            # add contradicts edge if result contradicts
            if result.verdict == "contradicts" and result.confidence > 0.6:
                contra = Edge(
                    type         = EdgeType.CONTRADICTS,
                    narration    = (f"Computational test contradicts: "
                                    f"{result.interpretation}"),
                    weight       = result.confidence,
                    confidence   = result.confidence,
                    source       = EdgeSource.SANDBOX,
                    decay_exempt = True
                )
                self.brain.add_edge(nid, hypothesis_node_id, contra)

        # only link to mission if test actually ran and is meaningful
        if result.verdict not in ("error",) and result.confidence > 0.6:
            self.brain.link_to_mission(
                nid,
                f"Empirical result: {result.implications}",
                strength=result.confidence * 0.7
            )

        # notify observer only on genuine supporting results
        if (self.observer and
                result.verdict == "supports" and
                result.confidence > 0.65):
            self.observer.record_mission_advance(
                nid,
                f"Computational test supports: {result.implications}",
                result.confidence * 0.8
            )

        print(f"   EMPIRICAL node created: {nid[:8]}")
        return nid

    # ── Scan and test ─────────────────────────────────────────────────────────

    def scan_and_test(self, max_tests: int = 3) -> list:
        print(f"\n── Sandbox scan: looking for testable hypotheses ──")

        mission_id = (self.brain.get_mission() or {}).get("id")

        candidates = []
        for nid, data in self.brain.nodes_by_type(NodeType.HYPOTHESIS):
            # skip already tested
            already_tested = any(
                edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                for _, _, edata in self.brain.graph.out_edges(nid, data=True)
            )
            if already_tested:
                continue

            score = data.get('importance', 0.5)
            if mission_id and self.brain.graph.has_edge(nid, mission_id):
                score += 0.3
            candidates.append((nid, data, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        results = []

        for nid, data, score in candidates[:max_tests]:
            stmt = data.get('statement', '')
            if not stmt:
                continue
            result = self.test_hypothesis(stmt, node_id=nid)
            results.append(result)
            time.sleep(1)

        print(f"\n── Sandbox complete: {len(results)} tests run ──")
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs("logs", exist_ok=True)
        data = {"results": [r.to_dict() for r in self.results]}
        atomic_write_json(SANDBOX_LOG_PATH, data)

    def _load(self):
        try:
            with open(SANDBOX_LOG_PATH, 'r') as f:
                data = json.load(f)
            self.results = [
                SandboxResult(**r) for r in data.get('results', [])
            ]
            print(f"Sandbox loaded — {len(self.results)} prior results")
        except FileNotFoundError:
            print("Sandbox: starting fresh")
