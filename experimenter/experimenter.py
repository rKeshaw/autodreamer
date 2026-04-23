import os
import json
import ast
import platform
import re
import sys
import subprocess
import time
import importlib.util
import importlib.metadata as importlib_metadata
import statistics
from dataclasses import asdict, dataclass, field

from llm_utils import llm_call, require_json
from persistence import atomic_write_json
from graph.brain import (
    Brain,
    Edge,
    EdgeSource,
    EdgeType,
    Node,
    NodeType,
    NodeStatus,
)
from scientist_workspace import ArtifactStatus

DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_BASE_SEED = 1729
MAX_CODE_SNIPPET_CHARS = 6000
MAX_SECURITY_REWRITE_PASSES = 2
PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")

PACKAGE_ALIAS_MAP = {
    "sklearn": "scikit-learn",
    "scikit_learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "pil": "pillow",
    "pytorch": "torch",
}

SECURITY_REWRITE_PROMPT = """You are a security observer for generated scientific experiment code.

Task:
- Fix ONLY security issues.
- Preserve the scientific logic, metrics computation, and experiment intent.
- Keep deterministic seed behavior and METRICS_JSON output.

Detected security issues:
{issues_json}

Original code:
```python
{code}
```

Output ONLY corrected Python code in a single ```python ... ``` block.
Do not add explanations.
"""

EXPERIMENT_PLAN_PROMPT = """You are planning a reproducible virtual-lab experiment.

Hypothesis:
{hypothesis}

Return ONLY JSON with this schema:
{{
  "hypothesis": "copy of hypothesis",
  "protocol": "3-6 sentence concrete protocol with what will be measured",
  "controls": ["control 1", "control 2"],
  "metrics": ["metric_name_1", "metric_name_2"],
  "success_criteria": ["criterion 1", "criterion 2"],
    "preferred_libraries": ["numpy", "scipy"],
  "stochastic": true or false,
  "recommended_runs": integer >= 1,
  "interpretation_focus": "how to decide confirms vs contradicts"
}}

Rules:
- Keep controls and metrics concrete and measurable.
- Recommend external scientific libraries in preferred_libraries when useful.
- If there is randomness, set stochastic=true and recommended_runs >= 2.
- Avoid vague goals like "looks better"; define measurable outcomes.
"""

EXPERIMENT_CODE_PROMPT = """You are the Experimenter generating executable Python for a virtual lab.

Hypothesis:
{hypothesis}

Experiment plan:
{plan_json}

Write ONE standalone Python script that follows the plan.

Hard requirements:
1. Deterministic under fixed seed.
2. Read seed from argv `--seed` if present, else from env var `EXPERIMENT_SEED`, else default 1729.
3. Print clear human-readable conclusions.
4. Print one machine-readable metrics line exactly in this format:
   METRICS_JSON: {{"metric_a": 0.0, "metric_b": 0.0}}
5. You may use external scientific packages as needed.
6. The runtime will install missing packages inferred from imports.
7. Avoid dangerous operations unrelated to science (shell injection, destructive system commands).

Return ONLY valid Python code in a single ```python ... ``` block.
"""

METRIC_EXTRACTION_PROMPT = """Extract measurable metrics from this experiment output.

Output:
{output}

Return ONLY a JSON object mapping metric names to numeric values.
If no metrics are present, return {{}}.
"""

INTERPRETATION_PROMPT = """You executed a structured virtual-lab experiment.

Hypothesis:
{hypothesis}

Experiment plan:
{plan_json}

Run summaries:
{run_summaries_json}

Metric consistency summary:
{consistency_json}

Interpret the evidence. Verdict must be grounded in measured metrics.

Return ONLY JSON:
{{
  "verdict": "confirms" | "contradicts" | "irrelevant",
  "explanation": "one sentence referencing concrete metrics",
  "specific_claim": "quantitative empirical claim",
  "metrics_used": [
    {{"name": "metric", "evidence": "value or trend used"}}
  ],
  "confidence": 0.0 to 1.0
}}
"""

EXPERIMENT_ADMISSIBILITY_PROMPT = """You are screening a hypothesis for virtual-lab testability.

Hypothesis:
{hypothesis}

Hypothesis metadata:
{metadata_json}

Return ONLY JSON:
{{
  "admissible": true or false,
  "reason": "one sentence",
  "required_prerequisite": "what is missing if inadmissible"
}}

Rules:
- admissible=true only if a standalone computational experiment or numerical analysis can test the claim without inventing missing equations or observables.
- If the claim mainly needs formal derivation, literature comparison, or externally measured physical inputs not present here, return admissible=false.
- If the hypothesis is speculative and lacks grounded support, return admissible=false.
"""


@dataclass
class ExperimentPlan:
    hypothesis: str
    protocol: str
    controls: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    preferred_libraries: list[str] = field(default_factory=list)
    stochastic: bool = True
    recommended_runs: int = 2
    interpretation_focus: str = ""

    @classmethod
    def from_payload(cls, payload: dict | None, hypothesis: str) -> "ExperimentPlan":
        payload = payload or {}

        protocol = str(payload.get("protocol", "") or "").strip()
        if not protocol:
            protocol = (
                "Run a reproducible simulation with explicit controls and "
                "record quantitative metrics for evaluation."
            )

        controls = [
            str(item).strip()
            for item in (payload.get("controls", []) or [])
            if str(item).strip()
        ]
        metrics = [
            str(item).strip()
            for item in (payload.get("metrics", []) or [])
            if str(item).strip()
        ]
        success_criteria = [
            str(item).strip()
            for item in (payload.get("success_criteria", []) or [])
            if str(item).strip()
        ]
        preferred_libraries = [
            str(item).strip().lower()
            for item in (payload.get("preferred_libraries", []) or [])
            if str(item).strip()
        ]

        stochastic = bool(payload.get("stochastic", True))
        try:
            recommended_runs = int(payload.get("recommended_runs", 2) or 2)
        except (TypeError, ValueError):
            recommended_runs = 2
        recommended_runs = max(1, recommended_runs)

        if stochastic:
            recommended_runs = max(2, recommended_runs)

        return cls(
            hypothesis=str(payload.get("hypothesis", "") or hypothesis).strip() or hypothesis,
            protocol=protocol,
            controls=controls,
            metrics=metrics,
            success_criteria=success_criteria,
            preferred_libraries=preferred_libraries,
            stochastic=stochastic,
            recommended_runs=recommended_runs,
            interpretation_focus=str(payload.get("interpretation_focus", "") or "").strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunArtifact:
    run_index: int
    seed: int
    run_dir: str
    script_path: str
    stdout_path: str
    stderr_path: str
    metrics_path: str
    environment_path: str
    return_code: int
    duration_sec: float
    metrics: dict = field(default_factory=dict)
    metrics_found: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

class Experimenter:
    def __init__(self, brain: Brain, sandbox_dir: str = "virtual_lab"):
        self.brain = brain
        self.sandbox_dir = os.path.abspath(sandbox_dir)
        os.makedirs(self.sandbox_dir, exist_ok=True)
        self.experiments_dir = os.path.join(self.sandbox_dir, "experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)

    def _llm(self, prompt: str, temperature: float = 0.3,
             role: str = "precise") -> str:
        return llm_call(prompt, temperature=temperature, role=role)

    def _extract_code(self, raw_text: str) -> str:
        if "```python" in raw_text:
            return raw_text.split("```python")[1].split("```")[0].strip()
        if "```" in raw_text:
            return raw_text.split("```")[1].split("```")[0].strip()
        return raw_text.strip()

    def _slug(self, text: str, max_len: int = 42) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower())
        cleaned = cleaned.strip("-")
        if not cleaned:
            cleaned = "experiment"
        return cleaned[:max_len].rstrip("-")

    def _deterministic_admissibility(self, hypothesis: str, metadata: dict) -> dict:
        text = " ".join(
            str(part or "")
            for part in (
                hypothesis,
                metadata.get("predicted_answer", ""),
                metadata.get("testable_by", ""),
            )
        ).lower()
        theory_markers = {
            "lagrangian", "hamiltonian", "operator", "coupling", "symmetry",
            "sector", "field", "axion", "qcd", "theta", "dipole moment",
            "curvature", "torsion", "geometry", "mirror sector",
        }
        computational_markers = {
            "accuracy", "loss", "reward", "dataset", "simulation", "trajectory",
            "classifier", "regression", "replay", "optimization", "control",
            "benchmark", "seed", "ablation", "performance", "policy",
        }
        if any(marker in text for marker in theory_markers):
            return {
                "admissible": False,
                "reason": "Hypothesis appears to require formal theoretical derivation rather than standalone computational testing.",
                "required_prerequisite": "derive explicit equations or constrained observables before virtual-lab testing",
            }
        if any(marker in text for marker in computational_markers):
            return {
                "admissible": True,
                "reason": "Hypothesis is framed in computationally testable terms.",
                "required_prerequisite": "",
            }
        return {
            "admissible": False,
            "reason": "Failed to establish computational testability.",
            "required_prerequisite": "",
        }

    def _write_text(self, path: str, content: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _package_version(self, package_name: str) -> str:
        try:
            return importlib_metadata.version(package_name)
        except Exception:
            return "not_installed"

    def _security_issues(self, code: str) -> list[dict]:
        issues = []
        checks = [
            {
                "id": "shell_true",
                "severity": "high",
                "message": "subprocess with shell=True can enable shell injection",
                "pattern": r"subprocess\.(run|Popen|call|check_call|check_output)\([^\n]*shell\s*=\s*True",
            },
            {
                "id": "os_system",
                "severity": "high",
                "message": "os.system allows uncontrolled shell execution",
                "pattern": r"\bos\.system\s*\(",
            },
            {
                "id": "dynamic_exec",
                "severity": "high",
                "message": "eval/exec can execute arbitrary code",
                "pattern": r"\b(eval|exec)\s*\(",
            },
            {
                "id": "destructive_command",
                "severity": "critical",
                "message": "destructive system command detected",
                "pattern": r"rm\s+-rf\s+/|mkfs\.|dd\s+if=|shutdown\s+-|reboot\b|poweroff\b",
            },
            {
                "id": "sensitive_path_write",
                "severity": "high",
                "message": "write operation against sensitive system path",
                "pattern": r"open\s*\(\s*[\"'](?:/etc/|/root/|/var/run/docker\.sock|~/.ssh)",
            },
        ]

        for check in checks:
            for match in re.finditer(check["pattern"], code, flags=re.IGNORECASE | re.MULTILINE):
                issues.append(
                    {
                        "id": check["id"],
                        "severity": check["severity"],
                        "message": check["message"],
                        "match": match.group(0),
                    }
                )
        return issues

    def _hard_security_patch(self, code: str) -> str:
        patched = code

        patched = re.sub(r"shell\s*=\s*True", "shell=False", patched)

        if re.search(r"\bos\.system\s*\(", patched):
            if "def _blocked_system_call(" not in patched:
                patched = (
                    "def _blocked_system_call(*args, **kwargs):\n"
                    "    raise RuntimeError('Blocked insecure os.system call by security observer')\n\n"
                ) + patched
            patched = re.sub(r"\bos\.system\s*\(", "_blocked_system_call(", patched)

        if re.search(r"\bexec\s*\(", patched):
            if "def _blocked_exec_call(" not in patched:
                patched = (
                    "def _blocked_exec_call(*args, **kwargs):\n"
                    "    raise RuntimeError('Blocked insecure exec call by security observer')\n\n"
                ) + patched
            patched = re.sub(r"\bexec\s*\(", "_blocked_exec_call(", patched)

        if re.search(r"\beval\s*\(", patched):
            if "import ast" not in patched:
                patched = "import ast\n" + patched
            patched = re.sub(r"\beval\s*\(", "ast.literal_eval(", patched)

        return patched

    def _apply_security_observer(self, hypothesis: str,
                                 code: str,
                                 session_dir: str) -> dict:
        sanitized = code
        initial_issues = self._security_issues(sanitized)
        rewrite_passes = []
        remaining_issues = list(initial_issues)

        for idx in range(MAX_SECURITY_REWRITE_PASSES):
            if not remaining_issues:
                break
            raw_rewrite = self._llm(
                SECURITY_REWRITE_PROMPT.format(
                    issues_json=json.dumps(remaining_issues, indent=2),
                    code=sanitized,
                ),
                temperature=0.05,
                role="code",
            )
            rewritten = self._extract_code(raw_rewrite)
            if not rewritten:
                break
            sanitized = rewritten
            remaining_issues = self._security_issues(sanitized)
            rewrite_passes.append(
                {
                    "pass": idx + 1,
                    "remaining_issue_count": len(remaining_issues),
                }
            )

        if remaining_issues:
            sanitized = self._hard_security_patch(sanitized)
            remaining_issues = self._security_issues(sanitized)

        status = "clean"
        if initial_issues and not remaining_issues:
            status = "corrected"
        elif remaining_issues:
            status = "blocked"

        report = {
            "hypothesis": hypothesis,
            "status": status,
            "initial_issues": initial_issues,
            "rewrite_passes": rewrite_passes,
            "remaining_issues": remaining_issues,
            "issue_count_before": len(initial_issues),
            "issue_count_after": len(remaining_issues),
        }
        report_path = os.path.join(session_dir, "security_report.json")
        atomic_write_json(report_path, report)
        report["path"] = report_path
        report["sanitized_code"] = sanitized
        return report

    def _stdlib_modules(self) -> set[str]:
        std = getattr(sys, "stdlib_module_names", None)
        if std:
            return set(std)
        return {
            "argparse", "collections", "csv", "datetime", "functools",
            "itertools", "json", "math", "os", "pathlib", "random",
            "re", "statistics", "subprocess", "sys", "time", "typing",
            "unittest",
        }

    def _module_available(self, module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _pip_install(self, package_name: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package_name],
                capture_output=True,
                text=True,
                timeout=240,
            )
            if result.returncode == 0:
                return True, (result.stdout or "").strip()
            stderr = (result.stderr or "").strip() or (result.stdout or "").strip()
            return False, stderr
        except Exception as exc:
            return False, str(exc)

    def _library_spec_for_request(self, request: str) -> dict | None:
        key = str(request or "").strip().lower()
        if not key:
            return None

        canonical = key.replace("_", "-")
        package = PACKAGE_ALIAS_MAP.get(canonical, canonical)
        module = key.replace("-", "_").split(".", 1)[0]
        if module in PACKAGE_ALIAS_MAP:
            package = PACKAGE_ALIAS_MAP[module]

        if not PACKAGE_NAME_RE.match(package):
            return None
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", module):
            return None

        return {
            "request": request,
            "package": package,
            "module": module,
        }

    def _infer_external_modules(self, code: str) -> list[str]:
        modules = []
        stdlib = self._stdlib_modules()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".", 1)[0]
                        if module and module not in stdlib:
                            modules.append(module)
                elif isinstance(node, ast.ImportFrom):
                    if not node.module:
                        continue
                    module = node.module.split(".", 1)[0]
                    if module and module not in stdlib:
                        modules.append(module)
        except SyntaxError:
            for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w\.]*)", code, re.MULTILINE):
                module = match.group(1).split(".", 1)[0]
                if module and module not in stdlib:
                    modules.append(module)

        deduped = []
        for module in modules:
            if module not in deduped:
                deduped.append(module)
        return deduped

    def _provision_dependencies(self, code: str,
                                plan: ExperimentPlan,
                                session_dir: str) -> dict:
        inferred_modules = self._infer_external_modules(code)
        requested_entries = list(inferred_modules)
        for lib in plan.preferred_libraries:
            if lib not in requested_entries:
                requested_entries.append(lib)

        recognized = []
        invalid_requests = []
        seen_packages = set()
        for request in requested_entries:
            spec = self._library_spec_for_request(request)
            if not spec:
                invalid_requests.append(request)
                continue
            if spec["package"] in seen_packages:
                continue
            seen_packages.add(spec["package"])
            recognized.append(spec)

        already_available = []
        installed = []
        failed = []
        for spec in recognized:
            package_name = spec["package"]
            module_name = spec["module"]
            if self._module_available(module_name):
                already_available.append(package_name)
                continue

            ok, message = self._pip_install(package_name)
            if ok and self._module_available(module_name):
                installed.append(package_name)
            else:
                failed.append({
                    "package": package_name,
                    "module": module_name,
                    "error": message,
                })

        report = {
            "requested_entries": requested_entries,
            "recognized": recognized,
            "invalid_requests": invalid_requests,
            "already_available": sorted(set(already_available)),
            "installed": sorted(set(installed)),
            "failed": failed,
        }
        dependency_report_path = os.path.join(session_dir, "dependency_report.json")
        atomic_write_json(dependency_report_path, report)
        report["path"] = dependency_report_path
        return report

    def _environment_metadata(self, seed: int,
                              tracked_packages: list[str] | None = None) -> dict:
        tracked = {
            "numpy", "scipy", "pandas", "networkx", "torch", "pypdf"
        }
        for pkg in tracked_packages or []:
            if pkg:
                tracked.add(str(pkg))

        package_versions = {
            pkg: self._package_version(pkg)
            for pkg in sorted(tracked)
        }
        return {
            "seed": seed,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": package_versions,
            "timestamp": time.time(),
        }

    def _assess_admissibility(self, hypothesis_node_id: str, hypothesis: str) -> dict:
        node = self.brain.get_node(hypothesis_node_id) or {}
        grounded_support = 0
        for source_id, _, edge in self.brain.graph.in_edges(hypothesis_node_id, data=True):
            source = self.brain.get_node(source_id) or {}
            if (
                source.get("epistemic_status") == ArtifactStatus.GROUNDED.value and
                edge.get("type") in {
                    EdgeType.CONFIRMED_BY.value,
                    EdgeType.SUPPORTS.value,
                    EdgeType.EMPIRICALLY_TESTED.value,
                }
            ):
                grounded_support += 1
        metadata = {
            "status": node.get("status", ""),
            "epistemic_status": node.get("epistemic_status", ""),
            "created_by": node.get("created_by", ""),
            "predicted_answer": node.get("predicted_answer", ""),
            "testable_by": node.get("testable_by", ""),
            "grounded_support_count": grounded_support,
        }
        raw = self._llm(
            EXPERIMENT_ADMISSIBILITY_PROMPT.format(
                hypothesis=hypothesis,
                metadata_json=json.dumps(metadata, indent=2),
            ),
            temperature=0.0,
            role="verifier",
        )
        parsed = require_json(raw, default={})
        if not isinstance(parsed, dict):
            parsed = {}
        if not parsed:
            parsed = self._deterministic_admissibility(hypothesis, metadata)
        admissible = parsed.get("admissible", False)
        if isinstance(admissible, str):
            admissible = admissible.strip().lower() in {"true", "1", "yes"}
        parsed["admissible"] = bool(admissible)
        parsed.setdefault("reason", "Failed to establish computational testability.")
        parsed.setdefault("required_prerequisite", "")
        return parsed

    def _build_plan(self, hypothesis: str) -> ExperimentPlan:
        raw = self._llm(
            EXPERIMENT_PLAN_PROMPT.format(hypothesis=hypothesis),
            temperature=0.15,
            role="precise",
        )
        payload = require_json(raw, default={})
        if not isinstance(payload, dict):
            payload = {}
        return ExperimentPlan.from_payload(payload, hypothesis=hypothesis)

    def _generate_code(self, hypothesis: str, plan: ExperimentPlan) -> str:
        raw = self._llm(
            EXPERIMENT_CODE_PROMPT.format(
                hypothesis=hypothesis,
                plan_json=json.dumps(plan.to_dict(), indent=2),
            ),
            temperature=0.2,
            role="code",
        )
        code = self._extract_code(raw)
        return code

    def _extract_metrics_from_output(self, stdout: str, stderr: str) -> dict:
        for line in (stdout or "").splitlines()[::-1]:
            if "METRICS_JSON:" not in line:
                continue
            payload = line.split("METRICS_JSON:", 1)[1].strip()
            parsed = require_json(payload, default={})
            if isinstance(parsed, dict):
                return parsed

        combined = (stdout or "") + "\n" + (stderr or "")
        parsed_fallback = require_json(combined, default=None)
        if isinstance(parsed_fallback, dict) and parsed_fallback:
            return parsed_fallback

        raw = self._llm(
            METRIC_EXTRACTION_PROMPT.format(output=combined[:5000]),
            temperature=0.0,
            role="precise",
        )
        parsed = require_json(raw, default={})
        return parsed if isinstance(parsed, dict) else {}

    def _execute_single_run(self, session_dir: str, run_index: int,
                            seed: int, code: str,
                            tracked_packages: list[str] | None = None,
                            timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> RunArtifact:
        run_dir = os.path.join(session_dir, f"run_{run_index:02d}")
        os.makedirs(run_dir, exist_ok=True)

        script_path = os.path.join(run_dir, "experiment.py")
        stdout_path = os.path.join(run_dir, "stdout.txt")
        stderr_path = os.path.join(run_dir, "stderr.txt")
        metrics_path = os.path.join(run_dir, "metrics.json")
        environment_path = os.path.join(run_dir, "environment.json")

        self._write_text(script_path, code)
        atomic_write_json(
            environment_path,
            self._environment_metadata(seed, tracked_packages=tracked_packages),
        )

        env = os.environ.copy()
        env["EXPERIMENT_SEED"] = str(seed)

        start = time.time()
        command = [sys.executable, script_path, "--seed", str(seed)]
        stdout_text = ""
        stderr_text = ""
        return_code = -1

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )

            # Fallback for scripts that do not accept --seed.
            if result.returncode != 0 and "unrecognized arguments: --seed" in (result.stderr or ""):
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=env,
                )

            stdout_text = result.stdout or ""
            stderr_text = result.stderr or ""
            return_code = int(result.returncode)
        except subprocess.TimeoutExpired:
            stdout_text = ""
            stderr_text = f"ERROR: Experiment timed out after {timeout_seconds} seconds."
            return_code = -124
        except Exception as exc:
            stdout_text = ""
            stderr_text = f"ERROR: Experiment execution failed: {exc}"
            return_code = -1

        duration_sec = time.time() - start
        self._write_text(stdout_path, stdout_text)
        self._write_text(stderr_path, stderr_text)

        metrics = self._extract_metrics_from_output(stdout_text, stderr_text)
        metrics_found = isinstance(metrics, dict) and bool(metrics)
        atomic_write_json(
            metrics_path,
            {
                "run_index": run_index,
                "seed": seed,
                "return_code": return_code,
                "metrics_found": metrics_found,
                "metrics": metrics if isinstance(metrics, dict) else {},
            },
        )

        return RunArtifact(
            run_index=run_index,
            seed=seed,
            run_dir=run_dir,
            script_path=script_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metrics_path=metrics_path,
            environment_path=environment_path,
            return_code=return_code,
            duration_sec=duration_sec,
            metrics=metrics if isinstance(metrics, dict) else {},
            metrics_found=metrics_found,
        )

    def _run_summary(self, runs: list[RunArtifact]) -> dict:
        summary = {
            "total_runs": len(runs),
            "successful_runs": sum(1 for run in runs if run.return_code == 0),
            "metrics_available_runs": sum(1 for run in runs if run.metrics_found),
            "metric_statistics": {},
        }

        values_by_metric: dict[str, list[float]] = {}
        for run in runs:
            if not isinstance(run.metrics, dict):
                continue
            for key, value in run.metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values_by_metric.setdefault(str(key), []).append(float(value))

        for metric_name, values in values_by_metric.items():
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            minimum = min(values)
            maximum = max(values)
            rel_std = abs(std / mean) if abs(mean) > 1e-9 else abs(std)

            summary["metric_statistics"][metric_name] = {
                "values": values,
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
                "relative_std": rel_std,
                "consistent": rel_std <= 0.25,
            }

        summary["has_quantitative_metrics"] = bool(summary["metric_statistics"])
        return summary

    def _interpret_runs(self, hypothesis: str, plan: ExperimentPlan,
                        runs: list[RunArtifact], run_summary: dict) -> dict:
        run_payload = [
            {
                "run_index": run.run_index,
                "seed": run.seed,
                "return_code": run.return_code,
                "duration_sec": run.duration_sec,
                "metrics": run.metrics,
            }
            for run in runs
        ]

        raw = self._llm(
            INTERPRETATION_PROMPT.format(
                hypothesis=hypothesis,
                plan_json=json.dumps(plan.to_dict(), indent=2),
                run_summaries_json=json.dumps(run_payload, indent=2),
                consistency_json=json.dumps(run_summary, indent=2),
            ),
            temperature=0.1,
            role="precise",
        )
        parsed = require_json(raw, default={})
        if not isinstance(parsed, dict):
            parsed = {}

        verdict = str(parsed.get("verdict", "irrelevant") or "irrelevant").strip().lower()
        if verdict not in {"confirms", "contradicts", "irrelevant"}:
            verdict = "irrelevant"

        explanation = str(parsed.get("explanation", "") or "").strip()
        specific_claim = str(parsed.get("specific_claim", "") or "").strip()
        metrics_used = parsed.get("metrics_used", [])
        if not isinstance(metrics_used, list):
            metrics_used = []

        try:
            confidence = float(parsed.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        metric_stats = run_summary.get("metric_statistics", {}) or {}
        metric_summary_text = "; ".join(
            f"{name}=mean {stats.get('mean', 0.0):.4f}, std {stats.get('std', 0.0):.4f}"
            for name, stats in metric_stats.items()
        )

        if metric_summary_text and metric_summary_text.lower() not in explanation.lower():
            explanation = (explanation + " " if explanation else "") + f"Measured metrics: {metric_summary_text}."
        if metric_summary_text and metric_summary_text.lower() not in specific_claim.lower():
            specific_claim = (
                specific_claim + " " if specific_claim else ""
            ) + f"Measured metrics indicate {metric_summary_text}."

        if not explanation:
            explanation = "Run outputs were evaluated but did not yield a strong conclusion."
        if not specific_claim:
            specific_claim = explanation

        return {
            "verdict": verdict,
            "explanation": explanation,
            "specific_claim": specific_claim,
            "metrics_used": metrics_used,
            "confidence": confidence,
        }

    def _attach_experiment_to_graph(self, hypothesis_node_id: str,
                                    hypothesis_text: str,
                                    verdict: str,
                                    interpretation: dict,
                                    run_summary: dict,
                                    session_dir: str,
                                    code: str,
                                    session_summary_path: str) -> str:
        summary_ref = f"virtual_lab://{os.path.basename(session_dir)}/session_summary"
        summary_excerpt = interpretation.get("explanation", "")

        source_id = self.brain.create_source_node(
            title=f"Virtual Lab Experiment: {self._slug(hypothesis_text, max_len=56)}",
            reference=summary_ref,
            source_type="virtual_lab_experiment",
            created_by="experimenter",
            excerpt=summary_excerpt[:800],
        )

        metric_summary_text = "; ".join(
            f"{name}=mean {stats.get('mean', 0.0):.4f}, std {stats.get('std', 0.0):.4f}"
            for name, stats in (run_summary.get("metric_statistics", {}) or {}).items()
        )
        quote = metric_summary_text or summary_excerpt or "Experiment completed with recorded outputs."

        empirical_node = Node(
            statement=interpretation.get("specific_claim", "Experiment completed."),
            node_type=NodeType.EMPIRICAL,
            cluster="empirical",
            status=NodeStatus.SETTLED.value,
            epistemic_status=ArtifactStatus.GROUNDED.value,
            source_quality=0.95,
            source_ids=[source_id],
            source_refs=[summary_ref],
            provenance_spans=[
                {
                    "source_id": source_id,
                    "source_ref": summary_ref,
                    "section_label": "run_summary",
                    "span_start": 0,
                    "span_end": 0,
                    "quote": quote[:320],
                    "extraction_confidence": float(interpretation.get("confidence", 0.75) or 0.75),
                }
            ],
            extraction_confidence=float(interpretation.get("confidence", 0.75) or 0.75),
            created_by="experimenter",
            importance=0.82,
            empirical_result=verdict,
            empirical_code=code[:MAX_CODE_SNIPPET_CHARS],
            empirical_metrics=run_summary.get("metric_statistics", {}),
            experiment_artifacts={
                "session_dir": session_dir,
                "session_summary": session_summary_path,
                "total_runs": run_summary.get("total_runs", 0),
                "successful_runs": run_summary.get("successful_runs", 0),
            },
        )
        empirical_id = self.brain.add_node(empirical_node)

        tested_edge = Edge(
            type=EdgeType.EMPIRICALLY_TESTED,
            narration=(
                f"Virtual-lab protocol executed over {run_summary.get('total_runs', 0)} runs; "
                f"{run_summary.get('successful_runs', 0)} successful."
            ),
            weight=0.84,
            confidence=float(interpretation.get("confidence", 0.75) or 0.75),
            source=EdgeSource.SANDBOX,
            decay_exempt=True,
        )
        self.brain.add_edge(hypothesis_node_id, empirical_id, tested_edge)

        if verdict == "contradicts":
            self.brain.update_node(
                hypothesis_node_id,
                status=NodeStatus.CONTRADICTED.value,
                epistemic_status=ArtifactStatus.CONTRADICTED.value,
            )
            correction_edge = Edge(
                type=EdgeType.CORRECTED_BY,
                narration=interpretation.get("explanation", ""),
                weight=0.9,
                confidence=float(interpretation.get("confidence", 0.75) or 0.75),
                source=EdgeSource.SANDBOX,
                decay_exempt=True,
            )
            self.brain.add_edge(empirical_id, hypothesis_node_id, correction_edge)
        elif verdict == "confirms":
            hypothesis_node = self.brain.get_node(hypothesis_node_id) or {}
            self.brain.update_node(
                hypothesis_node_id,
                status=NodeStatus.UNCERTAIN.value,
                epistemic_status=ArtifactStatus.OPEN.value,
                importance=min(1.0, float(hypothesis_node.get("importance", 0.5) or 0.5) + 0.2),
            )
            confirm_edge = Edge(
                type=EdgeType.CONFIRMED_BY,
                narration=interpretation.get("explanation", ""),
                weight=0.88,
                confidence=float(interpretation.get("confidence", 0.75) or 0.75),
                source=EdgeSource.SANDBOX,
            )
            self.brain.add_edge(empirical_id, hypothesis_node_id, confirm_edge)
        else:
            self.brain.update_node(
                hypothesis_node_id,
                status=NodeStatus.LACKS_EVIDENCE.value,
                epistemic_status=ArtifactStatus.LACKS_EVIDENCE.value,
            )

        return empirical_id

    def run_experiment(self, hypothesis_node_id: str,
                       force_runs: int | None = None,
                       timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
        hyp_node = self.brain.get_node(hypothesis_node_id)
        if not hyp_node:
            return {"status": "error", "message": "Node not found"}

        question_text = str(hyp_node.get("statement", "") or "").strip()
        print(f"\n  ── 🔬 Virtual Lab: structured experiment for hypothesis ──")
        print(f"     {question_text[:90]}...")

        admissibility = self._assess_admissibility(hypothesis_node_id, question_text)
        if not admissibility.get("admissible", False):
            print(f"     Blocked: {admissibility.get('reason', 'inadmissible hypothesis')}")
            prerequisite = str(admissibility.get("required_prerequisite", "") or "").strip()
            if prerequisite:
                task_node = Node(
                    statement=f"Prerequisite for testing: {prerequisite}",
                    node_type=NodeType.TASK,
                    status=NodeStatus.UNCERTAIN,
                    epistemic_status=ArtifactStatus.OPEN.value,
                    importance=0.78,
                    created_by="experimenter_gate",
                )
                task_id = self.brain.add_node(task_node)
                self.brain.add_edge(
                    task_id,
                    hypothesis_node_id,
                    Edge(
                        type=EdgeType.TESTS,
                        narration="Experimenter prerequisite task generated because the hypothesis is not yet computationally testable.",
                        weight=0.8,
                        confidence=0.9,
                        source=EdgeSource.SANDBOX,
                    ),
                )
            return {
                "status": "blocked",
                "message": admissibility.get("reason", "Hypothesis is not yet computationally testable."),
                "admissibility": admissibility,
                "run_count": 0,
            }

        plan = self._build_plan(question_text)
        requested_runs = force_runs if isinstance(force_runs, int) and force_runs > 0 else plan.recommended_runs
        run_count = max(2, requested_runs) if plan.stochastic else max(1, requested_runs)

        timestamp = int(time.time())
        session_name = f"exp_{timestamp}_{self._slug(hypothesis_node_id, max_len=8)}"
        session_dir = os.path.join(self.experiments_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        plan_path = os.path.join(session_dir, "plan.json")
        atomic_write_json(plan_path, plan.to_dict())

        code = self._generate_code(question_text, plan)
        if not code:
            return {
                "status": "error",
                "message": "Failed to generate experiment code.",
                "session_dir": session_dir,
                "plan_path": plan_path,
            }

        security_report = self._apply_security_observer(
            hypothesis=question_text,
            code=code,
            session_dir=session_dir,
        )
        code = security_report.pop("sanitized_code", code)
        if security_report.get("status") == "blocked":
            return {
                "status": "error",
                "message": "Security observer could not sanitize generated code safely.",
                "session_dir": session_dir,
                "plan_path": plan_path,
                "security_report": security_report,
            }

        script_path = os.path.join(session_dir, "experiment.py")
        self._write_text(script_path, code)

        dependency_report = self._provision_dependencies(
            code=code,
            plan=plan,
            session_dir=session_dir,
        )
        tracked_packages = sorted({
            spec.get("package", "")
            for spec in dependency_report.get("recognized", [])
            if isinstance(spec, dict) and spec.get("package")
        })

        print(f"     Plan saved: {plan_path}")
        print(f"     Script saved: {script_path}")
        print(
            "     Security observer: "
            f"status={security_report.get('status')} "
            f"issues={security_report.get('issue_count_before', 0)}->"
            f"{security_report.get('issue_count_after', 0)}"
        )
        print(
            "     Dependency provisioning: "
            f"installed={len(dependency_report.get('installed', []))} "
            f"available={len(dependency_report.get('already_available', []))} "
            f"failed={len(dependency_report.get('failed', []))}"
        )
        print(f"     Running {run_count} execution(s) for reproducibility...")

        run_artifacts = []
        for run_index in range(1, run_count + 1):
            seed = DEFAULT_BASE_SEED + run_index - 1
            run_artifact = self._execute_single_run(
                session_dir=session_dir,
                run_index=run_index,
                seed=seed,
                code=code,
                tracked_packages=tracked_packages,
                timeout_seconds=timeout_seconds,
            )
            run_artifacts.append(run_artifact)
            print(
                f"     Run {run_index}/{run_count} | seed={seed} "
                f"| return={run_artifact.return_code} "
                f"| metrics={bool(run_artifact.metrics)}"
            )

        run_summary = self._run_summary(run_artifacts)
        interpretation = self._interpret_runs(
            question_text,
            plan,
            run_artifacts,
            run_summary,
        )

        interpretation_path = os.path.join(session_dir, "interpretation.json")
        atomic_write_json(interpretation_path, interpretation)

        session_summary_path = os.path.join(session_dir, "session_summary.json")
        atomic_write_json(
            session_summary_path,
            {
                "hypothesis_node_id": hypothesis_node_id,
                "hypothesis": question_text,
                "plan": plan.to_dict(),
                "security_report": security_report,
                "dependency_report": dependency_report,
                "run_summary": run_summary,
                "runs": [artifact.to_dict() for artifact in run_artifacts],
                "interpretation": interpretation,
                "script_path": script_path,
            },
        )

        verdict = interpretation.get("verdict", "irrelevant")
        explanation = interpretation.get("explanation", "")
        print(f"     Verdict: [{verdict}] {explanation}")

        empirical_node_id = self._attach_experiment_to_graph(
            hypothesis_node_id=hypothesis_node_id,
            hypothesis_text=question_text,
            verdict=verdict,
            interpretation=interpretation,
            run_summary=run_summary,
            session_dir=session_dir,
            code=code,
            session_summary_path=session_summary_path,
        )

        return {
            "status": "ok",
            "session_dir": session_dir,
            "plan_path": plan_path,
            "script_path": script_path,
            "security_report": security_report,
            "dependency_report": dependency_report,
            "interpretation_path": interpretation_path,
            "session_summary_path": session_summary_path,
            "run_count": run_count,
            "verdict": verdict,
            "explanation": explanation,
            "run_summary": run_summary,
            "empirical_node_id": empirical_node_id,
            "output_snippet": explanation[:500],
        }
