"""
Template Brain Builder ? Pre-build foundational knowledge brains.

Creates reusable brain templates seeded with large corpora of foundational
knowledge that new research missions can fork from.

Usage:
    python build_template.py --builtin general_scientist
    python build_template.py --builtin physics_researcher --max-per-domain 4
    python build_template.py --manifest brain_templates/manifests/general_scientist.json

Templates are stored in brain_templates/ and loaded via:
    python bootstrap.py --template general_scientist "Your mission question"
"""

import os
import sys
import json
import time
import copy
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from graph.brain import Brain
from observer.observer import Observer
from ingestion.ingestor import Ingestor
from notebook.notebook import Notebook
from reader.reader import Reader
from embedding_index import EmbeddingIndex

TEMPLATE_DIR = "brain_templates"
MANIFEST_DIR = "brain_templates/manifests"

# ?? Built-in manifests ???????????????????????????????????????????????????????

GENERAL_MANIFEST = {
    "name": "general_scientist",
    "description": "Foundational knowledge for an educated scientist across core disciplines",
    "domains": [
        {
            "domain": "Scientific Method & Epistemology",
            "sources": [
                "https://en.wikipedia.org/wiki/Scientific_method",
                "https://en.wikipedia.org/wiki/Epistemology",
                "https://en.wikipedia.org/wiki/Falsifiability",
                "https://en.wikipedia.org/wiki/Paradigm_shift",
                "https://en.wikipedia.org/wiki/Peer_review",
            ]
        },
        {
            "domain": "Mathematics Foundations",
            "sources": [
                "https://en.wikipedia.org/wiki/Calculus",
                "https://en.wikipedia.org/wiki/Statistics",
                "https://en.wikipedia.org/wiki/Probability",
                "https://en.wikipedia.org/wiki/Logic",
                "https://en.wikipedia.org/wiki/Set_theory",
            ]
        },
        {
            "domain": "Physics Foundations",
            "sources": [
                "https://en.wikipedia.org/wiki/Classical_mechanics",
                "https://en.wikipedia.org/wiki/Thermodynamics",
                "https://en.wikipedia.org/wiki/Electromagnetism",
                "https://en.wikipedia.org/wiki/Quantum_mechanics",
                "https://en.wikipedia.org/wiki/General_relativity",
            ]
        },
        {
            "domain": "Biology Foundations",
            "sources": [
                "https://en.wikipedia.org/wiki/Evolution",
                "https://en.wikipedia.org/wiki/Cell_(biology)",
                "https://en.wikipedia.org/wiki/DNA",
                "https://en.wikipedia.org/wiki/Ecology",
                "https://en.wikipedia.org/wiki/Genetics",
            ]
        },
        {
            "domain": "Systems Thinking",
            "sources": [
                "https://en.wikipedia.org/wiki/Systems_theory",
                "https://en.wikipedia.org/wiki/Complexity",
                "https://en.wikipedia.org/wiki/Emergence",
                "https://en.wikipedia.org/wiki/Feedback",
                "https://en.wikipedia.org/wiki/Network_science",
            ]
        },
        {
            "domain": "History of Science",
            "sources": [
                "https://en.wikipedia.org/wiki/History_of_science",
                "https://en.wikipedia.org/wiki/Scientific_revolution",
                "https://en.wikipedia.org/wiki/Industrial_Revolution",
                "https://en.wikipedia.org/wiki/Philosophy_of_science",
            ]
        }
    ]
}

PHYSICS_MANIFEST = {
    "name": "physics_researcher",
    "description": "Core theoretical physics, mathematical formalism, and statistical reasoning",
    "domains": [
        {
            "domain": "Classical Mechanics",
            "sources": [
                "https://en.wikipedia.org/wiki/Classical_mechanics",
                "https://en.wikipedia.org/wiki/Lagrangian_mechanics",
                "https://en.wikipedia.org/wiki/Hamiltonian_mechanics",
                "https://en.wikipedia.org/wiki/Harmonic_oscillator",
            ]
        },
        {
            "domain": "Electromagnetism",
            "sources": [
                "https://en.wikipedia.org/wiki/Electromagnetism",
                "https://en.wikipedia.org/wiki/Maxwell%27s_equations",
                "https://en.wikipedia.org/wiki/Electromagnetic_radiation",
                "https://en.wikipedia.org/wiki/Gauge_theory",
            ]
        },
        {
            "domain": "Thermodynamics & Statistical Mechanics",
            "sources": [
                "https://en.wikipedia.org/wiki/Thermodynamics",
                "https://en.wikipedia.org/wiki/Statistical_mechanics",
                "https://en.wikipedia.org/wiki/Entropy",
                "https://en.wikipedia.org/wiki/Free_energy",
                "https://en.wikipedia.org/wiki/Phase_transition",
            ]
        },
        {
            "domain": "Quantum Theory",
            "sources": [
                "https://en.wikipedia.org/wiki/Quantum_mechanics",
                "https://en.wikipedia.org/wiki/Wave_function",
                "https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation",
                "https://en.wikipedia.org/wiki/Quantum_field_theory",
                "https://en.wikipedia.org/wiki/Quantum_information",
            ]
        },
        {
            "domain": "Relativity & Spacetime",
            "sources": [
                "https://en.wikipedia.org/wiki/Special_relativity",
                "https://en.wikipedia.org/wiki/General_relativity",
                "https://en.wikipedia.org/wiki/Spacetime",
                "https://en.wikipedia.org/wiki/Black_hole",
            ]
        },
        {
            "domain": "Mathematical Methods",
            "sources": [
                "https://en.wikipedia.org/wiki/Differential_equation",
                "https://en.wikipedia.org/wiki/Linear_algebra",
                "https://en.wikipedia.org/wiki/Fourier_analysis",
                "https://en.wikipedia.org/wiki/Tensor",
            ]
        }
    ]
}

BIOLOGY_MANIFEST = {
    "name": "biology_researcher",
    "description": "A strong biology foundation spanning molecular to evolutionary scales",
    "domains": [
        {
            "domain": "Molecular Biology",
            "sources": [
                "https://en.wikipedia.org/wiki/Molecular_biology",
                "https://en.wikipedia.org/wiki/DNA",
                "https://en.wikipedia.org/wiki/RNA",
                "https://en.wikipedia.org/wiki/Protein",
                "https://en.wikipedia.org/wiki/Gene_expression",
            ]
        },
        {
            "domain": "Cell Biology",
            "sources": [
                "https://en.wikipedia.org/wiki/Cell_(biology)",
                "https://en.wikipedia.org/wiki/Cell_signaling",
                "https://en.wikipedia.org/wiki/Cell_cycle",
                "https://en.wikipedia.org/wiki/Mitochondrion",
            ]
        },
        {
            "domain": "Genetics",
            "sources": [
                "https://en.wikipedia.org/wiki/Genetics",
                "https://en.wikipedia.org/wiki/Inheritance",
                "https://en.wikipedia.org/wiki/Mutation",
                "https://en.wikipedia.org/wiki/Population_genetics",
            ]
        },
        {
            "domain": "Evolutionary Biology",
            "sources": [
                "https://en.wikipedia.org/wiki/Evolution",
                "https://en.wikipedia.org/wiki/Natural_selection",
                "https://en.wikipedia.org/wiki/Adaptation",
                "https://en.wikipedia.org/wiki/Speciation",
            ]
        },
        {
            "domain": "Ecology & Systems Biology",
            "sources": [
                "https://en.wikipedia.org/wiki/Ecology",
                "https://en.wikipedia.org/wiki/Food_web",
                "https://en.wikipedia.org/wiki/Systems_biology",
                "https://en.wikipedia.org/wiki/Homeostasis",
            ]
        },
        {
            "domain": "Neuroscience Interface",
            "sources": [
                "https://en.wikipedia.org/wiki/Neuroscience",
                "https://en.wikipedia.org/wiki/Neuron",
                "https://en.wikipedia.org/wiki/Synaptic_plasticity",
                "https://en.wikipedia.org/wiki/Neural_circuit",
            ]
        }
    ]
}

COGNITIVE_SCIENCE_MANIFEST = {
    "name": "cognitive_scientist",
    "description": "Mind, learning, memory, language, and neuroscience for cognition-centered missions",
    "domains": [
        {
            "domain": "Cognitive Science",
            "sources": [
                "https://en.wikipedia.org/wiki/Cognitive_science",
                "https://en.wikipedia.org/wiki/Mental_representation",
                "https://en.wikipedia.org/wiki/Embodied_cognition",
                "https://en.wikipedia.org/wiki/Predictive_coding",
            ]
        },
        {
            "domain": "Psychology of Learning",
            "sources": [
                "https://en.wikipedia.org/wiki/Learning",
                "https://en.wikipedia.org/wiki/Memory",
                "https://en.wikipedia.org/wiki/Attention",
                "https://en.wikipedia.org/wiki/Problem_solving",
            ]
        },
        {
            "domain": "Neuroscience",
            "sources": [
                "https://en.wikipedia.org/wiki/Neuroscience",
                "https://en.wikipedia.org/wiki/Neuroplasticity",
                "https://en.wikipedia.org/wiki/Working_memory",
                "https://en.wikipedia.org/wiki/Hippocampus",
                "https://en.wikipedia.org/wiki/Sleep",
            ]
        },
        {
            "domain": "Language & Thought",
            "sources": [
                "https://en.wikipedia.org/wiki/Psycholinguistics",
                "https://en.wikipedia.org/wiki/Language_processing_in_the_brain",
                "https://en.wikipedia.org/wiki/Concept",
                "https://en.wikipedia.org/wiki/Analogy",
            ]
        },
        {
            "domain": "Decision & Reasoning",
            "sources": [
                "https://en.wikipedia.org/wiki/Decision-making",
                "https://en.wikipedia.org/wiki/Heuristic",
                "https://en.wikipedia.org/wiki/Bayesian_inference",
                "https://en.wikipedia.org/wiki/Rationality",
            ]
        },
        {
            "domain": "Consciousness & Philosophy",
            "sources": [
                "https://en.wikipedia.org/wiki/Consciousness",
                "https://en.wikipedia.org/wiki/Philosophy_of_mind",
                "https://en.wikipedia.org/wiki/Qualia",
                "https://en.wikipedia.org/wiki/Global_workspace_theory",
            ]
        }
    ]
}

AI_MANIFEST = {
    "name": "ai_researcher",
    "description": "Machine learning, deep learning, optimization, and AI alignment foundations",
    "domains": [
        {
            "domain": "Machine Learning",
            "sources": [
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Supervised_learning",
                "https://en.wikipedia.org/wiki/Unsupervised_learning",
                "https://en.wikipedia.org/wiki/Reinforcement_learning",
            ]
        },
        {
            "domain": "Deep Learning",
            "sources": [
                "https://en.wikipedia.org/wiki/Deep_learning",
                "https://en.wikipedia.org/wiki/Artificial_neural_network",
                "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
                "https://en.wikipedia.org/wiki/Representation_learning",
            ]
        },
        {
            "domain": "Optimization & Statistics",
            "sources": [
                "https://en.wikipedia.org/wiki/Gradient_descent",
                "https://en.wikipedia.org/wiki/Stochastic_gradient_descent",
                "https://en.wikipedia.org/wiki/Convex_optimization",
                "https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff",
            ]
        },
        {
            "domain": "Information & Computation",
            "sources": [
                "https://en.wikipedia.org/wiki/Information_theory",
                "https://en.wikipedia.org/wiki/Algorithm",
                "https://en.wikipedia.org/wiki/Computational_complexity_theory",
                "https://en.wikipedia.org/wiki/Bayesian_network",
            ]
        },
        {
            "domain": "AI Safety & Alignment",
            "sources": [
                "https://en.wikipedia.org/wiki/AI_alignment",
                "https://en.wikipedia.org/wiki/Control_problem",
                "https://en.wikipedia.org/wiki/Reward_hacking",
                "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence",
            ]
        },
        {
            "domain": "Cognitive Inspiration",
            "sources": [
                "https://en.wikipedia.org/wiki/Connectionism",
                "https://en.wikipedia.org/wiki/Computational_neuroscience",
                "https://en.wikipedia.org/wiki/Neuromorphic_engineering",
                "https://en.wikipedia.org/wiki/Predictive_coding",
            ]
        }
    ]
}

COMPLEX_SYSTEMS_MANIFEST = {
    "name": "complex_systems_researcher",
    "description": "Networks, emergence, adaptation, and multi-scale systems behavior",
    "domains": [
        {
            "domain": "Complex Systems",
            "sources": [
                "https://en.wikipedia.org/wiki/Complex_system",
                "https://en.wikipedia.org/wiki/Emergence",
                "https://en.wikipedia.org/wiki/Self-organization",
                "https://en.wikipedia.org/wiki/Nonlinear_system",
            ]
        },
        {
            "domain": "Network Science",
            "sources": [
                "https://en.wikipedia.org/wiki/Network_science",
                "https://en.wikipedia.org/wiki/Small-world_network",
                "https://en.wikipedia.org/wiki/Scale-free_network",
                "https://en.wikipedia.org/wiki/Percolation_theory",
            ]
        },
        {
            "domain": "Dynamical Systems",
            "sources": [
                "https://en.wikipedia.org/wiki/Dynamical_system",
                "https://en.wikipedia.org/wiki/Chaos_theory",
                "https://en.wikipedia.org/wiki/Attractor",
                "https://en.wikipedia.org/wiki/Bifurcation_theory",
            ]
        },
        {
            "domain": "Collective Behavior",
            "sources": [
                "https://en.wikipedia.org/wiki/Collective_behavior",
                "https://en.wikipedia.org/wiki/Swarm_intelligence",
                "https://en.wikipedia.org/wiki/Agent-based_model",
                "https://en.wikipedia.org/wiki/Game_theory",
            ]
        },
        {
            "domain": "Biological & Ecological Systems",
            "sources": [
                "https://en.wikipedia.org/wiki/Systems_biology",
                "https://en.wikipedia.org/wiki/Ecosystem",
                "https://en.wikipedia.org/wiki/Resilience_(ecology)",
                "https://en.wikipedia.org/wiki/Food_web",
            ]
        },
        {
            "domain": "Information & Control",
            "sources": [
                "https://en.wikipedia.org/wiki/Cybernetics",
                "https://en.wikipedia.org/wiki/Feedback",
                "https://en.wikipedia.org/wiki/Control_theory",
                "https://en.wikipedia.org/wiki/Information_theory",
            ]
        }
    ]
}

BUILTIN_MANIFESTS = {
    GENERAL_MANIFEST["name"]: GENERAL_MANIFEST,
    PHYSICS_MANIFEST["name"]: PHYSICS_MANIFEST,
    BIOLOGY_MANIFEST["name"]: BIOLOGY_MANIFEST,
    COGNITIVE_SCIENCE_MANIFEST["name"]: COGNITIVE_SCIENCE_MANIFEST,
    AI_MANIFEST["name"]: AI_MANIFEST,
    COMPLEX_SYSTEMS_MANIFEST["name"]: COMPLEX_SYSTEMS_MANIFEST,
}


def ensure_dirs():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)


def list_builtin_templates() -> list[dict]:
    return [
        {
            "name": manifest["name"],
            "description": manifest.get("description", ""),
            "domains": len(manifest.get("domains", [])),
        }
        for manifest in BUILTIN_MANIFESTS.values()
    ]


def print_builtin_templates():
    print()
    print("Available built-in templates:")
    print()
    for item in list_builtin_templates():
        print(f"- {item['name']}: {item['description']} ({item['domains']} domains)")


def save_default_manifests(force: bool = False):
    """Save built-in manifests to disk."""
    ensure_dirs()
    saved = 0
    for name, manifest in BUILTIN_MANIFESTS.items():
        path = os.path.join(MANIFEST_DIR, f"{name}.json")
        if force or not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"Saved manifest: {path}")
            saved += 1
    if saved == 0:
        print("Built-in manifests already exist. Use --force-save-defaults to overwrite.")


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_builtin_manifest(name: str) -> dict:
    manifest = BUILTIN_MANIFESTS.get(name)
    if not manifest:
        raise KeyError(f"Unknown built-in template: {name}")
    return copy.deepcopy(manifest)


def build_template(manifest: dict, max_per_domain: int = 5):
    """
    Build a template brain from a manifest.

    Args:
        manifest: dict with 'name', 'description', 'domains' (each with 'sources')
        max_per_domain: Max articles to absorb per domain
    """
    name = manifest["name"]
    description = manifest.get("description", "")

    print(f"\n{'='*60}")
    print(f"Building template brain: {name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    brain     = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer  = Observer(brain)
    notebook  = Notebook(brain, observer=observer)
    ingestor  = Ingestor(brain, embedding_index=emb_index)
    reader    = Reader(brain, observer=observer, notebook=notebook,
                       ingestor=ingestor)

    total_absorbed = 0
    domains = manifest.get("domains", [])

    for di, domain_info in enumerate(domains):
        domain_name = domain_info["domain"]
        sources = domain_info.get("sources", [])

        print()
        print(f"[{di+1}/{len(domains)}] Domain: {domain_name}")
        print(f"  Sources: {len(sources)}")

        for si, url in enumerate(sources[:max_per_domain]):
            try:
                is_wikipedia = "wikipedia" in url
                title = url.split("/wiki/")[-1].replace("_", " ") if is_wikipedia else url
                source_type = domain_info.get("source_type") or ("wikipedia" if is_wikipedia else "web")
                result = reader.absorb_url(url, title=title, source_type=source_type)
                if result.success:
                    total_absorbed += 1
                    print(f"  OK [{si+1}] {title} - {result.node_count} nodes")
                else:
                    print(f"  FAIL [{si+1}] {title} - {result.error}")
            except Exception as e:
                print(f"  ERROR [{si+1}]: {e}")
            time.sleep(1)

    ensure_dirs()
    brain_path = os.path.join(TEMPLATE_DIR, f"{name}.brain.json")
    index_path = os.path.join(TEMPLATE_DIR, f"{name}.index")

    brain.save(brain_path)
    emb_index.save(index_path)

    print(f"\n{'='*60}")
    print(f"Template '{name}' built successfully")
    print(f"  Absorbed: {total_absorbed} articles")
    print(f"  Brain: {brain.stats()['nodes']} nodes, {brain.stats()['edges']} edges")
    print(f"  Saved: {brain_path}")
    print(f"{'='*60}")

    return brain.stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a DREAMER template brain")
    parser.add_argument("--name", default=None,
                        help="Template name (overrides manifest or builtin name)")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest JSON")
    parser.add_argument("--builtin", default="general_scientist",
                        help="Built-in template name to use when --manifest is not provided")
    parser.add_argument("--list-builtins", action="store_true",
                        help="List built-in templates and exit")
    parser.add_argument("--save-defaults", action="store_true",
                        help="Save all built-in manifests to brain_templates/manifests")
    parser.add_argument("--force-save-defaults", action="store_true",
                        help="Overwrite built-in manifests when saving defaults")
    parser.add_argument("--max-per-domain", type=int, default=5,
                        help="Max articles to absorb per domain")

    args = parser.parse_args()

    if args.list_builtins:
        print_builtin_templates()
        raise SystemExit(0)

    if args.save_defaults or args.force_save_defaults:
        save_default_manifests(force=args.force_save_defaults)
        if not args.manifest and args.builtin is None:
            raise SystemExit(0)

    if args.manifest:
        manifest = load_manifest(args.manifest)
    else:
        manifest = get_builtin_manifest(args.builtin)

    if args.name:
        manifest["name"] = args.name

    build_template(manifest, max_per_domain=args.max_per_domain)
