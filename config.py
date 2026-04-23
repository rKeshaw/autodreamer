class ThresholdConfig:
    # Similarity thresholds (cosine, 0-1)
    MERGE_NODE          = 0.72
    DUPLICATE_MERGE     = 0.88
    WEAK_EDGE           = 0.58
    QUESTION_DEDUP_HIGH = 0.90
    QUESTION_DEDUP_LOW  = 0.70
    COHERENCE           = 0.65
    GAP_CONFIDENCE      = 0.75
    GAP_DEDUP           = 0.75
    CONTRADICTION       = 0.60
    AGENDA_PREFILTER    = 0.30
    MISSION_LINK        = 0.60
    SYNTHESIS_COHESION  = 0.60


THRESHOLDS = ThresholdConfig()


class ModelConfig:
    """
    Per-role model selection.

    Override any of these to use different models for different tasks:
        MODELS.CREATIVE = "llama3.1:70b"   # bigger model for dreaming
        MODELS.PRECISE  = "qwen2.5:7b"     # faster model for JSON extraction
        MODELS.CRITIC   = "llama3.1:70b"   # more rigorous model for System 2
    """
    # Creative tasks: dreaming, synthesis, analogies (System 1)
    CREATIVE     = "gemma4:latest"
    # Precise tasks: JSON extraction, factual answers
    PRECISE      = "gemma4:latest"
    # Code generation: sandbox experiments
    CODE         = "gemma4:latest"
    # Deliberate reasoning: thinker, chain-of-thought (System 1)
    REASONING    = "gemma4:latest"
    # Conversation: chat interface
    CONVERSATION = "gemma4:latest"
    # Critic / System 2: adversarial evaluation, gating
    CRITIC       = "gemma4:latest"
    # Structured notebook summaries and progress memos
    NOTEBOOK     = "gemma4:latest"
    # Conservative publication drafting
    PUBLICATION  = "gemma4:latest"
    # Strict verification / admissibility checks
    VERIFIER     = "gemma4:latest"


MODELS = ModelConfig()


class CriticConfig:
    """
    System 2 (Critic) — Dual-process gating configuration.

    The Critic applies adversarial scrutiny to high-stakes cognitive outputs
    before they enter the knowledge graph. Low-stakes outputs bypass it
    (Kahneman's "laziness principle").
    """
    # ── Activation gate ──
    # Minimum importance score for a claim to trigger System 2 review
    ACTIVATION_THRESHOLD   = 0.65

    # ── Dialogue parameters ──
    # Maximum adversarial challenge-defense rounds
    MAX_DIALOGUE_TURNS     = 3
    # Minimum confidence from System 2 to ACCEPT a claim
    ACCEPT_CONFIDENCE_FLOOR = 0.50

    # ── Type-based routing ──
    # Node/edge types that ALWAYS get reviewed (regardless of importance)
    ALWAYS_REVIEW_TYPES    = [
        "hypothesis", "synthesis",
        "structural_analogy", "deep_isomorphism",
    ]
    # Node/edge types that NEVER get reviewed (always bypass)
    BYPASS_TYPES           = ["concept", "associated", "surface_analogy"]

    # ── Refinement ──
    # Maximum times a REFINE loop can iterate before forcing DEFER
    MAX_REFINE_ITERATIONS  = 2


CRITIC = CriticConfig()
