class ThresholdConfig:
    # Similarity thresholds (cosine, 0-1)
    MERGE_NODE          = 0.80
    DUPLICATE_MERGE     = 0.88
    WEAK_EDGE           = 0.60
    QUESTION_DEDUP_HIGH = 0.90
    QUESTION_DEDUP_LOW  = 0.70
    COHERENCE           = 0.65
    GAP_CONFIDENCE      = 0.75
    GAP_DEDUP           = 0.75
    CONTRADICTION       = 0.65
    AGENDA_PREFILTER    = 0.30


THRESHOLDS = ThresholdConfig()

