"""Feature interaction modules for BIQA."""

from vnct.models.interactions.dual_source import (
    DualSourceInteraction,
    SpatialReducedCrossAttention,
    UnifiedEvidenceInteraction,
)

__all__ = [
    "DualSourceInteraction",
    "SpatialReducedCrossAttention",
    "UnifiedEvidenceInteraction",
]
