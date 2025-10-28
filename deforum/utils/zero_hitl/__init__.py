"""Zero-HITL Slopcore Generator

Qwen's playground for generating gloriously broken AI videos with zero human intervention.

ISOLATION: All chaos stays in the Zero-HITL tab. Normal Deforum remains unchanged.
"""

from .parameter_randomizer import randomize_parameters, SlopcoreParameters
from .orchestrator import orchestrate_slop, OrchestrationResult

__all__ = ['randomize_parameters', 'SlopcoreParameters', 'orchestrate_slop', 'OrchestrationResult']
