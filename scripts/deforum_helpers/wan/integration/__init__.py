"""
WAN Integration Module
Integration layer between WAN pipelines and Deforum
"""

from .unified_integration import WanUnifiedIntegration

# Try to import other integrations, fallback if dependencies missing
try:
    from .simple_integration import WanSimpleIntegration
    SIMPLE_INTEGRATION_AVAILABLE = True
except ImportError:
    WanSimpleIntegration = None
    SIMPLE_INTEGRATION_AVAILABLE = False

try:
    from .direct_integration import WanDirectIntegration
    DIRECT_INTEGRATION_AVAILABLE = True  
except ImportError:
    WanDirectIntegration = None
    DIRECT_INTEGRATION_AVAILABLE = False

__all__ = [
    "WanUnifiedIntegration",
    "WanSimpleIntegration", 
    "WanDirectIntegration",
    "SIMPLE_INTEGRATION_AVAILABLE",
    "DIRECT_INTEGRATION_AVAILABLE"
] 