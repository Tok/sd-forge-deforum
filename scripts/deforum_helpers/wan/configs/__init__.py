"""
WAN Configuration Module
Model configurations for different WAN variants
"""

from .wan_t2v_1_3B import t2v_1_3B as wan_t2v_1_3B_config
from .wan_t2v_14B import t2v_14B as wan_t2v_14B_config
from .wan_i2v_14B import i2v_14B as wan_i2v_14B_config
from .shared_config import wan_shared_cfg as WanBaseConfig

__all__ = [
    "wan_t2v_1_3B_config",
    "wan_t2v_14B_config", 
    "wan_i2v_14B_config",
    "WanBaseConfig"
]

def get_config_for_model(model_name: str):
    """Get configuration for a specific model"""
    config_map = {
        "wan_t2v_1_3b": wan_t2v_1_3B_config,
        "wan_t2v_14b": wan_t2v_14B_config,
        "wan_i2v_14b": wan_i2v_14B_config,
    }
    
    model_key = model_name.lower().replace("-", "_").replace(".", "_")
    return config_map.get(model_key, wan_t2v_1_3B_config)  # Default to 1.3B 