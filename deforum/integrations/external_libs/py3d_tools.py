# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deforum 3D Tools - Compatibility Module

This module provides a streamlined interface to 3D transformations, camera operations,
and rendering utilities. Previously a monolithic 1,802-line file, this functionality
is now organized into focused modules while maintaining full backward compatibility.

Architecture:
    - py3d_core: Core Transform3d class and fundamental operations
    - py3d_transformations: Specific transformation classes (Translate, Rotate, Scale, etc.)
    - py3d_rendering: Camera classes and rendering operations  
    - py3d_utilities: Helper classes, tensor operations, and mathematical functions

All original functionality remains accessible through this compatibility layer.
"""

# Core 3D transformation functionality
from .py3d_core import (
    Transform3d,
    Device,
    make_device,
    _broadcast_bmm,
    _R,
    _T,
)

# Transformation classes
from .py3d_transformations import (
    Translate,
    Rotate,
    Scale, 
    RotateAxisAngle,
    RotateEuler,
    get_world_to_view_transform,
    get_device,
)

# Camera and rendering functionality
from .py3d_rendering import (
    CamerasBase,
    FoVPerspectiveCameras,
    get_ndc_to_screen_transform,
)

# Utility classes and functions
from .py3d_utilities import (
    TensorAccessor,
    TensorProperties,
    _axis_angle_rotation,
    euler_angles_to_matrix,
    _safe_det_3x3,
    _check_valid_rotation_matrix,
    format_tensor,
    convert_to_tensors_and_broadcast,
    _handle_coord,
    _handle_input,
)

# Convenience imports for backward compatibility
__all__ = [
    # Core classes
    "Transform3d",
    "Device",
    
    # Transformation classes
    "Translate",
    "Rotate", 
    "Scale",
    "RotateAxisAngle",
    "RotateEuler",
    
    # Camera classes
    "CamerasBase",
    "FoVPerspectiveCameras",
    
    # Utility classes
    "TensorAccessor",
    "TensorProperties",
    
    # Core functions
    "make_device",
    "get_world_to_view_transform",
    "get_ndc_to_screen_transform",
    
    # Mathematical functions
    "euler_angles_to_matrix",
    "_axis_angle_rotation",
    "_safe_det_3x3",
    "_check_valid_rotation_matrix",
    
    # Tensor utilities
    "format_tensor",
    "convert_to_tensors_and_broadcast",
    "_handle_input",
    "_handle_coord",
    "_broadcast_bmm",
    "get_device",
    
    # Constants
    "_R",
    "_T",
]
