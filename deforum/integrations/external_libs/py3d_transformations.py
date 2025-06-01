# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union, Any

import torch

from .py3d_core import Transform3d, Device, make_device, _R, _T
from .py3d_utilities import _handle_input, _axis_angle_rotation, euler_angles_to_matrix, _check_valid_rotation_matrix


class Translate(Transform3d):
    """
    A translation transform.
    """

    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        """
        Create a new Translation transform with the given translation values.

        Args:
            x: Translation offset in the x-axis or a tensor of shape (N, 3)
            y: Translation offset in the y-axis
            z: Translation offset in the z-axis
            dtype: The data type of the translation matrix.
            device: The device on which the translation matrix should be stored.
        """
        xyz = _handle_input(x, y, z, dtype, device, "Translate", allow_singleton=True)
        N = xyz.shape[0]
        mat = torch.eye(4, dtype=dtype, device=xyz.device).view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 3, :3] = xyz
        super().__init__(dtype=dtype, device=device, matrix=mat)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse transformation matrix.
        For a translation matrix, the inverse is the negation of the translation.
        """
        mat_inv = self._matrix.clone()
        mat_inv[:, 3, :3] = -mat_inv[:, 3, :3]
        return mat_inv


class Rotate(Transform3d):
    """
    A rotation transform defined by a rotation matrix.
    """

    def __init__(
        self,
        R: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        """
        Create a new Rotation transform with the given rotation matrix.

        Args:
            R: rotation matrix of shape (3, 3) or (N, 3, 3)
            dtype: The data type of the transformation matrix.
            device: The device on which the transformation matrix should be stored.
            orthogonal_tol: tolerance for checking if R is a valid rotation matrix.
        """
        device_ = get_device(R, device)
        if R.dim() == 2:
            R = R[None]  # (3, 3) -> (1, 3, 3)
        if R.shape[-2:] != (3, 3):
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(R.shape))
        R = R.to(device=device_, dtype=dtype)
        _check_valid_rotation_matrix(R, tol=orthogonal_tol)
        N = R.shape[0]

        # Create 4x4 transformation matrix
        mat = torch.eye(4, dtype=dtype, device=device_).view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        super().__init__(dtype=dtype, device=device, matrix=mat)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse transformation matrix.
        For a rotation matrix, the inverse is the transpose.
        """
        mat_inv = self._matrix.clone()
        mat_inv[:, :3, :3] = mat_inv[:, :3, :3].transpose(-1, -2)
        return mat_inv


class Scale(Transform3d):
    """
    A scaling transform.
    """

    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        """
        Create a new Scale transform with the given scaling values.

        Args:
            x: Scaling factor in the x-axis or a tensor of shape (N, 3)
            y: Scaling factor in the y-axis
            z: Scaling factor in the z-axis
            dtype: The data type of the scaling matrix.
            device: The device on which the scaling matrix should be stored.
        """
        xyz = _handle_input(x, y, z, dtype, device, "Scale", allow_singleton=True)
        N = xyz.shape[0]
        
        # Check for zero scale factors
        if (xyz == 0).any():
            warnings.warn("Scale factors of 0 can cause numerical instability")
        
        mat = torch.eye(4, dtype=dtype, device=xyz.device).view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 0, 0] = xyz[:, 0]  # x scaling
        mat[:, 1, 1] = xyz[:, 1]  # y scaling
        mat[:, 2, 2] = xyz[:, 2]  # z scaling
        super().__init__(dtype=dtype, device=device, matrix=mat)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse transformation matrix.
        For a scale matrix, the inverse is 1/scale for each axis.
        """
        mat_inv = self._matrix.clone()
        mat_inv[:, 0, 0] = 1.0 / mat_inv[:, 0, 0]  # 1/x scaling
        mat_inv[:, 1, 1] = 1.0 / mat_inv[:, 1, 1]  # 1/y scaling  
        mat_inv[:, 2, 2] = 1.0 / mat_inv[:, 2, 2]  # 1/z scaling
        return mat_inv


class RotateAxisAngle(Transform3d):
    """
    A rotation transform defined by an axis and angle.
    """

    def __init__(
        self,
        angle,
        axis: str = "X",
        degrees: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        """
        Create a new AxisAngle rotation transform.

        Args:
            angle: Rotation angle(s) - scalar, vector or tensor of shape (N,)
            axis: Rotation axis - one of "X", "Y", "Z"
            degrees: If True, angles are in degrees, otherwise radians
            dtype: The data type of the transformation matrix.
            device: The device on which the transformation matrix should be stored.
        """
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            msg = "Expected axis to be one of ['X', 'Y', 'Z']; got %s"
            raise ValueError(msg % axis)

        angle = torch.as_tensor(angle, dtype=dtype, device=device)
        if angle.dim() == 0:
            angle = angle.view(1)
        if angle.dim() != 1:
            msg = "Expected angle to be a scalar or 1D tensor; got %s"
            raise ValueError(msg % angle.shape)

        # Convert to rotation matrix using axis-angle rotation
        R = _axis_angle_rotation(axis, angle)
        super().__init__(dtype=dtype, device=device)
        
        # Create 4x4 transformation matrix
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=R.device).view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat


class RotateEuler(Transform3d):
    """
    A rotation transform defined by Euler angles.
    """

    def __init__(
        self,
        x=0.0,
        y=0.0, 
        z=0.0,
        convention: str = "XYZ",
        degrees: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        """
        Create a new Euler rotation transform.

        Args:
            x: Rotation angle around x-axis
            y: Rotation angle around y-axis  
            z: Rotation angle around z-axis
            convention: Euler angle convention (e.g. "XYZ", "ZYX")
            degrees: If True, angles are in degrees, otherwise radians
            dtype: The data type of the transformation matrix.
            device: The device on which the transformation matrix should be stored.
        """
        euler_angles = _handle_input(x, y, z, dtype, device, "RotateEuler")
        
        if degrees:
            euler_angles = euler_angles * math.pi / 180.0
        
        # Convert Euler angles to rotation matrix
        R = euler_angles_to_matrix(euler_angles, convention)
        super().__init__(dtype=dtype, device=device)
        
        # Create 4x4 transformation matrix
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=R.device).view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat


def get_world_to_view_transform(
    R: torch.Tensor = _R, T: torch.Tensor = _T
) -> Transform3d:
    """
    Calculate the world to view transformation matrix given by:
        
    .. code-block:: python
        
        T_world_view = Translate(T).compose(Rotate(R))

    Args:
        R: Rotation matrix of shape (N, 3, 3) or (1, 3, 3) for the rotation.
            If R is None, then the rotation component is omitted.
        T: Translation matrix of shape (N, 3) or (1, 3) for the translation.
            If T is None, then the translation component is omitted.

    Returns:
        a Transform3d object which represents the composed RT transformation.
    """
    # TODO: also support the case where RT is specified as one matrix
    # of shape (N, 4, 4).

    if T.shape[0] != R.shape[0]:
        msg = "Expected R, T to have the same batch dimension; got %r, %r"
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = "Expected T to have shape (N, 3); got %r"
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = "Expected R to have shape (N, 3, 3); got %r"
        raise ValueError(msg % repr(R.shape))

    # Create a Transform3d object
    T_ = Translate(T, device=T.device)
    R_ = Rotate(R, device=R.device)
    return R_.compose(T_)


def get_device(x, device: Optional[Device] = None) -> torch.device:
    """
    Helper function to get the device from a tensor or device specification.

    Args:
        x: torch tensor
        device: Device (as str or torch.device) or None

    Returns:
        A torch.device object
    """
    if device is not None:
        return make_device(device)

    if torch.is_tensor(x):
        return x.device

    return torch.device("cpu") 