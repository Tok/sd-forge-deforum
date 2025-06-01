# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F

import copy
import inspect
import torch.nn as nn

Device = Union[str, torch.device]

# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)


# Provide get_origin and get_args even in Python 3.7.

if sys.version_info >= (3, 8, 0):
    from typing import get_args, get_origin
elif sys.version_info >= (3, 7, 0):

    def get_origin(cls):  # pragma: no cover
        return getattr(cls, "__origin__", None)

    def get_args(cls):  # pragma: no cover
        return getattr(cls, "__args__", None)


else:
    raise ImportError("This module requires Python 3.7+")


################################################################
##   ██████╗██╗      █████╗ ███████╗███████╗███████╗███████╗  ##
##  ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝  ##
##  ██║     ██║     ███████║███████╗███████╗█████╗  ███████╗  ##
##  ██║     ██║     ██╔══██║╚════██║╚════██║██╔══╝  ╚════██║  ##
##  ╚██████╗███████╗██║  ██║███████║███████║███████╗███████║  ##
##   ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝  ##
################################################################

class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3d objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))     # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))     # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3d objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3d objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t1.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points(x)
        y3 = t1.compose(t2, t3).transform_points(x)


    Composing transforms should broadcast.

    .. code-block:: python

        if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.

    We can also stack a sequence of Transform3d objects, which represents
    composition along the batch dimension; then the following should compute the
    same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat([tN.transform_points(xN), tM.transform_points(xM)], dim=0)
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3d objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3d().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3d().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = Transform3d().scale(s1_params).translate(t_params).scale(s2_params)
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are row vectors. The internal representation of the Nx4x4 transformation
    matrix is of the form:

    .. code-block:: python

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

    To apply the transformation to points which are row vectors, the M matrix
    can be pre multiplied by the points:

    .. code-block:: python

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        transformed_points = points * M

    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """

        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError('"matrix" has to be of shape (4, 4) or (N, 4, 4).')
            # Convert 2D to 3D by adding batch dimension
            if matrix.ndim == 2:
                matrix = matrix[None]
            self._matrix = matrix

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def __getitem__(
        self, index: Union[int, List[int], slice, torch.Tensor]
    ) -> "Transform3d":
        """
        Args:
            index: Specifying the indices of the transforms to retrieve.
                Can be an int, slice, list of ints, or a boolean tensor.

        Returns:
            Transform3d object with selected transforms. The tensors are not cloned.
        """
        if isinstance(index, int):
            index = [index]
        return self.__class__(matrix=self._matrix[index])

    def compose(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d representing the composition of self with the
        given other transforms, which will be stored as an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """
        out = Transform3d(dtype=self._matrix.dtype, device=self.device)
        out._matrix = self._matrix
        for other in others:
            out = out._compose_matrix(other)
        return out

    def get_matrix(self) -> torch.Tensor:
        """
        Return a copy of the transformation matrix.
        Returns:
            A tensor of shape (N, 4, 4)
        """
        return self._matrix.clone()

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse transformation matrix.
        Returns:
            A tensor of shape (N, 4, 4)
        """
        return torch.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        """
        Returns a new Transform3d object that represents an inverse of the
        current transformation.

        Args:
            invert_composed: 
                - True: First compose the list of stored transformations and then apply inverse to the result. 
                - False: Apply inverse to all stored transformations.

        Returns:
            A new Transform3d object containing the inverse.
        """

        tinv = Transform3d(dtype=self._matrix.dtype, device=self.device)
        tinv._matrix = self._get_matrix_inverse()
        return tinv

    def stack(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d which is a stack of self and others.
        The stack is done along the batch dimension.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d object with the stacked transforms.
        """
        transforms = [self] + list(others)
        matrices = [t._matrix for t in transforms]
        out_mat = torch.cat(matrices, dim=0)
        return Transform3d(matrix=out_mat)

    def transform_points(self, points, eps: Optional[float] = None) -> torch.Tensor:
        """
        Use this transform to transform a set of 3D points. Assumes row vector points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord == 0)) * torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 lead to NaNs after the division.

        Returns:
            points_out: Tensor of shape (N, P, 3) or (P, 3) based on the input dimensions
        """
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        T = self._matrix
        if T.shape[0] != N:
            msg = "Expected batch dim of points to be %r; got %r"
            raise ValueError(msg % (T.shape[0], N))

        # Transform points
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_hom = torch.cat([points_batch, ones], dim=2)
        points_out = torch.bmm(points_hom, T.transpose(1, 2))

        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points_batch.shape[1:])

        return points_out

    def transform_normals(self, normals) -> torch.Tensor:
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (N, P, 3) or (P, 3) based on the input
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        if normals.shape[-1] != 3:
            msg = "Expected normals to have shape (N, P, 3) or (P, 3): got %r"
            raise ValueError(msg % (normals.shape,))

        normals_batch = normals.clone()
        if normals_batch.dim() == 2:
            normals_batch = normals_batch[None]  # (P, 3) -> (1, P, 3)

        N, P, _3 = normals_batch.shape
        T = self._matrix
        if T.shape[0] != N:
            msg = "Expected batch dim of normals to be %r; got %r"
            raise ValueError(msg % (T.shape[0], N))

        # Transform normals using transpose of inverse transformation
        R = T[:, :3, :3]
        normals_out = torch.bmm(normals_batch, R)

        # When transform is (1, 4, 4) and normals is (P, 3) return (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals_batch.shape[1:])

        return normals_out

    def translate(self, *args, **kwargs) -> "Transform3d":
        """
        Compose this transform with a translation.
        """
        from .py3d_transformations import Translate
        return self.compose(Translate(*args, device=self.device, **kwargs))

    def scale(self, *args, **kwargs) -> "Transform3d":
        """
        Compose this transform with a scaling.
        """
        from .py3d_transformations import Scale
        return self.compose(Scale(*args, device=self.device, **kwargs))

    def rotate(self, *args, **kwargs) -> "Transform3d":
        """
        Compose this transform with a rotation.
        """
        from .py3d_transformations import Rotate
        return self.compose(Rotate(*args, device=self.device, **kwargs))

    def rotate_axis_angle(self, *args, **kwargs) -> "Transform3d":
        """
        Compose this transform with a rotation about an axis.
        """
        from .py3d_transformations import RotateAxisAngle
        return self.compose(RotateAxisAngle(*args, device=self.device, **kwargs))

    def clone(self) -> "Transform3d":
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d(dtype=self._matrix.dtype, device=self.device)
        other._matrix = self._matrix.clone()
        return other

    def to(
        self,
        device: Device,
        copy: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Transform3d":
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device (as str or torch.device) for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.
          dtype: If not None, casts the internal tensor to dtype.

        Returns:
          Transform3d object.
        """
        device_ = make_device(device)
        skip_to = self.device == device_ and not copy
        if skip_to and dtype is None:
            return self

        other = self.clone()

        if dtype is not None:
            other._matrix = other._matrix.to(device=device_, dtype=dtype)
        else:
            other._matrix = other._matrix.to(device=device_)
        return other

    @property
    def device(self) -> torch.device:
        return self._matrix.device

    def cpu(self) -> "Transform3d":
        return self.to("cpu")

    def cuda(self) -> "Transform3d":
        return self.to("cuda")

    def _compose_matrix(self, other: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d with a transformation matrix that is the
        composition of this Transform3d's transformation matrix and another.

        Args:
            other: A Transform3d object

        Returns:
            A new Transform3d with the stored transformation matrix
        """
        self_mat = self._matrix
        other_mat = other._matrix
        return Transform3d(matrix=_broadcast_bmm(self_mat, other_mat))


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specification.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """

    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If no index is specified, use the current device.
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    return device


def _broadcast_bmm(a, b) -> torch.Tensor:
    """
    Batch matrix multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K, M) or (K, M)
        b: torch tensor of shape (P, M, N) or (M, N)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(P, 1)
    """

    if a.dim() == 2:
        a = a[None]
    if b.dim() == 2:
        b = b[None]
    return torch.bmm(a, b) 