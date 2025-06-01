# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn

from .py3d_core import Device, make_device


class TensorAccessor(nn.Module):
    """
    A helper class to be used with the __getitem__ method. This can be used for
    getting/setting the values for an attribute of a class at one particular
    index.  This is only to be used when the attribute is a tensor which is not
    differentiable.
    """

    def __init__(self, class_object, index: Union[int, slice]) -> None:
        self.class_object = class_object
        self.index = index

    def __setattr__(self, name: str, value: Any):
        if not hasattr(self, "class_object"):
            return super().__setattr__(name, value)

        if hasattr(self.class_object, name):
            v = getattr(self.class_object, name)
            if torch.is_tensor(v):
                # Convert the attribute to a tensor if it is not a tensor.
                if not torch.is_tensor(value):
                    value = torch.tensor(value)

                # Check the shapes are correct for broadcasting.
                if v.dim() != value.dim():
                    msg = "Expected value to have %r dimensions but got %r."
                    raise ValueError(msg % (v.dim(), value.dim()))
                if not (v.ndim == 1 or value.shape[1:] == v.shape[1:]):
                    msg = "Expected value to have shape %r but got %r."
                    raise ValueError(msg % (v.shape, value.shape))

                v[self.index] = value
                return
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if hasattr(self.class_object, name):
            return getattr(self.class_object, name)[self.index]

        msg = "'%s' object has no attribute '%s'"
        raise AttributeError(msg % (self.class_object.__class__.__name__, name))


class TensorProperties(nn.Module):
    """
    A mix-in class for storing tensors as properties with helper methods.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        **kwargs,
    ) -> None:
        """
        Args:
            dtype: data type to use for storing the tensors.
            device: Device (as str or torch.device) on which the tensors should be stored.
            kwargs: any number of keyword arguments. Any arguments which are
                of type (float, int, list, tuple) are broadcasted and
                other keyword arguments are set as attributes.
        """
        super().__init__()
        self.device = make_device(device)

        if kwargs:
            names = []
            values = []
            for k, v in kwargs.items():
                if v is None or torch.is_tensor(v):
                    setattr(self, k, v)
                elif isinstance(v, (float, int, list, tuple)):
                    names.append(k)
                    values.append(v)
                else:
                    setattr(self, k, v)
            self._convert_to_tensors_and_broadcast(names, values, dtype, device)

    def _convert_to_tensors_and_broadcast(self, names, values, dtype, device):
        """
        Helper function to handle conversion and broadcasting of tensor properties.
        """
        # Convert values to tensors if necessary and broadcast to the same batch size
        from .py3d_core import _broadcast_bmm  # Import here to avoid circular imports
        
        converted_values = []
        for v in values:
            if not torch.is_tensor(v):
                v = torch.tensor(v, dtype=dtype, device=device)
            if v.dim() == 0:
                v = v.view(1)
            converted_values.append(v)

        # Find max batch size
        batch_sizes = [v.shape[0] for v in converted_values]
        N = max(batch_sizes) if batch_sizes else 1

        # Broadcast all values to the same batch size
        for name, v in zip(names, converted_values):
            if v.shape[0] == 1 and N > 1:
                # Broadcast to batch size N
                shape = (N,) + v.shape[1:]
                v = v.expand(shape)
            elif v.shape[0] != N and v.shape[0] != 1:
                msg = "Got non-broadcastable sizes %r" % batch_sizes
                raise ValueError(msg)
            setattr(self, name, v)

        self._N = N

    def __len__(self) -> int:
        return self._N

    def isempty(self) -> bool:
        return self._N == 0

    def __getitem__(self, index: Union[int, slice]) -> TensorAccessor:
        """

        Args:
            index: an int or slice used to index all the fields in the class.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original class.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(class_object=self, index=index)

        msg = "Expected index of type int or slice; got %r"
        raise ValueError(msg % type(index))

    def to(self, device: Device = "cpu") -> "TensorProperties":
        """
        In place operation to move class properties which are tensors to a
        specified device. If self has a property named device, update this as well.
        """
        device = make_device(device)
        for k in dir(self):
            v = getattr(self, k)
            if k.startswith("_") or not torch.is_tensor(v):
                continue
            setattr(self, k, v.to(device))

        self.device = device
        return self

    def cpu(self) -> "TensorProperties":
        return self.to("cpu")

    def cuda(self, device: Optional[int] = None) -> "TensorProperties":
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    def clone(self, other) -> "TensorProperties":
        """
        Update the tensor properties of other with the cloned properties of self.
        """
        for k in dir(self):
            v = getattr(self, k)
            if k.startswith("_") or not torch.is_tensor(v):
                continue
            setattr(other, k, v.clone())
        return other

    def gather_props(self, batch_idx) -> "TensorProperties":
        """
        This is an in place operation to reformat all tensor class attributes
        based on a set of given indices using torch.gather and the first
        dimension of the tensor. This is useful when attributes which are batched
        e.g. shape (N, 3) need to be multiplied with another set of attributes
        e.g. shape (N, 3). (TODO consider if this can be used during
        rasterization)

        Args:
            batch_idx: shape (B, K) where B is the batch size and K is
                the number of points per batch element.

        Returns:
            self with all properties reshaped. e.g. a property with shape (N, 3)
            is transformed to shape (B, K, 3).
        """
        # Iterate through the attributes of the class which are tensors.
        for k in dir(self):
            v = getattr(self, k)
            if k.startswith("_") or not torch.is_tensor(v):
                continue

            if v.dim() < 2:
                msg = "All tensor attributes must have shape (N, K) where K >= 1"
                raise ValueError(msg)

            # There are different use cases for the gather_props function.
            # When batch_idx has shape (B, K) and v has shape (N, ...) this
            # is to reformat v to have the first dim to be of batch size B
            # instead of N. The tensor v is indexed using batch_idx.
            # This use case is when the class is init with tensors of shape (N, ...)
            # and later we want to select only the values specified by an indices
            # tensor and reformat to (B, K, ...) where B is the leading dim of
            # the indices tensor.
            if batch_idx.dim() == 2:
                batch_idx_long = batch_idx.view(-1).long()  # (B*K, )
                v_idx = v.index_select(0, batch_idx_long)  # (B*K, ...)
                setattr(
                    self, k, v_idx.view(batch_idx.shape + v.shape[1:])
                )  # (B, K, ...)

            # When batch_idx has shape (K,) and v has shape (B, ...) this is when
            # a class has been init with tensors of shape (B, ...) and we want to
            # select only the values at the indices specified by batch_idx along
            # the first dimension.
            elif batch_idx.dim() == 1:
                v_idx = v.index_select(0, batch_idx.long())
                setattr(self, k, v_idx)

        return self


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError("Invalid convention %s." % convention)
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError("Invalid letter %s in convention string." % letter)

    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _safe_det_3x3(t: torch.Tensor):
    """
    Fast determinant calculation for 3x3 matrices.
    
    Args:
        t: Tensor of shape (..., 3, 3)
        
    Returns:
        Determinant tensor of shape (...)
    """
    det = (
        t[..., 0, 0] * (t[..., 1, 1] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1] * (t[..., 1, 0] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 0])
        + t[..., 0, 2] * (t[..., 1, 0] * t[..., 2, 1] - t[..., 1, 1] * t[..., 2, 0])
    )
    return det


def _check_valid_rotation_matrix(R, tol: float = 1e-7) -> None:
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = _safe_det_3x3(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    if not (orthogonal and no_distortion):
        msg = "R is not a valid rotation matrix"
        warnings.warn(msg)
    return


def format_tensor(
    input,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.
    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should be placed.
    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    device_ = make_device(device)
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device_)
    elif not input.device.type.startswith('mps'):
        input = torch.tensor(input, dtype=torch.float32, device=device_)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device_:
        return input

    input = input.to(device=device)
    return input


def convert_to_tensors_and_broadcast(
    *args,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.
    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.
    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd


def _handle_coord(c, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Helper function for _handle_input.

    Args:
        c: Python scalar, torch scalar, or 1D torch tensor

    Returns:
        c_vec: 1D torch tensor
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=dtype, device=device)
    if c.dim() == 0:
        c = c.view(1)
    if c.device != device or c.dtype != dtype:
        c = c.to(device=device, dtype=dtype)
    return c


def _handle_input(
    x,
    y,
    z,
    dtype: torch.dtype,
    device: Optional[Device],
    name: str,
    allow_singleton: bool = False,
) -> torch.Tensor:
    """
    Helper function to handle parsing logic for building transforms. The output
    is always a tensor of shape (N, 3), but there are several types of allowed
    input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here just
        return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)
    """
    device_ = make_device(device) if device is not None else torch.device("cpu")
    
    # If x is actually a tensor of shape (N, 3) then just return it
    if torch.is_tensor(x) and x.dim() == 2:
        if x.shape[1] != 3:
            msg = "Expected tensor of shape (N, 3); got %r (in %s)"
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = "Expected y and z to be None (in %s)" % name
            raise ValueError(msg)
        return x.to(device=device_, dtype=dtype)

    if allow_singleton and y is None and z is None:
        y = x
        z = x

    # Convert all to 1D tensors
    xyz = [_handle_coord(c, dtype, device_) for c in [x, y, z]]

    # Broadcast and concatenate
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r (in %s)" % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = torch.stack(xyz, dim=1)
    return xyz 