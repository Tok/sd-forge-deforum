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

from .py3d_core import Transform3d, Device, make_device, _R, _T
from .py3d_utilities import TensorAccessor, TensorProperties


class CamerasBase(TensorProperties):
    """
    `CamerasBase` implements a base class for all cameras.
    For cameras, there are four different coordinate systems (or spaces)
    - World coordinate system: This is the system the object lives - the world.
    - Camera view coordinate system: This is the system that has its origin on the camera
        and the and the Z-axis perpendicular to the image plane.
        In PyTorch3D, we assume that +X points left, and +Y points up and
        +Z points out from the image plane.
        The transformation from world --> view happens after applying a rotation (R)
        and translation (T)
    - NDC coordinate system: This is the normalized coordinate system that confines
        in a volume the rendered part of the object or scene. Also known as view volume.
        For square images, given the PyTorch3D convention, (+1, +1, znear)
        is the top left near corner, and (-1, -1, zfar) is the bottom right far
        corner of the volume.
        The transformation from view --> NDC happens after applying the camera
        projection matrix (P) if defined in NDC space.
        For non square images, we scale the points such that smallest side
        has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    - Screen coordinate system: This is another representation of the view volume with
        the XY coordinates defined in image space instead of a normalized space.
    A better illustration of the coordinate systems can be found in
    pytorch3d/docs/notes/cameras.md.
    It defines methods that are common to all camera models:
        - `get_camera_center` that returns the optical center of the camera in
            world coordinates
        - `get_world_to_view_transform` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `get_full_projection_transform` which composes the projection
            transform (P) with the world-to-view transform (R, T)
        - `transform_points` which takes a set of input points in world coordinates and
            projects to the space the camera is defined in (NDC or screen)
        - `get_ndc_camera_transform` which defines the transform from screen/NDC to
            PyTorch3D's NDC space
        - `transform_points_ndc` which takes a set of points in world coordinates and
            projects them to PyTorch3D's NDC space
        - `transform_points_screen` which takes a set of points in world coordinates and
            projects them to screen space
    For each new camera, one should implement the `get_projection_transform`
    routine that returns the mapping from camera view coordinates to camera
    coordinates (NDC or screen).
    Another useful function that is specific to each camera model is
    `unproject_points` which sends points from camera coordinates (NDC or screen)
    back to camera view or world coordinates depending on the `world_coordinates`
    boolean argument of the function.
    """

    # Used in __getitem__ to index the relevant fields
    # When creating a new camera, this should be set in the __init__
    _FIELDS: Tuple[str, ...] = ()

    # Names of fields which are a constant property of the whole batch, rather
    # than themselves a batch of data.
    # When joining objects into a batch, they will have to agree.
    _SHARED_FIELDS: Tuple[str, ...] = ()

    def get_projection_transform(self):
        """
        Calculate the projection matrix to convert from camera view coordinates
        to camera coordinates (NDC or screen space).
        This method should be implemented for each camera model.
        """
        raise NotImplementedError()

    def unproject_points(self, xy_depth: torch.Tensor, **kwargs):
        """
        Unproject 2D camera coordinates (with depth) to 3D points in world coordinates.
        This method should be implemented for each camera model.

        Args:
            xy_depth: 2D camera coordinates with depth values

        Returns:
            Tensor of 3D points in world coordinates
        """
        # world_coordinates = kwargs.get("world_coordinates", True)
        # if world_coordinates:
        #     to_camera_transform = self.get_full_projection_transform()
        # else:
        #     to_camera_transform = self.get_projection_transform()

        # return to_camera_transform.inverse().transform_points(xy_depth)
        raise NotImplementedError()

    def get_camera_center(self, **kwargs) -> torch.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value gets propagated through the transforms.

        Returns:
            C: a tensor of shape (N, 3) or (1, 3) representing the 3D locations
            of the center of each camera in the batch in world coordinates.
        """

        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # The camera center is given by C = -R_inv @ T = -R.T @ T.
        # where R is the rotation matrix and T is the translation.
        # C is the camera center in world coordinates.
        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                to override the default values set in __init__.
                Valid parameters depend on the camera type.

        Setting R and T here will update the values set in init as this
        value gets propagated through the transforms.

        Returns:
            A Transform3d object with a batch dimension of N where N is the
            number of cameras in the batch.
        """
        R = kwargs.get("R", self.R)  # (N, 3, 3)
        T = kwargs.get("T", self.T)  # (N, 3)

        from .py3d_transformations import get_world_to_view_transform
        world_to_view_transform = get_world_to_view_transform(R=R, T=T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-screen transform composing the
        world-to-view and view-to-screen transforms.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.

        Setting R and T here will update the values set in init as this
        value gets propagated through the transforms.

        Returns:
            a Transform3d object with a batch dimension of N where N is the
            number of cameras in the batch.
        """
        self.get_projection_transform(**kwargs)
        world_to_view_transform = self.get_world_to_view_transform(**kwargs)
        view_to_proj_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_proj_transform)

    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to camera coordinates using the
        camera-to-world transform.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the divisor
                in the homogeneous division operation.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_proj_transform = self.get_full_projection_transform(**kwargs)
        return world_to_proj_transform.transform_points(points, eps=eps)

    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Return the transform from camera projection coordinates (screen or NDC)
        to the PyTorch3D NDC coordinate frame.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in as
                keyword arguments to override the default values
                set in __init__.

        For cameras that are already in NDC coordinates, this transform
        is the identity transform.

        Returns:
            a Transform3d object which represents the transformation
            from the camera projection coordinates to PyTorch3D NDC coordinates.
        """

        # By default, assume NDC space
        ndc_transform = Transform3d(device=self.device, dtype=torch.float32)
        return ndc_transform

    def transform_points_ndc(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to PyTorch3D NDC coordinates.
        This function converts from world coordinates to camera coordinates,
        then to camera coordinates in NDC space.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the divisor
                in the homogeneous division operation.

        Returns
            new_points: transformed points with the same shape as the input.
        """

        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        ndc_to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
        world_to_ndc_transform = world_to_ndc_transform.compose(
            ndc_to_ndc_transform
        )
        return world_to_ndc_transform.transform_points(points, eps=eps)

    def transform_points_screen(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to screen coordinates.
        This function converts from world coordinates to camera coordinates,
        then to screen coordinates.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the divisor
                in the homogeneous division operation.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        points_ndc = self.transform_points_ndc(points, eps=eps, **kwargs)
        image_size = kwargs.get("image_size", self.get_image_size())
        return get_ndc_to_screen_transform(self, image_size).transform_points(
            points_ndc, eps=eps
        )

    def clone(self):
        """
        Return a copy of the camera object.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)

    def is_perspective(self):
        raise NotImplementedError()

    def in_ndc(self):
        """
        Specifies whether the camera is defined in NDC space
        or in screen space
        """
        raise NotImplementedError()

    def get_znear(self):
        return self.znear

    def get_image_size(self):
        """
        return the image size, if provided, expected format: (height, width)
        """
        return getattr(self, "image_size", None)

    def __getitem__(
        self, index: Union[int, List[int], torch.LongTensor]
    ) -> "CamerasBase":
        """
        Override for the __getitem__ method in TensorProperties which needs to be
        refactored.

        Args:
            index: an int or tensor used to index all the fields in the cameras given in
            self._FIELDS.

        Returns:
            an instance of the current cameras class with only the values at the selected
            index.
        """
        kwargs = {}

        if not isinstance(index, (list, tuple)):
            index = [index]

        # Check if all fields are tensors and have batch dimension > the max index
        if any(
            not torch.is_tensor(getattr(self, field))
            for field in self._FIELDS
            if hasattr(self, field)
        ):
            raise ValueError("Cannot index a camera with non-tensor fields")

        if any(
            getattr(self, field).shape[0] <= max(index)
            for field in self._FIELDS
            if hasattr(self, field)
        ):
            raise ValueError("Cannot index a camera with batch size smaller than index")

        # Index into each field
        for field in self._FIELDS:
            if hasattr(self, field):
                val = getattr(self, field)
                # e.g. self.image_size = (5, 4) -> no need to index it
                if torch.is_tensor(val):
                    kwargs[field] = val[index]
                else:
                    kwargs[field] = val

        # Copy over the shared fields
        for field in self._SHARED_FIELDS:
            if hasattr(self, field):
                kwargs[field] = getattr(self, field)

        kwargs["device"] = self.device
        cam_type = type(self)
        return cam_type(**kwargs)


class FoVPerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices by specifying the field of view.
    The definition of the parameters follow the OpenGL perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> ndc.

    The `transform_points` method calculates the full world -> ndc transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.

    * Setting the Aspect Ratio for Non Square Images *

    If the desired output image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration: There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The `aspect_ratio` setting in the FoVPerspectiveCameras sets the
    pixel aspect ratio. When using this camera with the differentiable rasterizer
    be aware that in the rasterizer we assume square pixels, but allow
    variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera `aspect_ratio=1.0`
    (i.e. square pixels) and only vary the output image dimensions in pixels
    for rasterization.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "znear",
        "zfar",
        "aspect_ratio",
        "fov",
        "R",
        "T",
        "degrees",
    )

    _SHARED_FIELDS = ("degrees",)

    def __init__(
        self,
        znear=1.0,
        zfar=100.0,
        aspect_ratio=1.0,
        fov=60.0,
        degrees: bool = True,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
    ) -> None:
        """

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            fov: field of view angle of the camera.
            degrees: bool, set to True if fov is specified in degrees.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need znear, zfar, fov, aspect_ratio, degrees
            device: Device (as str or torch.device)
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            K=K,
            degrees=degrees,
        )

        if self.K is not None and self.K.shape != (self._N, 4, 4):
            msg = "Expected K to have shape of (%r, 4, 4)"
            raise ValueError(msg % (self._N,))

    def compute_projection_matrix(
        self, znear, zfar, fov, aspect_ratio, degrees: bool
    ) -> torch.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = torch.zeros((self._N, 4, 4), dtype=torch.float32, device=self.device)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)
        if degrees:
            fov = (math.pi / 180) * fov

        if not torch.is_tensor(fov):
            fov = torch.tensor(fov, device=self.device)
        tanHalfFov = torch.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the camera space
        # +z axis points out from the screen while the NDC space +z axis points
        # into the screen. Therefore we negate the Z coordinate.
        z_sign = -1.0

        K[:, 0, 0] = (2.0 * znear) / (max_x - min_x)
        K[:, 1, 1] = (2.0 * znear) / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate to the range [0, 1] instead of [-1, 1].
        # This differs from the OpenGL perspective projection matrix.
        # For more details, see the doc string.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix to convert from camera view coordinates
        to NDC coordinates.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.

        Returns:
            a Transform3d object which represents a batch of projection
            transforms of shape (N, 4, 4)

        """
        K = kwargs.get("K", self.K)
        if K is not None:
            transform = Transform3d(matrix=K.transpose(-1, -2).contiguous(), device=self.device)
        else:
            znear = kwargs.get("znear", self.znear)
            zfar = kwargs.get("zfar", self.zfar)
            fov = kwargs.get("fov", self.fov)
            aspect_ratio = kwargs.get("aspect_ratio", self.aspect_ratio)
            degrees = kwargs.get("degrees", self.degrees)

            K = self.compute_projection_matrix(
                znear, zfar, fov, aspect_ratio, degrees
            )

            # Transpose the projection matrix as expected by the Transform3d class.
            transform = Transform3d(
                matrix=K.transpose(-1, -2).contiguous(), device=self.device
            )
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """>!
        FoV cameras further allow for passing depth in world units
        (`scaled_depth_input=False`) or in the [0, 1]-normalized units
        (`scaled_depth_input=True`)

        Args:
            scaled_depth_input: If `True`, assumes the input depth is in
                the [0, 1]-normalized units. If `False` the input depth is in
                the world units.
        """

        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)

        if scaled_depth_input:
            # the input is scaled depth, so we don't need to do anything
            xy_sdepth = xy_depth
        else:
            # parse out the depth which is world coordinate depth
            xy_wdepth = xy_depth
            d_world = xy_wdepth[..., 2:3]

            # get the znear and zfar parameters of the camera
            znear = kwargs.get("znear", self.znear)  # pyre-ignore
            zfar = kwargs.get("zfar", self.zfar)  # pyre-ignore

            # Transform the world depth to the scaled depth
            d_scaled = (d_world - znear) / (zfar - znear)

            # concatenate xy with scaled depth
            xy_sdepth = torch.cat((xy_wdepth[..., 0:2], d_scaled), dim=-1)

        unprojected_points = to_camera_transform.inverse().transform_points(xy_sdepth)
        return unprojected_points

    def is_perspective(self):
        return True

    def in_ndc(self):
        return True


def get_ndc_to_screen_transform(cameras, image_size) -> Transform3d:
    """
    Transform from NDC coordinates to screen coordinates.
    NDC: +X is right, +Y is up, +Z is away from the camera.
    Screen: +X is right, +Y is down, +Z is away from the camera.
    """
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer
    H, W = image_size
    scale = min(H, W) / 2.0
    
    # NDC to screen transform
    # x_screen = (x_ndc + 1) * W / 2
    # y_screen = (-y_ndc + 1) * H / 2
    # z_screen = z_ndc
    
    transform_matrix = torch.tensor([
        [W/2.0, 0.0, 0.0, W/2.0],
        [0.0, -H/2.0, 0.0, H/2.0], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=cameras.device, dtype=torch.float32)
    
    # Expand to batch size
    N = len(cameras)
    transform_matrix = transform_matrix.unsqueeze(0).expand(N, -1, -1)
    
    return Transform3d(matrix=transform_matrix.transpose(-1, -2).contiguous()) 