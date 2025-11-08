import torch
import torch.nn.functional as F


def warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
    """Warp image using optical flow (RIFE v4.26 signature).

    Args:
        tenInput: Input tensor to warp (image or feature map)
        tenFlow: Optical flow tensor
        tenFlow_div: Flow division factor for normalization
        backwarp_tenGrid: Pre-computed warp grid

    Returns:
        Warped tensor with preserved dtype
    """
    dtype = tenInput.dtype
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)

    tenFlow = torch.cat([tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1)
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True).to(dtype)
