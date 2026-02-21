#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fastgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None, return_2d=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    
    Args:
        return_2d: 如果为True，返回每个高斯点的2D屏幕坐标和可见点信息
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts
    }
    
    # 如果需要返回2D坐标，添加屏幕坐标信息（优化版）
    if return_2d:
        # screenspace_points 包含每个点的齐次坐标 [x, y, z, w]
        # 计算归一化设备坐标 (NDC)
        points_2d = screenspace_points[:, :2] / (screenspace_points[:, 3:4] + 1e-8)
        result["points_2d_ndc"] = points_2d
        
        # 获取可见点的索引
        visible_indices = (radii > 0).nonzero(as_tuple=True)[0]
        result["visible_indices"] = visible_indices
        
        if len(visible_indices) > 0:
            # 直接计算可见点的像素坐标（极速版）
            visible_points_2d = points_2d[visible_indices]
            H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
            
            # 一次性计算所有可见点的像素坐标
            visible_px_x = ((visible_points_2d[:, 0] + 1) / 2 * W).long()
            visible_px_y = ((visible_points_2d[:, 1] + 1) / 2 * H).long()
            
            # 确保坐标在有效范围内
            visible_px_x = torch.clamp(visible_px_x, 0, W-1)
            visible_px_y = torch.clamp(visible_px_y, 0, H-1)
            
            result["visible_points_px"] = torch.stack([visible_px_x, visible_px_y], dim=1)
            result["visible_points_2d_ndc"] = visible_points_2d
            result["visible_count"] = len(visible_indices)
        else:
            result["visible_points_px"] = torch.zeros((0, 2), dtype=torch.long, device="cuda")
            result["visible_points_2d_ndc"] = torch.zeros((0, 2), device="cuda")
            result["visible_count"] = 0

    return result