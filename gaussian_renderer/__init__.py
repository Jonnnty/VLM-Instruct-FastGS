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
from utils.graphics_utils import geom_transform_points

from utils.sh_utils import eval_sh

def render_fastgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
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

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}


def render_pointcloud_only(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    """
    只渲染高斯中心点云 - 完全复用现有的投影函数
    """
    # 获取图像尺寸
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    
    # 创建空白图像（背景色）
    point_image = bg_color.clone().view(3, 1, 1).expand(3, H, W).contiguous()
    
    # 获取所有高斯点的世界坐标
    means3D = pc.get_xyz  # [N, 3]
    
    if means3D.shape[0] == 0:
        print("Warning: No points to render")
        return {
            "render": point_image,
            "point_cloud_info": {
                "total_points": 0,
                "visible_points": 0
            }
        }
    
    
    points_ndc = geom_transform_points(means3D, viewpoint_camera.full_proj_transform)
    
    # 转换为像素坐标
    # points_ndc 是 [N, 3]，其中 x,y,z 在 [-1, 1] 范围内（NDC空间）
    pixel_x = ((points_ndc[:, 0] + 1) / 2 * W).long()
    pixel_y = ((1 - (points_ndc[:, 1] + 1) / 2) * H).long()  # 翻转Y轴
    
    # 检查可见性：在NDC范围内且在相机前方
    in_front = points_ndc[:, 2] > 0
    in_image = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
    visible = in_front & in_image
    
    visible_indices = torch.where(visible)[0]
    
    if len(visible_indices) > 0:
        visible_px_x = pixel_x[visible_indices]
        visible_px_y = pixel_y[visible_indices]
        
        # 获取可见点的颜色（使用SH系数）
        visible_xyz = means3D[visible_indices]
        dir_pp = visible_xyz - viewpoint_camera.camera_center
        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
        
        # 使用SH计算颜色
        from utils.sh_utils import eval_sh
        features = pc.get_features[visible_indices]
        if len(features.shape) == 3:
            shs_view = features.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
            colors = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors = torch.clamp_min(colors + 0.5, 0.0)
        else:
            colors = features[:, :3] + 0.5
            colors = torch.clamp_min(colors, 0.0)
        
        # 绘制点
        for i in range(len(visible_indices)):
            x, y = visible_px_x[i], visible_px_y[i]
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                point_image[:, y, x] = colors[i]
    
    return {
        "render": point_image,
        "point_cloud_info": {
            "total_points": means3D.shape[0],
            "visible_points": len(visible_indices),
            "visible_indices": visible_indices,
            "visible_px_coords": torch.stack([pixel_x[visible_indices], pixel_y[visible_indices]], dim=1) if len(visible_indices) > 0 else torch.zeros((0, 2), device="cuda")
        }
    }
