import torch
import numpy as np

def compute_centroid_distance_penalty(gaussians, alpha=1.0, power=4.0):
    """
    计算基于点云质心的距离惩罚
    离质心越远的点，惩罚越大

    Args:
        gaussians: 高斯模型
        alpha: 惩罚系数（加大到1.0）
        power: 幂次，越大则远点惩罚越剧烈（默认4.0）
    """
    points = gaussians.get_xyz
    
    # 点数太少时不计算
    if points.shape[0] < 10:
        return torch.tensor(0.0, device=points.device), None
    
    # 计算质心
    centroid = points.mean(dim=0)
    
    # 计算每个点到质心的距离
    distances = torch.norm(points - centroid, dim=1)
    
    # 使用高次幂放大远距离的影响
    # 归一化距离到0-1范围
    max_dist = distances.max()
    if max_dist > 0:
        # 归一化距离
        normalized_dist = distances / max_dist
        
        # 使用高次幂：距离0.5的点惩罚只有0.5^4=0.0625
        # 距离0.8的点惩罚有0.8^4=0.4096
        # 距离1.0的点惩罚有1.0
        # 这样远点的惩罚会剧烈增大
        weights = normalized_dist ** power
        
        # 额外惩罚：对于距离超过平均值的点，再增加惩罚
        mean_dist = distances.mean()
        far_mask = distances > mean_dist
        if far_mask.any():
            weights[far_mask] = weights[far_mask] * 2.0
        
        # 计算加权平均惩罚
        penalty = (weights * distances).mean()
    else:
        penalty = torch.tensor(0.0, device=points.device)
    
    return penalty * alpha, centroid

def fit_bounding_ellipse(points_xy, expand_factor=1.2):
    """
    拟合二维点的边界椭圆

    Args:
        points_xy: N x 2 的点坐标
        expand_factor: 放大系数，让椭圆比实际点云更大

    Returns:
        center: 椭圆中心
        axes: 椭圆的两个轴方向 (2x2 矩阵)
        radii: 椭圆的两个轴半径
    """
    # 计算协方差矩阵
    cov = np.cov(points_xy.T)
    
    # SVD 分解得到主方向
    U, S, Vt = np.linalg.svd(cov)
    
    # 主方向
    axes = U.T
    
    # 计算每个方向上的投影范围
    projections = points_xy @ axes.T
    
    # 计算半径（取最大投影距离，并放大）
    radii = np.max(np.abs(projections), axis=0) * expand_factor
    
    # 中心
    center = np.mean(points_xy, axis=0)
    
    return center, axes, radii

def create_hollow_ellipsoid_shell(gaussians, num_points=5000, expand_factor=1.2, height_padding=0.2):
    """
    根据 Phase 0 重建的物体，创建纯空心椭圆筒（只有侧面）

    Args:
        gaussians: 当前的 GaussianModel 对象（包含 Phase 0 重建的物体）
        num_points: 要生成的椭圆筒高斯点数量
        expand_factor: 横截面椭圆放大系数
        height_padding: 高度方向额外padding（相对于物体高度）

    Returns:
        dict: 包含所有高斯属性的字典（颜色为白色，高不透明度）
    """
    from utils.sh_utils import RGB2SH
    from utils.general_utils import inverse_sigmoid
    
    print(f"\n{'='*60}")
    print(f"Creating hollow ellipsoid shell (side only)")
    print(f"{'='*60}")
    
    # 获取 Phase 0 重建的点的坐标
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    # 计算高度范围
    z_min = np.min(xyz[:, 2])
    z_max = np.max(xyz[:, 2])
    z_height = z_max - z_min
    
    # 高度方向扩展（上下各加 padding）
    z_min_ext = z_min - height_padding * z_height
    z_max_ext = z_max + height_padding * z_height
    
    print(f"Object height range: [{z_min:.3f}, {z_max:.3f}] (height: {z_height:.3f})")
    print(f"Extended height range: [{z_min_ext:.3f}, {z_max_ext:.3f}]")
    
    # 在 XY 平面拟合椭圆
    center_xy, axes_xy, radii_xy = fit_bounding_ellipse(xyz[:, :2], expand_factor)
    
    print(f"XY plane ellipse:")
    print(f"  Center: ({center_xy[0]:.3f}, {center_xy[1]:.3f})")
    print(f"  Axes: [{axes_xy[0]}, {axes_xy[1]}]")
    print(f"  Radii: [{radii_xy[0]:.3f}, {radii_xy[1]:.3f}] (expanded by {expand_factor}x)")
    
    # 获取已有高斯的平均尺度作为参考
    existing_scales = gaussians.get_scaling.detach().cpu().numpy()
    mean_scale = np.mean(existing_scales, axis=0)
    print(f"Existing Gaussians - mean scale: [{mean_scale[0]:.4f}, {mean_scale[1]:.4f}, {mean_scale[2]:.4f}]")
    
    # 生成空心椭圆筒的点（只有侧面）
    points = []
    scales_list = []
    
    # 旋转矩阵
    R = axes_xy.T
    
    # 在椭圆上均匀采样角度
    n_circum = int(np.sqrt(num_points * 2))  # 圆周方向点数
    n_vertical = num_points // n_circum  # 垂直方向点数
    
    # 生成圆周方向的角度
    angles = np.linspace(0, 2 * np.pi, n_circum, endpoint=False)
    
    # 生成垂直方向的高度
    z_positions = np.linspace(z_min_ext, z_max_ext, n_vertical)
    
    # 椭圆周长近似
    ellipse_circumference = np.pi * (3*(radii_xy[0] + radii_xy[1]) - np.sqrt((3*radii_xy[0] + radii_xy[1])*(radii_xy[0] + 3*radii_xy[1])))
    arc_length = ellipse_circumference / n_circum
    
    # 为每个点设置缩放 - 使用已有高斯尺度作为参考
    for i, angle in enumerate(angles):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 椭圆上的点
        point_ellipse_local = np.array([radii_xy[0] * cos_a, radii_xy[1] * sin_a])
        point_ellipse = point_ellipse_local @ R + center_xy
        
        for j, z in enumerate(z_positions):
            points.append([point_ellipse[0], point_ellipse[1], z])
            
            # 使用已有高斯的平均尺度，但调整方向：
            # - 径向方向：保持原有尺度
            # - 切线方向：略大于弧长，形成连续栅栏
            # - 垂直方向：略大于垂直间距，形成连续栅栏
            scale_radial = mean_scale[0] * 0.8  # 径向稍薄
            scale_tangent = max(arc_length * 1.2, mean_scale[1] * 1.2)  # 切线方向
            scale_vertical = max((z_max_ext - z_min_ext) / n_vertical * 1.2, mean_scale[2] * 1.2)  # 垂直方向
            
            scales_list.append([scale_radial, scale_tangent, scale_vertical])
    
    # 转换为torch张量
    fused_point_cloud = torch.tensor(np.array(points), dtype=torch.float32, device="cuda")
    scales_tensor = torch.tensor(np.array(scales_list), dtype=torch.float32, device="cuda")
    
    # 白色
    white_sh_value = 0.2820947917  # RGB2SH(1.0)
    features_dc = torch.full((len(points), 1, 3), white_sh_value, dtype=torch.float, device="cuda")
    features_rest = torch.zeros((len(points), (gaussians.max_sh_degree+1)**2 - 1, 3), device="cuda")
    
    # 随机旋转
    rots = torch.randn((len(points), 4), device="cuda")
    rots = rots / torch.norm(rots, dim=1, keepdim=True)
    
    # 高不透明度（确保可见）
    opacities = inverse_sigmoid(0.9 * torch.ones((len(points), 1), dtype=torch.float, device="cuda"))
    
    print(f"\n✅ Created {len(points)} hollow ellipsoid shell points:")
    print(f"  - Circumferential resolution: {n_circum} points")
    print(f"  - Vertical resolution: {n_vertical} points")
    print(f"  - Total points: {n_circum * n_vertical}")
    print(f"  - Color: white (opacity: 0.9)")
    print(f"  - Height range: [{z_min_ext:.3f}, {z_max_ext:.3f}]")
    print(f"  - Scale settings (based on existing Gaussians):")
    print(f"    - Radial: {scale_radial:.4f}")
    print(f"    - Tangent: {scale_tangent:.4f}")
    print(f"    - Vertical: {scale_vertical:.4f}")
    
    return {
        'xyz': fused_point_cloud,
        'features_dc': features_dc,
        'features_rest': features_rest,
        'scaling': torch.log(scales_tensor.clamp(min=1e-6)),
        'rotation': rots,
        'opacity': opacities
    }

def project_points_to_pixels_correct(means3D, camera):
    """
    - X轴：((ndc_x + 1) * W - 1) * 0.5
    - Y轴：((ndc_y + 1) * H - 1) * 0.5  
    """
    H = camera.image_height
    W = camera.image_width
    
    # 转换为齐次坐标
    ones = torch.ones((means3D.shape[0], 1), device=means3D.device)
    homogenous_points = torch.cat([means3D, ones], dim=1)
    
    # 应用 full_proj_transform (世界->裁剪空间)
    clip_space = homogenous_points @ camera.full_proj_transform
    
    # 透视除法 (裁剪空间->NDC)
    w = clip_space[:, 3:4] + 1e-8
    ndc = clip_space[:, :3] / w
    
    # 检查可见性：在NDC范围内且在相机前方
    in_view = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2] > 0) & (ndc[:, 2] < 1)
    
    # CUDA公式：((v + 1.0) * S - 1.0) * 0.5
    pixel_x = ((ndc[:, 0] + 1) * W - 1) * 0.5
    pixel_y = ((ndc[:, 1] + 1) * H - 1) * 0.5  # CUDA公式，无翻转
    
    # 转换为整数坐标
    pixel_x = pixel_x.long()
    pixel_y = pixel_y.long()
    
    # 确保坐标在图像范围内
    in_image = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
    visible = in_view & in_image
    
    return pixel_x, pixel_y, visible

def is_point_in_bbox(pixel_x, pixel_y, bbox):
    """检查像素点是否在bbox内"""
    x1, y1, x2, y2 = bbox
    return (pixel_x >= x1) & (pixel_x <= x2) & (pixel_y >= y1) & (pixel_y <= y2)

def compute_frustum_outside_loss(camera, gaussians, render_pkg, bbox, loss_weight=0.5):
    """
    计算当前视锥体外点的损失惩罚（增强版）

    Args:
        camera: 当前相机
        gaussians: 高斯模型
        render_pkg: 渲染结果包
        bbox: 2D边界框 [x1, y1, x2, y2]
        loss_weight: 惩罚权重（默认0.5，比之前更大）

    Returns:
        penalty_loss: 标量张量
    """
    if bbox is None:
        return torch.tensor(0.0, device="cuda")
    
    # 获取可见点的2D投影
    if "visible_points_px" not in render_pkg or "visible_indices" not in render_pkg:
        return torch.tensor(0.0, device="cuda")
    
    visible_indices = render_pkg["visible_indices"]
    visible_points_px = render_pkg["visible_points_px"]
    
    if len(visible_indices) == 0:
        return torch.tensor(0.0, device="cuda")
    
    x1, y1, x2, y2 = bbox
    
    # 获取图像尺寸用于归一化
    h, w = camera.image_height, camera.image_width
    
    # 获取所有可见点的2D坐标
    points_x = visible_points_px[:, 0]
    points_y = visible_points_px[:, 1]
    
    # 计算每个点到bbox边界的距离
    # 如果点在bbox内，距离为0；如果在bbox外，距离为正
    dist_left = torch.clamp(x1 - points_x, min=0)
    dist_right = torch.clamp(points_x - x2, min=0)
    dist_x = dist_left + dist_right
    
    dist_top = torch.clamp(y1 - points_y, min=0)
    dist_bottom = torch.clamp(points_y - y2, min=0)
    dist_y = dist_top + dist_bottom
    
    # 计算欧氏距离
    dist = torch.sqrt(dist_x**2 + dist_y**2)
    
    # 归一化距离（除以图像对角线长度，使距离在0-1范围）
    image_diag = torch.sqrt(torch.tensor(w**2 + h**2, device=dist.device))
    dist_normalized = dist / image_diag
    
    # 只有距离>0的点才贡献损失
    outside_mask = dist_normalized > 0
    if not outside_mask.any():
        return torch.tensor(0.0, device="cuda")
    
    # 增强惩罚：

    dist_cubic = dist_normalized[outside_mask] ** 3
    

    far_mask = dist_normalized[outside_mask] > 0.1
    dist_cubic[far_mask] = dist_cubic[far_mask] * 2.0
    

    penalty = dist_cubic.mean()
    

    if len(outside_mask) > 0:
        penalty = penalty + 0.01 * (outside_mask.float().mean())
    
    return penalty * loss_weight
