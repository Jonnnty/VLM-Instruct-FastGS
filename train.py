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
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import gc
import traceback

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

# ===== 语义指导相关导入（可选） =====
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from scene.semantic_guidance import find_smooth_regions, find_subject_regions
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: Semantic guidance modules not available. Run without semantic guidance.")


def print_cuda_memory():
    """打印CUDA显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"CUDA Memory - Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")


def clear_cuda_cache():
    """清理CUDA缓存"""
    torch.cuda.empty_cache()
    gc.collect()


def find_points_in_2d_region(camera, gaussians, mask_2d, pipe, bg, opt):
    """在2D区域内查找对应的3D高斯点"""
    with torch.no_grad():
        render_pkg = render_fastgs(camera, gaussians, pipe, bg, opt.mult, return_2d=True)

        if "visible_points_px" not in render_pkg or "radii" not in render_pkg:
            return torch.zeros(len(gaussians._xyz), dtype=bool, device="cuda")

        visible_indices = render_pkg["visible_indices"]
        visible_points_px = render_pkg["visible_points_px"]
        all_radii = render_pkg["radii"]
        visible_radii = all_radii[visible_indices]
        visible_count = render_pkg.get("visible_count", 0)

        if visible_count == 0:
            return torch.zeros(len(gaussians._xyz), dtype=bool, device="cuda")

        if not isinstance(mask_2d, torch.Tensor):
            mask_2d = torch.tensor(mask_2d, device="cuda")

        mask_coords = torch.where(mask_2d)
        if len(mask_coords[0]) == 0:
            return torch.zeros(len(gaussians._xyz), dtype=bool, device="cuda")

        mask_min_y, mask_max_y = mask_coords[0].min(), mask_coords[0].max()
        mask_min_x, mask_max_x = mask_coords[1].min(), mask_coords[1].max()

        points_x = visible_points_px[:, 0]
        points_y = visible_points_px[:, 1]
        radii_expanded = visible_radii + 2

        # 快速粗筛
        min_x = points_x - radii_expanded
        max_x = points_x + radii_expanded
        min_y = points_y - radii_expanded
        max_y = points_y + radii_expanded

        overlap_x = (max_x >= mask_min_x) & (min_x <= mask_max_x)
        overlap_y = (max_y >= mask_min_y) & (min_y <= mask_max_y)
        potential_mask = overlap_x & overlap_y

        if not potential_mask.any():
            return torch.zeros(len(gaussians._xyz), dtype=bool, device="cuda")

        # 精细检查
        in_region = torch.zeros(len(gaussians._xyz), dtype=bool, device="cuda")
        potential_indices = visible_indices[potential_mask]
        potential_points_x = points_x[potential_mask]
        potential_points_y = points_y[potential_mask]
        potential_radii = radii_expanded[potential_mask]

        for idx, x, y, r in zip(potential_indices, potential_points_x, potential_points_y, potential_radii):
            x, y = int(x.item()), int(y.item())
            r = int(r.item())

            y_min = max(0, y - r)
            y_max = min(mask_2d.shape[0], y + r + 1)
            x_min = max(0, x - r)
            x_max = min(mask_2d.shape[1], x + r + 1)

            if mask_2d[y_min:y_max, x_min:x_max].any():
                in_region[idx] = True

        return in_region


def get_uniform_camera_indices(total_cameras, num_samples, phase_idx, total_phases):
    """
    获取均匀分布的相机索引

    Args:
        total_cameras: 总相机数量
        num_samples: 每阶段采样的相机数量
        phase_idx: 当前阶段索引 (0-based)
        total_phases: 总阶段数

    Returns:
        list of indices: 选中的相机索引列表
    """
    if total_cameras <= num_samples:
        return list(range(total_cameras))
    
    step = total_cameras / (total_phases * num_samples)
    
    indices = []
    for i in range(num_samples):
        base_offset = phase_idx * num_samples + i
        idx = int(base_offset * step)
        idx = min(idx, total_cameras - 1)
        indices.append(idx)
    
    return indices


def training(dataset, opt, pipe, init_ply_path, testing_iterations, saving_iterations, checkpoint_iterations, 
             checkpoint, debug_from, websockets,
             # 语义指导参数（可选）
             qwen_model_path=None,
             semantic_densify_factor=2.0,
             semantic_num_samples=5,
             save_region_vis=False,
             phase1_start=500, phase1_end=8000, phase1_interval=500,
             phase2_start=8000, phase2_end=14000, phase2_interval=2000,
             prune_protection=2000, total_iterations=20000,
             phase1_max_points_per_view=2000,
             phase2_max_points_per_view=8000):
    """
    训练函数（支持可选的语义指导）
    """
    try:
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        
        # 创建场景
        scene = Scene(dataset, gaussians, init_ply_path=init_ply_path)
        
        # 将训练相机传递给高斯模型，用于随机初始化
        gaussians.train_cameras = scene.getTrainCameras()
        
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        # 创建日志文件 - 初始写入，使用实际的iteration和gaussian数量
        log_file_path = os.path.join(dataset.model_path, "training_log.txt")
        with open(log_file_path, 'w') as f:
            f.write("iteration,loss,gaussian_count\n")
            # 使用实际的first_iter和当前高斯点数量
            current_gaussian_count = gaussians._xyz.shape[0]
            f.write(f"{first_iter},{0.0},{current_gaussian_count}\n")
        print(f"Training log will be saved to: {log_file_path}")

        # ===== 初始化语义模型（如果提供路径） =====
        qwen_model = None
        qwen_processor = None
        use_semantic = False
        
        if qwen_model_path and SEMANTIC_AVAILABLE:
            print("=" * 50)
            print("Loading Qwen3-VL model for semantic guidance...")
            print(f"Model path: {qwen_model_path}")

            if not os.path.exists(qwen_model_path):
                raise FileNotFoundError(f"Qwen3-VL model not found at {qwen_model_path}")

            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)
            use_semantic = True
            print("Qwen3-VL model loaded successfully!")
            print("=" * 50)
        elif qwen_model_path and not SEMANTIC_AVAILABLE:
            print("Warning: Semantic guidance requested but modules not available. Running without semantic guidance.")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))

        optim_start = torch.cuda.Event(enable_timing=True)
        optim_end = torch.cuda.Event(enable_timing=True)
        total_time = 0.0

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ===== 语义相关变量 =====
        disable_gs_pruning = False
        phase2_iterations = list(range(phase2_start, phase2_end + 1, phase2_interval)) if use_semantic else []
        total_phases = len(phase2_iterations)
        total_cameras = len(viewpoint_stack)
        
        if use_semantic:
            print(f"\n===== Semantic Guidance Enabled =====")
            print(f"===== Phase 1 Settings =====")
            print(f"  - Start: {phase1_start}")
            print(f"  - End: {phase1_end}")
            print(f"  - Interval: {phase1_interval}")
            print(f"  - Max points per view: {phase1_max_points_per_view}")
            print(f"===== Phase 2 Settings =====")
            print(f"  - Start: {phase2_start}")
            print(f"  - End: {phase2_end}")
            print(f"  - Interval: {phase2_interval}")
            print(f"  - Iterations: {phase2_iterations}")
            print(f"  - Max points per view: {phase2_max_points_per_view}")
            print(f"===== Pruning Settings =====")
            print(f"  - Protection after Phase 2: {prune_protection} iterations")
            print("=" * 50)
        else:
            print("\n===== Running without Semantic Guidance =====")

        print("\n🚀 开始训练...")
        print_cuda_memory()

        for iteration in range(first_iter, opt.iterations + 1):
            try:
                if websockets:
                    if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                        cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                        net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                        network_gui_ws.latest_width = cam.image_width
                        network_gui_ws.latest_height = cam.image_height
                        network_gui_ws.latest_result = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                iter_start.record()
                
                gaussians.update_learning_rate(iteration)

                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_indices = list(range(len(viewpoint_stack)))
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_idx)
                _ = viewpoint_indices.pop(rand_idx)

                if (iteration - 1) == debug_from:
                    pipe.debug = True

                # 检查输入数据
                if gaussians._xyz.shape[0] == 0:
                    raise ValueError("Gaussian points are empty!")

                render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
                
                # 检查loss是否为有效值
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Warning: Invalid loss value at iteration {iteration}: {loss.item()}")
                    continue

                # 处理无可见点的情况
                if visibility_filter.sum() == 0:
                    print(f"\n⚠️ 迭代 {iteration}: 没有可见点，跳过反向传播")
                    
                    # 清空梯度
                    if hasattr(gaussians, '_xyz') and gaussians._xyz is not None and gaussians._xyz.grad is not None:
                        gaussians._xyz.grad = None
                    if hasattr(gaussians, '_features_dc') and gaussians._features_dc is not None and gaussians._features_dc.grad is not None:
                        gaussians._features_dc.grad = None
                    if hasattr(gaussians, '_features_rest') and gaussians._features_rest is not None and gaussians._features_rest.grad is not None:
                        gaussians._features_rest.grad = None
                    if hasattr(gaussians, '_scaling') and gaussians._scaling is not None and gaussians._scaling.grad is not None:
                        gaussians._scaling.grad = None
                    if hasattr(gaussians, '_rotation') and gaussians._rotation is not None and gaussians._rotation.grad is not None:
                        gaussians._rotation.grad = None
                    if hasattr(gaussians, '_opacity') and gaussians._opacity is not None and gaussians._opacity.grad is not None:
                        gaussians._opacity.grad = None
                    
                    iter_end.record()
                    
                    with torch.no_grad():
                        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                        if iteration % 10 == 0:
                            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                            progress_bar.update(10)
                        
                        iter_time = iter_start.elapsed_time(iter_end)
                        
                        if iteration % 100 == 0:
                            current_loss = loss.item()
                            current_gaussian_count = gaussians._xyz.shape[0]
                            with open(log_file_path, 'a') as f:
                                f.write(f"{iteration},{current_loss:.6f},{current_gaussian_count}\n")
                            print(f"\n[LOG] Iter {iteration}: Loss={current_loss:.6f}, Gaussians={current_gaussian_count} (无可见点)")
                        
                        optim_start.record()
                        
                        if iteration < opt.densify_until_iter:
                            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                                my_viewpoint_stack = scene.getTrainCameras().copy()
                                camlist = sampling_cameras(my_viewpoint_stack)

                                importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                                gaussians.densify_and_prune_fastgs(max_screen_size=size_threshold, 
                                                            min_opacity=0.005, 
                                                            extent=scene.cameras_extent, 
                                                            radii=radii,
                                                            args=opt,
                                                            importance_score=importance_score,
                                                            pruning_score=pruning_score)
                                
                                clear_cuda_cache()

                            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                                gaussians.reset_opacity()

                        optim_end.record()
                        torch.cuda.synchronize()
                        optim_time = optim_start.elapsed_time(optim_end)
                        total_time += (iter_time + optim_time) / 1e3
                        
                        continue
                
                # 正常执行反向传播
                loss.backward()

                iter_end.record()

                with torch.no_grad():
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(10)
                    if iteration == opt.iterations:
                        progress_bar.close()

                    iter_time = iter_start.elapsed_time(iter_end)
                    
                    if iteration % 100 == 0:
                        current_loss = loss.item()
                        current_gaussian_count = gaussians._xyz.shape[0]
                        with open(log_file_path, 'a') as f:
                            f.write(f"{iteration},{current_loss:.6f},{current_gaussian_count}\n")
                        print(f"\n[LOG] Iter {iteration}: Loss={current_loss:.6f}, Gaussians={current_gaussian_count}")
                        
                        if iteration % 1000 == 0:
                            print_cuda_memory()
                    
                    if iteration in saving_iterations:
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                    
                    optim_start.record()

                    # ===== 语义指导（如果启用） =====
                    if use_semantic and iteration < total_iterations:

                        # Phase 1: 模糊区域检测
                        if iteration >= phase1_start and iteration < phase1_end and (iteration - phase1_start) % phase1_interval == 0:
                            all_cams = scene.getTrainCameras()
                            num_cams = min(semantic_num_samples, len(all_cams))
                            if num_cams >= 3:
                                selected_cams = random.sample(all_cams, num_cams)

                                rendered_images = []
                                image_names = []
                                for cam in selected_cams:
                                    render_pkg_semantic = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                    rendered = render_pkg_semantic["render"]
                                    rendered_images.append(rendered)
                                    image_names.append(cam.image_name)

                                print(f"\n[ITER {iteration}] Phase 1 - Running smooth region analysis on {num_cams} images...")

                                vis_dir = None
                                if save_region_vis:
                                    output_dir = os.path.dirname(dataset.model_path.rstrip('/'))
                                    vis_dir = os.path.join(output_dir, "region_vis")
                                    os.makedirs(vis_dir, exist_ok=True)

                                smooth_masks = find_smooth_regions(
                                    rendered_images,
                                    qwen_model,
                                    qwen_processor,
                                    max_region_ratio=0.125,
                                    save_visualization=save_region_vis,
                                    save_dir=vis_dir,
                                    image_names=[f"phase1_iter{iteration}_{name}" for name in image_names]
                                )

                                # 取反：方框外区域作为要处理的模糊区域
                                smooth_masks = [~mask for mask in smooth_masks]

                                all_affected_points = []
                                valid_masks_count = 0
                                sampled_points_list = []

                                for cam_idx, (cam, mask) in enumerate(zip(selected_cams, smooth_masks)):
                                    if mask.any():
                                        valid_masks_count += 1
                                        affected_points = find_points_in_2d_region(
                                            cam, gaussians, mask, pipe, bg, opt
                                        )
                                        if affected_points.any():
                                            points_in_view = affected_points.sum().item()
                                            print(f"      - View {cam_idx}: {points_in_view} points in region")
                                            
                                            if points_in_view > phase1_max_points_per_view:
                                                sample_ratio = phase1_max_points_per_view / points_in_view
                                                print(f"        - Points exceed {phase1_max_points_per_view}, sampling with ratio: {sample_ratio:.4f}")
                                                
                                                random_mask = torch.rand(len(affected_points), device="cuda") < sample_ratio
                                                sampled_mask = affected_points & random_mask
                                                
                                                final_count = sampled_mask.sum().item()
                                                print(f"        - Sampled to {final_count} points")
                                                sampled_points_list.append(sampled_mask)
                                            else:
                                                print(f"        - Points within limit ({points_in_view} <= {phase1_max_points_per_view})")
                                                sampled_points_list.append(affected_points)

                                print(f"  - Found {valid_masks_count}/{num_cams} images with regions")

                                if sampled_points_list:
                                    combined_mask = torch.stack(sampled_points_list).any(dim=0)
                                    total_combined = combined_mask.sum().item()
                                    print(f"  - Total combined points after per-view sampling: {total_combined}")

                                    # Phase 1 增密：只分裂，不剪枝
                                    net_change = gaussians.densify_semantic_regions(
                                        combined_mask,
                                        opt,
                                        scene.cameras_extent,
                                        densify_factor=semantic_densify_factor
                                    )

                        # Phase 2: 主体区域检测（均匀采样）
                        if iteration in phase2_iterations:
                            all_cams = scene.getTrainCameras()
                            total_cams = len(all_cams)
                            
                            phase_idx = phase2_iterations.index(iteration)
                            
                            camera_indices = get_uniform_camera_indices(
                                total_cams, 
                                semantic_num_samples, 
                                phase_idx, 
                                total_phases
                            )
                            
                            selected_cams = [all_cams[idx] for idx in camera_indices]
                            
                            print(f"\n[ITER {iteration}] Phase 2 - Detecting subject regions (uniform sampling)")
                            print(f"  - Selected camera indices: {camera_indices}")

                            rendered_images = []
                            image_names = []
                            for cam in selected_cams:
                                render_pkg_semantic = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                rendered = render_pkg_semantic["render"]
                                rendered_images.append(rendered)
                                image_names.append(cam.image_name)

                            vis_dir = None
                            if save_region_vis:
                                output_dir = os.path.dirname(dataset.model_path.rstrip('/'))
                                vis_dir = os.path.join(output_dir, "region_vis")
                                os.makedirs(vis_dir, exist_ok=True)

                            subject_masks = find_subject_regions(
                                rendered_images,
                                qwen_model,
                                qwen_processor,
                                save_visualization=save_region_vis,
                                save_dir=vis_dir,
                                image_names=[f"phase2_iter{iteration}_{name}" for name in image_names]
                            )

                            # 取反得到背景区域
                            background_masks = [~mask for mask in subject_masks]

                            all_background_points = []
                            valid_masks_count = 0
                            sampled_points_list = []

                            for cam_idx, (cam, mask) in enumerate(zip(selected_cams, background_masks)):
                                if mask.any():
                                    valid_masks_count += 1
                                    affected_points = find_points_in_2d_region(
                                        cam, gaussians, mask, pipe, bg, opt
                                    )
                                    if affected_points.any():
                                        points_in_view = affected_points.sum().item()
                                        print(f"      - View {cam_idx} (cam {camera_indices[cam_idx]}): {points_in_view} background points")
                                        
                                        if points_in_view > phase2_max_points_per_view:
                                            sample_ratio = phase2_max_points_per_view / points_in_view
                                            print(f"        - Background points exceed {phase2_max_points_per_view}, sampling with ratio: {sample_ratio:.4f}")
                                            
                                            random_mask = torch.rand(len(affected_points), device="cuda") < sample_ratio
                                            sampled_mask = affected_points & random_mask
                                            
                                            final_count = sampled_mask.sum().item()
                                            print(f"        - Sampled to {final_count} points")
                                            sampled_points_list.append(sampled_mask)
                                        else:
                                            print(f"        - Background points within limit ({points_in_view} <= {phase2_max_points_per_view})")
                                            sampled_points_list.append(affected_points)

                            print(f"  - Found {valid_masks_count}/{len(selected_cams)} images with background regions")

                            if sampled_points_list:
                                combined_background_mask = torch.stack(sampled_points_list).any(dim=0)
                                
                                total_combined = combined_background_mask.sum().item()
                                print(f"  - Total combined background points after per-view sampling: {total_combined}")

                                # Phase 2 期间禁用GS修剪
                                disable_gs_pruning = True
                                print(f"  - ⚠️ Disabling GS pruning during Phase 2 to protect new background points")

                                net_change = gaussians.refine_difference_regions(
                                    combined_background_mask,
                                    opt,
                                    scene.cameras_extent,
                                    densify_factor=semantic_densify_factor,
                                    prune_after=False,
                                    target_points=None
                                )

                    # ===== 原有的densification（使用当前激活的相机集） =====
                    if iteration < opt.densify_until_iter:
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            # 如果处于第二阶段禁用修剪期，跳过densify_and_prune
                            if not disable_gs_pruning:
                                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                                my_viewpoint_stack = scene.getTrainCameras().copy()
                                camlist = sampling_cameras(my_viewpoint_stack)

                                importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                                gaussians.densify_and_prune_fastgs(max_screen_size=size_threshold, 
                                                            min_opacity=0.005, 
                                                            extent=scene.cameras_extent, 
                                                            radii=radii,
                                                            args=opt,
                                                            importance_score=importance_score,
                                                            pruning_score=pruning_score)
                            else:
                                print(f"  - GS pruning temporarily disabled (Phase 2 protection)")

                            clear_cuda_cache()

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

                    # ===== 剪枝阶段 =====
                    if iteration % 3000 == 0 and iteration > (phase2_end + prune_protection) and iteration < total_iterations:
                        if iteration > phase2_end:
                            disable_gs_pruning = False
                            print(f"  - ✅ Re-enabling GS pruning after Phase 2")
                        
                        my_viewpoint_stack = scene.getTrainCameras().copy()
                        camlist = sampling_cameras(my_viewpoint_stack)

                        _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)
                        
                        clear_cuda_cache()

                    # ===== 优化步骤 =====
                    if iteration < opt.iterations:
                        if opt.optimizer_type == "default":
                            gaussians.optimizer_step(iteration)
                        elif opt.optimizer_type == "sparse_adam":
                            visible = radii > 0
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none=True)

                    optim_end.record()
                    torch.cuda.synchronize()
                    optim_time = optim_start.elapsed_time(optim_end)
                    total_time += (iter_time + optim_time) / 1e3

            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"\n❌ CUDA error at iteration {iteration}: {e}")
                    print("Attempting to recover...")
                    
                    print("\n📊 错误发生时状态:")
                    print(f"  - 迭代次数: {iteration}")
                    print(f"  - 高斯点数量: {gaussians._xyz.shape[0]}")
                    print(f"  - 可见点数量: {visibility_filter.sum().item() if 'visibility_filter' in locals() else 'N/A'}")
                    print(f"  - 当前loss: {loss.item() if 'loss' in locals() else 'N/A'}")
                    print_cuda_memory()
                    
                    clear_cuda_cache()
                    
                    if iteration > 1000:
                        print(f"Saving checkpoint at iteration {iteration} before exiting...")
                        scene.save(iteration)
                    raise e
                else:
                    raise e

        # 最终日志保存
        final_loss = loss.item() if 'loss' in locals() else 0.0
        final_gaussian_count = gaussians._xyz.shape[0]
        with open(log_file_path, 'a') as f:
            f.write(f"{iteration},{final_loss:.6f},{final_gaussian_count}\n")
        print(f"\n✅ Training log saved to: {log_file_path}")
        
        print(f"\n📊 最终统计:")
        print(f"  - Gaussian number: {gaussians._xyz.shape[0]}")
        print(f"  - Training time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--init_ply_path", type=str, default=None,
                        help="Path to the initial point cloud PLY file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training (default: 1)")
    parser.add_argument("--clear_cache_interval", type=int, default=500,
                        help="Interval to clear CUDA cache (default: 500)")

    # ===== 语义指导参数（可选） =====
    parser.add_argument("--qwen_model_path", type=str, default=None,
                        help="Path to Qwen3-VL model. If not provided, semantic guidance is disabled.")
    parser.add_argument("--semantic_densify_factor", type=float, default=2.0,
                        help="Densification factor for semantic guidance")
    parser.add_argument("--semantic_num_samples", type=int, default=5,
                        help="Number of images to analyze each time (default: 5)")
    parser.add_argument("--save_region_vis", action='store_true', default=False,
                        help="Save visualization images of detected smooth regions")
    
    parser.add_argument("--phase1_start", type=int, default=500,
                        help="Phase 1 start iteration (default: 500)")
    parser.add_argument("--phase1_end", type=int, default=8000,
                        help="Phase 1 end iteration (default: 8000)")
    parser.add_argument("--phase1_interval", type=int, default=1500,
                        help="Phase 1 interval between analyses (default: 1500)")
    parser.add_argument("--phase1_max_points", type=int, default=2000,
                        help="Phase 1 max points per view (default: 2000)")
    
    parser.add_argument("--phase2_start", type=int, default=8000,
                        help="Phase 2 start iteration (default: 8000)")
    parser.add_argument("--phase2_end", type=int, default=14000,
                        help="Phase 2 end iteration (default: 14000)")
    parser.add_argument("--phase2_interval", type=int, default=2000,
                        help="Phase 2 interval between analyses (default: 2000)")
    parser.add_argument("--phase2_max_points", type=int, default=8000,
                        help="Phase 2 max points per view (default: 8000)")
    
    parser.add_argument("--prune_protection", type=int, default=2000,
                        help="Iterations to protect after Phase 2 before pruning (default: 2000)")
    parser.add_argument("--total_iterations", type=int, default=20000,
                        help="Total training iterations (default: 20000)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("="*60)
    print("Optimizing " + args.model_path)
    print("="*60)
    
    if args.init_ply_path:
        print(f"Using initial point cloud from: {args.init_ply_path}")
    else:
        print("No initial point cloud specified. Will use random initialization with camera frustum sampling.")
    
    if args.qwen_model_path:
        print(f"Semantic guidance enabled with Qwen model: {args.qwen_model_path}")
    else:
        print("Semantic guidance disabled. Running FastGS without semantic control.")

    if args.detect_anomaly:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("CUDA_LAUNCH_BLOCKING enabled for debugging")

    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.init_ply_path,
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets,
        # 语义指导参数
        args.qwen_model_path,
        args.semantic_densify_factor,
        args.semantic_num_samples,
        args.save_region_vis,
        args.phase1_start,
        args.phase1_end,
        args.phase1_interval,
        args.phase2_start,
        args.phase2_end,
        args.phase2_interval,
        args.prune_protection,
        args.total_iterations,
        args.phase1_max_points,
        args.phase2_max_points
    )

    print("\n✅ Training complete.")
