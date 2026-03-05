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
from gaussian_renderer import render_fastgs, network_gui_ws, render_pointcloud_only
import sys
import json
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, inverse_sigmoid, safe_tensor_1d, save_gaussian_to_ply, print_cuda_memory, clear_cuda_cache
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from plyfile import PlyData, PlyElement
from torch import nn
from scene.cameras import Camera
import gc
import traceback
import matplotlib.pyplot as plt
import cv2

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.sh_utils import RGB2SH


from utils.geometry_utils import (
    compute_centroid_distance_penalty, 
    create_hollow_ellipsoid_shell,
    project_points_to_pixels_correct,
    is_point_in_bbox,
    compute_frustum_outside_loss
)
from utils.pruning_utils import prune_large_gaussians, update_mask_to_current_size
from utils.phase2_loss import compute_phase2_weighted_loss, save_phase2_comparison

# ===== Qwen3-VL导入 =====
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from scene.semantic_guidance import find_improvement_regions_with_qwen
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: Qwen3-VL not available. Qwen detection will be disabled.")
    
    # 创建空函数避免导入错误
    def find_improvement_regions_with_qwen(*args, **kwargs):
        print("Qwen3-VL not available, returning empty results")
        return []


def prepare_output_and_logger(args):
    """准备输出目录和日志记录器"""
    if args.model_path:
        output_path = args.model_path
    else:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        output_path = os.path.join("./output/", unique_str)
        args.model_path = output_path

    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene, renderFunc, renderArgs, all_test_cameras=None):
    """训练报告生成函数"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        test_cameras = all_test_cameras if all_test_cameras is not None else scene.getTestCameras()
        validation_configs = (
            {'name': 'test', 'cameras': test_cameras},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
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



def training(dataset, opt, pipe, init_ply_path, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, websockets,
             # Phase 0参数
             phase0_start, phase0_end,
             # Phase 1参数
             phase1_start, phase1_end,
             # Phase 2参数 
             phase2_start, phase2_end, phase2_base_weight,
             # Qwen检测参数
             qwen_model_path,
             qwen_detection_iteration,
             qwen_detection_interval,
             save_qwen_visualization,
             # 椭圆筒初始化参数
             ellipsoid_expand_factor, ellipsoid_height_padding, ellipsoid_num_points,
             # 大高斯裁剪参数
             large_gaussian_scale_multiplier=3.0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    if init_ply_path:
        print(f"Loading initial point cloud from: {init_ply_path}")
        scene = Scene(dataset, gaussians, init_ply_path=init_ply_path)
    else:
        print("Warning: No initial PLY path provided. Using default initialization.")
        gaussians.create_random_initialization(num_points=100, spatial_lr_scale=1.0)
        scene = Scene(dataset, gaussians)

    # ===== 创建日志文件 =====
    log_file_path = os.path.join(dataset.model_path, "training_log.txt")
    with open(log_file_path, 'w') as f:
        f.write("iteration,loss,gaussian_count,phase,region_weight,region_ratio\n")
        f.write(f"0,{0.0},{gaussians._xyz.shape[0]},init,1.0,0.0\n")
    print(f"Training log will be saved to: {log_file_path}")

    # ===== 所有相机 =====
    all_train_cameras = scene.getTrainCameras()
    all_test_cameras = scene.getTestCameras()
    current_train_cameras = all_train_cameras

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    print("=" * 50)

    # ===== 初始化Qwen3-VL模型 =====
    qwen_model = None
    qwen_processor = None
    use_qwen = False

    if qwen_model_path and QWEN_AVAILABLE:
        print(f"\n===== Loading Qwen3-VL model from {qwen_model_path} =====")
        try:
            if not os.path.exists(qwen_model_path):
                raise FileNotFoundError(f"Qwen3-VL model not found at {qwen_model_path}")

            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)
            use_qwen = True
            print("✅ Qwen3-VL model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Qwen3-VL model: {e}")
            print(f"Error details: {traceback.format_exc()}")
            use_qwen = False
    elif qwen_model_path and not QWEN_AVAILABLE:
        print("⚠️ Qwen3-VL not available. Please install transformers and related packages.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 初始化当前使用的相机集
    viewpoint_stack = current_train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # Qwen检测相关变量
    qwen_detection_executed = False
    qwen_detected_cameras = []  # 保存检测到的相机
    qwen_detection_results = {}  # 保存检测结果
    qwen_detection_images = {}  # 保存检测时的渲染图
    qwen_region_masks = {}  # 保存每个相机的区域mask

    
    phase2_initialized = False

    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # ===== 生成观察点云保存迭代列表（从5000到10000，每隔1000）=====
    observation_iterations = list(range(5000, 10001, 1000))
    if 10000 not in observation_iterations:
        observation_iterations.append(10000)

    print(f"\n===== Phase 0 Settings =====")
    print(f"  - Start: {phase0_start}")
    print(f"  - End: {phase0_end}")
    print(f"===== Phase 0 Normal Mode Centroid Penalty =====")
    print(f"  - Enabled for normal mode with enhanced penalty")
    print(f"  - Penalty alpha: 1.0 (high)")
    print(f"  - Power: 4.0 (quadruple penalty for far points)")
    print(f"===== Large Gaussian Pruning at Phase 0 End =====")
    print(f"  - Scale multiplier: {large_gaussian_scale_multiplier}x of mean max axis")
    print(f"===== Observation Point Cloud Saves =====")
    print(f"  - Saving point clouds at iterations: {observation_iterations}")
    print(f"===== Phase 1 Settings =====")
    print(f"  - Start: {phase1_start}")
    print(f"  - End: {phase1_end}")
    print(f"===== Phase 2 Settings (Qwen Region-weighted Training) =====")
    print(f"  - Start: {phase2_start}")
    print(f"  - End: {phase2_end}")
    print(f"  - Region base weight: {phase2_base_weight}x")
    print(f"  - Background weight: 1.0x")
    print(f"  - Detection interval: Every {qwen_detection_interval}th image")
    print(f"  - Training on ALL images, with higher weight on detected regions")
    print(f"  - NO cloning of Gaussians - relying on weighted loss only")
    print(f"  - Densification: Original 3DGS (no importance_score filter)")

    # ===== Qwen检测设置打印 =====
    if use_qwen:
        print(f"===== Qwen Detection at Iteration {qwen_detection_iteration} =====")
        print(f"  - Detection interval: Every {qwen_detection_interval}th image")
        print(f"  - Save visualizations: {save_qwen_visualization}")
        print(f"  - Will NOT clone Gaussians in detected regions")
    else:
        print(f"===== Qwen Detection =====")
        print(f"  - Disabled (Qwen model not available)")

    print(f"===== Hollow Ellipsoid Shell Initialization =====")
    print(f"  - Expand factor: {ellipsoid_expand_factor}")
    print(f"  - Height padding: {ellipsoid_height_padding}")
    print(f"  - Num points: {ellipsoid_num_points}")
    print(f"===== Multi-view Consistency Pruning =====")
    print(f"  - Active from iteration 15000 to 20000 in Phase 1")
    print(f"  - Interval: 3000 iterations")
    print(f"===== Total Iterations: {opt.iterations} =====")
    print("=" * 50)

    # ===== 用于记录目标物品点的mask =====
    target_object_mask = None

    # ===== 设置最小点云数量保护 =====
    MIN_POINTS_THRESHOLD = 100  
    CRITICAL_POINTS_THRESHOLD = 50 

    # ===== 记录Phase 0结束时的点云 =====
    phase0_completed = False
    object_center = None

    # ===== 开始训练循环 =====
    for iteration in range(first_iter, opt.iterations + 1):

        # 动态设置背景颜色（每iteration随机）
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 每100次迭代强制同步一次缓冲区
        if iteration % 100 == 0:
            gaussians.force_sync_buffers()

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

        # ===== 根据阶段选择相机采样策略 =====
        in_phase0 = (iteration >= phase0_start and iteration < phase0_end)
        in_phase1 = (iteration >= phase1_start and iteration < phase1_end)
        
        # ===== 判断是否处于Phase 2 =====
        in_phase2 = (qwen_detection_executed and 
                    iteration >= phase2_start and 
                    iteration <= phase2_end)

        # ===== 根据阶段选择相机 =====
        if in_phase2:
  
            if not phase2_initialized:
                print(f"\n[ITER {iteration}] 🎯 Entering PHASE 2 (Region-weighted Training)")
                print(f"  - Training on ALL images")
                print(f"  - Detected cameras with regions: {len(qwen_detected_cameras)}")
                print(f"  - Region weight: {phase2_base_weight}x, background: 1.0x")
                print(f"  - NO cloning of Gaussians - relying on weighted loss only")
                print(f"  - Densification: Original 3DGS (no importance_score filter)")
                phase2_initialized = True
            
            # 使用所有相机，正常从堆栈中采样
            if not viewpoint_stack:
                viewpoint_stack = current_train_cameras.copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            view_idx = viewpoint_indices.pop(rand_idx)
            
        else:
            # Phase 0常规、Phase 1和其他阶段: 使用所有相机
            if phase2_initialized:
                viewpoint_stack = current_train_cameras.copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
                phase2_initialized = False
            
            # 从堆栈中采样
            if not viewpoint_stack:
                viewpoint_stack = current_train_cameras.copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            view_idx = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # ===== 渲染图像 =====
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)

        # 立即检查并修复visibility_filter和radii
        current_num_points = gaussians.get_xyz.shape[0]

        # 处理visibility_filter
        if "visibility_filter" not in render_pkg:
            render_pkg["visibility_filter"] = torch.ones(current_num_points, dtype=bool, device="cuda")
        elif isinstance(render_pkg["visibility_filter"], torch.Tensor):
            if render_pkg["visibility_filter"].dim() == 0:
                val = render_pkg["visibility_filter"].item()
                render_pkg["visibility_filter"] = torch.ones(current_num_points, dtype=bool, device="cuda") * (val > 0)
            elif render_pkg["visibility_filter"].dim() > 1:
                render_pkg["visibility_filter"] = render_pkg["visibility_filter"].flatten()

        # 处理radii
        if "radii" not in render_pkg:
            render_pkg["radii"] = torch.zeros(current_num_points, device="cuda")
        elif isinstance(render_pkg["radii"], torch.Tensor):
            if render_pkg["radii"].dim() == 0:
                val = render_pkg["radii"].item()
                render_pkg["radii"] = torch.ones(current_num_points, device="cuda") * val
            elif render_pkg["radii"].dim() > 1:
                render_pkg["radii"] = render_pkg["radii"].flatten()

        # 确保长度正确
        if len(render_pkg["visibility_filter"]) != current_num_points:
            old_filter = render_pkg["visibility_filter"]
            new_filter = torch.ones(current_num_points, dtype=bool, device="cuda")
            copy_len = min(len(old_filter), current_num_points)
            if len(old_filter) > 0:
                new_filter[:copy_len] = old_filter[:copy_len]
            render_pkg["visibility_filter"] = new_filter

        if len(render_pkg["radii"]) != current_num_points:
            old_radii = render_pkg["radii"]
            new_radii = torch.zeros(current_num_points, device="cuda")
            copy_len = min(len(old_radii), current_num_points)
            if len(old_radii) > 0:
                new_radii[:copy_len] = old_radii[:copy_len]
            render_pkg["radii"] = new_radii

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        visibility_filter = safe_tensor_1d(visibility_filter, current_num_points)
        radii = safe_tensor_1d(radii, current_num_points)

        gt_image = viewpoint_cam.original_image.cuda()

        # ===== 根据阶段选择Loss计算方式 =====
        if in_phase0:
            # Phase 0 常规模式：全图Loss + 基于质心的距离惩罚（增强版）
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # ===== 增强的质心距离惩罚 =====
            # 只在Phase 0且点数足够时添加
            if gaussians.get_xyz.shape[0] >= 30:
                # 使用增强的惩罚参数
                alpha = 1.0  # 惩罚系数加大到1.0
                power = 4.0  # 4次幂，远点惩罚急剧增大

                penalty, centroid = compute_centroid_distance_penalty(gaussians, alpha=alpha, power=power)
                loss = image_loss + penalty

                # 动态调整：随着迭代进行，逐渐加大惩罚
                progress = (iteration - phase0_start) / (phase0_end - phase0_start)
                if progress > 0.5:
                    # 后半段再增加50%惩罚
                    loss = image_loss + penalty * 1.5

                if iteration % 500 == 0 and penalty.item() > 0:
                    print(f"\n[Phase 0 Normal] Enhanced centroid penalty: {penalty.item():.6f} (alpha={alpha}, power={power})")
                    if centroid is not None:
                        print(f"  Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
            else:
                loss = image_loss
        elif in_phase2:
     
            cam_name = viewpoint_cam.image_name

        
            region_masks = qwen_region_masks.get(cam_name, [])
            

            loss, debug_info = compute_phase2_weighted_loss(
                image,
                gt_image,
                region_masks,
                region_weight=phase2_base_weight,  # 区域内权重
                background_weight=1.0  # 背景权重（1x）
            )

            # 每100次迭代打印一次调试信息（只对有检测区域的图）
            if iteration % 100 == 0 and region_masks:
                print(f"\n[Phase 2] Iter {iteration}, Camera: {cam_name}")
                print(f"  - Weighted L1: {debug_info['weighted_l1']:.6f}")
                print(f"  - SSIM loss: {debug_info['ssim_loss']:.6f}")
                print(f"  - Total loss: {debug_info['total_loss']:.6f}")
                print(f"  - Mode: {debug_info['mode']}")
                if 'region_pixels' in debug_info:
                    print(f"  - Region pixels: {debug_info['region_pixels']} ({debug_info['region_ratio']:.2%} of image)")
                    print(f"  - Weights: region x{debug_info['region_weight']:.1f}, bg x{debug_info['background_weight']:.1f}")
        else:
            # 标准Phase 1损失
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

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

            # ===== 每500次迭代记录loss和高斯点数 =====
            if iteration % 500 == 0:
                current_loss = loss.item()
                current_gaussian_count = gaussians._xyz.shape[0]
                phase_name = "phase0" if in_phase0 else "phase2" if in_phase2 else "phase1"
                with open(log_file_path, 'a') as f:
                    f.write(f"{iteration},{current_loss:.6f},{current_gaussian_count},{phase_name},1.0,0.0\n")
                print(f"\n[LOG] Iter {iteration}: Loss={current_loss:.6f}, Gaussians={current_gaussian_count}, Phase={phase_name}")

            # ===== 保存观察点云（从5000到10000，每隔1000）=====
            if iteration in observation_iterations and iteration >= phase0_end:
                obs_save_path = os.path.join(dataset.model_path, f"point_cloud_obs_iter{iteration}.ply")
                save_gaussian_to_ply(gaussians, obs_save_path)
                print(f"  - 📸 Saved observation point cloud at iteration {iteration}")

            # ===== Qwen检测 =====
            if use_qwen and iteration == qwen_detection_iteration and not qwen_detection_executed:
                print(f"\n{'='*60}")
                print(f"[ITER {iteration}] Running Qwen detection (ONE-TIME ANALYSIS)")
                print(f"{'='*60}")

                # 清理缓存
                clear_cuda_cache()

                all_cams = current_train_cameras
                selected_indices = list(range(0, len(all_cams), qwen_detection_interval))
                selected_cams = [all_cams[idx] for idx in selected_indices]

                print(f"  - Selected {len(selected_cams)} cameras for Qwen analysis (every {qwen_detection_interval}th image):")
                for i, (idx, cam) in enumerate(zip(selected_indices, selected_cams)):
                    print(f"      {i+1}: {cam.image_name} (index {idx})")


                rendered_images = []
                gt_images = []
                image_names = []

                for cam in selected_cams:
                    render_pkg_semantic = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                    rendered = render_pkg_semantic["render"]
                    rendered_images.append(rendered)
                    gt_images.append(cam.original_image.cuda())
                    image_names.append(cam.image_name)

                # 使用Qwen分析改进区域
                print(f"  - Running Qwen analysis on {len(selected_cams)} images...")

                analysis_results = find_improvement_regions_with_qwen(
                    rendered_images,
                    gt_images,
                    qwen_model,
                    qwen_processor,
                    save_visualization=save_qwen_visualization,
                    save_dir=os.path.join(dataset.model_path, f"qwen_detection_iter{iteration}") if save_qwen_visualization else None,
                    image_names=image_names
                )

                # 保存检测到的相机和结果
                detected_cameras_count = 0
                
                for cam_idx, result in enumerate(analysis_results):
                    cam = selected_cams[cam_idx]
                    cam_name = cam.image_name

                    if result['num_bboxes'] > 0:
                        detected_cameras_count += 1
                        print(f"      - Camera {cam_name}: {result['num_bboxes']} improvement regions")
                        for bbox_idx, bbox in enumerate(result['bboxes']):
                            print(f"          Box {bbox_idx+1}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

                        # 保存检测结果
                        qwen_detected_cameras.append(cam_name)
                        qwen_detection_results[cam_name] = result
                        qwen_detection_images[cam_name] = rendered_images[cam_idx].clone()
                        
                        # 保存区域mask用于加权损失
                        if 'mask' in result and result['mask'] is not None:
                            if cam_name not in qwen_region_masks:
                                qwen_region_masks[cam_name] = []
                            qwen_region_masks[cam_name].append(result['mask'])

           
                        print(f"          - Saved region mask for weighted loss in Phase 2")
                        
                    else:
                        print(f"      - Camera {cam_name}: No improvement regions found")

                print(f"  - Total images analyzed: {len(selected_cams)}")
                print(f"  - Detected cameras with regions: {detected_cameras_count}")

                # 打印Phase 2训练设置信息
                if detected_cameras_count > 0:
                    print(f"\n  - 🎯 {detected_cameras_count} cameras with detected regions will have weighted loss in Phase 2")
                    print(f"  - Phase 2 training from iteration {phase2_start} to {phase2_end}")
                    print(f"  - Training on ALL images, with region weight {phase2_base_weight}x")
                    print(f"  - NO cloning of Gaussians - relying on weighted loss only")
                    print(f"  - Densification: Original 3DGS (no importance_score filter)")

                # 保存检测结果到JSON文件
                results_json = []
                for cam_name, result in qwen_detection_results.items():
                    results_json.append({
                        'image_name': cam_name,
                        'num_bboxes': result['num_bboxes'],
                        'bboxes': result['bboxes']
                    })

                json_path = os.path.join(dataset.model_path, f"qwen_detection_iter{iteration}.json")
                with open(json_path, 'w') as f:
                    json.dump(results_json, f, indent=2)
                print(f"  - Saved detection results to: {json_path}")

                qwen_detection_executed = True

                # 清理缓存
                clear_cuda_cache()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            optim_start.record()

            # ===== 检查是否是添加椭圆筒后的第一次迭代 =====
            skip_densification = False

            # ===== Phase 0 结束时：先裁剪体积大的高斯，再添加空心椭圆筒 =====
            if not phase0_completed and iteration >= phase0_end:
                phase0_completed = True
                print(f"\n{'='*60}")
                print(f"Phase 0 completed at iteration {iteration}")
                print(f"{'='*60}")

                # ===== 第一步：裁剪体积大的高斯 =====
                print(f"\n--- Step 1: Pruning large Gaussians ---")
                prune_large_gaussians(
                    gaussians,
                    scale_multiplier=large_gaussian_scale_multiplier
                )

                # ===== 第二步：保存Phase 0结束时的点云（裁剪大高斯后）=====
                print(f"\n--- Step 2: Saving Phase 0 end point cloud ---")
                phase0_save_path = os.path.join(dataset.model_path, f"point_cloud_phase0_end.ply")
                save_gaussian_to_ply(gaussians, phase0_save_path)

                # ===== 第三步：根据当前点云拟合椭圆筒并添加高斯点 =====
                print(f"\n--- Step 3: Adding hollow ellipsoid shell ---")
                shell_data = create_hollow_ellipsoid_shell(
                    gaussians,
                    num_points=ellipsoid_num_points,
                    expand_factor=ellipsoid_expand_factor,
                    height_padding=ellipsoid_height_padding
                )

                # 确保 tmp_radii 被正确初始化
                if gaussians.tmp_radii is None:
                    gaussians.tmp_radii = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")

                # 将椭圆筒高斯添加到当前模型中
                new_tmp_radii = torch.zeros(shell_data['xyz'].shape[0], device="cuda")
                gaussians.densification_postfix(
                    shell_data['xyz'],
                    shell_data['features_dc'],
                    shell_data['features_rest'],
                    shell_data['opacity'],
                    shell_data['scaling'],
                    shell_data['rotation'],
                    new_tmp_radii
                )

                print(f"\n✅ Added {shell_data['xyz'].shape[0]} hollow ellipsoid shell points around object")
                print(f"Total points now: {gaussians._xyz.shape[0]}")

                # 保存添加椭圆筒后的点云
                shell_save_path = os.path.join(dataset.model_path, f"point_cloud_with_hollow_shell.ply")
                save_gaussian_to_ply(gaussians, shell_save_path)

   
                skip_densification = True
                print(f"  - Skipping densification for current iteration to avoid dimension mismatch")
                print(f"{'='*60}\n")

            # ===== densification =====
            if iteration < opt.densify_until_iter and not skip_densification:
                current_num_points = gaussians.get_xyz.shape[0]

                visibility_filter = safe_tensor_1d(visibility_filter, current_num_points)
                radii = safe_tensor_1d(radii, current_num_points)

                if visibility_filter.any():  
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 检查是否应该进行 densification
                should_densify = (iteration > opt.densify_from_iter and
                                  iteration % opt.densification_interval == 0)

                if should_densify:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = current_train_cameras.copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt,
                                                                                    DENSIFY=True)

                    # 根据阶段选择剪枝函数
                    if in_phase0:
                 
                        if current_num_points < CRITICAL_POINTS_THRESHOLD:
                            print(f"  - ⚠️ Phase 0: CRITICAL point count ({current_num_points}), temporarily disabling pruning")
                            gaussians.densify_and_prune(
                                max_screen_size=size_threshold,
                                min_opacity=0.0,
                                extent=scene.cameras_extent,
                                radii=radii,
                                args=opt,
                                importance_score=importance_score,
                                pruning_score=pruning_score * 0.0
                            )
                        elif current_num_points < MIN_POINTS_THRESHOLD:
                            print(f"  - Phase 0: Point count ({current_num_points}) below threshold, strongly reducing pruning")
                            gaussians.densify_and_prune(
                                max_screen_size=size_threshold,
                                min_opacity=0.02,
                                extent=scene.cameras_extent,
                                radii=radii,
                                args=opt,
                                importance_score=importance_score,
                                pruning_score=pruning_score * 0.2
                            )
                        else:
                            gaussians.densify_and_prune(
                                max_screen_size=size_threshold,
                                min_opacity=0.01,
                                extent=scene.cameras_extent,
                                radii=radii,
                                args=opt,
                                importance_score=importance_score,
                                pruning_score=pruning_score * 0.5
                            )
                    elif in_phase2:
      
                        gaussians.densify_and_prune_phase2(
                            max_grad=opt.grad_thresh, 
                            min_opacity=0.001, 
                            extent=scene.cameras_extent,
                            max_screen_size=size_threshold,
                            radii=radii
                        )
                    else:
               
                        gaussians.densify_and_prune_fastgs(
                            max_screen_size=size_threshold,
                            min_opacity=0.005,
                            extent=scene.cameras_extent,
                            radii=radii,
                            args=opt,
                            importance_score=importance_score,
                            pruning_score=pruning_score
                        )

                    current_num_points = gaussians._xyz.shape[0]
                    if target_object_mask is not None:
                        target_object_mask = update_mask_to_current_size(target_object_mask, current_num_points)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    current_num_points = gaussians._xyz.shape[0]
                    if target_object_mask is not None:
                        target_object_mask = update_mask_to_current_size(target_object_mask, current_num_points)

            # ===== 多视角一致性剪枝=====
            if iteration % 3000 == 0 and iteration > 15000 and iteration < 20000 and in_phase1:
                print(f"\n[ITER {iteration}] Running multi-view consistency pruning...")
                my_viewpoint_stack = current_train_cameras.copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)

                current_num_points = gaussians._xyz.shape[0]
                if target_object_mask is not None:
                    target_object_mask = update_mask_to_current_size(target_object_mask, current_num_points)
                print(f"  - ✅ After pruning: {current_num_points} Gaussians")

            # ===== 优化步骤 =====
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # ===== 在Phase1训练结束时，生成Qwen检测相机的对比图 =====
            if iteration == opt.iterations - 1 or iteration == opt.iterations:
                if qwen_detection_executed and len(qwen_detected_cameras) > 0:
                    print(f"\n  - Generating final comparison visualizations for Qwen-detected cameras...")

                    comparison_dir = os.path.join(dataset.model_path, f"qwen_detection_comparison")
                    os.makedirs(comparison_dir, exist_ok=True)

                    for cam_name in qwen_detected_cameras:
                        if cam_name in qwen_detection_results and cam_name in qwen_detection_images:
                            # 找到对应的相机
                            for cam in current_train_cameras:
                                if cam.image_name == cam_name:
                                    # 渲染当前的图像
                                    end_render_pkg = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                    end_render = end_render_pkg["render"]

                                    # 保存对比图
                                    save_phase2_comparison(
                                        qwen_detection_images[cam_name],
                                        end_render,
                                        cam.original_image.cuda(),
                                        qwen_detection_results[cam_name]['bboxes'],
                                        comparison_dir,
                                        cam_name
                                    )
                                    break

            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")

    with open(log_file_path, 'a') as f:
        f.write(f"{iteration},{loss.item():.6f},{gaussians._xyz.shape[0]},final,1.0,0.0\n")
    print(f"Training log saved to: {log_file_path}")


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
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)

    # 指定初始点云PLY文件路径
    parser.add_argument("--init_ply_path", type=str, default=None,
                        help="Path to the initial point cloud PLY file")

    # ===== Phase 0 参数 =====
    parser.add_argument("--phase0_start", type=int, default=0,
                        help="Phase 0 start iteration (default: 0)")
    parser.add_argument("--phase0_end", type=int, default=4000,
                        help="Phase 0 end iteration (default: 4000)")

    # ===== 大高斯裁剪参数 =====
    parser.add_argument("--large_gaussian_scale_multiplier", type=float, default=3.0,
                        help="Threshold multiplier for large Gaussian pruning at phase0 end (default: 3.0)")

    # ===== Phase 1 参数 =====
    parser.add_argument("--phase1_start", type=int, default=4000,
                        help="Phase 1 start iteration (default: 4000)")
    parser.add_argument("--phase1_end", type=int, default=40000,
                        help="Phase 1 end iteration (default: 40000)")

    # ===== Qwen检测参数 =====
    parser.add_argument("--qwen_model_path", type=str, default=None,
                        help="Path to Qwen3-VL model for detection")
    parser.add_argument("--qwen_detection_iteration", type=int, default=20000,
                        help="Iteration to run Qwen detection (default: 20000)")
    parser.add_argument("--qwen_detection_interval", type=int, default=15,
                        help="Interval for selecting images for Qwen detection (default: 15)")
    parser.add_argument("--save_qwen_visualization", action='store_true', default=True,
                        help="Save Qwen detection visualizations")

    # ===== Phase 2 参数 =====
    parser.add_argument("--phase2_start", type=int, default=20000,
                        help="Phase 2 start iteration (default: 20000)")
    parser.add_argument("--phase2_end", type=int, default=30000,
                        help="Phase 2 end iteration (default: 30000)")
    parser.add_argument("--phase2_base_weight", type=float, default=20.0,
                        help="Weight multiplier for detected regions in Phase 2 (default: 20.0)")

    # ===== 椭圆筒初始化参数 =====
    parser.add_argument("--ellipsoid_expand_factor", type=float, default=1.2,
                        help="Expand factor for XY ellipse (default: 1.2)")
    parser.add_argument("--ellipsoid_height_padding", type=float, default=0.2,
                        help="Height padding ratio relative to object height (default: 0.2)")
    parser.add_argument("--ellipsoid_num_points", type=int, default=5000,
                        help="Number of points for hollow ellipsoid shell (default: 5000)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    if args.init_ply_path:
        print(f"Using initial point cloud from: {args.init_ply_path}")
    else:
        print("Warning: No initial point cloud specified. Will use default initialization.")

    safe_state(args.quiet)

    if (args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)

    training(
        lp_args,
        op_args,
        pp_args,
        args.init_ply_path,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets,
        # Phase 0参数
        args.phase0_start,
        args.phase0_end,
        # Phase 1参数
        args.phase1_start,
        args.phase1_end,
        # Phase 2参数
        args.phase2_start,
        args.phase2_end,
        args.phase2_base_weight,
        # Qwen检测参数
        args.qwen_model_path,
        args.qwen_detection_iteration,
        args.qwen_detection_interval,
        args.save_qwen_visualization,
        # 椭圆筒参数
        args.ellipsoid_expand_factor,
        args.ellipsoid_height_padding,
        args.ellipsoid_num_points,
        # 大高斯裁剪参数
        args.large_gaussian_scale_multiplier
    )

    print("\nTraining complete.")
