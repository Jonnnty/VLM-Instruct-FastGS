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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import numpy as np
import random
import json
import re
from PIL import Image
import cv2
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from scene.cameras import Camera


def create_random_camera_from_stats(existing_cameras):
    """从现有相机的统计数据生成随机相机位姿"""
    all_R = []
    all_T = []
    all_fovx = []
    all_fovy = []
    
    for cam in existing_cameras:
        all_R.append(torch.tensor(cam.R))
        all_T.append(torch.tensor(cam.T))
        all_fovx.append(cam.FoVx)
        all_fovy.append(cam.FoVy)
    
    all_R = torch.stack(all_R)
    all_T = torch.stack(all_T)
    all_fovx = torch.tensor(all_fovx)
    all_fovy = torch.tensor(all_fovy)
    
    # 计算均值和标准差
    mean_R = all_R.mean(dim=0)
    mean_T = all_T.mean(dim=0)
    mean_fovx = all_fovx.mean().item()
    mean_fovy = all_fovy.mean().item()
    
    std_R = all_R.std(dim=0).mean().item()
    std_T = all_T.std(dim=0).mean().item()
    
    # 生成随机扰动
    noise_R = torch.randn_like(mean_R) * std_R * 0.3
    noise_T = torch.randn_like(mean_T) * std_T * 0.5
    
    # 保持旋转矩阵正交性
    new_R = mean_R + noise_R
    U, _, V = torch.svd(new_R)
    new_R = torch.mm(U, V.t())
    
    new_T = mean_T + noise_T
    
    # 随机微调FOV
    new_fovx = mean_fovx * (1 + (random.random() - 0.5) * 0.1)
    new_fovy = mean_fovy * (1 + (random.random() - 0.5) * 0.1)
    
    # 从现有相机中随机选择一个作为基础
    base_cam = random.choice(existing_cameras)
    
    # 创建新的相机
    new_camera = Camera(
        colmap_id=len(existing_cameras) + random.randint(1, 1000),
        R=new_R.cpu().numpy(),
        T=new_T.cpu().numpy(),
        FoVx=new_fovx,
        FoVy=new_fovy,
        image=base_cam.original_image,
        gt_alpha_mask=None,
        image_name=f"random_cam_{random.randint(0, 99999):05d}",
        uid=len(existing_cameras) + random.randint(1, 1000),
        trans=base_cam.trans,
        scale=base_cam.scale,
        data_device=torch.device("cuda")
    )
    
    return new_camera


def generate_random_cameras(existing_cameras, num_cameras=5):
    """生成随机相机位姿"""
    new_cameras = []
    for i in range(num_cameras):
        new_cam = create_random_camera_from_stats(existing_cameras)
        new_cameras.append(new_cam)
    return new_cameras


def create_non_black_mask(rendered_image, threshold=0.05):
    """创建非黑色区域的掩码（排除未渲染的黑色区域）"""
    if rendered_image.is_cuda:
        rendered_image = rendered_image.cpu()
    
    img_np = rendered_image.numpy().transpose(1, 2, 0)
    brightness = np.mean(img_np, axis=2)
    non_black_mask = brightness > threshold
    
    return torch.tensor(non_black_mask, device="cuda")


def extract_bboxes_from_text(text):
    """从模型输出中提取边界框JSON"""
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except:
        print(f"  Warning: Failed to parse model output: {text[:100]}...")
        return []


def analyze_with_qwen(rendered_image, model, processor):
    """
    使用Qwen3-VL模型分析渲染图像中不合理的地方
    """
    # 将tensor转换为PIL Image
    if rendered_image.is_cuda:
        rendered_image = rendered_image.cpu()
    
    render_np = rendered_image.numpy().transpose(1, 2, 0)
    render_np = (np.clip(render_np, 0, 1) * 255).astype(np.uint8)
    render_pil = Image.fromarray(render_np)
    
    width, height = render_pil.size
    
    # 创建非黑色区域掩码
    non_black_mask = create_non_black_mask(rendered_image)
    
    # 提示词 - 找不合理区域
    prompt = """You are analyzing a 3D Gaussian Splatting rendered image.
    Your task: Identify areas that look UNREASONABLE or have RENDERING ARTIFACTS.
    
    Look for:
    1. Floating artifacts - disconnected pieces floating in air
    2. Distorted shapes - objects with wrong geometry
    3. Texture problems - blurry regions that should be sharp
    4. Color artifacts - strange colors, color bleeding
    5. Inconsistencies - parts that don't match the scene
    
    IMPORTANT: ONLY consider areas that are ACTUALLY RENDERED (ignore black/unrendered regions).
    
    Output a JSON array of bounding boxes for unreasonable areas:
    [
        {"bbox": [x1, y1, x2, y2]},
        ...
    ]
    Coordinates normalized (0-1000). Return empty array if the image looks reasonable."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": render_pil},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    results = extract_bboxes_from_text(output_text)

    # 转换边界框为像素坐标和掩码
    pixel_bboxes = []
    mask = np.zeros((height, width), dtype=bool)

    for item in results:
        if "bbox" in item:
            bbox = item["bbox"]
            x1 = int(bbox[0] / 1000 * width)
            y1 = int(bbox[1] / 1000 * height)
            x2 = int(bbox[2] / 1000 * width)
            y2 = int(bbox[3] / 1000 * height)
            
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
            
            if x2 > x1 and y2 > y1:
                pixel_bboxes.append([x1, y1, x2, y2])
                mask[y1:y2, x1:x2] = True
    
    # 转换为tensor并确保只在非黑色区域内
    mask_tensor = torch.tensor(mask, device="cuda")
    mask_tensor = mask_tensor & non_black_mask
    
    return {
        "raw_output": output_text,
        "parsed_results": results,
        "pixel_bboxes": pixel_bboxes,
        "mask": mask_tensor,
        "non_black_mask": non_black_mask
    }


def save_visualization(image, analysis_result, save_path, image_name):
    """保存可视化结果，在原图上标记不合理区域"""
    os.makedirs(save_path, exist_ok=True)
    
    if image.is_cuda:
        image = image.cpu()
    img_np = image.numpy().transpose(1, 2, 0)
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    
    # 创建带框的图像
    img_with_boxes = img_np.copy()
    
    # 绘制边界框
    for bbox in analysis_result["pixel_bboxes"]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 保存图像
    cv2.imwrite(os.path.join(save_path, f"{image_name}_analysis.jpg"), 
                cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    # 保存原始图像
    cv2.imwrite(os.path.join(save_path, f"{image_name}_original.jpg"), 
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


def render_and_analyze(model_path, iteration, views, gaussians, pipeline, background, 
                       qwen_model, qwen_processor, args):
    """渲染随机生成的相机位姿并用Qwen分析不合理区域"""
    # 创建输出目录
    base_path = os.path.join(model_path, "random_views_analysis", "ours_{}".format(iteration))
    render_path = os.path.join(base_path, "renders")
    analysis_path = os.path.join(base_path, "analysis")
    viz_path = os.path.join(base_path, "visualizations")
    
    makedirs(render_path, exist_ok=True)
    makedirs(analysis_path, exist_ok=True)
    makedirs(viz_path, exist_ok=True)
    
    # 生成随机相机
    print(f"\n=== Generating {args.num_random_views} random camera poses ===")
    print(f"Using {len(views)} source cameras for statistics")
    random_cameras = generate_random_cameras(views, num_cameras=args.num_random_views)
    
    total_render_time = 0.0
    total_analysis_time = 0.0
    all_issues = []
    
    print(f"\n=== Rendering and Analyzing {len(random_cameras)} random views ===")
    for idx, view in enumerate(tqdm(random_cameras, desc="Processing")):
        # 渲染
        start_render = time.time()
        render_pkg = render_fastgs(view, gaussians, pipeline, background, args.mult)
        rendering = render_pkg["render"]
        end_render = time.time()
        render_time = end_render - start_render
        total_render_time += render_time
        
        # 保存渲染结果
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'random_{idx:05d}.png'))
        
        # 分析
        start_analysis = time.time()
        analysis_result = analyze_with_qwen(rendering, qwen_model, qwen_processor)
        end_analysis = time.time()
        analysis_time = end_analysis - start_analysis
        total_analysis_time += analysis_time
        
        # 保存分析结果
        with open(os.path.join(analysis_path, f'analysis_{idx:05d}.json'), 'w') as f:
            json.dump({
                'image_index': idx,
                'image_name': view.image_name,
                'num_issues': len(analysis_result['pixel_bboxes']),
                'raw_output': analysis_result['raw_output'],
                'pixel_bboxes': analysis_result['pixel_bboxes']
            }, f, indent=2)
        
        # 保存可视化
        save_visualization(rendering, analysis_result, viz_path, f'random_{idx:05d}')
        
        # 收集统计
        for bbox in analysis_result['pixel_bboxes']:
            all_issues.append({'image_idx': idx, 'bbox': bbox})
        
        # 打印当前结果
        print(f"\n  Image {idx}: {view.image_name}")
        print(f"    Render time: {render_time*1000:.2f} ms")
        print(f"    Analysis time: {analysis_time*1000:.2f} ms")
        print(f"    Issues found: {len(analysis_result['pixel_bboxes'])}")
    
    # 打印汇总
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total images: {len(random_cameras)}")
    print(f"Total rendering time: {total_render_time:.2f}s")
    print(f"Total analysis time: {total_analysis_time:.2f}s")
    print(f"Average render time: {total_render_time/len(random_cameras)*1000:.2f} ms")
    print(f"Average analysis time: {total_analysis_time/len(random_cameras)*1000:.2f} ms")
    print(f"Total issues found: {len(all_issues)}")
    print(f"Images with issues: {len(set([issue['image_idx'] for issue in all_issues]))}/{len(random_cameras)}")
    print(f"{'='*50}\n")


def load_qwen_model(model_path):
    """加载Qwen3-VL模型"""
    print(f"\n=== Loading Qwen3-VL model from {model_path} ===")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Qwen3-VL model loaded successfully!")
    return model, processor


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    """
    主函数：加载模型和数据，生成随机视图，渲染，分析不合理区域
    """
    with torch.no_grad():
        # 初始化高斯模型
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        
        # 加载场景
        print(f"\n=== Loading scene from {dataset.model_path} ===")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # 获取相机数据
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        
        print(f"\n=== Scene loaded successfully ===")
        print(f"Training cameras: {len(train_cameras)}")
        print(f"Test cameras: {len(test_cameras)}")
        print(f"Gaussian model points: {gaussians._xyz.shape[0]}")
        print(f"Loaded iteration: {scene.loaded_iter}")
        
        # 设置背景色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 选择源相机
        if args.use_test_cameras and len(test_cameras) > 0:
            source_cameras = test_cameras
            print(f"\nUsing test cameras as source ({len(source_cameras)} cameras)")
        else:
            source_cameras = train_cameras
            print(f"\nUsing training cameras as source ({len(source_cameras)} cameras)")
        
        # 加载Qwen模型
        if args.qwen_model_path:
            qwen_model, qwen_processor = load_qwen_model(args.qwen_model_path)
        else:
            print("No Qwen model path provided, skipping analysis")
            return
        
        # 渲染并分析随机视图
        render_and_analyze(
            dataset.model_path,
            scene.loaded_iter,
            source_cameras,
            gaussians,
            pipeline,
            background,
            qwen_model,
            qwen_processor,
            args
        )


if __name__ == "__main__":
    # 设置命令行参数
    parser = ArgumentParser(description="Render random views and find unreasonable areas with Qwen3-VL")
    
    # 基础参数
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # 渲染参数
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    
    # 随机视图参数
    parser.add_argument("--num_random_views", type=int, default=5)
    parser.add_argument("--use_test_cameras", action="store_true", default=False)
    
    # Qwen模型参数
    parser.add_argument("--qwen_model_path", type=str, required=True,
                        help="Path to Qwen3-VL model")
    
    # 解析参数
    args = get_combined_args(parser)
    
    print("\n" + "="*60)
    print("RANDOM VIEWS RENDERING - FIND UNREASONABLE AREAS")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Number of random views: {args.num_random_views}")
    print(f"Qwen model: {args.qwen_model_path}")
    print("="*60 + "\n")
    
    # 初始化系统状态
    safe_state(args.quiet)
    
    # 执行渲染和分析
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)