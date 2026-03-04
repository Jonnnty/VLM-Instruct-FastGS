# scene/semantic_guidance.py
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
import cv2


def _extract_bboxes_from_text(text):
    """
    从模型输出中提取边界框JSON
    
    Args:
        text: 模型输出的文本
    
    Returns:
        list: 解析出的边界框列表
    """
    # 尝试提取JSON数组
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    # 尝试清理markdown格式
    try:
        # 移除```json和```标记
        clean_text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except:
        # 如果解析失败，返回空列表
        print(f"  Warning: Failed to parse model output: {text[:100]}...")
        return []


def _tensor_to_pil(tensor):
    """将torch tensor转换为PIL Image"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 从 [C, H, W] 转换为 [H, W, C]
    if tensor.dim() == 3:
        np_img = tensor.numpy().transpose(1, 2, 0)
    else:
        np_img = tensor.numpy()
    
    # 确保值在0-255范围内
    np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def _tensor_to_numpy(tensor):
    """将torch tensor转换为numpy数组 (H, W, C)"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if tensor.dim() == 3:
        np_img = tensor.numpy().transpose(1, 2, 0)
    else:
        np_img = tensor.numpy()
    
    return (np.clip(np_img, 0, 1) * 255).astype(np.uint8)


def save_improvement_visualization(render, gt, pixel_bboxes, save_dir, image_name):
    """
    保存改进区域的对比可视化：
    1. 对比图（render带框 + GT）
    2. 单独带框的render图
    
    Args:
        render: 渲染图 tensor (C, H, W)
        gt: 真实图 tensor (C, H, W)
        pixel_bboxes: 像素坐标边界框列表
        save_dir: 保存目录
        image_name: 图像名称
    """
    import matplotlib.pyplot as plt
    
    # 转换为numpy
    render_np = _tensor_to_numpy(render)
    gt_np = _tensor_to_numpy(gt)
    
    # 创建带框的渲染图
    render_with_boxes = render_np.copy()
    for bbox in pixel_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(render_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # 1. 保存单独带框的render图
    render_with_boxes_bgr = cv2.cvtColor(render_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f"{image_name}_render_with_boxes.jpg"), render_with_boxes_bgr)
    
    # 2. 创建对比图（render带框 + GT）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Render带框图
    axes[0].imshow(render_with_boxes)
    axes[0].set_title('Render with Improvement Regions')
    axes[0].axis('off')
    
    # GT图
    axes[1].imshow(gt_np)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    plt.tight_layout()
    compare_path = os.path.join(save_dir, f"{image_name}_compare.jpg")
    plt.savefig(compare_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"      - Saved visualizations: {image_name}_render_with_boxes.jpg and {image_name}_compare.jpg")


def find_improvement_regions_with_qwen(rendered_images, gt_images, model, processor,
                                        save_visualization=False, save_dir=None, image_names=None):
    """
    使用Qwen3-VL对比渲染图和GT图，找出需要改进的区域
    
    Args:
        rendered_images: list of torch tensors [3, H, W] (渲染图)
        gt_images: list of torch tensors [3, H, W] (GT图)
        model: 已加载的Qwen3-VL模型
        processor: 已加载的Qwen3-VL处理器
        save_visualization: 是否保存可视化结果
        save_dir: 保存目录
        image_names: 图片名称列表
    
    Returns:
        list of dict: 每个图像的检测结果，包含:
            - 'mask': 改进区域mask (bool tensor [H, W])
            - 'bboxes': 像素坐标的边界框列表
            - 'raw_output': 模型原始输出文本
            - 'num_bboxes': 边界框数量
    """
    all_results = []

    # 创建保存目录
    if save_visualization and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"  - Saving improvement region visualizations to: {save_dir}")

    for img_idx, (render, gt) in enumerate(zip(rendered_images, gt_images)):
        img_name = image_names[img_idx] if image_names and img_idx < len(image_names) else f"image_{img_idx + 1}"
        
        print(f"  - Analyzing image {img_idx + 1}: {img_name}")

        # 转换为PIL图像
        render_pil = _tensor_to_pil(render)
        gt_pil = _tensor_to_pil(gt)
        
        # 获取图像尺寸
        width, height = render_pil.size

        # 创建对比图（左右拼接）- 用于Qwen输入
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(render_pil, (0, 0))
        combined.paste(gt_pil, (width, 0))

        # 准备提示
        prompt = """You are comparing a 3D Gaussian Splatting rendered image (LEFT) with the ground truth photo (RIGHT).
Your task: Identify areas in the RENDERED image that need improvement.

Guidelines:
1. Compare left and right images carefully
2. Look for rendering artifacts, missing details, blurry regions, or incorrect geometry
3. Focus on areas where the render differs significantly from the ground truth
4. Provide bounding boxes for the MOST PROBLEMATIC areas that need improvement
5.Output up to 3 bounding boxes focusing on regions clear in ground truth but blurry in render, especially distant backgrounds. Each box shall be no larger than 1/8 of the image.
Output a JSON array of bounding boxes for areas needing improvement:
[
    {"bbox": [x1, y1, x2, y2]},
    ...
]
Coordinates normalized (0-1000). Return empty array if the rendered image looks perfect."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": combined},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        # 调用模型
        try:
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

            print(f"    Qwen response: {output_text[:150]}...")

            # 提取边界框
            bboxes = _extract_bboxes_from_text(output_text)

            # 转换边界框为像素坐标
            pixel_bboxes = []
            mask = torch.zeros(height, width, dtype=torch.bool, device=render.device)

            for item in bboxes:
                if "bbox" in item:
                    bbox = item["bbox"]
                    # 归一化坐标转像素坐标
                    x1 = int(bbox[0] / 1000 * width)
                    y1 = int(bbox[1] / 1000 * height)
                    x2 = int(bbox[2] / 1000 * width)
                    y2 = int(bbox[3] / 1000 * height)

                    # 确保坐标有效
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(y1 + 1, min(y2, height))

                    if x2 > x1 and y2 > y1:
                        pixel_bboxes.append([x1, y1, x2, y2])
                        mask[y1:y2, x1:x2] = True
                        print(f"      Bbox found: [{x1},{y1},{x2},{y2}] (size: {x2-x1}x{y2-y1})")

            # 如果没有检测到bbox，使用差异图作为后备
            if not mask.any():
                print("    No bbox detected, using difference map as fallback")
                # 计算差异图
                diff = torch.abs(render - gt).mean(dim=0)
                threshold = diff.mean() + diff.std() * 1.5
                mask = diff > threshold
                
                # 将mask转换为边界框（简化处理）
                mask_np = mask.cpu().numpy()
                if mask_np.any():
                    # 找到所有非零区域的边界
                    y_indices, x_indices = np.where(mask_np)
                    x1, x2 = x_indices.min(), x_indices.max()
                    y1, y2 = y_indices.min(), y_indices.max()
                    
                    # 稍微扩展一下
                    pad = 10
                    x1 = max(0, x1 - pad)
                    x2 = min(width - 1, x2 + pad)
                    y1 = max(0, y1 - pad)
                    y2 = min(height - 1, y2 + pad)
                    
                    pixel_bboxes.append([x1, y1, x2, y2])
                    print(f"      Fallback bbox: [{x1},{y1},{x2},{y2}]")

            # 保存可视化结果（只保存两种图片）
            if save_visualization and save_dir and pixel_bboxes:
                save_improvement_visualization(
                    render, gt, pixel_bboxes,
                    save_dir, img_name
                )

            # 保存结果
            result = {
                'mask': mask,
                'bboxes': pixel_bboxes,
                'raw_output': output_text,
                'num_bboxes': len(pixel_bboxes)
            }
            all_results.append(result)

            # 打印找到的边界框数量
            if len(pixel_bboxes) > 0:
                print(f"      ✅ Found {len(pixel_bboxes)} improvement regions")
            else:
                print(f"      ⚠️ No improvement regions found")

            # 清理缓存
            del inputs, generated_ids
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ❌ Error processing image {img_idx}: {e}")
            import traceback
            traceback.print_exc()
            # 返回空结果
            all_results.append({
                'mask': torch.zeros(height, width, dtype=torch.bool, device=render.device),
                'bboxes': [],
                'raw_output': '',
                'num_bboxes': 0
            })

    # 打印统计信息
    total_bboxes = sum(r['num_bboxes'] for r in all_results)
    print(f"  - Total images analyzed: {len(all_results)}")
    print(f"  - Total bboxes found: {total_bboxes}")
    if len(all_results) > 0:
        print(f"  - Avg bboxes per image: {total_bboxes/len(all_results):.2f}")

    return all_results


def combine_improvement_masks(results):
    """
    将多个图像的改进区域mask合并为一个mask（用于训练）
    
    Args:
        results: find_improvement_regions_with_qwen 返回的结果列表
    
    Returns:
        torch.Tensor: 合并后的mask (N,)，N为高斯点数量
    """
    if not results:
        return None
    
    # 这里需要根据具体训练代码来整合，通常在训练循环中处理
    pass


def filter_bboxes_by_confidence(bboxes, min_area=100):
    """
    根据面积过滤边界框
    
    Args:
        bboxes: 像素坐标边界框列表
        min_area: 最小面积阈值
    
    Returns:
        过滤后的边界框列表
    """
    filtered = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            filtered.append(bbox)
    return filtered


def expand_bboxes(bboxes, expand_ratio=0.1, image_size=None):
    """
    扩展边界框
    
    Args:
        bboxes: 像素坐标边界框列表
        expand_ratio: 扩展比例（相对于原框大小）
        image_size: (width, height) 用于裁剪
    
    Returns:
        扩展后的边界框列表
    """
    expanded = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # 计算扩展量
        dx = int(w * expand_ratio)
        dy = int(h * expand_ratio)
        
        # 扩展
        x1 = max(0, x1 - dx)
        x2 = x2 + dx
        y1 = max(0, y1 - dy)
        y2 = y2 + dy
        
        # 如果提供了图像尺寸，确保不超出边界
        if image_size:
            width, height = image_size
            x2 = min(x2, width - 1)
            y2 = min(y2, height - 1)
        
        if x2 > x1 and y2 > y1:
            expanded.append([x1, y1, x2, y2])
    
    return expanded
