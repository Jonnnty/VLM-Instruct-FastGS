# scene/semantic_guidance.py
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
import cv2


def find_smooth_regions(rendered_images, model, processor, max_region_ratio=0.125,
                        save_visualization=False, save_dir=None, image_names=None):
    """
    分析渲染图，找出需要增强的平滑区域

    Args:
        rendered_images: list of torch tensors [3, H, W] 或 list of PIL Images
        model: 已加载的Qwen3-VL模型
        processor: 已加载的Qwen3-VL处理器
        max_region_ratio: 保留参数但不再使用
        save_visualization: 是否保存可视化结果
        save_dir: 保存目录
        image_names: 图像名称列表（用于保存文件）

    Returns:
        list of boolean masks, 每个图片的平滑区域掩码 [H, W]
    """
    all_masks = []

    # 如果需要保存可视化，创建保存目录
    if save_visualization and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"  - Saving region visualizations to: {save_dir}")

    for i, render_img in enumerate(rendered_images):
        # 保存原始图片信息用于打印
        img_info = ""
        img_name = image_names[i] if image_names and i < len(image_names) else f"image_{i + 1}"

        # 如果是torch tensor，转换为PIL Image
        if isinstance(render_img, torch.Tensor):
            # 确保在CPU上并转换为numpy
            if render_img.is_cuda:
                render_img = render_img.cpu()

            # 从 [C, H, W] 转换为 [H, W, C]
            render_np = render_img.numpy().transpose(1, 2, 0)
            # 确保值在0-255范围内
            render_np = (np.clip(render_np, 0, 1) * 255).astype(np.uint8)
            render_pil = Image.fromarray(render_np)
            img_info = f"Image {i + 1} (tensor)"
        else:
            render_pil = render_img
            if hasattr(render_img, 'filename'):
                img_info = f"Image {i + 1}: {render_img.filename}"
            else:
                img_info = f"Image {i + 1} (PIL Image)"

        # 改进的提示词 - 更精确地描述任务
        prompt = """You are analyzing a 3D Gaussian Splatting rendered image in early training.
        Your task: Identify areas that are BEGINNING TO SHOW TEXTURE DETAIL - the first places where details become visible.

        Guidelines:
        1. Look for areas that are starting to look less blurry than the rest
        2. These are the "early success" regions where Gaussians are converging well
        3. Focus on regions where you can start to see edges, patterns, or structure
        4. No size restrictions - can be any size
    

        Output a JSON array of bounding boxes for areas showing early texture detail:
        [
            {"bbox": [x1, y1, x2, y2]},
            ...
        ]
        Coordinates normalized (0-1000). Return empty array if the whole image is still blurry."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": render_pil},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        # 调用模型
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

        # 提取JSON数据
        smooth_areas = _extract_bboxes_from_text(output_text)

        # 打印图片信息和找到的区域
        print(f"  - {img_info}")

        # 不再过滤区域大小，直接使用所有检测到的区域
        valid_bboxes = []  # 保存所有边界框用于可视化
        if len(smooth_areas) > 0:
            width, height = render_pil.size
            
            for area in smooth_areas:
                if "bbox" in area:
                    bbox = area["bbox"]
                    valid_bboxes.append(bbox)
                    print(f"      Region: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            
            print(f"    Found {len(smooth_areas)} smooth regions")
        else:
            print(f"    No smooth regions found")

        # 将边界框转换为掩码（不再进行大小过滤）
        mask = _bboxes_to_mask(smooth_areas, render_pil.size)

        # 如果有区域，打印转换后的像素坐标
        pixel_bboxes = []  # 保存像素坐标用于可视化
        if len(smooth_areas) > 0:
            width, height = render_pil.size
            print(f"    Image size: {width}x{height} pixels")
            for j, area in enumerate(smooth_areas):
                if "bbox" in area:
                    bbox = area["bbox"]
                    # 转换到像素坐标
                    x1 = int(bbox[0] / 1000 * width)
                    y1 = int(bbox[1] / 1000 * height)
                    x2 = int(bbox[2] / 1000 * width)
                    y2 = int(bbox[3] / 1000 * height)
                    pixel_bboxes.append([x1, y1, x2, y2])
                    print(f"      Pixel coordinates: [{x1}, {y1}, {x2}, {y2}]")

        # 保存可视化结果
        if save_visualization and save_dir and len(pixel_bboxes) > 0:
            save_region_visualization(
                render_img,
                pixel_bboxes,
                save_dir,
                img_name
            )

        all_masks.append(mask)

    return all_masks


def find_subject_regions(rendered_images, model, processor,
                        save_visualization=False, save_dir=None, image_names=None):
    """
    分析渲染图，识别主体区域（不取反，只返回主体mask）

    Args:
        rendered_images: list of torch tensors [3, H, W] 渲染图
        model: 已加载的Qwen3-VL模型
        processor: 已加载的Qwen3-VL处理器
        save_visualization: 是否保存可视化结果
        save_dir: 保存目录
        image_names: 图像名称列表（用于保存文件）

    Returns:
        list of boolean masks, 每个图片的主体区域掩码 [H, W] (True表示主体)
    """
    all_masks = []

    # 如果需要保存可视化，创建保存目录
    if save_visualization and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"  - Saving subject region visualizations to: {save_dir}")

    for i, render_img in enumerate(rendered_images):
        # 保存原始图片信息用于打印
        img_name = image_names[i] if image_names and i < len(image_names) else f"image_{i + 1}"

        # 将tensor转换为PIL Image
        if isinstance(render_img, torch.Tensor):
            if render_img.is_cuda:
                render_img = render_img.cpu()
            render_np = render_img.numpy().transpose(1, 2, 0)
            render_np = (np.clip(render_np, 0, 1) * 255).astype(np.uint8)
            render_pil = Image.fromarray(render_np)
        else:
            render_pil = render_img

        print(f"  - Image {i + 1}: {img_name}")

        # 提示词 - 识别主体区域
        prompt = """You are analyzing a 3D Gaussian Splatting rendered image.
        Your task: Identify the MAIN SUBJECT of the image.

        Guidelines:
        1. The main subject is the primary object, person, or area of interest in the scene
        2. Provide a SINGLE bounding box that tightly encloses the main subject
        3. If there are multiple subjects, choose the most prominent one
        4. The bounding box should not exceed 1/2 of the entire image

        Output a JSON array with ONE bounding box:
        [
            {"bbox": [x1, y1, x2, y2]}
        ]
        Coordinates normalized (0-1000). Return empty array if there's no clear subject."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": render_pil},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        # 调用模型
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

        # 提取JSON数据
        subject_areas = _extract_bboxes_from_text(output_text)

        # 打印找到的区域
        subject_bboxes = []  # 保存所有边界框用于可视化
        pixel_bboxes = []  # 保存像素坐标用于可视化
        width, height = render_pil.size

        if len(subject_areas) > 0:
            print(f"    Found {len(subject_areas)} subject regions:")
            for area in subject_areas:
                if "bbox" in area:
                    bbox = area["bbox"]
                    subject_bboxes.append(bbox)
                    
                    # 转换到像素坐标
                    x1 = int(bbox[0] / 1000 * width)
                    y1 = int(bbox[1] / 1000 * height)
                    x2 = int(bbox[2] / 1000 * width)
                    y2 = int(bbox[3] / 1000 * height)
                    pixel_bboxes.append([x1, y1, x2, y2])
                    
                    print(f"      Subject region: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] -> pixel=[{x1}, {y1}, {x2}, {y2}]")
        else:
            print(f"    No clear subject found, using empty mask")

        # 将边界框转换为掩码（主体区域）
        subject_mask = _bboxes_to_mask(subject_areas, (width, height))

        # 保存可视化结果 - 只画主体框
        if save_visualization and save_dir and len(pixel_bboxes) > 0:
            save_subject_visualization(
                render_img, pixel_bboxes,
                save_dir, f"{img_name}_subject"
            )

        all_masks.append(subject_mask)

    return all_masks


def save_subject_visualization(image, pixel_bboxes, save_dir, image_name):
    """
    在图像上绘制主体边界框并保存（只画框，不填色）

    Args:
        image: 渲染的图像 tensor (C, H, W) 或 numpy array
        pixel_bboxes: 像素坐标边界框列表 [[x1, y1, x2, y2], ...]
        save_dir: 保存目录
        image_name: 图像名称
    """
    # 将 tensor 转换为 numpy 图像
    if torch.is_tensor(image):
        if image.is_cuda:
            image = image.cpu()
        # 假设图像是 (C, H, W) 范围 [0, 1]
        img_np = (image.detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img_np = image

    # 确保是 RGB 格式（OpenCV 使用 BGR）
    if img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    # 在图像上绘制所有边界框（绿色，线宽2）
    for i, bbox in enumerate(pixel_bboxes):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存图像
    save_path = os.path.join(save_dir, f"{image_name}_boxes_only.jpg")
    cv2.imwrite(save_path, img_bgr)
    print(f"      - Saved subject visualization: {save_path}")


def save_region_visualization(image, pixel_bboxes, save_dir, image_name):
    """
    在图像上绘制边界框并保存

    Args:
        image: 渲染的图像 tensor (C, H, W) 或 numpy array
        pixel_bboxes: 像素坐标边界框列表 [[x1, y1, x2, y2], ...]
        save_dir: 保存目录
        image_name: 图像名称
    """
    # 将 tensor 转换为 numpy 图像
    if torch.is_tensor(image):
        if image.is_cuda:
            image = image.cpu()
        # 假设图像是 (C, H, W) 范围 [0, 1]
        img_np = (image.detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img_np = image

    # 确保是 RGB 格式（OpenCV 使用 BGR）
    if img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    # 在图像上绘制所有边界框
    for i, bbox in enumerate(pixel_bboxes):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # 绘制矩形框（绿色，线宽2）
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存图像
    save_path = os.path.join(save_dir, f"{image_name}_boxes_only.jpg")
    cv2.imwrite(save_path, img_bgr)
    print(f"      - Saved region visualization: {save_path}")


def _extract_bboxes_from_text(text):
    """从模型输出中提取边界框JSON"""
    # 尝试提取JSON数组
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    # 尝试清理markdown格式
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except:
        # 如果解析失败，返回空列表
        print(f"  Warning: Failed to parse model output: {text[:100]}...")
        return []


def _bboxes_to_mask(bboxes, image_size):
    """将边界框列表转换为二值掩码（不再进行大小过滤）"""
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)

    for item in bboxes:
        if "bbox" in item:
            bbox = item["bbox"]
            # 归一化坐标转像素坐标
            x1 = int(bbox[0] / 1000 * width)
            y1 = int(bbox[1] / 1000 * height)
            x2 = int(bbox[2] / 1000 * width)
            y2 = int(bbox[3] / 1000 * height)

            # 确保坐标有效
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))

            # 标记区域（不再检查大小）
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

    return torch.tensor(mask, device="cuda")


def post_process_mask(mask, min_region_size=100):
    """
    后处理掩码：移除太小的孤立区域，避免噪声

    Args:
        mask: 布尔掩码
        min_region_size: 最小区域像素数

    Returns:
        处理后的掩码
    """
    if not mask.any():
        return mask

    # 转换为numpy进行处理
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask

    from scipy import ndimage
    # 标记连通区域
    labeled, num_features = ndimage.label(mask_np)

    # 创建新掩码
    new_mask = np.zeros_like(mask_np, dtype=bool)

    for i in range(1, num_features + 1):
        region = (labeled == i)
        if region.sum() >= min_region_size:
            new_mask = np.logical_or(new_mask, region)

    return torch.tensor(new_mask, device="cuda") if torch.is_tensor(mask) else new_mask