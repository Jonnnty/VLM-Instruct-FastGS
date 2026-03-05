import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim

def compute_phase2_weighted_loss(image, gt_image, region_masks, region_weight=20.0, background_weight=1.0):
    """
    计算Phase 2加权损失 - 区域内像素权重高，背景像素权重低
    对所有图都使用这个损失，但只有检测到的图有region_masks

    Args:
        image: 渲染图像 (3, H, W)
        gt_image: 真实图像 (3, H, W)
        region_masks: 该相机检测到的区域mask列表（如果为None或空，则使用普通权重）
        region_weight: 区域内像素的权重倍数（默认20倍）
        background_weight: 背景像素的权重（默认1.0）

    Returns:
        weighted_loss: 加权后的总损失
        debug_info: 调试信息
    """
    # 如果没有区域mask，使用普通权重（全图权重都是1.0）
    if not region_masks:
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        full_loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)  # 使用默认的lambda_dssim=0.2
        return full_loss, {"loss": full_loss.item(), "mode": "standard_full_image"}

    # 合并所有区域mask
    combined_mask = torch.zeros_like(region_masks[0], dtype=torch.bool)
    for mask in region_masks:
        combined_mask = combined_mask | mask

    # 计算区域像素和背景像素的数量
    region_pixels = combined_mask.sum().item()
    total_pixels = image.shape[1] * image.shape[2]
    background_pixels = total_pixels - region_pixels
    region_ratio = region_pixels / total_pixels

    # 创建权重图：区域内高权重，背景普通权重
    # 扩展到3通道
    weight_map = torch.ones_like(image) * background_weight
    mask_3ch = combined_mask.unsqueeze(0).expand(3, -1, -1)
    weight_map[mask_3ch] = region_weight

    # 计算L1损失（逐像素）
    l1_per_pixel = torch.abs(image - gt_image)  # [3, H, W]

    # 加权L1损失
    weighted_l1 = (l1_per_pixel * weight_map).mean()

    # SSIM损失（全图）
    ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    ssim_loss = 1.0 - ssim_value

    # 组合损失（可以调整lambda参数）
    lambda_dssim = 0.2  # 与默认值保持一致
    weighted_loss = (1.0 - lambda_dssim) * weighted_l1 + lambda_dssim * ssim_loss

    debug_info = {
        "weighted_l1": weighted_l1.item(),
        "ssim_loss": ssim_loss.item(),
        "total_loss": weighted_loss.item(),
        "region_pixels": region_pixels,
        "background_pixels": background_pixels,
        "region_ratio": region_ratio,
        "region_weight": region_weight,
        "background_weight": background_weight,
        "mode": "weighted_full_image"
    }

    return weighted_loss, debug_info

def save_phase2_comparison(render_start, render_end, gt, pixel_bboxes, save_dir, image_name):
    """
    保存phase2开始和结束的对比图，确保边界框正确显示

    Args:
        render_start: phase2开始时的渲染图 tensor (3, H, W)
        render_end: phase2结束时的渲染图 tensor (3, H, W)
        gt: 真实图 tensor (3, H, W)
        pixel_bboxes: 像素坐标边界框列表
        save_dir: 保存目录
        image_name: 图像名称
    """
    
    # 转换为numpy
    def tensor_to_numpy(tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        np_img = tensor.numpy().transpose(1, 2, 0)
        return (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
    
    render_start_np = tensor_to_numpy(render_start)
    render_end_np = tensor_to_numpy(render_end)
    gt_np = tensor_to_numpy(gt)
    
    # 在渲染图上绘制边界框（使用绿色，线宽3）
    render_start_with_boxes = render_start_np.copy()
    render_end_with_boxes = render_end_np.copy()
    
    print(f"      - Drawing {len(pixel_bboxes)} bounding boxes on comparison images")
    for i, bbox in enumerate(pixel_bboxes):
        x1, y1, x2, y2 = bbox
        print(f"        - Box {i+1}: [{x1}, {y1}, {x2}, {y2}]")
        # 使用更粗的线宽和亮绿色
        cv2.rectangle(render_start_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.rectangle(render_end_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Phase2开始
    axes[0].imshow(render_start_with_boxes)
    axes[0].set_title('Phase 2 Start (with boxes)', fontsize=14)
    axes[0].axis('off')
    
    # Phase2结束
    axes[1].imshow(render_end_with_boxes)
    axes[1].set_title('Phase 2 End (with boxes)', fontsize=14)
    axes[1].axis('off')
    
    # GT
    axes[2].imshow(gt_np)
    axes[2].set_title('Ground Truth', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{image_name}_phase2_comparison.jpg")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # 也保存单独带框的render图
    render_end_with_boxes_bgr = cv2.cvtColor(render_end_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f"{image_name}_render_end_with_boxes.jpg"), render_end_with_boxes_bgr)
    
    print(f"      - Saved phase2 comparison to: {save_path}")
    print(f"      - Saved render with boxes to: {save_dir}/{image_name}_render_end_with_boxes.jpg")
