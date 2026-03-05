import torch
import numpy as np

def prune_large_gaussians(gaussians, scale_multiplier=3.0):
    """
    裁剪体积大的高斯点
    对于每个高斯，取三个轴中最长的轴，如果大于平均值 * scale_multiplier，则裁剪掉

    Args:
        gaussians: GaussianModel对象
        scale_multiplier: 阈值乘数，默认3.0

    Returns:
        pruned_count: 裁剪掉的高斯数量
    """
    with torch.no_grad():
        if gaussians._scaling.shape[0] == 0:
            print("  - No Gaussians to prune")
            return 0
            
       
        scales = torch.exp(gaussians._scaling)  # [N, 3]
        
        # 对于每个高斯，取最长的轴
        max_axis = torch.max(scales, dim=1)[0]  # [N]
        
        # 计算所有最长轴的平均值
        mean_max_axis = torch.mean(max_axis)
        
        # 计算阈值
        threshold = mean_max_axis * scale_multiplier
        
        # 找出需要裁剪的高斯（最长轴 > 阈值）
        prune_mask = max_axis > threshold
        
        pruned_count = prune_mask.sum().item()
        
        if pruned_count > 0:
            print(f"\n=== Large Gaussian Pruning ===")
            print(f"  - Total Gaussians: {len(scales)}")
            print(f"  - Mean max axis: {mean_max_axis:.6f}")
            print(f"  - Threshold (mean * {scale_multiplier}): {threshold:.6f}")
            print(f"  - Gaussians to prune: {pruned_count} ({pruned_count/len(scales)*100:.2f}%)")
            
            # 打印一些统计信息
            if pruned_count > 0:
                pruned_scales = max_axis[prune_mask]
                print(f"  - Pruned max axis range: [{pruned_scales.min():.6f}, {pruned_scales.max():.6f}]")
                print(f"  - Pruned max axis mean: {pruned_scales.mean():.6f}")
            
            # 执行裁剪
            gaussians.prune_points(prune_mask)
            gaussians.force_sync_buffers()
            
            print(f"  - ✅ Pruned {pruned_count} large Gaussians")
            print(f"  - Remaining Gaussians: {gaussians._scaling.shape[0]}")
        else:
            print(f"\n=== Large Gaussian Pruning ===")
            print(f"  - No large Gaussians found (threshold: {threshold:.6f})")
        
        return pruned_count

def update_mask_to_current_size(mask, current_num_points):
    """将mask扩展到当前高斯点数量"""
    if mask is None:
        return None
    if len(mask) == current_num_points:
        return mask
    # 创建新的mask，原来的点保持原值，新点默认为False
    new_mask = torch.zeros(current_num_points, dtype=bool, device=mask.device)
    min_len = min(len(mask), current_num_points)
    new_mask[:min_len] = mask[:min_len]
    return new_mask
