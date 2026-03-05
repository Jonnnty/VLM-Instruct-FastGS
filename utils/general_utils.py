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
import sys
from datetime import datetime
import numpy as np
import random
from plyfile import PlyData, PlyElement
import gc

def identity_gate(x):
    return x

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def safe_tensor_1d(tensor, expected_len=None):
    """确保tensor是1维且长度正确"""
    if tensor is None:
        return torch.zeros(expected_len if expected_len else 0, dtype=bool, device="cuda") if expected_len else None
    
    # 检查是否为0维张量（标量）
    if tensor.dim() == 0:
        # 0维tensor转换为1维（长度为1的tensor）
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() > 1:
        # 如果维度大于1，尝试压缩
        tensor = tensor.squeeze()
        # 如果压缩后还是大于1维，取第一个维度
        if tensor.dim() > 1:
            tensor = tensor.flatten()
    
    # 确保现在是一维的
    if tensor.dim() != 1:
        tensor = tensor.flatten()
    
    # 处理长度不匹配的情况
    if expected_len is not None and len(tensor) != expected_len:
        new_tensor = torch.zeros(expected_len, dtype=tensor.dtype, device=tensor.device)
        copy_len = min(len(tensor), expected_len)
        if len(tensor) > 0:
            new_tensor[:copy_len] = tensor[:copy_len]
        tensor = new_tensor
    
    return tensor

def save_gaussian_to_ply(gaussians, save_path):
    """保存高斯点云到PLY文件"""
    print(f"Saving Gaussian point cloud to {save_path}")
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    
    # 获取特征
    features_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
    
    # 获取其他属性
    opacities = gaussians._opacity.detach().cpu().numpy()
    scales = gaussians._scaling.detach().cpu().numpy()
    rotations = gaussians._rotation.detach().cpu().numpy()
    
    # 创建PLY属性
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
    
    # 添加特征属性
    for i in range(features_dc.shape[1]):
        dtype_full.append((f'f_dc_{i}', 'f4'))
    
    # 添加不透明度
    dtype_full.append(('opacity', 'f4'))
    
    # 添加缩放
    for i in range(scales.shape[1]):
        dtype_full.append((f'scale_{i}', 'f4'))
    
    # 添加旋转
    for i in range(rotations.shape[1]):
        dtype_full.append((f'rot_{i}', 'f4'))
    
    # 构建元素
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    
    # 填充特征
    for i in range(features_dc.shape[1]):
        elements[f'f_dc_{i}'] = features_dc[:, i]
    
    elements['opacity'] = opacities[:, 0]
    
    for i in range(scales.shape[1]):
        elements[f'scale_{i}'] = scales[:, i]
    
    for i in range(rotations.shape[1]):
        elements[f'rot_{i}'] = rotations[:, i]
    
    # 保存PLY
    plydata = PlyData([PlyElement.describe(elements, 'vertex')], text=False)
    plydata.write(save_path)
    print(f"Saved {xyz.shape[0]} Gaussian points to {save_path}")

def print_cuda_memory():
    """打印CUDA显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        cached = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"CUDA Memory - Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")

def clear_cuda_cache():
    """清理CUDA缓存"""
    torch.cuda.empty_cache()
    gc.collect()
