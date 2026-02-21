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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.tmp_radii = None
        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.xyz_gradient_accum_abs,
                self.denom,
                self.optimizer.state_dict(),
                self.shoptimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.xyz_gradient_accum_abs,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         xyz_gradient_accum_abs,
         denom,
         opt_dict,
         shopt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd_source, spatial_lr_scale: float):
        """
        Initialize Gaussian model from point cloud data

        Args:
            pcd_source: Either a path to a PLY file or a BasicPointCloud object
            spatial_lr_scale: Spatial learning rate scale
        """
        from scene.dataset_readers import fetchPly

        self.spatial_lr_scale = spatial_lr_scale

        # Check if pcd_source is a string (file path) or a BasicPointCloud object
        if isinstance(pcd_source, str):
            # It's a file path, load the PLY file
            print(f"Loading point cloud from PLY file: {pcd_source}")
            pcd = fetchPly(pcd_source)
        else:
            # It's already a BasicPointCloud object
            pcd = pcd_source

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_random_initialization(self, num_points=100, spatial_lr_scale=1.0):
        """
        创建随机初始化的点云，用于没有提供PLY文件的情况
        
        Args:
            num_points: 初始化的点数（默认100）
            spatial_lr_scale: 空间学习率缩放
        """
        self.spatial_lr_scale = spatial_lr_scale
        print(f"Creating random initial Gaussian model with {num_points} points")
        
        # 在单位球体内随机生成点（范围-1到1）
        fused_point_cloud = torch.rand((num_points, 3), device="cuda") * 2 - 1
        
        # 随机颜色（转换为SH系数）
        random_colors = torch.rand((num_points, 3), device="cuda")
        fused_color = RGB2SH(random_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        # 随机缩放（较小，避免太大）
        random_scales = torch.rand((num_points, 3), device="cuda") * 0.01
        scales = torch.log(random_scales)
        
        # 随机旋转（单位四元数）
        rots = torch.randn((num_points, 4), device="cuda")
        rots = rots / torch.norm(rots, dim=1, keepdim=True)
        
        # 随机不透明度（0.2-0.8之间）
        opacities = self.inverse_opacity_activation(
            0.5 * torch.ones((num_points, 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        print(f"Random initialization complete: {num_points} points created")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none=True)
        elif iteration <= 20000:
            if iteration % 32 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none=True)
        else:
            if iteration % 64 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none=True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_tmp_radii)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_tmp_radii)

    # ============== 新增：剪枝方法 ==============

    def prune_low_quality(self, min_opacity=0.005, max_size_ratio=0.1, extent=1.0,
                          prune_screen_size=15, use_gradient=False):
        """
        综合剪枝低质量的高斯点

        Args:
            min_opacity: 最小透明度阈值
            max_size_ratio: 最大尺寸比例
            extent: 场景范围
            prune_screen_size: 屏幕空间最大尺寸
            use_gradient: 是否使用梯度信息

        Returns:
            剪枝的点数量
        """
        before_prune = self.get_xyz.shape[0]

        # 1. 透明度太低
        opacity_mask = (self.get_opacity < min_opacity).squeeze()

        # 2. 尺寸太大
        size_mask = self.get_scaling.max(dim=1).values > max_size_ratio * extent

        # 3. 屏幕空间太大
        screen_mask = torch.zeros_like(opacity_mask)
        if hasattr(self, 'max_radii2D') and self.max_radii2D is not None:
            screen_mask = self.max_radii2D > prune_screen_size

        # 4. 梯度太小（可选）
        grad_mask = torch.zeros_like(opacity_mask)
        if use_gradient and hasattr(self, 'xyz_gradient_accum') and self.denom is not None:
            grad_norm = torch.norm(self.xyz_gradient_accum / (self.denom + 1e-8), dim=-1)
            grad_mask = grad_norm < 0.0001

        # 合并剪枝条件
        prune_mask = opacity_mask | size_mask | screen_mask | grad_mask

        # 限制剪枝比例，避免过度剪枝
        max_prune_ratio = 0.3
        if prune_mask.sum() > before_prune * max_prune_ratio:
            quality_score = torch.zeros(before_prune, device="cuda")

            opacity_score = 1.0 - self.get_opacity.squeeze()
            quality_score += opacity_score * 2.0

            size_score = self.get_scaling.max(dim=1).values / (max_size_ratio * extent)
            quality_score += size_score.clamp(max=2.0)

            threshold = torch.quantile(quality_score, 1 - max_prune_ratio)
            prune_mask = quality_score > threshold

        if prune_mask.any():
            self.prune_points(prune_mask)
            new_num = self.get_xyz.shape[0]
            if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
                self.tmp_radii = torch.zeros(new_num, device="cuda")

        return before_prune - self.get_xyz.shape[0]

    def balance_prune(self, extent, target_ratio=0.8):
        """
        强制平衡剪枝，保持点数量在合理范围

        Args:
            extent: 场景范围
            target_ratio: 保留比例

        Returns:
            剪枝的点数量
        """
        before = self.get_xyz.shape[0]

        quality_score = torch.zeros(before, device="cuda")

        opacity = self.get_opacity.squeeze()
        quality_score += (1.0 - opacity) * 2.0

        size_ratio = self.get_scaling.max(dim=1).values / (0.1 * extent)
        quality_score += size_ratio.clamp(max=2.0)

        if hasattr(self, 'max_radii2D') and self.max_radii2D is not None:
            screen_ratio = self.max_radii2D / 15.0
            quality_score += screen_ratio.clamp(max=2.0)

        num_keep = int(before * target_ratio)
        num_remove = before - num_keep

        if num_remove > 0:
            _, indices = torch.topk(quality_score, num_remove, largest=True)
            prune_mask = torch.zeros(before, dtype=bool, device="cuda")
            prune_mask[indices] = True
            self.prune_points(prune_mask)

            if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
                self.tmp_radii = torch.zeros(self.get_xyz.shape[0], device="cuda")

        return before - self.get_xyz.shape[0]

    # ============== 语义增密相关方法（修改版） ==============

    def densify_semantic_regions(self, points_mask, args, extent, densify_factor=2.0, prune_after=True,
                                 max_process_ratio=0.125):
        """
        对语义指定的区域进行增密（按质量排序选择克隆点）

        Args:
            points_mask: 布尔张量，表示哪些高斯点需要增密
            args: 优化参数
            extent: 场景范围
            densify_factor: 增密强度因子（克隆时的复制倍数）
            prune_after: 增密后是否立即剪枝
            max_process_ratio: 最大处理比例，总处理点数不超过候选点的1/8
        """
        if points_mask is None or points_mask.sum() == 0:
            return 0

        before_densify = self.get_xyz.shape[0]
        total_candidates = points_mask.sum().item()
        print(f"Semantic densification: {total_candidates} candidate points")

        if not hasattr(self, 'tmp_radii') or self.tmp_radii is None:
            self.tmp_radii = torch.zeros(self.get_xyz.shape[0], device="cuda")

        # 根据尺度判断哪些点需要分裂（大尺寸），哪些需要克隆（小尺寸）
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense * extent

        current_num_points = self.get_xyz.shape[0]
        if points_mask.shape[0] != current_num_points:
            if points_mask.shape[0] < current_num_points:
                new_mask = torch.zeros(current_num_points, dtype=bool, device="cuda")
                new_mask[:points_mask.shape[0]] = points_mask
                points_mask = new_mask
            else:
                points_mask = points_mask[:current_num_points]

        # ===== 第一步：处理分裂（所有需要分裂的点都处理，因为分裂会删除原点多出2个新点，净增1倍） =====
        semantic_splits = torch.logical_and(split_qualifiers, points_mask)
        if semantic_splits.sum() > 0:
            self._densify_split_semantic(semantic_splits, densify_factor)
            print(f"  - Split {semantic_splits.sum()} points")
            # 分裂后点的数量变了，需要更新points_mask
            points_mask = self._update_mask_after_densification(points_mask, semantic_splits, densify_factor,
                                                                is_split=True)
            clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent

        # ===== 第二步：处理克隆（按质量排序，只选最好的1/8） =====
        semantic_clones_all = torch.logical_and(clone_qualifiers, points_mask)
        if semantic_clones_all.sum() > 0:
            num_clones_all = semantic_clones_all.sum().item()

            # 计算总处理点数不能超过总候选点的 max_process_ratio
            # 已经处理的分裂点数 + 将要处理的克隆点数 <= total_candidates * max_process_ratio
            splits_done = semantic_splits.sum().item() if 'semantic_splits' in locals() else 0
            max_clones = max(0, int(total_candidates * max_process_ratio) - splits_done)
            num_clones = min(num_clones_all, max_clones)

            if num_clones > 0:
                # 获取所有克隆候选点的索引
                clone_indices = torch.where(semantic_clones_all)[0]

                # 计算质量分数（透明度越低越需要增强）
                clone_opacities = self.get_opacity[clone_indices].squeeze()
                # 分数越高越需要增强（透明度低）
                quality_scores = 1.0 - clone_opacities

                # 按质量分数排序，选择最好的 num_clones 个
                _, top_indices = torch.topk(quality_scores, min(num_clones, len(clone_indices)))
                selected_clone_indices = clone_indices[top_indices]

                # 创建新的mask
                semantic_clones = torch.zeros_like(points_mask)
                semantic_clones[selected_clone_indices] = True

                print(
                    f"  - Clone candidates: {num_clones_all} points, selecting {num_clones} (best {num_clones / num_clones_all * 100:.1f}%)")

                if semantic_clones.sum() > 0:
                    self._densify_clone_semantic(semantic_clones, densify_factor)
                    print(f"  - Cloned {semantic_clones.sum()} points")
            else:
                print(
                    f"  - No clone budget available (splits used {splits_done}/{int(total_candidates * max_process_ratio)})")
        else:
            print(f"  - No clone candidates")

        # 增密后立即剪枝
        if prune_after:
            pruned = self.prune_low_quality(
                min_opacity=0.005,
                max_size_ratio=0.1,
                extent=extent,
                prune_screen_size=15
            )
            if pruned > 0:
                print(f"  - Pruned {pruned} low-quality points after densification")

        after_all = self.get_xyz.shape[0]
        net_change = after_all - before_densify

        if after_all != before_densify:
            self.tmp_radii = torch.zeros(after_all, device="cuda")
            print(f"  - Net change: {net_change} points (now: {after_all})")

        return net_change

    def _update_mask_after_densification(self, old_mask, split_mask, densify_factor, is_split=False):
        N = 2 if is_split else 1
        if is_split:
            new_points_count = split_mask.sum() * (N - 1)
        else:
            new_points_count = split_mask.sum() * N

        new_mask = torch.zeros(old_mask.shape[0] + new_points_count, dtype=bool, device="cuda")
        keep_indices = ~split_mask
        new_mask[:keep_indices.sum()] = old_mask[keep_indices]
        return new_mask

    def _densify_clone_semantic(self, mask, densify_factor=2.0):
        N = int(densify_factor)
        new_xyz = self._xyz[mask].repeat(N, 1)
        new_features_dc = self._features_dc[mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[mask].repeat(N, 1, 1)
        new_opacities = self._opacity[mask].repeat(N, 1)
        new_scaling = self._scaling[mask].repeat(N, 1)
        new_rotation = self._rotation[mask].repeat(N, 1)

        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            new_tmp_radii = self.tmp_radii[mask].repeat(N)
        else:
            new_tmp_radii = torch.ones(N * mask.sum(), device="cuda")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def _densify_split_semantic(self, mask, densify_factor=2.0):
        N = 2
        stds = self.get_scaling[mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[mask].repeat(N, 1)

        scale_factor = 0.8 * N / densify_factor
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[mask].repeat(N, 1) / scale_factor
        )

        new_rotation = self._rotation[mask].repeat(N, 1)
        new_features_dc = self._features_dc[mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[mask].repeat(N, 1, 1)
        new_opacity = self._opacity[mask].repeat(N, 1)

        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            new_tmp_radii = self.tmp_radii[mask].repeat(N)
        else:
            new_tmp_radii = torch.ones(N * mask.sum(), device="cuda")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((mask, torch.zeros(N * mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # ============== 原有的 densify_and_prune_fastgs（修改版） ==============

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score=None,
                                 pruning_score=None):
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0

        current_num_points = self.get_xyz.shape[0]
        if radii.shape[0] != current_num_points:
            if radii.shape[0] < current_num_points:
                new_radii = torch.zeros(current_num_points, device="cuda")
                new_radii[:radii.shape[0]] = radii
                self.tmp_radii = new_radii
            else:
                self.tmp_radii = radii[:current_num_points]
        else:
            self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense * extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        metric_mask = importance_score > 5

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        # 修改：直接剪枝，不设预算
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if prune_mask.any():
            self.prune_points(prune_mask)

        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1,
                                                                 keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score=None):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)