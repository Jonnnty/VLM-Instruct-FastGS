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

        如果传入了cameras参数，则在相机视锥相交的空间内采样
        否则回退到基于场景范围的随机采样
        """
        self.spatial_lr_scale = spatial_lr_scale

        # 检查是否有相机信息可用（通过外部传入或从类属性获取）
        cameras = getattr(self, 'train_cameras', None)

        if cameras is not None and len(cameras) > 0:
            print(f"Creating initial Gaussian model from camera frustums with {num_points} points")
            points = self._sample_from_camera_frustums(cameras, num_points)
        else:
            print(f"Creating random initial Gaussian model with {num_points} points (scene-based)")
            # 回退到基于场景范围的随机采样
            scale = spatial_lr_scale * 0.5
            points = (torch.rand((num_points, 3), device="cuda") * 2 - 1) * scale

        # 随机颜色（转换为SH系数）
        random_colors = torch.rand((num_points, 3), device="cuda")
        fused_color = RGB2SH(random_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # 随机缩放
        random_scales = torch.rand((num_points, 3), device="cuda") * 0.01 * spatial_lr_scale * 0.1
        scales = torch.log(random_scales.clamp(min=1e-6))

        # 随机旋转
        rots = torch.randn((num_points, 4), device="cuda")
        rots = rots / torch.norm(rots, dim=1, keepdim=True)

        # 随机不透明度
        opacities = self.inverse_opacity_activation(
            0.5 * torch.ones((num_points, 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        print(f"Random initialization complete: {num_points} points created")

    def _sample_from_camera_frustums(self, cameras, num_points):
        """在相机视锥相交的空间内采样点"""
        import numpy as np

        points_list = []
        num_cameras = len(cameras)

        # 对每个点，随机选择两个相机，在它们的视锥相交区域内采样
        for _ in range(num_points):
            # 随机选两个不同的相机
            idx1, idx2 = np.random.choice(num_cameras, 2, replace=False)
            cam1 = cameras[idx1]
            cam2 = cameras[idx2]

            # 获取相机位置
            pos1 = cam1.camera_center.cpu()
            pos2 = cam2.camera_center.cpu()

            # 在两相机连线上随机采样一个点
            t = torch.rand(1, device="cuda")
            point_on_line = pos1 * t + pos2 * (1 - t)

            # 添加一些随机扰动
            noise = torch.randn(3, device="cuda") * 0.2 * self.spatial_lr_scale
            point = point_on_line + noise

            points_list.append(point)

        return torch.stack(points_list)

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

        # 同步所有缓冲区
        self._sync_buffers_after_modification(valid_points_mask)

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

    def _sync_buffers_after_modification(self, valid_points_mask=None):
        """
        在修改点数后同步所有相关缓冲区
        Args:
            valid_points_mask: 如果是剪枝操作，传入有效点的mask
        """
        current_num = self.get_xyz.shape[0]

        if valid_points_mask is not None:
            # 剪枝操作：直接索引
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            if self.tmp_radii is not None:
                self.tmp_radii = self.tmp_radii[valid_points_mask]
        else:
            # 增密操作：需要扩展缓冲区
            old_num = self.xyz_gradient_accum.shape[0]
            if old_num != current_num:
                # 扩展梯度累积
                new_grad = torch.zeros((current_num, 1), device=self._xyz.device)
                new_grad[:old_num] = self.xyz_gradient_accum[:old_num]
                self.xyz_gradient_accum = new_grad

                new_grad_abs = torch.zeros((current_num, 1), device=self._xyz.device)
                new_grad_abs[:old_num] = self.xyz_gradient_accum_abs[:old_num]
                self.xyz_gradient_accum_abs = new_grad_abs

                new_denom = torch.zeros((current_num, 1), device=self._xyz.device)
                new_denom[:old_num] = self.denom[:old_num]
                self.denom = new_denom

                new_max = torch.zeros(current_num, device=self._xyz.device)
                new_max[:old_num] = self.max_radii2D[:old_num]
                self.max_radii2D = new_max

                if self.tmp_radii is not None:
                    new_tmp = torch.zeros(current_num, device=self._xyz.device)
                    new_tmp[:old_num] = self.tmp_radii[:old_num]
                    self.tmp_radii = new_tmp

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_tmp_radii):
        old_num_points = self.get_xyz.shape[0]

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

        # 同步所有缓冲区（增密后扩展）
        self._sync_buffers_after_modification()

        return self.get_xyz.shape[0]

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

        new_num_points = self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
                                                    new_scaling,
                                                    new_rotation, new_tmp_radii)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        return new_num_points

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        new_num_points = self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
                                                    new_scaling,
                                                    new_rotation, new_tmp_radii)

        return new_num_points

    # ============== 剪枝方法 ==============

    def prune_low_quality(self, min_opacity=0.005, max_size_ratio=0.1, extent=1.0,
                          prune_screen_size=15, use_gradient=False, target_points=None):
        """
        综合剪枝低质量的高斯点

        Args:
            min_opacity: 最小透明度阈值（默认0.005）
            max_size_ratio: 最大尺寸比例
            extent: 场景范围
            prune_screen_size: 屏幕空间最大尺寸
            use_gradient: 是否使用梯度信息
            target_points: 目标总点数（如果提供且当前点数超过此值，则强制剪枝到此数值）

        Returns:
            剪枝的点数量
        """
        before_prune = self.get_xyz.shape[0]

        # ===== 第一步：常规剪枝（基于固定阈值） =====
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

        # 合并常规剪枝条件
        regular_prune_mask = opacity_mask | size_mask | screen_mask | grad_mask

        # 执行常规剪枝
        if regular_prune_mask.any():
            self.prune_points(regular_prune_mask)
            after_regular = self.get_xyz.shape[0]
            print(f"Regular pruning removed {before_prune - after_regular} points "
                  f"(opacity<{min_opacity}, size>{max_size_ratio * extent:.4f})")
        else:
            after_regular = before_prune

        # ===== 第二步：如果需要，强制剪枝到目标点数 =====
        if target_points is not None and after_regular > target_points:
            need_prune = after_regular - target_points
            print(f"Current points {after_regular} exceeds target {target_points}, "
                  f"need to prune {need_prune} more points")

            # 计算每个点的质量分数（分数越低表示质量越高，越应该保留）
            # 分数越高，越应该被剪枝

            # 1. 透明度分数（透明度越低，分数越高）
            opacity = self.get_opacity.squeeze()
            opacity_score = 1.0 - opacity  # 透明度越低，分数越高

            # 2. 尺寸分数（尺寸越大，分数越高）
            size_score = self.get_scaling.max(dim=1).values / (max_size_ratio * extent)
            size_score = size_score.clamp(max=2.0)

            # 3. 屏幕空间大小分数（投影越大，分数越高）
            screen_score = torch.zeros_like(opacity_score)
            if hasattr(self, 'max_radii2D') and self.max_radii2D is not None:
                screen_score = self.max_radii2D / prune_screen_size
                screen_score = screen_score.clamp(max=2.0)

            # 4. 梯度分数（可选，梯度越小，分数越高）
            grad_score = torch.zeros_like(opacity_score)
            if use_gradient and hasattr(self, 'xyz_gradient_accum') and self.denom is not None:
                grad_norm = torch.norm(self.xyz_gradient_accum / (self.denom + 1e-8), dim=-1)
                grad_score = 1.0 / (grad_norm + 1e-8)  # 梯度越小，分数越高
                grad_score = grad_score.clamp(max=2.0)

            # 综合质量分数（加权和）
            quality_score = (
                    opacity_score * 2.0 +  # 透明度权重最高
                    size_score * 1.0 +
                    screen_score * 1.0 +
                    grad_score * 0.5
            )

            # 找到质量分数最高的 need_prune 个点（最应该被剪枝的）
            _, indices_to_prune = torch.topk(quality_score, min(need_prune, len(quality_score)), largest=True)

            # 创建剪枝mask
            force_prune_mask = torch.zeros(after_regular, dtype=bool, device="cuda")
            force_prune_mask[indices_to_prune] = True

            # 执行强制剪枝
            self.prune_points(force_prune_mask)

            after_force = self.get_xyz.shape[0]
            print(f"Forced pruning removed {after_regular - after_force} points to reach target {target_points}")

        # 最终剪枝数量
        total_pruned = before_prune - self.get_xyz.shape[0]

        return total_pruned

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

        return before - self.get_xyz.shape[0]

    # ============== 第一阶段：模糊区域处理（只分裂，不剪枝） ==============

    def densify_semantic_regions(self, points_mask, args, extent, densify_factor=2.0):
        """
        对模糊区域进行增密（第一阶段）
        - 只分裂，不剪枝
        - 基于梯度选择候选点

        Args:
            points_mask: 布尔张量，表示哪些高斯点需要增密
            args: 优化参数
            extent: 场景范围
            densify_factor: 增密强度因子（固定为2.0）

        Returns:
            net_change: 净增点数
        """
        if points_mask is None or points_mask.sum() == 0:
            return 0

        before_densify = self.get_xyz.shape[0]

        print(f"Phase 1 - Semantic densification: {points_mask.sum().item()} candidate points")

        if not hasattr(self, 'tmp_radii') or self.tmp_radii is None:
            self.tmp_radii = torch.zeros(self.get_xyz.shape[0], device="cuda")

        # 只使用分裂（第一阶段只需要分裂）
        current_num_points = self.get_xyz.shape[0]
        if points_mask.shape[0] != current_num_points:
            if points_mask.shape[0] < current_num_points:
                new_mask = torch.zeros(current_num_points, dtype=bool, device="cuda")
                new_mask[:points_mask.shape[0]] = points_mask
                points_mask = new_mask
            else:
                points_mask = points_mask[:current_num_points]

        # 获取所有候选点中尺寸大的点（需要分裂的）
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense * extent
        semantic_splits_all = torch.logical_and(split_qualifiers, points_mask)

        if semantic_splits_all.sum() == 0:
            print(f"  - No split candidates found")
            return 0

        # 获取所有分裂候选点的索引
        split_indices = torch.where(semantic_splits_all)[0]

        # 计算梯度（使用累积梯度）
        if hasattr(self, 'xyz_gradient_accum') and self.denom is not None:
            grad_norm = torch.norm(self.xyz_gradient_accum[split_indices] /
                                   (self.denom[split_indices] + 1e-8), dim=-1)
        else:
            # 如果没有梯度信息，随机选择
            grad_norm = torch.ones(len(split_indices), device="cuda")

        # 按梯度降序排序（梯度大的优先分裂）
        sorted_indices = split_indices[torch.argsort(grad_norm, descending=True)]

        # 所有候选点都进行分裂
        num_to_split = len(sorted_indices)

        # 创建分裂mask
        split_mask = torch.zeros_like(points_mask)
        split_mask[sorted_indices] = True

        print(f"  - Split candidates: {semantic_splits_all.sum()} points, "
              f"splitting all {num_to_split} points")

        # 执行分裂（不剪枝）
        self._densify_split_semantic_no_prune(split_mask, densify_factor)
        print(f"  - Split {split_mask.sum()} points")

        after_all = self.get_xyz.shape[0]
        net_change = after_all - before_densify

        if after_all != before_densify:
            self.tmp_radii = torch.zeros(after_all, device="cuda")
            print(f"  - Net change: {net_change} points (now: {after_all})")

        return net_change

    # ============== 第二阶段：差异区域处理（分裂+克隆，无剪枝） ==============

    def refine_difference_regions(self, points_mask, args, extent, densify_factor=2.0, prune_after=False,
                                  target_points=None):
        """
        对差异区域进行局部细化（第二阶段）
        - 大尺寸点：分裂
        - 小尺寸点：克隆（根据梯度决定数量）
        - 无剪枝（prune_after固定为False，只保留target_points限制）
        - 如果提供target_points，只会在最后检查并强制剪枝到目标点数

        Args:
            points_mask: 布尔张量，表示哪些高斯点需要处理
            args: 优化参数
            extent: 场景范围
            densify_factor: 增密强度因子
            prune_after: 固定为False，表示不执行常规剪枝
            target_points: 目标总点数上限（如果提供且当前点数超过此值，则强制剪枝到此数值）

        Returns:
            net_change: 净增点数
        """
        if points_mask is None or points_mask.sum() == 0:
            return 0

        before_densify = self.get_xyz.shape[0]
        total_candidates = points_mask.sum().item()
        print(f"Phase 2 - Refining difference regions: {total_candidates} candidate points")

        if not hasattr(self, 'tmp_radii') or self.tmp_radii is None:
            self.tmp_radii = torch.zeros(self.get_xyz.shape[0], device="cuda")

        # 对齐mask维度
        current_num_points = self.get_xyz.shape[0]
        if points_mask.shape[0] != current_num_points:
            if points_mask.shape[0] < current_num_points:
                new_mask = torch.zeros(current_num_points, dtype=bool, device="cuda")
                new_mask[:points_mask.shape[0]] = points_mask
                points_mask = new_mask
            else:
                points_mask = points_mask[:current_num_points]

        # 获取差异区域内的所有点索引
        diff_indices = torch.where(points_mask)[0]

        if len(diff_indices) == 0:
            return 0

        # 获取这些点的尺度，区分大尺寸和小尺寸
        scales = torch.max(self.get_scaling[diff_indices], dim=1).values
        split_qualifiers = scales > args.dense * extent  # 大尺寸 -> 分裂
        clone_qualifiers = ~split_qualifiers  # 小尺寸 -> 克隆

        total_split = 0
        total_clone = 0

        # 保存分裂后需要更新的信息
        split_done = False
        split_count = 0

        # ===== 第一步：处理大尺寸点（分裂） =====
        if split_qualifiers.sum() > 0:
            split_indices = diff_indices[split_qualifiers]
            split_mask = torch.zeros_like(points_mask)
            split_mask[split_indices] = True

            print(f"  - Split candidates: {len(split_indices)} large points")
            self._densify_split_semantic(split_mask, densify_factor)  # 注意：这个函数内部会剪枝原始点
            total_split = split_mask.sum().item()
            split_done = True
            split_count = len(split_indices)
            print(f"    Split {total_split} points")

        # ===== 第二步：处理小尺寸点（克隆，按梯度决定数量） =====
        if clone_qualifiers.sum() > 0:
            # 如果已经执行了分裂，需要重新获取当前的点索引
            if split_done:
                # 重新计算clone_indices（因为点数已经变了）
                current_num_points = self.get_xyz.shape[0]
                original_clone_indices = diff_indices[clone_qualifiers]
                # 检查哪些索引仍然有效（小于当前点数）
                valid_mask = original_clone_indices < current_num_points
                valid_clone_indices = original_clone_indices[valid_mask]

                if len(valid_clone_indices) > 0:
                    # 计算这些点的梯度
                    if hasattr(self, 'xyz_gradient_accum') and self.denom is not None:
                        grad_norm = torch.norm(self.xyz_gradient_accum[valid_clone_indices] /
                                               (self.denom[valid_clone_indices] + 1e-8), dim=-1)
                        if grad_norm.max() > 0:
                            grad_norm = grad_norm / grad_norm.max()
                    else:
                        grad_norm = torch.ones(len(valid_clone_indices), device="cuda") * 0.5

                    # 根据梯度决定克隆数量
                    clone_counts = (grad_norm * 2 + 1).int()  # 范围：1-3

                    # 逐点克隆
                    total_clones_this_round = 0
                    for idx, count in zip(valid_clone_indices, clone_counts):
                        if count > 0:
                            mask = torch.zeros(current_num_points, dtype=bool, device="cuda")
                            mask[idx] = True
                            self._densify_clone_semantic(mask, densify_factor=count.item())
                            total_clones_this_round += count.item()

                    total_clone = total_clones_this_round
                    print(f"  - Clone candidates: {len(valid_clone_indices)} small points, "
                          f"cloned {total_clone} copies (adaptive based on gradient)")
            else:
                # 没有分裂，直接用原来的indices
                clone_indices = diff_indices[clone_qualifiers]

                # 计算梯度分数
                if hasattr(self, 'xyz_gradient_accum') and self.denom is not None:
                    grad_norm = torch.norm(self.xyz_gradient_accum[clone_indices] /
                                           (self.denom[clone_indices] + 1e-8), dim=-1)
                    if grad_norm.max() > 0:
                        grad_norm = grad_norm / grad_norm.max()
                else:
                    grad_norm = torch.ones(len(clone_indices), device="cuda") * 0.5

                # 根据梯度决定克隆数量
                clone_counts = (grad_norm * 2 + 1).int()  # 范围：1-3

                # 逐点克隆
                total_clones_this_round = 0
                for idx, count in zip(clone_indices, clone_counts):
                    if count > 0:
                        mask = torch.zeros_like(points_mask)
                        mask[idx] = True
                        self._densify_clone_semantic(mask, densify_factor=count.item())
                        total_clones_this_round += count.item()

                total_clone = total_clones_this_round
                print(f"  - Clone candidates: {len(clone_indices)} small points, "
                      f"cloned {total_clone} copies (adaptive based on gradient)")

        # ===== 第三步：跳过常规剪枝，只检查目标点数限制 =====
        # 注意：prune_after参数被忽略，我们固定为False
        after_densify = self.get_xyz.shape[0]
        
        # 如果提供了target_points且当前点数超过目标，强制剪枝
        if target_points is not None and after_densify > target_points:
            print(f"  - Current points {after_densify} exceeds target {target_points}, enforcing limit")
            
            # 使用prune_low_quality的强制剪枝功能，但不进行常规剪枝
            # 我们创建一个临时mask，标记所有点，然后让prune_low_quality只执行target_points限制
            pruned = self.prune_low_quality(
                min_opacity=0.0,  # 不基于透明度剪枝
                max_size_ratio=1.0,  # 不基于尺寸剪枝
                extent=extent,
                prune_screen_size=1000,  # 不基于屏幕大小剪枝
                target_points=target_points  # 只使用目标点数限制
            )
            if pruned > 0:
                print(f"  - Enforced target points limit, removed {pruned} points")
        else:
            # 完全不执行任何剪枝
            pass

        # 计算净变化
        after_all = self.get_xyz.shape[0]
        net_change = after_all - before_densify

        if after_all != before_densify:
            self.tmp_radii = torch.zeros(after_all, device="cuda")
            print(f"  - Net change: {net_change} points (now: {after_all})")

        return net_change

    # ============== 基础增密函数 ==============

    def _densify_clone_semantic(self, mask, densify_factor=2.0):
        """克隆语义区域内的点"""
        N = int(densify_factor)

        # 确保mask的维度匹配
        if mask.shape[0] != self._xyz.shape[0]:
            if mask.shape[0] < self._xyz.shape[0]:
                new_mask = torch.zeros(self._xyz.shape[0], dtype=bool, device="cuda")
                new_mask[:mask.shape[0]] = mask
                mask = new_mask
            else:
                mask = mask[:self._xyz.shape[0]]

        if mask.sum() == 0:
            return

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
        """分裂语义区域内的点（每个点分裂成2个，并剪枝原始点）"""
        N = 2

        # 确保mask的维度匹配
        if mask.shape[0] != self._xyz.shape[0]:
            if mask.shape[0] < self._xyz.shape[0]:
                new_mask = torch.zeros(self._xyz.shape[0], dtype=bool, device="cuda")
                new_mask[:mask.shape[0]] = mask
                mask = new_mask
            else:
                mask = mask[:self._xyz.shape[0]]

        if mask.sum() == 0:
            return

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

    def _densify_split_semantic_no_prune(self, mask, densify_factor=2.0):
        """分裂语义区域内的点（每个点分裂成2个，但保留原始点 - 用于第一阶段）"""
        N = 2

        # 确保mask的维度匹配
        if mask.shape[0] != self._xyz.shape[0]:
            if mask.shape[0] < self._xyz.shape[0]:
                new_mask = torch.zeros(self._xyz.shape[0], dtype=bool, device="cuda")
                new_mask[:mask.shape[0]] = mask
                mask = new_mask
            else:
                mask = mask[:self._xyz.shape[0]]

        if mask.sum() == 0:
            return

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

        # 不剪枝原始点

    # ============== 原有的 densify_and_prune_fastgs ==============

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

        # 使用带目标点数的剪枝
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

    # ============== 修复后的 add_densification_stats 方法 ==============

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        添加稠密化统计信息
        当维度不匹配时，重置统计信息以匹配当前点数
        """
        # 确保update_filter是一维的
        if update_filter.dim() > 1:
            update_filter = update_filter.squeeze()

        # 确保update_filter是布尔类型
        if update_filter.dtype != torch.bool:
            update_filter = update_filter.bool()

        current_num_points = self.get_xyz.shape[0]
        filter_num_points = update_filter.shape[0]

        # 处理update_filter维度不匹配
        if filter_num_points != current_num_points:
            if filter_num_points < current_num_points:
                # 用False填充
                new_filter = torch.zeros(current_num_points, dtype=bool, device="cuda")
                new_filter[:filter_num_points] = update_filter
                update_filter = new_filter
            else:
                # 截取前current_num_points个
                update_filter = update_filter[:current_num_points]

        # 处理梯度维度不匹配
        if viewspace_point_tensor.grad is None:
            return

        grad = viewspace_point_tensor.grad
        if grad.shape[0] != current_num_points:
            # 创建新的梯度张量
            new_grad = torch.zeros((current_num_points, grad.shape[1]),
                                   device=grad.device, dtype=grad.dtype)

            # 复制旧梯度（如果有点）
            if grad.shape[0] > 0:
                copy_len = min(grad.shape[0], current_num_points)
                new_grad[:copy_len] = grad[:copy_len]

            # 使用新的梯度进行计算
            grad = new_grad

        # 确保缓冲区大小匹配
        if self.xyz_gradient_accum.shape[0] != current_num_points:
            new_accum = torch.zeros((current_num_points, 1), device="cuda")
            copy_len = min(self.xyz_gradient_accum.shape[0], current_num_points)
            new_accum[:copy_len] = self.xyz_gradient_accum[:copy_len]
            self.xyz_gradient_accum = new_accum

        if self.xyz_gradient_accum_abs.shape[0] != current_num_points:
            new_accum_abs = torch.zeros((current_num_points, 1), device="cuda")
            copy_len = min(self.xyz_gradient_accum_abs.shape[0], current_num_points)
            new_accum_abs[:copy_len] = self.xyz_gradient_accum_abs[:copy_len]
            self.xyz_gradient_accum_abs = new_accum_abs

        if self.denom.shape[0] != current_num_points:
            new_denom = torch.zeros((current_num_points, 1), device="cuda")
            copy_len = min(self.denom.shape[0], current_num_points)
            new_denom[:copy_len] = self.denom[:copy_len]
            self.denom = new_denom

        if self.max_radii2D.shape[0] != current_num_points:
            new_max = torch.zeros(current_num_points, device="cuda")
            copy_len = min(self.max_radii2D.shape[0], current_num_points)
            new_max[:copy_len] = self.max_radii2D[:copy_len]
            self.max_radii2D = new_max

        # 更新统计信息
        if update_filter.any():
            try:
                self.xyz_gradient_accum[update_filter] += torch.norm(
                    grad[update_filter, :2], dim=-1, keepdim=True
                )
                self.xyz_gradient_accum_abs[update_filter] += torch.norm(
                    grad[update_filter, 2:], dim=-1, keepdim=True
                )
                self.denom[update_filter] += 1
            except RuntimeError as e:
                print(f"Error in add_densification_stats:")
                print(f"  - update_filter shape: {update_filter.shape}, sum: {update_filter.sum().item()}")
                print(f"  - current_num_points: {current_num_points}")
                print(f"  - grad shape: {grad.shape}")
                print(f"  - xyz_gradient_accum shape: {self.xyz_gradient_accum.shape}")
                raise e

    def final_prune_fastgs(self, min_opacity, pruning_score=None):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)

    # ============== 新增：强制同步所有缓冲区的方法 ==============

    def force_sync_buffers(self):
        """强制同步所有缓冲区到当前点数"""
        current_num = self.get_xyz.shape[0]

        # 同步xyz_gradient_accum
        if self.xyz_gradient_accum.shape[0] != current_num:
            new_accum = torch.zeros((current_num, 1), device="cuda")
            copy_len = min(self.xyz_gradient_accum.shape[0], current_num)
            new_accum[:copy_len] = self.xyz_gradient_accum[:copy_len]
            self.xyz_gradient_accum = new_accum

        # 同步xyz_gradient_accum_abs
        if self.xyz_gradient_accum_abs.shape[0] != current_num:
            new_accum_abs = torch.zeros((current_num, 1), device="cuda")
            copy_len = min(self.xyz_gradient_accum_abs.shape[0], current_num)
            new_accum_abs[:copy_len] = self.xyz_gradient_accum_abs[:copy_len]
            self.xyz_gradient_accum_abs = new_accum_abs

        # 同步denom
        if self.denom.shape[0] != current_num:
            new_denom = torch.zeros((current_num, 1), device="cuda")
            copy_len = min(self.denom.shape[0], current_num)
            new_denom[:copy_len] = self.denom[:copy_len]
            self.denom = new_denom

        # 同步max_radii2D
        if self.max_radii2D.shape[0] != current_num:
            new_max = torch.zeros(current_num, device="cuda")
            copy_len = min(self.max_radii2D.shape[0], current_num)
            new_max[:copy_len] = self.max_radii2D[:copy_len]
            self.max_radii2D = new_max

        # 同步tmp_radii
        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            if self.tmp_radii.shape[0] != current_num:
                new_tmp = torch.zeros(current_num, device="cuda")
                copy_len = min(self.tmp_radii.shape[0], current_num)
                new_tmp[:copy_len] = self.tmp_radii[:copy_len]
                self.tmp_radii = new_tmp

    # ============== 新增：安全更新 max_radii2D 的方法 ==============

    def safe_update_max_radii2D(self, visibility_filter, radii):
        """
        安全地更新 max_radii2D，处理所有可能的维度不匹配
        """
        current_num = self.get_xyz.shape[0]

        # 确保 max_radii2D 大小正确
        if self.max_radii2D.shape[0] != current_num:
            new_max = torch.zeros(current_num, device=self._xyz.device)
            copy_len = min(self.max_radii2D.shape[0], current_num)
            new_max[:copy_len] = self.max_radii2D[:copy_len]
            self.max_radii2D = new_max

        # 处理 visibility_filter
        if visibility_filter.shape[0] != current_num:
            new_filter = torch.zeros(current_num, dtype=bool, device=self._xyz.device)
            copy_len = min(visibility_filter.shape[0], current_num)
            new_filter[:copy_len] = visibility_filter[:copy_len]
            visibility_filter = new_filter

        # 处理 radii
        if radii.shape[0] != current_num:
            if radii.shape[0] < current_num:
                new_radii = torch.cat([radii, torch.zeros(current_num - radii.shape[0], device=self._xyz.device)])
            else:
                new_radii = radii[:current_num]
            radii = new_radii

        # 安全更新
        if visibility_filter.any():
            visible_indices = torch.where(visibility_filter)[0]
            self.max_radii2D[visible_indices] = torch.max(
                self.max_radii2D[visible_indices],
                radii[visible_indices]
            )