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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, init_ply_path=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param args: ModelParams object containing dataset parameters
        :param gaussians: GaussianModel object to be initialized
        :param load_iteration: If specified, loads a model checkpoint at that iteration
        :param init_ply_path: If specified, loads initial point cloud from this PLY file instead of using default initialization
        :param shuffle: If True, shuffles the camera order
        :param resolution_scales: List of resolution scales to load cameras at
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # 修改：如果提供了init_ply_path，使用指定的PLY文件
            if init_ply_path and os.path.exists(init_ply_path):
                print(f"Using specified initial point cloud from: {init_ply_path}")
                # 复制指定的PLY文件到模型路径
                with open(init_ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                # 如果没有指定init_ply_path，使用随机初始化，不需要复制PLY文件
                if init_ply_path and not os.path.exists(init_ply_path):
                    print(f"Warning: Specified PLY file {init_ply_path} does not exist. Using random initialization.")
                else:
                    print("No initial PLY provided. Will use random initialization.")
                # 创建一个空的input.ply标记文件（可选）
                with open(os.path.join(self.model_path, "input_random_init.txt"), 'w') as f:
                    f.write("Random initialization used")
            
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 修改：根据是否提供了init_ply_path来决定使用哪个点云
            if init_ply_path and os.path.exists(init_ply_path):
                # 使用指定的PLY文件创建高斯模型
                print(f"Creating Gaussians from specified PLY: {init_ply_path}")
                self.gaussians.create_from_pcd(init_ply_path, self.cameras_extent)
            else:
                # 使用随机初始化创建高斯模型
                print("=" * 50)
                print("No initial PLY provided. Creating random initial Gaussian model with 100 points.")
                print("This is normal for training from scratch.")
                print("=" * 50)
                self.gaussians.create_random_initialization(num_points=100, spatial_lr_scale=self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]