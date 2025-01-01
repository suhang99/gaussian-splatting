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
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d, CubicSpline
import mediapy as media
from PIL import Image
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import copy
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def write_tum_traj(traj_path, tum_traj):
    with open(traj_path, "w") as f:
        f.write("# image_id tx ty tz qx qy qz qw\n")
        for line in tum_traj:
            f.write(" ".join(map(str, line)) + "\n")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, interpol_step: int, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "traj", "ours_{}".format(iteration), "renders")
        traj_path = os.path.join(dataset.model_path, "traj", "ours_{}".format(iteration))

        train_views = scene.getTrainCameras()
        test_views = scene.getTestCameras()
        llffhold = 8
        views = []
        train_id, test_id= 0, 0
        # Drop the last 10% views (since their trajectory is not smooth)
        for view_id in range(int((len(train_views) + len(test_views)) * 0.9)):
            if view_id % llffhold == 0:
                if test_id < len(test_views):
                    views.append(test_views[test_id])
                    test_id += 1
            else:
                if train_id < len(train_views):
                    views.append(train_views[train_id])
                    train_id += 1
        tum_traj = []

        makedirs(traj_path, exist_ok=True)
        
        for i, view in enumerate(views):
            R_Cam_World = view.R
            t_Cam_World = view.T
            T_Cam_World = np.eye(4)
            T_Cam_World[:3, :3] = R_Cam_World
            T_Cam_World[:3, 3] = t_Cam_World
            T_World_Cam = np.linalg.inv(T_Cam_World)
            R_World_Cam = T_World_Cam[:3, :3]
            t_World_Cam = T_World_Cam[:3, 3]
            qx, qy, qz, qw = map(float, Rotation.from_matrix(R_World_Cam).as_quat())
            tx, ty, tz = map(float, t_World_Cam)
            tum_traj.append([i, tx, ty, tz, qx, qy, qz, qw])

        tum_traj.sort(key=lambda x: x[0])        
        write_tum_traj(traj_path + "/traj.tum", tum_traj)

        makedirs(render_path, exist_ok=True)
        
        # Perform pose interpolation
        translation_interp = CubicSpline(np.arange(len(views)), np.array([view.T for view in views]), axis=0)
        rotation_interp = Slerp(np.arange(len(views)), Rotation.from_matrix([view.R for view in views]))

        # Compute interpolate times
        interp_times = []
        for i in range(len(views)-1):
            num_interp = int(np.linalg.norm(views[i].T - views[i+1].T) / interpol_step) + 1
            step = 1.0 / num_interp
            interp_times += list(np.arange(i, i + 1.0, step))
        interp_times.append(len(views)-1)
        
        # Get interpolated poses
        T_interpolated = translation_interp(interp_times)
        R_interpolated = rotation_interp(interp_times)

        # num_frames = 60
        num_frames = len(interp_times)

        for i in tqdm(range(num_frames), desc="Rendering progress"):
            view = copy.deepcopy(views[0])
            view.R = R_interpolated[i].as_matrix()
            view.T = T_interpolated[i]
            # Don't know why this works, but it does
            Transformation = np.eye(4)
            Transformation[:3, :3] = view.R
            Transformation[3, :3] = view.T
            view.world_view_transform = torch.from_numpy(Transformation).float().cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3,:3]
            
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh)["render"]
            if args.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]

            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))

        # Create video
        video_path = os.path.join(traj_path, "render_traj.mp4")
        video_kwargs = {
            'shape': (views[0].image_height, views[0].image_width),
            'codec': 'h264',
            'fps': 60,
            'crf': 18,
        }
        with media.VideoWriter(video_path, **video_kwargs, input_format="rgb") as writer:
            for i in tqdm(range(num_frames), desc="Exporting video"):
                image_file = os.path.join(render_path, '{0:05d}'.format(i) + ".png")
                with open(image_file, 'rb') as f:
                    image = np.array(Image.open(f), dtype=np.float32) / 255
                frame = (np.clip(np.nan_to_num(image), 0., 1.) * 255.).astype(np.uint8)
                writer.add_image(frame)
        print("Video saved to", video_path)
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--interp_step", default=0.05, type=int, help="Interpolation step between views in meters")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.interp_step, SPARSE_ADAM_AVAILABLE)