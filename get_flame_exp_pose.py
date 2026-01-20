import os
import os.path as osp
import math
import pickle
import imageio
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import moviepy.editor as mp
from ffmpy import FFmpeg

import torch
import torch.nn.functional as F

from threestudio.utils.typing import *
from threestudio.utils.head_v2 import FlamePointswRandomExp

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel

device = torch.device('cuda')


def get_c2w(dist, elev, azim, device=torch.device('cuda')):
    elev = elev * math.pi / 180
    azim = azim * math.pi / 180
    batch_size = dist.shape[0]
    camera_positions: Float[Tensor, "B 3"] = torch.stack(
        [
            dist * torch.cos(elev) * torch.cos(azim),
            dist * torch.cos(elev) * torch.sin(azim),
            dist * torch.sin(elev),
        ],
        dim=-1
    ).to(device)
    center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions, device=device)
    up: Float[Tensor, "B 3"] = torch.as_tensor(
        [0, 0, 1], dtype=torch.float32, device=device)[None, :].repeat(batch_size, 1)
    lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
    right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w: Float[Tensor, "B 4 4"] = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1], device=device)], dim=1
    )
    c2w[:, 3, 3] = 1.0
    return c2w


class Avatar:
    def __init__(self, ply_path, gender="generic"):
        self.ply_path = ply_path

        gaussian = GaussianFlameUVModel(sh_degree=0, model_folder=flame_path)
        skel = FlamePointswRandomExp(
            flame_path,
            gender=gender,
            device="cuda",
            batch_size=1,
            flame_scale=-10
        )
        cameras_extent = 4.0
        flame_scale = -10.0
        gaussian.create_from_flame(cameras_extent, flame_scale)
        gaussian.load_ply(ply_path)

        self.black_background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.renderbackground = self.black_background
        parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(parser)

        self.gaussian = gaussian
        self.skel = skel
        self.skel.betas = self.gaussian.get_shape.detach()

    def get_cond(self, dist, elev_deg, azim_deg, fovy_deg, expression, jaw_pose, neck_pose, leye_pose, reye_pose,
                 at=torch.tensor(((0, 0, 0),)), up=torch.tensor(((0, 0, 1),))):
        at = at.to(torch.float)
        up = up.to(torch.float)
        # mesh_vis = True -> 渲染FLAME；
        # lmk = True & mediapipe = Ture -> 渲染Mediapipe
        flame_depths = self.skel.get_cond(
            dist, elev_deg, azim_deg, at, up, fovy_deg, expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose,
            leye_pose=leye_pose, reye_pose=reye_pose, # lmk=True, mediapipe=True,
            mesh_vis=True
        )
        return flame_depths

    def get_depth(self, dist, elev_deg, azim_deg, fovy_deg, expression, jaw_pose, neck_pose, leye_pose, reye_pose,
                 at=torch.tensor(((0, 0, 0),)), up=torch.tensor(((0, 0, 1),))):
        at = at.to(torch.float)
        up = up.to(torch.float)
        # mesh_vis = True -> 渲染FLAME；
        # lmk = True & mediapipe = Ture -> 渲染Mediapipe
        flame_depths = self.skel.get_cond(
            dist, elev_deg, azim_deg, at, up, fovy_deg, expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose,
            leye_pose=leye_pose, reye_pose=reye_pose, lmk=True, mediapipe=True,
        )
        return flame_depths

    def get_camera(self, dist, elev, azim, fovy_deg=70.0):
        c2w = get_c2w(dist=dist, elev=elev, azim=azim)
        fovy_deg = torch.full_like(elev, fovy_deg)
        fovy = fovy_deg * math.pi / 180
        height = 1024
        width = 1024
        viewpoint_cam = Camera(c2w=c2w[0], FoVy=fovy[0], height=height, width=width)
        return viewpoint_cam

    def set_pose(self, expression, jaw_pose, leye_pose=None, reye_pose=None, neck_pose=None):
        self.gaussian._expression = expression.detach()
        self.gaussian._jaw_pose = jaw_pose.detach()
        if leye_pose is not None:
            self.gaussian._leye_pose = leye_pose.detach()
        if reye_pose is not None:
            self.gaussian._reye_pose = reye_pose.detach()
        self.gaussian._neck_pose = neck_pose.detach()

    def render_mesh(self, dist, elev, azim, expression, jaw_pose, neck_pose, leye_pose, reye_pose, fovy_deg=70.0):
        fovy_deg = torch.full_like(elev, fovy_deg)
        mesh = self.get_cond(dist, elev, azim, fovy_deg, expression, jaw_pose, neck_pose, leye_pose, reye_pose,)
        return mesh

    def render(self, dist, elev, azim):
        viewpoint_cam = self.get_camera(dist, elev, azim)
        render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, self.renderbackground)
        image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
        image = image.permute(1, 2, 0)
        return image


class NerSemble:
    def __init__(self, exp_path, pose_path):
        self.exp_path = exp_path
        self.pose_path = pose_path
        self.expression = torch.from_numpy(np.load(self.exp_path)).to('cuda')
        self.neck_pose = torch.from_numpy(np.load(self.pose_path))[:, 3:6].to('cuda')
        self.jaw_pose = torch.from_numpy(np.load(self.pose_path))[:, 6:9].to('cuda')
        self.leye_pose = torch.from_numpy(np.load(self.pose_path))[:, 9:12].to('cuda')
        self.reye_pose = torch.from_numpy(np.load(self.pose_path))[:, 12:15].to('cuda')

        n_exp = self.expression.shape[0]

        self.n_frames = n_exp
        azimuth_deg = torch.linspace(60., 120, 180)
        azimuth_deg = azimuth_deg.repeat((self.n_frames + 180 - 1) // 180)[:self.n_frames]  # 循环填充到长度 n
        elevation_deg = torch.full_like(azimuth_deg, 15.0)
        camera_distances = torch.full_like(elevation_deg, 2.0)
        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, index):
        return {
            'expression': self.expression[index: index + 1],
            'jaw_pose': self.jaw_pose[index: index + 1],
            'leye_pose': self.leye_pose[index: index + 1],
            'reye_pose': self.reye_pose[index: index + 1],
            'neck_pose': self.neck_pose[index: index + 1],
            'elev': self.elevation_deg[index: index + 1],
            'azim': self.azimuth_deg[index: index + 1],
            'dist': self.camera_distances[index: index + 1],
        }


def batch_gs_render(flame_sequences, save_path, static=False, dynamic=False, eye_pose=False):
    images = []
    n_views = len(flame_sequences)
    imgs_path = osp.join(save_path, 'imgs')
    flame_path = osp.join(save_path, 'flame')
    os.makedirs(imgs_path, exist_ok=True)
    os.makedirs(flame_path, exist_ok=True)
    if static:
        n_views = 120
        azims = torch.linspace(60., 120.0, n_views + 1)[: n_views]
    if dynamic:
        azims = torch.linspace(60., 120.0, n_views + 1)[: n_views]
    for i in tqdm(range(n_views)):
        s = flame_sequences[i]

        expression = s['expression']
        jaw_pose = s['jaw_pose']
        neck_pose = s['neck_pose']
        leye_pose = s['leye_pose']
        reye_pose = s['reye_pose']
        elev = s['elev']
        azim = s['azim']
        dist = s['dist']
        if not eye_pose:
            leye_pose = None
            reye_pose = None
        avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose,
                        reye_pose=reye_pose)
        image = avatar.render(dist, elev, azim)

        image = image.detach().cpu().numpy()
        image = image.clip(min=0, max=1)
        image = (image * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(imgs_path, f"{i:06d}.png"), image)

        flame_conds = avatar.get_depth(dist, elev, azim, 70.0, expression, jaw_pose, neck_pose, leye_pose, reye_pose)[0]
        flame_conds = flame_conds.detach().cpu().numpy()
        flame_conds = flame_conds.clip(min=0, max=1)
        flame_conds = (flame_conds * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(flame_path, f"flame_{i:06d}.png"), flame_conds)

        # images.append(image)
    # images = np.stack(images, axis=0)
    return images


def batch_mesh_render(flame_sequences, static=False, dynamic=False):
    images = []
    n_views = len(flame_sequences)
    # n_views = 30
    if static:
        azims = torch.linspace(-180., 180.0, n_views + 1)[: n_views]
    if dynamic:
        azims = torch.linspace(60., 120.0, n_views + 1)[: n_views]

    for i in tqdm(range(len(flame_sequences))):
        s = flame_sequences[i]

        expression = s['expression']
        jaw_pose = s['jaw_pose']
        neck_pose = s['neck_pose']
        leye_pose = s['leye_pose']
        reye_pose = s['reye_pose']
        elev = s['azim']
        azim = s['azim']
        dist = s['dist']

        image = avatar.render_mesh(dist, elev, azim, expression, jaw_pose, neck_pose, leye_pose, reye_pose)
        image = image[0].detach().cpu().numpy()
        image = image.clip(min=0, max=1)
        image = (image * 255.0).astype(np.uint8)
        images.append(image)
    images = np.stack(images, axis=0)
    return images


def save_mp4_w_audio(images, tag='gs', w_audio=False):
    def get_model_name(path):
        return path.split("/")[-3]

    def get_talkshow_name(path):
        return osp.splitext(osp.split(path)[1])[0]

    mp4_path = os.path.join(save_path, f"{tag}-{get_model_name(ply_path)}-{get_talkshow_name(exp_path)}.mp4")
    imageio.mimwrite(mp4_path, images, fps=30)

    if w_audio:
        # Add Audio
        audio_path = exp_path.replace(".pkl", ".wav")
        result = os.path.join(save_path,
                              f"{tag}-{get_model_name(ply_path)}-{get_talkshow_name(exp_path)}-audio.mp4")
        _codec = 'aac'
        ff = FFmpeg(inputs={mp4_path: None, audio_path: None},
                    outputs={result: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
        print(ff.cmd)
        ff.run()


if __name__ == '__main__':
    # dist = torch.tensor([2.0, ], dtype=torch.float, device=device)
    # elev = torch.tensor([15.0, ], dtype=torch.float, device=device)
    # azim = torch.tensor([90, ], dtype=torch.float, device=device)

    # path to flame folder, e.g. flame_path = "/path/to/FLAME-2020"
    flame_path = 'ckpts/FLAME-2020'

    exp_path = 'assets/exp.npy'
    pose_path = 'assets/pose.npy'

    ply_path = 'outputs/headstudio/20260119-071629/save/step_5000.ply'
    root_dir = os.path.dirname(os.path.dirname(os.path.normpath(ply_path)))
    save_path = os.path.join(root_dir, 'nersamble')
    os.makedirs(save_path, exist_ok=True)

    nersamble = NerSemble(exp_path, pose_path)
    avatar = Avatar(ply_path)

    mesh_render = False

    # Render the dynamic avatar with fix front view (azim=90 degree): static=False, dynamic=False
    # Render the static avatar from azim 60 degree to 90 degre: static=True, dynamic=False
    images = batch_gs_render(nersamble, save_path, eye_pose=True)
    # save_mp4_w_audio(images, tag='gs-static', w_audio=False)

    if mesh_render:
        images = batch_mesh_render(nersamble)
        # save_mp4_w_audio(images, tag='mesh')
