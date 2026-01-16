import math
import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

import threestudio
from animation import get_c2w
from gaussiansplatting.arguments import OptimizationParams
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_flame_uv_tega import GaussianFlameUVModel
from stablediff_finetune_control import StableDiffusion
from threestudio.utils.head_v2 import FlamePointswRandomExp
from threestudio.utils.loss_utils import ssim
from threestudio.utils.perceptual.vgg_feature import VGGPerceptualLoss
from threestudio.utils.typing_ import *

device = torch.device('cuda')


def load_yaml_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def to_cuda(batch: Dict[str, Any], device: torch.device, non_blocking: bool = True) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out

def _uv_unblend_from_uv_render(uv_render: torch.Tensor, eps: float = 1e-6):
    """
    uv_render: (B,3,H,W) where last channel is composited from constant -1 with white bg=1.
              We recover alpha as: alpha = 1 - (ch2 + 1)/2  (same trick as GGHead code)
    return:
        uv_nb: (B,3,H,W)  unblended uv render (background pixels may be inf/nan)
        alpha: (B,1,H,W)  in [0,1]
        bg_mask: (B,1,H,W) alpha==0
    """
    ch2 = uv_render[:, 2:3]  # (B,1,H,W)
    alpha = 1.0 - (ch2 + 1.0) / 2.0
    alpha = alpha.clamp(0.0, 1.0)
    bg_mask = alpha <= eps

    # Undo alpha compositing with white bg=1:
    # uv_nb = (uv_render - (1 - alpha)*1) / alpha
    uv_nb = (uv_render - (1.0 - alpha)) / alpha.clamp_min(eps)
    return uv_nb, alpha, bg_mask

def tv_uv_loss_seam_aware(
    uv_render: torch.Tensor,
    alpha_eps: float = 1e-6,
    uv_jump_thresh: float = 0.25,
) -> torch.Tensor:
    """
    Seam-aware + boundary-aware UV TV loss.
    - Start from GGHead UV-TV: unblend UV render to ignore alpha blending.
    - Additionally ignore pixel pairs whose UV jump is large (likely UV seam / disocclusion boundary).
      This prevents punishing correct UV discontinuities.

    uv_render: (B,3,H,W)
    """
    uv_nb, alpha, bg_mask = _uv_unblend_from_uv_render(uv_render, eps=alpha_eps)
    uv_nb = torch.nan_to_num(uv_nb, nan=0.0, posinf=0.0, neginf=0.0)

    # background adjacency masks (GGHead-style)
    bg_y = (bg_mask[:, :, 1:] | bg_mask[:, :, :-1])  # (B,1,H-1,W)
    bg_x = (bg_mask[:, :, :, 1:] | bg_mask[:, :, :, :-1])  # (B,1,H,W-1)

    # UV jumps (only look at (u,v) channels)
    duv_y = uv_nb[:, :2, 1:] - uv_nb[:, :2, :-1]  # (B,2,H-1,W)
    duv_x = uv_nb[:, :2, :, 1:] - uv_nb[:, :2, :, :-1]  # (B,2,H,W-1)
    jump_y = (duv_y.pow(2).sum(dim=1, keepdim=True).sqrt() > uv_jump_thresh)  # (B,1,H-1,W)
    jump_x = (duv_x.pow(2).sum(dim=1, keepdim=True).sqrt() > uv_jump_thresh)  # (B,1,H,W-1)

    # final masks for TV: exclude background-adjacent and seam-jump pairs
    mask_y = (bg_y | jump_y).repeat(1, 3, 1, 1)  # (B,3,H-1,W)
    mask_x = (bg_x | jump_x).repeat(1, 3, 1, 1)  # (B,3,H,W-1)

    diff_y = uv_nb[:, :, 1:] - uv_nb[:, :, :-1]
    diff_x = uv_nb[:, :, :, 1:] - uv_nb[:, :, :, :-1]

    tv_y = diff_y[~mask_y].abs().mean() if (~mask_y).any() else diff_y.abs().mean() * 0.0
    tv_x = diff_x[~mask_x].abs().mean() if (~mask_x).any() else diff_x.abs().mean() * 0.0
    return tv_x + tv_y

@torch.no_grad()
def splat_rgb_to_uv_atlas(
    rgb: torch.Tensor,
    uv_render: torch.Tensor,
    atlas_res: int = 256,
    alpha_thresh: float = 0.2,
    flip_v: bool = False,
    eps: float = 1e-8,
):
    """
    Build a UV atlas by splatting per-pixel RGB into UV space using uv_render.
    rgb:       (B,3,H,W) in [0,1]  (teacher/pseudo target, e.g. batch['img'])
    uv_render: (B,3,H,W) uv rendering (as produced by render_uv)
    returns:
        atlas_rgb: (B,3,R,R) normalized
        atlas_w:   (B,1,R,R) weights
    Notes:
      - This atlas is a *teacher*; we keep it no-grad to avoid degenerate 'warp UV to cheat' solutions.
      - flip_v depends on your vt convention; if you see vertical flip artifacts in atlas reprojection, set True.
    """
    assert rgb.ndim == 4 and uv_render.ndim == 4
    B, _, H, W = rgb.shape
    device = rgb.device

    uv_nb, alpha, bg_mask = _uv_unblend_from_uv_render(uv_render, eps=1e-6)
    uv = uv_nb[:, :2].clamp(0.0, 1.0)  # (B,2,H,W)
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]

    valid = (alpha > alpha_thresh) & (~bg_mask)  # (B,1,H,W)
    valid = valid.squeeze(1)  # (B,H,W)

    atlas_rgb = torch.zeros((B, 3, atlas_res, atlas_res), device=device, dtype=rgb.dtype)
    atlas_w = torch.zeros((B, 1, atlas_res, atlas_res), device=device, dtype=rgb.dtype)

    for b in range(B):
        m = valid[b]  # (H,W)
        if not m.any():
            continue

        u = uv[b, 0][m]  # (K,)
        v = uv[b, 1][m]  # (K,)
        a = alpha[b, 0][m]  # (K,)
        col = rgb[b, :, m]  # (3,K)

        x = u * (atlas_res - 1)
        y = v * (atlas_res - 1)

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = (x0 + 1).clamp(0, atlas_res - 1)
        y1 = (y0 + 1).clamp(0, atlas_res - 1)
        x0 = x0.clamp(0, atlas_res - 1)
        y0 = y0.clamp(0, atlas_res - 1)

        dx = (x - x0.float()).clamp(0.0, 1.0)
        dy = (y - y0.float()).clamp(0.0, 1.0)

        w00 = (1 - dx) * (1 - dy) * a
        w01 = (1 - dx) * dy * a
        w10 = dx * (1 - dy) * a
        w11 = dx * dy * a

        idx00 = y0 * atlas_res + x0
        idx01 = y1 * atlas_res + x0
        idx10 = y0 * atlas_res + x1
        idx11 = y1 * atlas_res + x1

        rgb_flat = atlas_rgb[b].view(3, -1)
        w_flat = atlas_w[b].view(1, -1)

        for idx, ww in ((idx00, w00), (idx01, w01), (idx10, w10), (idx11, w11)):
            rgb_flat.scatter_add_(1, idx.unsqueeze(0).expand(3, -1), col * ww.unsqueeze(0))
            w_flat.scatter_add_(1, idx.unsqueeze(0), ww.unsqueeze(0))

    atlas_rgb = atlas_rgb / atlas_w.clamp_min(eps)
    return atlas_rgb, atlas_w

def sample_uv_atlas_to_image(
    atlas_rgb: torch.Tensor,
    uv_render: torch.Tensor,
    flip_v: bool = False,
):
    """
    Sample UV atlas back to image plane using per-pixel UV from uv_render.
    atlas_rgb: (B,3,R,R)
    uv_render: (B,3,H,W)
    return: atlas_img (B,3,H,W)
    """
    uv_nb, alpha, bg_mask = _uv_unblend_from_uv_render(uv_render, eps=1e-6)
    uv = uv_nb[:, :2].clamp(0.0, 1.0)  # (B,2,H,W)
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]

    # grid_sample expects grid in [-1,1], shape (B,H,W,2) with order (x,y)=(u,v)
    grid = uv.permute(0, 2, 3, 1) * 2.0 - 1.0
    atlas_img = F.grid_sample(atlas_rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return atlas_img


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred,tgt: (B,3,H,W)
    mask: (B,1,H,W) or (B,H,W) in {0,1}
    """
    if mask.ndim == 3:
        mask = mask[:, None]
    mask = mask.to(dtype=pred.dtype)
    num = (mask.sum() * pred.shape[1]).clamp_min(eps)
    return ((pred - tgt).abs() * mask).sum() / num

class Avatar:
    def __init__(self, cfg, gender="generic"):
        self.ply_path = cfg['ply_path']

        gaussian = GaussianFlameUVModel(sh_degree=0, model_folder=cfg['flame_path'])
        skel = FlamePointswRandomExp(
            cfg['flame_path'],
            gender=gender,
            device="cuda",
            batch_size=1,
            flame_scale=-10
        )
        cameras_extent = 4.0
        flame_scale = -10.0
        gaussian.create_from_flame(cameras_extent, flame_scale)
        gaussian.load_ply(self.ply_path)

        self.black_background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.renderbackground = self.black_background
        parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(parser)

        self.gaussian = gaussian
        self.skel = skel
        self.skel.betas = self.gaussian.get_shape.detach()

        self.parser = ArgumentParser(description="Training script parameters")
        opt = OptimizationParams(self.parser)
        self.gaussian.training_setup(opt)

    def get_cond(self, dist, elev_deg, azim_deg, fovy_deg, expression, jaw_pose, neck_pose,
                 at=torch.tensor(((0, 0, 0),)), up=torch.tensor(((0, 0, 1),))):
        at = at.to(torch.float)
        up = up.to(torch.float)
        # mesh_vis = True -> 渲染FLAME；
        # lmk = True & mediapipe = Ture -> 渲染Mediapipe
        flame_depths = self.skel.get_cond(
            dist, elev_deg, azim_deg, at, up, fovy_deg, expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose,
            # lmk=True, mediapipe=True,
            mesh_vis=True
        )
        return flame_depths

    def get_flame_cond(self, dist, elev_deg, azim_deg, fovy_deg, expression, jaw_pose, neck_pose, leye_pose, reye_pose,
                 at=torch.tensor(((0, 0, 0),)), up=torch.tensor(((0, 0, 1),))):
        at = at.to(torch.float)
        up = up.to(torch.float)
        # mesh_vis = True -> 渲染FLAME
        # lmk = True & mediapipe = Ture -> 渲染Mediapipe
        flame_depths = self.skel.get_cond(
            dist, elev_deg, azim_deg, at, up, fovy_deg, expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose,
            lmk=True, mediapipe=True,
            mesh_vis=True
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
        if neck_pose is not None:
            self.gaussian._neck_pose = neck_pose.detach()

    def render_mesh(self, dist, elev, azim, expression, jaw_pose, neck_pose, fovy_deg=70.0):
        fovy_deg = torch.full_like(elev, fovy_deg)
        mesh = self.get_cond(dist, elev, azim, fovy_deg, expression, jaw_pose, neck_pose)
        return mesh

    def render(self, dist, elev, azim):
        viewpoint_cam = self.get_camera(dist, elev, azim)
        render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, self.renderbackground)
        image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
        image = image.permute(1, 2, 0)
        return image, viewspace_point_tensor, radii

    def render_uv(self, dist, elev, azim):
        """
        Render a UV image R_uv, by overriding per-Gaussian color with (u,v,-1).
        The -1 with white background (1) allows alpha recovery:
            alpha = 1 - (ch2 + 1)/2
        This mirrors the UV rendering idea used in your Head3DGSLKs forward :contentReference[oaicite:4]{index=4}.
        """
        if not hasattr(self.gaussian, "get_uv"):
            raise RuntimeError("Gaussian has no get_uv. Use your UV-parameterized Gaussian model first.")

        viewpoint_cam = self.get_camera(dist, elev, azim)

        uv = self.gaussian.get_uv  # (N,2) in [0,1]
        uv_color = torch.cat([uv, -torch.ones_like(uv[:, :1])], dim=1)  # (N,3)

        # IMPORTANT: use white bg=1 to match alpha recovery math
        pkg_uv = render(viewpoint_cam, self.gaussian, self.pipe, bg_color=self.white_background, override_color=uv_color)
        uv_img = pkg_uv["render"]  # (3,H,W)
        return uv_img


class NerSemble(Dataset):
    def __init__(
            self,
            cfg,
            is_train=True,
            return_image=None,
            img_path_override=None,
    ):
        self.is_train = is_train
        self.return_image = return_image
        if is_train:
            self.exp_path = osp.join(cfg["train_data"], 'exp200.npy')
            self.pose_path = osp.join(cfg["train_data"], 'pose200.npy')
        else:
            self.exp_path = osp.join(cfg["test_data"], 'exp.npy')
            self.pose_path = osp.join(cfg["test_data"], 'pose.npy')
        self.image_path = img_path_override if img_path_override is not None else cfg["img_path"]
        # self.image_path = cfg["img_path"]

        self.expression = torch.from_numpy(np.load(self.exp_path))
        self.neck_pose = torch.from_numpy(np.load(self.pose_path))[:, 3:6]
        self.jaw_pose = torch.from_numpy(np.load(self.pose_path))[:, 6:9]
        self.leye_pose = torch.from_numpy(np.load(self.pose_path))[:, 9:12]
        self.reye_pose = torch.from_numpy(np.load(self.pose_path))[:, 12:15]
        # self.fix_pose_path = 'assets/fix_pose.npy'
        # self.fix_exp = torch.zeros_like(self.expression)[0:1]
        # self.fix_neck = torch.zeros_like(self.neck_pose)[0:1]
        # self.fix_jaw = torch.from_numpy(np.load(self.fix_pose_path))[0:3].unsqueeze(0)
        # self.fix_leye = torch.from_numpy(np.load(self.fix_pose_path))[3:6].unsqueeze(0)
        # self.fix_reye = torch.from_numpy(np.load(self.fix_pose_path))[6:9].unsqueeze(0)

        n_pose = self.expression.shape[0]

        self.n_frames = n_pose

        azimuth_deg = torch.linspace(60., 120, 200)
        azimuth_deg = azimuth_deg.repeat((self.n_frames + 200 - 1) // 200)[:self.n_frames]
        # elevation_deg = torch.full_like(azimuth_deg, 15.0)
        elevation_deg = torch.linspace(5., 25, 200)
        elevation_deg = elevation_deg.repeat((self.n_frames + 200 - 1) // 200)[:self.n_frames]
        perm = torch.randperm(elevation_deg.size(0))  # 生成随机排列的索引
        elevation_shuffled = elevation_deg[perm]
        camera_distances = torch.full_like(elevation_deg, 2.0)
        self.fix_azim = torch.tensor([90.0])
        self.fix_elev = torch.tensor([15.0])

        self.elevation_deg, self.azimuth_deg = elevation_shuffled, azimuth_deg
        self.camera_distances = camera_distances
        self.image_list = []
        self._build_image_index()

    def switch_image_path(self, new_path):
        self.image_path = new_path
        self._build_image_index()

    def _build_image_index(self):

        names = [f for f in os.listdir(self.image_path) if f.lower().endswith('.png')]
        self.image_list = sorted(names, key=lambda x: int(Path(x).stem))
        self.n_img = len(self.image_list)

    def _read_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)      # HWC, BGR, uint8 [0,255]
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> HWC, RGB
        img = img.astype(np.float32) / 255.0
        return img

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx) -> Dict[str, Any]:

        item = {
            'expression': self.expression[idx: idx + 1],
            'jaw_pose':   self.jaw_pose[idx: idx + 1],
            'leye_pose':  self.leye_pose[idx: idx + 1],
            'reye_pose':  self.reye_pose[idx: idx + 1],
            'neck_pose':  self.neck_pose[idx: idx + 1],
            'elev':       self.elevation_deg[idx: idx + 1],
            'azim':       self.azimuth_deg[idx: idx + 1],
            'dist':       self.camera_distances[idx: idx + 1],
            # 'fix_exp': self.fix_exp,
            # 'fix_jaw': self.fix_jaw,
            # 'fix_leye': self.fix_leye,
            # 'fix_reye': self.fix_reye,
            # 'fix_neck': self.fix_neck,
            # 'fix_azim': self.fix_azim,
            # 'fix_elev': self.fix_elev,
            'idx': idx,
        }
        if self.is_train:
            if self.image_list == []:
                return item
            img_name = self.image_list[idx]
            img_fp = osp.join(self.image_path, img_name)
            img_np = self._read_image(img_fp)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            item['img'] = img_t

        return item


class Trainer:
    def __init__(
        self,
        avatar,
        train_loader,
        test_loader=None,
        device=torch.device('cuda'),
        log_every=20,
        log_dir='logs',
        cfg=None,
    ):
        self.avatar = avatar
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_every = log_every
        self.cfg = cfg

        self._vgg_loss = VGGPerceptualLoss().to(self.device)
        self.cameras_extent = 4.0

        self.densify_prune_start_step = cfg['avatar']['densify_prune_start_step']
        self.densify_prune_end_step = cfg['avatar']['densify_prune_end_step']
        self.densify_prune_interval = cfg['avatar']['densify_prune_interval']
        self.prune_only_start_step = cfg['avatar']['prune_only_start_step']
        self.prune_only_end_step = cfg['avatar']['prune_only_end_step']
        self.prune_only_interval = cfg['avatar']['prune_only_interval']
        self.prune_size_threshold = cfg['avatar']['prune_size_threshold']
        self.size_threshold = cfg['avatar']['size_threshold']
        self.size_threshold_fix_step = cfg['avatar']['size_threshold_fix_step']
        self.max_grad = cfg['avatar']['max_grad']

        # diffusion
        self.sd = StableDiffusion(device, False, False, hf_key='../HeadStudio_lib/realistic-vision-51')
        # guidance
        c = {'pretrained_model_name_or_path': '../HeadStudio_lib/realistic-vision-51',
             'control_type': 'mediapipe',
             'min_step_percent': 0.05,
             'max_step_percent': 0.8,
             }
        # self.guidance = threestudio.find(c.system.guidance_type)(c.system.guidance)
        self.guidance = threestudio.find('controlnet-depth-guidance')(c)

        self.writer = SummaryWriter(log_dir=log_dir)

    def update_data(self, path, stage=False):
        os.makedirs(path, exist_ok=True)
        noise = torch.randn(1, 4, 128, 128).to(self.device)
        for batch in self.train_loader:
            batch = to_cuda(batch, self.device)
            B = batch['expression'].shape[0]
            for b in range(B):
                expression = batch['expression'][b]
                jaw_pose = batch['jaw_pose'][b]
                neck_pose = batch['neck_pose'][b]
                leye_pose = batch['leye_pose'][b]
                reye_pose = batch['reye_pose'][b]

                dist, elev, azim = batch['dist'][b], batch['elev'][b], batch['azim'][b]
                idx = batch['idx'][b]
                self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose,
                                     reye_pose=reye_pose)
                #
                pred_img, viewspace_point_tensor, radii = self.avatar.render(dist, elev, azim)  # H,W,C  in [0,1]
                mesh = self.avatar.get_flame_cond(dist, elev, azim, 70.0, expression, jaw_pose, neck_pose, leye_pose, reye_pose)[0]
                img = pred_img.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
                flame_cond = mesh.permute(2, 0, 1).unsqueeze(0).to(device)  # 1,3,H,W
                # finetune image
                fine_img = self.sd.denoise_img(self.guidance, img, flame_cond, self.cfg['paths']['config_path'], noise)
                # fine_img = (img[0].permute(1,2,0).clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
                out_path = os.path.join(path, f"{idx:06d}.png")
                imageio.imwrite(out_path, fine_img)

        self.train_loader.dataset.switch_image_path(path)

    def train(self, epochs=1, save_every_n_epochs=10, update_iter=30, ckpt_dir=None, img_dir=None):
        self.global_step = 0
        ep_iter = trange(epochs, desc="Epochs", dynamic_ncols=True)
        for ep in ep_iter:
            if self.global_step % update_iter == 0 and self.global_step < self.cfg['train']['fix_exp_step']:
                self.update_data(self.cfg['paths']['img_path'], stage=True)
            # if self.global_step % update_iter == 0 and self.global_step >= self.cfg['train']['fix_exp_step']:
            #     self.update_data(self.cfg['paths']['img_path_stage2'])
            batch_iter = tqdm(self.train_loader, desc=f"Train {ep + 1}/{epochs}", leave=False, dynamic_ncols=True)
            for batch in batch_iter:
                B = batch['expression'].shape[0]
                self._optim_zero_grad()
                # total_loss = 0.0
                batch = to_cuda(batch, self.device)
                preds = []
                uv_renders = []
                self.viewspace_point_list = []
                for b in range(B):

                    expression = batch['expression'][b]
                    jaw_pose = batch['jaw_pose'][b]
                    neck_pose = batch['neck_pose'][b]
                    leye_pose = batch['leye_pose'][b]
                    reye_pose = batch['reye_pose'][b]

                    dist, elev, azim = batch['dist'][b], batch['elev'][b], batch['azim'][b]

                    self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose, reye_pose=reye_pose)
                    #
                    pred_img, viewspace_point_tensor, radii = self.avatar.render(dist, elev, azim)  # H,W,C  in [0,1]
                    uv_img = self.avatar.render_uv(dist, elev, azim)  # (3,H,W)
                    uv_renders.append(uv_img)
                    # pred = pred_img.permute(2, 0, 1)[[2, 1, 0]]  # 3,H,W
                    pred = pred_img.permute(2, 0, 1)
                    preds.append(pred)
                    self.viewspace_point_list.append(viewspace_point_tensor)

                    if b == 0:
                        self.radii = radii
                    else:
                        self.radii = torch.max(radii, self.radii)

                # loss = F.mse_loss(pred, gt_t)
                self.visibility_filter = self.radii > 0.0
                preds = torch.stack(preds)
                uv_renders = torch.stack(uv_renders, dim=0)
                loss_l1 = F.l1_loss(preds, batch['img']) * cfg['loss']['l1_weight']
                loss_vgg = self._vgg_loss(preds, batch['img']) * cfg['loss']['vgg_weight']
                loss_ssim = (1-ssim(preds, batch['img'])) * cfg['loss']['ssim_weight']
                total_loss = loss_l1 + loss_ssim + loss_vgg
                tv_start = cfg['loss'].get('tv_uv_start_step', 2400)  # prune-only 从 2400 开始
                tv_w = cfg['loss'].get('tv_uv_weight', 100.0)  #
                uv_jump = cfg['loss'].get('tv_uv_jump_thresh', 0.25)
                if self.global_step >= tv_start and tv_w > 0:
                    loss_tv_uv = tv_uv_loss_seam_aware(uv_renders, uv_jump_thresh=uv_jump)
                    total_loss = total_loss + tv_w * loss_tv_uv
                    self.writer.add_scalar("train/loss_tv_uv", loss_tv_uv.item(), self.global_step)
                else:
                    loss_tv_uv = preds.sum() * 0.0

                atlas_start = cfg['loss'].get('uv_atlas_start_step', 2400)
                atlas_w = cfg['loss'].get('uv_atlas_weight', 1.0)  # 从小到大调（0.2~2.0 常见）
                atlas_res = cfg['loss'].get('uv_atlas_res', 256)
                atlas_alpha_th = cfg['loss'].get('uv_atlas_alpha_thresh', 0.2)
                atlas_flip_v = cfg['loss'].get('uv_atlas_flip_v', False)  # 若出现上下颠倒再置 True
                if self.global_step >= atlas_start and atlas_w > 0:
                    with torch.no_grad():
                        atlas_rgb, atlas_w_map = splat_rgb_to_uv_atlas(
                            rgb=batch['img'],  # teacher = sdedit后的 pseudo 图
                            uv_render=uv_renders,
                            atlas_res=atlas_res,
                            alpha_thresh=atlas_alpha_th,
                            flip_v=atlas_flip_v,
                        )
                        atlas_img = sample_uv_atlas_to_image(atlas_rgb, uv_renders, flip_v=atlas_flip_v)  # (B,3,H,W)

                        # mask：只在前景区域计算（避免背景引导）
                        _, alpha, bg_mask = _uv_unblend_from_uv_render(uv_renders, eps=1e-6)
                        fg = (alpha > atlas_alpha_th) & (~bg_mask)

                    loss_uv_atlas = masked_l1(preds, atlas_img, fg)
                    total_loss = total_loss + atlas_w * loss_uv_atlas
                    self.writer.add_scalar("train/loss_uv_atlas", loss_uv_atlas.item(), self.global_step)
                else:
                    loss_uv_atlas = preds.sum() * 0.0

                total_loss.backward()
                self._lr_step(self.global_step)
                self.on_before_optimizer_step()
                self.avatar.gaussian.optimizer.step()

                # running += total_loss.item()
                self.global_step += 1
                self.writer.add_scalar("train/loss_l1", loss_l1.item(), self.global_step)
                self.writer.add_scalar("train/loss_ssim", loss_ssim.item(), self.global_step)
                self.writer.add_scalar("train/loss_vgg", loss_vgg.item(), self.global_step)
                self.writer.add_scalar("train/loss_total", total_loss.item(), self.global_step)

                batch_iter.set_postfix(
                    loss=f"{total_loss.item():.6f}",
                    l1=f"{loss_l1.item():.6f}",
                    l_ssim=f"{loss_ssim.item():.6f}",
                    loss_vgg=f"{loss_vgg.item():.6f}",
                    num=f"{self.avatar.gaussian._xyz.shape[0]}",
                )

            if save_every_n_epochs and ((ep % save_every_n_epochs == 0) or (ep == epochs-1)):
                frames = self.test(save_frames=True, save_dir=img_dir, return_numpy=True, epoch=ep)
                # save_mp4_w_audio(frames, tag='gs-static', w_audio=False)
                if not ckpt_dir:
                    raise ValueError("There is no folder ckpt_dir")
                ply_path = os.path.join(ckpt_dir, f"epoch_{ep:04d}.ply")
                self.avatar.gaussian.save_ply(ply_path)
                print(f"[ckpt] saved {ply_path}")

    @torch.no_grad()
    def test(self, save_frames=False, save_dir=None, return_numpy=True, epoch="0"):

        frames = []
        os.makedirs(save_dir, exist_ok=True) if (save_frames and save_dir) else None

        frame_idx = 0
        for batch in self.test_loader:
            B = batch['expression'].shape[0]
            batch = to_cuda(batch, self.device)
            for b in range(B):
                expression = batch['expression'][b]
                jaw_pose = batch['jaw_pose'][b]
                neck_pose = batch['neck_pose'][b]
                leye_pose = batch['leye_pose'][b]
                reye_pose = batch['reye_pose'][b]
                dist, elev, azim = batch['dist'][b], batch['elev'][b], batch['azim'][b]
                self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose,
                                reye_pose=reye_pose)
                pred, _, _ = self.avatar.render(dist, elev, azim)  # HWC, [0,1]

                if return_numpy:
                    img = (pred.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
                    frames.append(img)
                    if save_frames and save_dir:
                        out_path = os.path.join(save_dir, f"{epoch:04d}_{frame_idx:06d}.png")
                        imageio.imwrite(out_path, img)  # RGB
                else:
                    frames.append(pred.detach().cpu())

                frame_idx += 1

        return frames

    def on_before_optimizer_step(self):
        with torch.no_grad():

            if self.global_step < self.densify_prune_end_step:  # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.avatar.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.avatar.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.avatar.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.global_step > self.densify_prune_start_step and self.global_step % self.densify_prune_interval == 0:  # 500 100
                    size_threshold = self.size_threshold if self.global_step > self.size_threshold_fix_step else None  # 3000
                    self.avatar.gaussian.densify_and_prune(0.002, 0.05, self.cameras_extent, size_threshold)

                    # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.global_step > self.prune_only_start_step and self.global_step < self.prune_only_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.avatar.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.avatar.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.avatar.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.global_step % self.prune_only_interval == 0:
                    self.avatar.gaussian.prune_only(extent=self.cameras_extent)

    def _optim_zero_grad(self):
        self.avatar.gaussian.optimizer.zero_grad(set_to_none=True)

    def _lr_step(self, global_step: int):
        gaussian = self.avatar.gaussian
        if hasattr(gaussian, "xyz_scheduler_args"):
            new_lr = gaussian.xyz_scheduler_args(global_step)
            for g in gaussian.optimizer.param_groups:
                if g.get("name") == "xyz":
                    g["lr"] = new_lr


if __name__ == '__main__':

    parser = ArgumentParser(description="Gaussian Head avatar training & testing")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    train_dataset = NerSemble(cfg["paths"], is_train=True)
    test_dataset = NerSemble(cfg["paths"], is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train"]["train_bs"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["train"]["test_bs"], shuffle=False)

    avatar = Avatar(cfg["paths"])

    cur_dir = Path(cfg["paths"]["output_root"]).resolve()

    optim_dir = cur_dir / "optim"
    os.makedirs(optim_dir, exist_ok=True)
    log_dir = optim_dir / "logs"
    img_dir = optim_dir / "imgs"
    ckpt_dir = optim_dir / "ckpt"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = Trainer(
        avatar=avatar,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=device,
        log_every=20,
        log_dir=log_dir,
        cfg=cfg,
    )

    trainer.train(epochs=cfg['train']['epochs'], save_every_n_epochs=cfg['train']['save_every_n_epochs'],
                  update_iter=cfg['train']['update_iter'], ckpt_dir=ckpt_dir, img_dir=img_dir)
    trainer.writer.flush()
    trainer.writer.close()
