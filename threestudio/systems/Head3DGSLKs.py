import io
import math

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from dataclasses import dataclass, field
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

import threestudio
# from threestudio.utils.poser import Skeleton
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.clip_eval import CLIPTextImageEvaluator

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussiansplatting.scene.cameras import Camera, MiniCam
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel
from gaussiansplatting.scene.gaussian_flame_model import GaussianFlameModel

# from gaussiansplatting.gaussian_renderer.render_2d import render


@threestudio.register("head-3dgs-lks-rig-system")
class Head3DGSLKsRig(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        texture_structure_joint: bool = False
        controlnet: bool = False
        flame_path: str = "/path/to/flame/model"
        flame_gender: str = 'generic'
        pts_num: int = 100000

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_size_threshold: float = 0.008

        apose: bool = True
        bg_white: bool = False
        bg_random: bool = True
        bg_random_gray_prob: float = 0.7  #

        area_relax: bool = False
        shape_update_end_step: int = 12000
        training_w_animation: bool = True

        # Text-image CLIP evaluation
        clip_eval: bool = True
        # Load three CLIP variants by default
        clip_model_names: tuple = ("ViT-B/16", "ViT-B/32", "ViT-L/14")
        clip_model_root: str = "../HeadStudio_lib/clip"
        # area scaling factor
        # area_scaling_factor: float = 1

    cfg: Config

    def configure(self) -> None:
        self.radius = self.cfg.radius
        # self.gaussian = GaussianModel(sh_degree=0)
        self.gaussian = GaussianFlameUVModel(sh_degree=0, gender=self.cfg.flame_gender, model_folder=self.cfg.flame_path,
                                 jacobian_mode='autodiff', jacobian_create_graph=False, update_faces_on_densify=True)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32,
                                              device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0],
                                                                                                    dtype=torch.float32,
                                                                                                    device="cuda")

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.controlnet = self.cfg.controlnet

        self.cameras_extent = 4.0

        self.cfg.loss.lambda_position = 0.01 * self.cfg.loss.lambda_position
        self.cfg.loss.lambda_scaling = 0.01 * self.cfg.loss.lambda_scaling
        if self.cfg.area_relax:
            reduction = 'none'
        else:
            reduction = 'mean'
        self.smoothl1_position = torch.nn.SmoothL1Loss(beta=1.0, reduction=reduction)
        self.l1_scaling = torch.nn.L1Loss(reduction=reduction)

        # Initialize CLIP evaluator (optional)
        self.clip_evaluator = None
        if getattr(self.cfg, "clip_eval", False):
            try:
                threestudio.info(f"Loading CLIP evaluation ...")
                self.clip_evaluator = CLIPTextImageEvaluator(
                    device=self.device,
                    model_name=self.cfg.clip_model_names,
                    model_root=self.cfg.clip_model_root,
                )
            except Exception as e:
                threestudio.info(f"[CLIP Eval] Failed to initialize CLIP models: {e}. CLIP evaluation will be disabled")
                self.clip_evaluator = None

    def save_gif_to_file(self, images, output_file):
        with io.BytesIO() as writer:
            images[0].save(
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
            )
            writer.seek(0)
            with open(output_file, 'wb') as file:
                file.write(writer.read())

    def get_c2w(self, dist, elev, azim):
        elev = elev * math.pi / 180
        azim = azim * math.pi / 180
        batch_size = dist.shape[0]
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                dist * torch.cos(elev) * torch.cos(azim),
                dist * torch.cos(elev) * torch.sin(azim),
                dist * torch.sin(elev),
            ],
            dim=-1,
        )
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions, device=self.device)
        up: Float[Tensor, "B 3"] = torch.as_tensor(
            [0, 0, 1], dtype=torch.float32, device=self.device)[None, :].repeat(batch_size, 1)
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1], device=self.device)], dim=1
        )
        c2w[:, 3, 3] = 1.0
        return c2w

    def set_pose(self, expression, jaw_pose, leye_pose, reye_pose, neck_pose=None):
        self.gaussian._expression = expression.detach()
        self.gaussian._jaw_pose = jaw_pose.detach()
        self.gaussian._leye_pose = leye_pose.detach()
        self.gaussian._reye_pose = reye_pose.detach()
        if neck_pose is not None:
            self.gaussian._neck_pose = neck_pose.detach()

    def forward(self, batch: Dict[str, Any], renderbackground=None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        alphas = []
        uv_images = []
        self.viewspace_point_list = []
        self.radii_list = []
        uv = self.gaussian.get_uv  # (N,2)
        uv_color = torch.cat([uv, -torch.ones_like(uv[:, :1])], dim=1)  # (N,3)

        if self.cfg.training_w_animation:
            self.set_pose(batch['expression'], batch['jaw_pose'], batch['leye_pose'], batch['reye_pose'])

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w=batch['c2w'][id], FoVy=batch['fovy'][id].item(), height=batch['height'], width=batch['width'])

            with torch.cuda.amp.autocast(False):
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
                pkg_uv = render(viewpoint_cam, self.gaussian, self.pipe, bg_color=self.background_tensor, override_color=uv_color)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)
            self.radii_list.append(radii)
            uv_img = pkg_uv["render"]  # (3,H,W)
            uv_images.append(uv_img)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            alpha = render_pkg["alpha_3dgs"]

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0).clamp(0, 1)
            alpha = alpha.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        alphas = torch.stack(alphas, 0)
        uv_images = torch.stack(uv_images, 0)
        # depth_min = torch.amin(depths, dim=[1, 2, 3], keepdim=True)
        # depth_max = torch.amax(depths, dim=[1, 2, 3], keepdim=True)
        # depths = (depths - depth_min) / (depth_max - depth_min + 1e-10)
        # depths = depths.repeat(1, 1, 1, 3)

        self.visibility_filter = self.radii > 0.0

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        # render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["alpha"] = alphas
        render_pkg["opacity"] = alphas
        render_pkg["uv_img"] = uv_images
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)


    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)

        if self.true_global_step > self.cfg.half_scheduler_max_step:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        bg = None
        r = torch.rand((), device=self.device)
        if r < 0.33:
            bg = torch.ones(3, device=self.device)
        elif r < 0.66:
            bg = torch.zeros(3, device=self.device)
        else:
            bg = torch.rand(3, device=self.device)

        control_images = batch["flame_conds"]
        if self.true_global_step < 2000:
            self.cfg.training_w_animation = False
            control_images = batch["neutral_flame_conds"]
        else:
            self.cfg.training_w_animation = True

        out = self(batch, renderbackground=bg)
        # out = self(batch)

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        # control_images = out["depth"]

        guidance_eval = False

        guidance_out = self.guidance(self.true_global_step,
            images.permute(0, 3, 1, 2), control_images.permute(0, 3, 1, 2), prompt_utils,
            **batch, rgb_as_latents=False,
        )

        loss = 0.0

        self.log("train/loss_sds", guidance_out['loss_sds'])
        loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])

        scaling = self.gaussian.get_world_scale_max_approx()
        big_points_ws = scaling > 0.01
        loss_scaling = self.l1_scaling(scaling[big_points_ws], torch.zeros_like(scaling[big_points_ws]))

        self.log("train/loss_scaling", loss_scaling)
        loss += loss_scaling * self.C(self.cfg.loss.lambda_scaling)

        if self.true_global_step >= self.cfg.prune_only_start_step:
            xyz = self.gaussian.get_xyz
            position = torch.norm(xyz, dim=1)
            # mask = position > position_threshold
            loss_position = self.smoothl1_position(position, torch.zeros_like(position)).mean()
            self.log("train/loss_position", loss_position)
            loss += loss_position * self.C(self.cfg.loss.lambda_position)

            uv_render = out["uv_img"]
            loss_uv_tv = self.tv_uv_loss(uv_render)
            loss = loss + loss_uv_tv * self.cfg.loss.lambda_uv_tv
            self.log("train/loss_tv", loss_uv_tv)

            p = self.gaussian.get_opacity.clamp(1e-4, 1 - 1e-4)  # (N,1)
            eps = 0.1
            loss_opaque = (torch.log(p + eps) + torch.log((1.0 + eps) - p) - math.log(eps) - math.log(1.0 + eps)).mean()
            # opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            # loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        loss_shape = torch.norm(self.gaussian._shape)
        self.log("train/loss_shape", loss_shape)
        loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        alpha = out["opacity"][..., 0]  # (B,H,W)
        B, H, W = alpha.shape
        w = int(0.05 * min(H, W))  #

        border = torch.zeros_like(alpha)
        border[:, :w, :] = 1
        border[:, -w:, :] = 1
        border[:, :, :w] = 1
        border[:, :, -w:] = 1
        loss_sparsity = (alpha * border).mean()
        # bg_mask = (alpha < 0.05)
        # loss_sparsity = alpha[bg_mask].mean()
        # loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def tv_uv_loss(self, uv_renderings, eps=1e-6):
        """
        uv_renderings: (B,3,H,W)  where ch2 is blended third channel (was -1 before blending)
        returns scalar loss
        """
        # Recover alpha(mask) from 3rd channel
        mask = 1.0 - (uv_renderings[:, [2]] + 1.0) / 2.0  # (B,1,H,W) == alpha

        mask = mask.clamp(0.0, 1.0)
        mask_safe = mask.clamp_min(eps)

        # Undo alpha blending: uv = (render - (1-alpha)*bg) / alpha, with bg=1
        uv_no_blend = (uv_renderings - (1.0 - mask)) / mask_safe  # (B,3,H,W)

        background_mask = mask <= eps  # (B,1,H,W) bool

        # Exclude pixel-pairs that touch background
        mask_y = (background_mask[:, :, 1:] | background_mask[:, :, :-1]).repeat(1, 3, 1, 1)
        mask_x = (background_mask[:, :, :, 1:] | background_mask[:, :, :, :-1]).repeat(1, 3, 1, 1)

        # TV diffs
        diff_y = uv_no_blend[:, :, 1:] - uv_no_blend[:, :, :-1]  # (B,3,H-1,W)
        diff_x = uv_no_blend[:, :, :, 1:] - uv_no_blend[:, :, :, :-1]  # (B,3,H,W-1)

        tv_y = diff_y[~mask_y].abs().mean()
        tv_x = diff_x[~mask_x].abs().mean()
        return tv_x + tv_y

    def uv_knn_smooth_loss(
            self, uv, face_idx, attr,
            S=4096, k=8, sigma=0.02,
            same_face_only=True,
            mode="l1",  # "l1" (TV) or "l2"
    ):
        """
        uv: (N,2) in [0,1]
        face_idx: (N,) long
        attr: (N,C) float
        """
        N = uv.shape[0]
        device = uv.device
        if N <= S:
            idx = torch.arange(N, device=device)
        else:
            idx = torch.randint(0, N, (S,), device=device)

        uv_s = uv[idx]  # (S,2)
        f_s = face_idx[idx]  # (S,)
        a_s = attr[idx]  # (S,C)

        # pairwise distances in UV
        d = torch.cdist(uv_s, uv_s, p=2)  # (S,S)
        d2 = d * d
        d2.fill_diagonal_(1e9)

        if same_face_only:
            same = (f_s[:, None] == f_s[None, :])
            d2 = d2 + (~same) * 1e6  # mask out different faces

        nn_d2, nn_j = torch.topk(d2, k, largest=False)  # (S,k)
        valid = (nn_d2 < 1e5)  # rows with enough same-face neighbors

        neigh = a_s[nn_j]  # (S,k,C)
        diff = a_s[:, None, :] - neigh  # (S,k,C)

        w = torch.exp(-nn_d2 / (sigma * sigma)).clamp_min(1e-6)  # (S,k)
        w = w * valid

        if mode == "l2":
            loss = (w[..., None] * (diff * diff)).sum() / (w.sum() * a_s.shape[-1] + 1e-8)
        else:  # "l1"
            loss = (w[..., None] * diff.abs()).sum() / (w.sum() * a_s.shape[-1] + 1e-8)
        return loss

    def _accumulate_grad_norm_and_radii(self):
        N = self.gaussian.num_gs
        grad_sum = torch.zeros((N, 1), device=self.device)
        vis_cnt = torch.zeros((N, 1), device=self.device)
        for idx in range(len(self.viewspace_point_list)):
            g = self.viewspace_point_list[idx].grad
            if g is None:
                continue
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            # 关键：用当前视角自己的 radii 来算 vis_i（别用 union，否则 cnt 不准）
            radii_i = self.radii_list[idx] if hasattr(self, "radii_list") else None
            if radii_i is None:
                # 如果你没有保存每视角 radii，就退化为 union（能跑，但 cnt 不精准）
                vis_i = self.visibility_filter
            else:
                vis_i = radii_i > 0.0

            gn = torch.norm(g[:, :2], dim=-1, keepdim=True)  # (N,1) 先 norm 再加
            grad_sum[vis_i] += gn[vis_i]
            vis_cnt[vis_i] += 1.0
        # 用 mean（每点按可见视角数平均），避免视角数影响阈值
        grad_mean = grad_sum / vis_cnt.clamp_min(1.0)
        return grad_mean

    def get_sds_grad_proxy(self):
        """
        返回 (N, D) 的 per-Gaussian 梯度 proxy
        """
        N = self.gaussian.num_gs
        grad_sum = torch.zeros((N, 2), device=self.device)
        cnt = torch.zeros((N, 1), device=self.device)

        for i, vpt in enumerate(self.viewspace_point_list):
            if vpt.grad is None:
                continue
            g = torch.nan_to_num(vpt.grad[:, :2], 0.0)  # screen xy grad
            vis = self.radii_list[i] > 0
            grad_sum[vis] += g[vis]
            cnt[vis] += 1

        return grad_sum / cnt.clamp_min(1.0)

    @torch.no_grad()
    def smooth_gradients_via_grid(self):
        """
        [Gradient Surgery] 利用 UV Grid 对显式参数的梯度进行平滑。
        修复了维度不匹配问题 (支持 [N, 3] 和 [N, 1, 3] 等任意形状)。
        """
        # 1. 获取数据
        uv = self.gaussian.get_uv  # (N, 2)

        # 2. 定义需要平滑的显式参数及其对应的平滑强度
        # 注意：这里直接引用 self.gaussian 的 Parameter 张量
        target_params = [
            (self.gaussian._scaling, "scaling"),  # [N, 3]
            (self.gaussian._features_dc, "color"),  # [N, 1, 3] --> 会导致之前报错的元凶
            (self.gaussian._rotation, "rotation"),  # [N, 4]
            (self.gaussian._d, "offset"),     #
            (self.gaussian._opacity, "opacity"),  #
        ]

        # UV Grid 分辨率
        res = self.gaussian.uv_grid_res  # 256

        # 计算每个点落在哪个 grid cell
        uv_idx = (uv * res).long().clamp(0, res - 1)
        grid_indices = uv_idx[:, 1] * res + uv_idx[:, 0]  # (N,) 扁平化索引

        # 准备计数器 (所有属性共享同一个 grid 计数，避免重复计算)
        # count: 每个 cell 里有多少个点
        N = uv.shape[0]
        count = torch.zeros((res * res, 1), device=self.device)
        count.index_add_(0, grid_indices, torch.ones(N, 1, device=self.device))
        count_clamped = count.clamp_min(1.0)

        for param, name in target_params:
            if param.grad is None:
                continue

            grad = param.grad  # 原始梯度，形状可能是 [N, 3] 或 [N, 1, 3]
            original_shape = grad.shape

            # --- 【核心修复】维度拍平 ---
            # 无论原来是 [N, 3] 还是 [N, 1, 3]，都变成 [N, C_total]
            grad_flat = grad.reshape(N, -1)
            C_total = grad_flat.shape[1]

            # --- Scatter Reduce (求和) ---
            grad_sum = torch.zeros((res * res, C_total), device=self.device)
            # 现在 source=[N, C_total], self=[res*res, C_total]，除第0维外维度一致，不会报错
            grad_sum.index_add_(0, grid_indices, grad_flat)

            # --- Average ---
            grid_avg_grad = grad_sum / count_clamped

            # --- Gather (写回) ---
            # 将 Grid 的平均梯度分配回每个点
            smoothed_grad_flat = grid_avg_grad[grid_indices]  # [N, C_total]

            # --- Hard Surgery (混合) ---
            # 几何属性 (Scaling/Rotation) 给强平滑 (0.9)，消除针状
            # 颜色属性 (Color) 给中等平滑 (0.5)，消除噪点但保留部分纹理
            mix_rate = 0.9 if name in ["scaling", "rotation", "offset", "opacity"] else 0.2

            final_grad_flat = mix_rate * smoothed_grad_flat + (1 - mix_rate) * grad_flat

            # --- 【核心修复】维度还原并赋值 ---
            # 变回 [N, 1, 3] 或 [N, 3]
            param.grad = final_grad_flat.view(original_shape)

    def on_before_optimizer_step(self, optimizer):
        self.smooth_gradients_via_grid()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)

        with torch.no_grad():

            if self.true_global_step < self.cfg.densify_prune_end_step:
                viewspace_point_tensor_grad = self._accumulate_grad_norm_and_radii()
                # viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                # for idx in range(len(self.viewspace_point_list)):
                #     g = self.viewspace_point_list[idx].grad
                #     if g is None:
                #         continue
                #     g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + g
                    # viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)
                # densify_and_prune
                if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:  # 500 100
                    size_threshold = self.cfg.size_threshold if self.true_global_step > self.cfg.size_threshold_fix_step else None  # 3000
                    self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, size_threshold)

            # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                viewspace_point_tensor_grad = self._accumulate_grad_norm_and_radii()
                # viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                # for idx in range(len(self.viewspace_point_list)):
                #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step % self.cfg.prune_only_interval == 0:
                    self.gaussian.prune_only(extent=self.cameras_extent)

            if self.true_global_step > self.cfg.shape_update_end_step:
                for param_group in self.gaussian.optimizer.param_groups:
                    if param_group['name'] == 'flame_shape':
                        param_group['lr'] = 1e-10

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 注意：这里已经在 optimizer.step() 之后
        with torch.no_grad():
            self.gaussian.clamp_uv_logits_()
            # 每 1~5 步一次：
            if self.true_global_step % 1 == 0:
                cnt = self.gaussian.update_face_idx_from_uv(return_stats=True)
                threestudio.info(f"[Step {self.true_global_step}] Updated face indices from UVs, changed {cnt['updated']} Gaussians.")

    def on_after_backward(self):
        self.dataset.skel.betas = self.gaussian.get_shape.detach()
        # uv一致性loss
        # with torch.enable_grad():
        #     uv = self.gaussian.get_uv  # (N,2) in [0,1]
        #     face_idx = self.gaussian._face_idx  # (N,)
        #     grad_proxy = self.get_sds_grad_proxy()  # (N,2)
        #
        #     loss_uv_grad = self.uv_knn_smooth_loss(
        #         uv=uv,
        #         face_idx=face_idx,
        #         attr=grad_proxy,
        #         S=4096,
        #         k=8,
        #         sigma=0.02,
        #         same_face_only=True,
        #         mode="l1",
        #     )
        #     (loss_uv_grad * 0.05).backward()
        #     self.log("train/loss_uv_grad", loss_uv_grad)
        # pass

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self(batch)
            # prompt_utils = self.prompt_processor()
            # images = out["comp_rgb"]
            # control_images = batch["flame_conds"]
            # t = torch.tensor([[800], [600], [400], [200], [20]]).to(control_images.device)
            # grads = []
            # for t_ in t:
            #     grad_heat = self.guidance.save_t_grad(
            #         images.permute(0, 3, 1, 2), control_images.permute(0, 3, 1, 2), prompt_utils, t_,
            #         batch['elevation'], batch['azimuth'], batch['camera_distances'],
            #     )
            #     heat_path = self.get_save_path(f"grad_it{self.true_global_step}_t{t_[0]}-{batch['index'][0]}.png")
            #     grads.append(grad_heat)
            #     cv2.imwrite(heat_path, grad_heat[...,[2,1,0]]*255)
            # for i, grad_t in enumerate(grads):
            #     heatmap_colored = grad_t[..., [2, 1, 0]] * 255  # BGR -> RGB
            #     original_img = out["comp_rgb"][0].detach().cpu().numpy() * 255  # 假设是 [0,1]
            #     overlay = 0.6 * original_img + 0.4 * heatmap_colored
            #     overlay_path = self.get_save_path(f"overlay_it{self.true_global_step}_t{t[i, 0]}-{batch['index'][0]}.png")
            #     cv2.imwrite(overlay_path, overlay)
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="validation_step",
                step=self.true_global_step,
            )
            # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
            # self.gaussian.save_ply(save_path)
            # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))
            if self.true_global_step % 500 == 0:
                # save_path = self.get_save_path(f"last.ply")
                save_path = self.get_save_path(f"step_{self.true_global_step}.ply")
                self.gaussian.save_ply(save_path)
                weigth_path = self.get_save_path(f"ckpts/step_{self.true_global_step}.pt")
                # self.gaussian.save_ckpt(weigth_path, self.gaussian.optimizer, step=self.true_global_step)

            # 在验证阶段计算文本-图像 CLIP 相似度（多个模型）
            if self.clip_evaluator is not None:
                # 渲染图像转为 NCHW
                clip_imgs = out["comp_rgb"].permute(0, 3, 1, 2)
                prompts = [self.cfg.prompt_processor.prompt]

                clip_sims_dict = self.clip_evaluator.compute_similarity(clip_imgs, prompts)
                for model_name, sims in clip_sims_dict.items():
                    tag = model_name.replace("/", "").replace("ViT-", "vit_").lower()
                    self.log(f"val/clip_sim_{tag}", sims.mean().item())

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)
        weigth_path = self.get_save_path(f"ckpts/step_{self.true_global_step}.pt")
        # self.gaussian.save_ckpt(weigth_path, self.gaussian.optimizer, step=self.true_global_step)

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        self.gaussian.create_from_flame(self.cameras_extent, -10, N=self.cfg.pts_num)
        # with torch.no_grad():
        #     self.gaussian.clamp_uv_logits_()
        #     self.gaussian.update_face_idx_from_uv()
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )