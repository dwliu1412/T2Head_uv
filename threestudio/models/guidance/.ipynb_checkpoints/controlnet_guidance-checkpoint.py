import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


@threestudio.register("controlnet-depth-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "stablediffusionapi/realistic-vision-51"
        ddim_scheduler_name_or_path: str = "../HeadStudio_lib/stable-diffusion-v1-5"
        control_type: str = "normal"  # normal/canny

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_nfsd: bool = False
        use_dsd: bool = False
        edit_image: bool = False

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path: str
        if self.cfg.control_type == "normal":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_canny"
        elif self.cfg.control_type == "depth":
            controlnet_name_or_path = "lllyasviel/control_v11f1p_sd15_depth"
        elif self.cfg.control_type == "openpose":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_openpose"
        elif self.cfg.control_type == "mediapipe":
            # controlnet_name_or_path = "CrucibleAI/ControlNetMediaPipeFace"
            controlnet_name_or_path = "../HeadStudio_lib/ControlNetMediaPipeFace"

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        if self.cfg.control_type == "mediapipe":
            if self.cfg.pretrained_model_name_or_path in ["stablediffusionapi/realistic-vision-51", "../HeadStudio_lib/realistic-vision-51",
                                                          "runwayml/stable-diffusion-v1-5", "../HeadStudio_lib/stable-diffusion-v1-5",
                                                          "../HeadStudio_lib/Realistic_Vision_V5.1_noVAE"]:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    subfolder="diffusion_sd15",
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                    use_safetensors=False
                )
            else:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                )

        else:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name_or_path,
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
            )
        # import pdb; pdb.set_trace()
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)
        # import pdb; pdb.set_trace()
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        if self.cfg.control_type == "normal":
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.total_steps = 10000
        self.t_max = 800
        self.t_min = 50
        self.step_len = 1000
        self.end_max = 600
        self.end_min = 200

        # DreamTime 调度
        s = 100
        m = 400

        t_range = np.arange(0, self.num_train_timesteps)
        w_p = np.exp(-((t_range - m) ** 2) / (2 * s ** 2))
        w_p = torch.from_numpy(w_p).to(self.device)
        w_d = torch.sqrt((1 - self.alphas) / self.alphas)
        # 组合权重
        w_total = w_d * w_p
        # 归一化
        w_total = w_total / (w_total.sum() + 1e-8)
        cdf = torch.cumsum(w_total, dim=0) # 从 t=0 到 t=1000
        self.timestep_map = cdf / cdf[-1]  # 归一化到 [0, 1]
        # self.plot_t_schedule()
        # cur = self.get_t(1, 4)

        threestudio.info(f"Loaded ControlNet!")

    def plot_t_schedule(self, max_step: int = None, interval: int = 1, save_path: str = None):
        """
        绘制 t-i 曲线，其中 i 为 global_step，t 为 get_t 返回的时间步（取 batch 中第一个样本）。
        \- max_step: 要绘制的最大 global_step（默认用 self.total_steps）
        \- interval: 采样间隔，比如每隔 100 步画一个点
        \- save_path: 如果不为 None，则保存到指定路径，否则直接 plt.show()
        """
        if max_step is None:
            max_step = self.total_steps

        steps = []
        t_values = []

        # 这里用 batch_size=1 即可，只看曲线形状
        batch_size = 1
        device = self.device

        for step in range(0, max_step + 1, interval):
            t_tensor = self.get_t(step, batch_size, device=device)
            t_scalar = int(t_tensor[0].item())
            steps.append(step)
            t_values.append(t_scalar)

        plt.figure(figsize=(6, 4))
        plt.plot(steps, t_values, marker="o")
        plt.xlabel("global\_step i")
        plt.ylabel("t")
        plt.title("t-i Schedule (DreamTime)")
        plt.grid(True)

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def get_t(self, global_step, batch_size, device='cuda'):
        """
        获取当前步的时间步
        """
        progress = global_step / self.total_steps
        progress = float(np.clip(progress, 0.0, 1.0))

        target_cdf = 1.0 - progress

        t_idx = torch.searchsorted(self.timestep_map, torch.tensor([target_cdf], device=self.timestep_map.device), right=False)[0].item()
        t_idx = int(np.clip(t_idx, self.t_min, self.t_max))

        t_tensor = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

        jitter = torch.randint(-50, 50, (batch_size,), device=device)
        t_tensor = torch.clamp(t_tensor + jitter, self.t_min, self.t_max)

        return t_tensor

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            image_cond: Float[Tensor, "..."],
            condition_scale: float,
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
            cross_attention_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
            self,
            latents: Float[Tensor, "B 4 H W"],
            latent_height: int = 64,
            latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 64 64"]:
        # self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.config.num_train_timesteps = 1000
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            image_cond_input = torch.cat([image_cond] * 2)
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            # threestudio.debug("Start editing...")
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 2)
                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.forward_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        image_cond=image_cond_input,
                        condition_scale=self.cfg.condition_scale,
                    )

                    noise_pred = self.forward_control_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )
                # perform classifier-free guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )
                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # threestudio.debug("Editing finished.")
        return latents

    def compute_grad_sds(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        noise_pred = noise_pred_null + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_null
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def compute_grad_nfsd(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
    ):
        batch_size = latents.shape[0]
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        # Eq.6 in Noise-free Score Distillation, Katzir et al., arXiv preprint arXiv:2310.17590, 2023.
        delta_c = self.cfg.guidance_scale * (noise_pred_text - noise_pred_null)
        mask = (t < 200).int().view(batch_size, 1, 1, 1)
        if self.cfg.use_dsd:
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null + (noise_pred_null - noise_pred_neg))
        else:
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null - noise_pred_neg)

        # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
        #     noise_pred_text - noise_pred_uncond
        # )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (delta_c + delta_d)
        return grad

    def save_t_grad(self, rgb, control_image, prompt_utils, t, elevation, azimuth, camera_distances):

        rgb_BCHW = rgb
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW_512)
        image_cond = F.interpolate(
            control_image, (512, 512), mode="bilinear", align_corners=False
        )
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, True
        )
        if self.cfg.use_nfsd or self.cfg.use_dsd:
            grad = self.compute_grad_nfsd(text_embeddings, latents, image_cond, t)
        else:
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t)
        grad = torch.nan_to_num(grad)

        # target = (latents - grad).detach()
        # diff = torch.abs(latents - target)
        # with torch.no_grad():
        #     grad_pixel = self.decode_latents(grad)

        heatmap = self.grad_to_heatmap(grad, upsample_size=(1024, 1024))
        # heatmap = self.grad_to_heatmap(grad_pixel, upsample_size=None)

        return heatmap

    def grad_to_heatmap(self, grad, upsample_size=None):
        """
        grad: (B, 4, 64, 64) 的梯度张量（对 latent 的 grad）
        upsample_size: 如果希望放大到原图大小，比如 (512,512)，就传这个；否则就用 64x64
        return: heatmap: (H, W, 3)，numpy，[0,1]
        """
        # 先取一张（第 0 个样本）
        g = grad[0]  # (4, 64, 64)
        #
        grad_norm = torch.norm(g, dim=0, keepdim=True)

        if upsample_size is not None:
            # 如果需要上采样
            g_processed = F.interpolate(
                grad_norm[None, ...],
                size=upsample_size,
                mode="bilinear",
                align_corners=False
            )[0, 0]  # 结果 (H, W)
        else:
            # 如果不需要上采样，直接去掉 channel 维
            g_processed = grad_norm[0]  # 结果 (64, 64)

        # 3. 归一化 (Min-Max Normalization)
        g_min = g_processed.min()
        g_max = g_processed.max()
        g_normalized = (g_processed - g_min) / (g_max - g_min + 1e-8)

        g_np = g_normalized.detach().cpu().numpy()  # (H, W)

        # 用 colormap 映射到 RGB
        heatmap = cm.jet(g_np)[..., :3]  # (H, W, 3)，[0,1]
        return heatmap

    def __call__(
            self,
            global_step,
            rgb: Float[Tensor, "B H W C"],
            control_image: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
            rgb_as_latents=False,
            **kwargs,
    ):
        batch_size = rgb.shape[0]
        # assert batch_size == 1

        rgb_BCHW = rgb
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # image_cond = control_image
        image_cond = F.interpolate(
            control_image, (512, 512), mode="bilinear", align_corners=False
        )

        # temp = torch.zeros(batch_size).to(rgb.device)
        # text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, True
        )
        # 计算当前进度的线性衰减 delta
        # progress = global_step / self.total_steps
        # current_delta = self.delta_start - (self.delta_start - self.delta_end) * progress
        # current_delta = max(self.delta_end, current_delta)  # 确保不小于最小值
        # t_mid = int(self.t_max - (self.t_max - self.t_min) * np.log2(1 + (global_step // self.step_len) * self.step_len / self.total_steps))
        # t_low_raw = t_mid - current_delta
        # t_high_raw = t_mid + current_delta
        # t_low = max(self.t_min, t_low_raw)
        # t_high = min(self.t_max, t_high_raw)
        # if t_high <= t_low:
        #     t_high = t_low + 1
        # t = self.get_t(global_step, batch_size)
        # t = torch.randint(
        #     int(t_low),
        #     int(t_high),
        #     [batch_size],
        #     dtype=torch.long,
        #     device=self.device,
        # )
        # text_embeddings = text_embeddings[:batch_size * 2]
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        progress = global_step / self.total_steps
        progress = float(np.clip(progress, 0.0, 1.0))
        self.min_step = self.t_min + int((self.end_min - self.t_min) * progress)
        self.max_step = self.t_max - int((self.end_max - self.t_min) * progress)
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.edit_image:
            edit_latents = self.edit_latents(text_embeddings, latents, image_cond, t)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (512, 512), mode="bilinear")
            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

        if self.cfg.use_nfsd or self.cfg.use_dsd:
            grad = self.compute_grad_nfsd(text_embeddings, latents, image_cond, t)
        else:
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t)

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/experimental/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = cv2.resize(rgb_image, (512, 512))
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
