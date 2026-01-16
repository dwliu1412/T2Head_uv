
import argparse
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image
import copy
import torchvision


# ============================
# 1. 定义 Joint UNet 架构 (保持不变)
# ============================
class HumanGaussianUNet(nn.Module):
    def __init__(self, base_model_path="stablediffusionapi/realistic-vision-51"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")

        # === RGB Branch (Level 0) ===
        self.conv_in_rgb = self.unet.conv_in
        self.down_rgb = self.unet.down_blocks[0]
        self.up_rgb = self.unet.up_blocks[-1]
        self.conv_out_rgb = self.unet.conv_out

        # === Depth Branch (Level 0) ===
        self.conv_in_depth = copy.deepcopy(self.unet.conv_in)
        self.down_depth = copy.deepcopy(self.unet.down_blocks[0])
        self.up_depth = copy.deepcopy(self.unet.up_blocks[-1])
        self.conv_out_depth = copy.deepcopy(self.unet.conv_out)

        # === Shared Layers (Level 1, 2, 3) ===
        self.mid_block = self.unet.mid_block
        self.down_blocks_shared = self.unet.down_blocks[1:]  # Indices 1, 2, 3
        self.up_blocks_shared = self.unet.up_blocks[:-1]  # Indices 0, 1, 2

    def forward(self, sample_rgb, sample_depth, timestep, encoder_hidden_states):
        # 1. Embedding
        t_emb = self.unet.time_proj(timestep)
        emb = self.unet.time_embedding(t_emb)

        # =========================
        # Down 0 (RGB)
        # =========================
        h_rgb = self.conv_in_rgb(sample_rgb)
        rgb_res_stack = [h_rgb]  # ✅ 一定要把 conv_in 输出放进去（原版 UNet 就是这么做的）

        h_rgb, res_rgb = self.down_rgb(h_rgb, emb, encoder_hidden_states)

        # res_rgb 通常 = (res1, res2, downsample_out)；我们把 downsample_out 留给 shared
        if hasattr(self.down_rgb, "downsamplers") and self.down_rgb.downsamplers is not None and len(
                self.down_rgb.downsamplers) > 0:
            rgb_res_stack.extend(list(res_rgb[:-1]))  # ✅ 只保留 resnet 输出
            # res_rgb[-1] == downsample_out == h_rgb
        else:
            rgb_res_stack.extend(list(res_rgb))

        # =========================
        # Down 0 (Depth)
        # =========================
        h_depth = self.conv_in_depth(sample_depth)
        depth_res_stack = [h_depth]  # ✅ conv_in 输出

        h_depth, res_depth = self.down_depth(h_depth, emb, encoder_hidden_states)

        if hasattr(self.down_depth, "downsamplers") and self.down_depth.downsamplers is not None and len(
                self.down_depth.downsamplers) > 0:
            depth_res_stack.extend(list(res_depth[:-1]))
        else:
            depth_res_stack.extend(list(res_depth))

        # =========================
        # Shared Down (1,2,3)
        # =========================
        h_joint = torch.cat([h_rgb, h_depth], dim=0)
        emb_joint = torch.cat([emb, emb], dim=0)
        enc_joint = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)

        shared_res_stack = [h_joint]  # ✅ 关键：把 Down0 的 downsample 输出作为 shared 的“起始 residual”
        #    这正是某些 upblock 会跨界需要的那一个

        for block in self.down_blocks_shared:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                h_joint, res_tuple = block(hidden_states=h_joint, temb=emb_joint, encoder_hidden_states=enc_joint)
            else:
                h_joint, res_tuple = block(hidden_states=h_joint, temb=emb_joint)

            shared_res_stack.extend(list(res_tuple))

        # =========================
        # Mid
        # =========================
        h_joint = self.mid_block(h_joint, emb_joint, encoder_hidden_states=enc_joint)

        # =========================
        # Up shared (0,1,2)
        # =========================
        for block in self.up_blocks_shared:
            need = len(block.resnets)
            # ✅ 建议加个断言，后续再出错能第一时间定位
            # assert len(shared_res_stack) >= need, f"shared_res_stack not enough: need {need}, got {len(shared_res_stack)}"

            res_samples = shared_res_stack[-need:]
            shared_res_stack = shared_res_stack[:-need]
            res_samples = tuple(res_samples)

            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                h_joint = block(
                    hidden_states=h_joint,
                    temb=emb_joint,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=enc_joint
                )
            else:
                h_joint = block(
                    hidden_states=h_joint,
                    temb=emb_joint,
                    res_hidden_states_tuple=res_samples
                )

        # split
        h_rgb_mid, h_depth_mid = torch.chunk(h_joint, 2, dim=0)

        # =========================
        # RGB Exit Up (last)
        # =========================
        res_rgb_len = len(self.up_rgb.resnets)
        res_rgb = rgb_res_stack[-res_rgb_len:]

        h_rgb = self.up_rgb(
            hidden_states=h_rgb_mid,
            temb=emb,
            res_hidden_states_tuple=tuple(res_rgb),
            encoder_hidden_states=encoder_hidden_states
        )

        # ✅ 修正顺序：norm->act->conv_out（你原来是 conv_out 后 act，顺序不对）
        h_rgb = self.unet.conv_norm_out(h_rgb)
        h_rgb = self.unet.conv_act(h_rgb)
        out_rgb = self.conv_out_rgb(h_rgb)

        # =========================
        # Depth Exit Up (last)
        # =========================
        res_depth_len = len(self.up_depth.resnets)
        res_depth = depth_res_stack[-res_depth_len:]

        h_depth = self.up_depth(
            hidden_states=h_depth_mid,
            temb=emb,
            res_hidden_states_tuple=tuple(res_depth),
            encoder_hidden_states=encoder_hidden_states
        )

        h_depth = self.unet.conv_norm_out(h_depth)
        h_depth = self.unet.conv_act(h_depth)
        out_depth = self.conv_out_depth(h_depth)

        return out_rgb, out_depth


# ============================
# 2. 数据集加载 (保持不变)
# ============================
class HumanGaussianDataset(Dataset):
    def __init__(self, json_path, tokenizer, size=512):
        self.data = []
        with open(json_path, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict):
                self.data = list(content.values())
            else:
                self.data = content
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image']
        # 简单处理路径，确保它是绝对路径或者相对于脚本的路径
        if not os.path.exists(img_path):
            # 假设 json 里是相对路径，根据需要调整
            pass

        image = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5) - 1.0

        depth_path = item['depth']
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / 65535.0
        depth = (depth * 2.0) - 1.0
        depth = np.expand_dims(depth, axis=-1)

        caption = item['caption']
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values_rgb": torch.tensor(image).permute(2, 0, 1),
            "pixel_values_depth": torch.tensor(depth).permute(2, 0, 1),
            "input_ids": inputs.input_ids[0]
        }


# ============================
# 3. 可视化采样函数 (新增)
# ============================
def save_image_logs(args, step, model, vae, text_encoder, tokenizer, scheduler, accelerator):
    """
    运行一个简化的推理循环来生成图片进行可视化验证
    """
    print(f"\n[Validation] Running sampling at step {step}...")

    # 1. 设置验证 Prompt
    validation_prompts = [
        "a woman with curly hair and hoop earrings",
        "a portrait photo of a man, realistic, high detail",
    ]

    model.eval()
    vae.eval()
    text_encoder.eval()

    # 2. 准备 Latents
    height = 512
    width = 512
    num_inference_steps = 20
    guidance_scale = 7.5

    # 使用独立的 Scheduler 进行推理 (避免影响训练状态)
    eval_scheduler = copy.deepcopy(scheduler)
    eval_scheduler.set_timesteps(num_inference_steps)

    # 随机噪声
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    latents_rgb = torch.randn((len(validation_prompts), 4, height // 8, width // 8), generator=generator,
                              device=accelerator.device, dtype=torch.float16)
    latents_depth = torch.randn((len(validation_prompts), 4, height // 8, width // 8), generator=generator,
                                device=accelerator.device, dtype=torch.float16)

    # Encode Prompts
    text_input = tokenizer(validation_prompts, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]

        # Uncond embeddings for Classifier Free Guidance
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * len(validation_prompts), padding="max_length", max_length=max_length,
                                 return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]

        # Concat [uncond, cond]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 3. 采样循环
    for t in tqdm(eval_scheduler.timesteps, disable=True):
        # 扩展 Latents 以适应 CFG (Batch Size * 2)
        latent_model_input_rgb = torch.cat([latents_rgb] * 2)
        latent_model_input_depth = torch.cat([latents_depth] * 2)

        latent_model_input_rgb = eval_scheduler.scale_model_input(latent_model_input_rgb, t)
        latent_model_input_depth = eval_scheduler.scale_model_input(latent_model_input_depth, t)

        t_batch = torch.full(
            (latent_model_input_rgb.shape[0],),
            int(t.item()) if torch.is_tensor(t) else int(t),
            device=latent_model_input_rgb.device,
            dtype=torch.long
        )

        # 预测噪声
        with torch.no_grad():
            noise_pred_rgb, noise_pred_depth = model(
                latent_model_input_rgb,
                latent_model_input_depth,
                t_batch,
                encoder_hidden_states=text_embeddings
            )

        # CFG
        noise_pred_uncond_rgb, noise_pred_text_rgb = noise_pred_rgb.chunk(2)
        noise_pred_rgb = noise_pred_uncond_rgb + guidance_scale * (noise_pred_text_rgb - noise_pred_uncond_rgb)

        noise_pred_uncond_depth, noise_pred_text_depth = noise_pred_depth.chunk(2)
        noise_pred_depth = noise_pred_uncond_depth + guidance_scale * (noise_pred_text_depth - noise_pred_uncond_depth)

        # Step
        latents_rgb = eval_scheduler.step(noise_pred_rgb, t, latents_rgb).prev_sample
        latents_depth = eval_scheduler.step(noise_pred_depth, t, latents_depth).prev_sample
        latents_rgb = latents_rgb.to(dtype=torch.float16)
        latents_depth = latents_depth.to(dtype=torch.float16)

    # 4. 解码 Latents
    with torch.no_grad():
        latents_rgb = (latents_rgb / vae.config.scaling_factor).to(dtype=vae.dtype)
        image_rgb = vae.decode(latents_rgb).sample

        latents_depth = (latents_depth / vae.config.scaling_factor).to(dtype=vae.dtype)
        image_depth = vae.decode(latents_depth).sample  # 注意：这里 Depth 解码出来是 3 通道的

    # 5. 后处理与保存
    # RGB: [-1, 1] -> [0, 1]
    image_rgb = (image_rgb / 2 + 0.5).clamp(0, 1)
    # Depth: [-1, 1] -> [0, 1] (取第一通道，或者保持3通道如果是灰度)
    image_depth = (image_depth / 2 + 0.5).clamp(0, 1)
    # 为了可视化清楚，如果 depth 是 3 通道且通过 rgb-vae 解码，通常它是灰度的，直接保存即可

    save_dir = os.path.join(args.output_dir, "validation_images")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(validation_prompts)):
        # 保存 RGB
        torchvision.utils.save_image(image_rgb[i], os.path.join(save_dir, f"step_{step}_img_{i}_rgb.png"))
        # 保存 Depth
        torchvision.utils.save_image(image_depth[i], os.path.join(save_dir, f"step_{step}_img_{i}_depth.png"))

    print(f"[Validation] Images saved to {save_dir}")
    model.train()  # 恢复训练模式


# ============================
# 4. 训练主流程 (修改版)
# ============================
def train(args):
    # Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Models
    model_id = "../HeadStudio_lib/realistic-vision-51"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.to(accelerator.device, dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Load Custom UNet
    model = HumanGaussianUNet(model_id)

    # Freeze VAE & Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Dataset
    dataset = HumanGaussianDataset(args.json_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Prepare
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Variables for tracking
    global_step = 0
    save_interval = 2000

    print(f"Start training... Total Epochs: {args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                # VAE Encode
                latents_rgb = vae.encode(batch["pixel_values_rgb"].to(dtype=torch.float16)).latent_dist.sample()
                latents_rgb = latents_rgb * vae.config.scaling_factor

                depth_3ch = batch["pixel_values_depth"].repeat(1, 3, 1, 1).to(dtype=torch.float16)
                latents_depth = vae.encode(depth_3ch).latent_dist.sample()
                latents_depth = latents_depth * vae.config.scaling_factor

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Sample Noise
            noise_rgb = torch.randn_like(latents_rgb)
            noise_depth = torch.randn_like(latents_depth)
            bsz = latents_rgb.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_rgb.device)

            # Add Noise
            noisy_latents_rgb = noise_scheduler.add_noise(latents_rgb, noise_rgb, timesteps)
            noisy_latents_depth = noise_scheduler.add_noise(latents_depth, noise_depth, timesteps)

            # Predict
            with accelerator.autocast():
                pred_rgb, pred_depth = model(noisy_latents_rgb, noisy_latents_depth, timesteps, encoder_hidden_states)

                # Loss
                loss_rgb = F.mse_loss(pred_rgb.float(), noise_rgb.float(), reduction="mean")
                loss_depth = F.mse_loss(pred_depth.float(), noise_depth.float(), reduction="mean")
                loss = loss_rgb + loss_depth

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # Update Progress Bar
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

            # === 保存与可视化逻辑 ===
            if global_step % save_interval == 0:
                if accelerator.is_local_main_process:
                    # 1. 保存权重
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                    print(f"\nSaving model checkpoint to {checkpoint_path}")
                    # unwrap model to save pure state_dict
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), checkpoint_path)

                    # 2. 运行可视化
                    save_image_logs(
                        args=args,
                        step=global_step,
                        model=unwrapped_model,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=noise_scheduler,
                        accelerator=accelerator
                    )

        # Epoch 结束也可以保存一次
        if accelerator.is_local_main_process:
            epoch_ckpt_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}.pth")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), epoch_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="H:/dataset/Filtered-Laion-Face/filtered_laion_faces/new_face_rgb_depth.json",
                        help="Path to dataset json")
    parser.add_argument("--output_dir", type=str, default="outputs/depth_unet", help="Dir to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args)
