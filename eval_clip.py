import torch
import torch.nn.functional as F
from typing import List, Union, Dict, Iterable
from PIL import Image
from torchvision import transforms
from threestudio.utils.clip_eval import CLIPTextImageEvaluator

# --- 尝试导入 clip ---
try:
    import clip  # type: ignore
except ImportError as e:
    clip = None


# ==========================================
#              主执行逻辑
# ==========================================

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """加载图像并转换为 (1, 3, H, W) 的 Tensor，值域 [0, 1]"""
    try:
        image = Image.open(image_path).convert("RGB")
        # 转换为 Tensor，ToTensor() 自动将 [0, 255] 映射到 [0.0, 1.0]
        transform = transforms.ToTensor()
        tensor = transform(image)
        # 增加 Batch 维度 -> (1, 3, H, W)
        return tensor.unsqueeze(0)
    except Exception as e:
        print(f"无法加载图像 '{image_path}': {e}")
        print("使用随机噪声图像代替进行演示...")
        return torch.rand(1, 3, 512, 512)


def main():
    # 1. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 定义要评估的模型列表
    target_models = ["ViT-B/16", "ViT-B/32", "ViT-L/14"]
    clip_model_root = "../HeadStudio_lib/clip"

    # 3. 初始化评估器
    try:
        evaluator = CLIPTextImageEvaluator(
            device=device,
            model_name=target_models,
            model_root=clip_model_root,
        )
    except RuntimeError as e:
        print(e)
        return

    # 4. 准备输入数据
    # 这里请替换为你真实的图片路径
    image_path = r"F:\Media_Department\head_avatar\text2avatar\baseline\T2Head_uv\outputs\headstudio\20260109-070652\save\it5300-2.png"
    image_tensor = load_image_as_tensor(image_path)

    # 定义一组文本提示 (Prompts)
    prompts = [
        "a DSLR portrait of Lionel Messi, masterpiece, Studio Quality, 8k, ultra-HD, next generation",
        "a photo of a dog",
    ]

    print(f"\nEvaluating image against prompts: {prompts}")

    # 5. 计算相似度
    # 返回的字典格式: { "model_name": tensor([max_score_for_image_1, ...]) }
    similarity_results = evaluator.compute_similarity(image_tensor, prompts)

    # 6. 打印结果
    print("\n" + "=" * 40)
    print(f"Max Similarity Scores (Image vs Prompts)")
    print("=" * 40)

    for model_name, score_tensor in similarity_results.items():
        # score_tensor 是 (Batch_Size,)，这里 Batch=1，所以取第一个元素
        score = score_tensor.item()
        print(f"Model: {model_name:<10} | Max Similarity: {score:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()