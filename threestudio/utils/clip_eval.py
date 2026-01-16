import torch
import torch.nn.functional as F
from typing import List, Union, Dict, Iterable

try:
    import clip  # type: ignore
except ImportError as e:  # pragma: no cover - runtime dependency
    clip = None


class CLIPTextImageEvaluator:
    """Utility to evaluate text-image similarity with one or more CLIP models.

    By default it loads a single model (e.g. "ViT-L/14"), but you can also
    pass a list of model names such as ["ViT-B/16", "ViT-B/32", "ViT-L/14"].
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        model_name: Union[str, Iterable[str]] = "ViT-L/14",
        model_root: str = None,
    ) -> None:
        if clip is None:
            raise RuntimeError(
                "CLIP library is not installed. Install it via 'pip install openai-clip' or 'pip install git+https://github.com/openai/CLIP.git'."
            )
        self.device = torch.device(device)
        self.model_root = model_root

        # Normalize model_name to a list
        if isinstance(model_name, str):
            model_names = [model_name]
        else:
            model_names = list(model_name)

        self.models: Dict[str, torch.nn.Module] = {}
        for name in model_names:
            # If model_root is provided, let clip.load cache/load there; otherwise use default cache
            if self.model_root is not None:
                model, _ = clip.load(name, device=self.device, download_root=self.model_root)
            else:
                model, _ = clip.load(name, device=self.device)
            model.eval()
            self.models[name] = model

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text once and reuse across models (they share tokenizer)."""
        text_tokens = clip.tokenize(prompts).to(self.device)
        # Note: text encoder weights differ per model, so we call per-model later.
        return text_tokens

    @torch.no_grad()
    def _encode_text_with_model(self, model: torch.nn.Module, text_tokens: torch.Tensor) -> torch.Tensor:
        text_features = model.encode_text(text_tokens)
        return F.normalize(text_features, dim=-1)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Pre-normalize images to [0,1] and 224x224; CLIP-specific norm is inside model.

        This returns the resized tensor; actual encoding is per-model.
        """
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.0 or images.min() < -1.0:
            images = images.clamp(-1.0, 1.0)
        # bring to [0,1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        images = images.to(self.device)
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return images

    @torch.no_grad()
    def _encode_images_with_model(self, model: torch.nn.Module, images_resized: torch.Tensor) -> torch.Tensor:
        image_features = model.encode_image(images_resized)
        return F.normalize(image_features, dim=-1)

    @torch.no_grad()
    def compute_similarity(self, images: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Compute cosine similarity between images and text for all loaded models.

        Args:
            images: (B,3,H,W) tensor.
            prompts: list of prompt strings.
        Returns:
            A dict mapping model name -> similarity tensor of shape (B,).
        """
        text_tokens = self.encode_text(prompts)  # shared tokens
        images_resized = self.encode_images(images)

        sims: Dict[str, torch.Tensor] = {}
        for name, model in self.models.items():
            text_features = self._encode_text_with_model(model, text_tokens)  # (P,D)
            image_features = self._encode_images_with_model(model, images_resized)  # (B,D)
            sim = image_features @ text_features.t()  # (B,P)
            sim_max, _ = sim.max(dim=1)
            sims[name] = sim_max
        return sims
