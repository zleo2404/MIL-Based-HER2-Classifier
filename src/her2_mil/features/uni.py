"""
UNI foundation model feature extractor.

UNI is a ViT-L/16 pretrained on ~100M histopathology patches. Weights are
hosted on a *gated* HuggingFace repo, so before using this extractor you
must request access at https://huggingface.co/MahmoodLab/UNI

Reference: Chen et al., "Towards a general-purpose foundation model for
computational pathology", Nature Medicine 2024.
"""
import torch

from her2_mil.features.base import FeatureExtractor


class UNIFeatureExtractor(FeatureExtractor):
    feature_dim = 1024

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "UNI requires `timm`. Install with `pip install timm huggingface_hub` "
                "and authenticate with `huggingface-cli login` (gated repo)."
            ) from e

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.model.eval()
        self.model = self.model.to(device)

    def get_transform(self):
        """Use UNI's own expected preprocessing."""
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        data_config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        return create_transform(**data_config)

    @torch.no_grad()
    def extract(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(patch_tensor).squeeze()
