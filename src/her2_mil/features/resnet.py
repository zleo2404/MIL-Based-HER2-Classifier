"""ResNet50 (ImageNet-pretrained) feature extractor -- current baseline."""
import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

from her2_mil.features.base import FeatureExtractor


class ResNet50FeatureExtractor(FeatureExtractor):
    feature_dim = 2048

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        backbone = create_feature_extractor(
            resnet50(weights=ResNet50_Weights.DEFAULT),
            return_nodes={"avgpool": "features"},
        )
        backbone.eval()
        self.model = backbone.to(device)

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        out = self.model(patch_tensor)
        return out["features"].squeeze()
