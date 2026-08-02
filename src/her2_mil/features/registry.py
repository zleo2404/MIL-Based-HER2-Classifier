"""Factory for feature extractors.

To plug in a new foundation model:
  1. Write a class in this package implementing `FeatureExtractor` (base.py).
  2. Add one line to FEATURE_EXTRACTORS below.
  3. Set `features.extractor_name: <your_key>` in the config YAML.
No other file in the pipeline needs to change.
"""

from her2_mil.features.base import FeatureExtractor
from her2_mil.features.resnet import ResNet50FeatureExtractor
from her2_mil.features.uni import UNIFeatureExtractor

FEATURE_EXTRACTORS = {
    "resnet50": ResNet50FeatureExtractor,
    "uni": UNIFeatureExtractor
}


def build_feature_extractor(name: str, device: str = "cpu") -> FeatureExtractor:
    if name not in FEATURE_EXTRACTORS:
        available = ", ".join(FEATURE_EXTRACTORS)
        raise ValueError(f"Unknown feature extractor '{name}'. Available: {available}")
    return FEATURE_EXTRACTORS[name](device=device)
