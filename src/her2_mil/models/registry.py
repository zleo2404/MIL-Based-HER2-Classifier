"""Factory for MIL aggregators.

To plug in a new MIL architecture:
  1. Write a class in this package implementing `MILModel` (base.py) with
     constructor signature (input_dim, hidden_dim, num_classes, dropout).
  2. Add one line to MIL_MODELS below.
  3. Set `model.name: <your_key>` in the config YAML.
No other file in the pipeline needs to change.
"""
from her2_mil.models.abmil import ABMIL
from her2_mil.models.base import MILModel
from her2_mil.models.transmil import TransMIL

MIL_MODELS = {
    "abmil": ABMIL,
    "transmil": TransMIL,
}


def build_model(name: str, **kwargs) -> MILModel:
    if name not in MIL_MODELS:
        available = ", ".join(MIL_MODELS)
        raise ValueError(f"Unknown MIL model '{name}'. Available: {available}")
    return MIL_MODELS[name](**kwargs)
