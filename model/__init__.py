from model.cfm import CFM
from model.dit import DiT

# Trainer is only needed for training, not inference
try:
    from model.trainer import Trainer
    __all__ = ["CFM", "DiT", "Trainer"]
except ImportError:
    # Trainer requires dataset module which is only available during training
    __all__ = ["CFM", "DiT"]
