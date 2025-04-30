# src/model/models/FNO.py
import torch.nn as nn
from src.model.layers.fno_components import FNO  # Absolute import

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = FNO(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            n_modes=cfg.n_modes,
            hidden_channels=cfg.hidden_channels
        )

    def forward(self, x):
        if x.dim() == 3:  # Fixed typo: din() -> dim()
            x = x.unsqueeze(-1)
        return self.model(x)
