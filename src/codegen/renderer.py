import torch
import torch.nn as nn
from typing import Callable
from .blueprint import Blueprint, Stage

def conv_block(in_ch, out_ch, kernel=3, stride=1, norm=True, activation=nn.ReLU):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, stride=stride, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(activation(inplace=True))
    return nn.Sequential(*layers)

class SimpleBackbone(nn.Module):
    def __init__(self, bp: Blueprint):
        super().__init__()
        in_ch = bp.input_shape[0]
        layers = []
        for s in bp.stages:
            for i in range(s.depth):
                out_ch = s.filters
                layers.append(conv_block(in_ch, out_ch, kernel=s.kernel))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, bp.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

def render_blueprint(bp: Blueprint) -> nn.Module:
    # For Phase 1: a simple mapping — support conv-based blueprints
    if bp.backbone in ["convnet", "conv_mixer", "simple"]:
        return SimpleBackbone(bp)
    else:
        raise NotImplementedError(f"Backbone '{bp.backbone}' not supported in MVP")
