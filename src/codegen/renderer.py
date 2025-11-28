# src/codegen/renderer.py
import torch
import torch.nn as nn
from typing import Callable, Any
from .blueprint import Blueprint, Stage

def conv3x3(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=bias)

class SEBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride=1, se_ratio=0.0, downsample=None):
        super().__init__()
        mid_ch = out_ch
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv3x3(mid_ch, mid_ch, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = conv1x1(mid_ch, out_ch * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = SEBlock(out_ch * self.expansion, se_ratio) if se_ratio and se_ratio > 0 else None

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCustomBackbone(nn.Module):
    def __init__(self, bp: Blueprint):
        super().__init__()
        # Stem
        in_ch = bp.input_shape[0]
        stem_filters = bp.__dict__.get("stem", {}).get("filters", 32) if hasattr(bp, "__dict__") else 32
        # If blueprint.stem exists as dict (from JSON), handle both dict or Stage forms.
        try:
            stem_cfg = bp.__dict__.get("stem", {}) if hasattr(bp, "__dict__") else {}
            stem_filters = int(stem_cfg.get("filters", stem_filters))
        except Exception:
            stem_filters = stem_filters

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_filters),
            nn.ReLU(inplace=True)
        )
        in_ch = stem_filters

        layers = []
        # bp.stages can be list of Stage dataclasses or dicts
        for s in bp.stages:
            # normalize stage dict-like access
            if isinstance(s, Stage):
                typ = s.type
                filters = int(s.filters)
                depth = int(s.depth)
                stride = int(s.stride) if hasattr(s, "stride") else 1
                se_ratio = float(getattr(s, "se_ratio", 0.0) or 0.0)
            else:
                typ = s.get("type", "bottleneck_block")
                filters = int(s.get("filters", 64))
                depth = int(s.get("depth", 1))
                stride = int(s.get("stride", 1))
                # support both 'se_ratio' and 'squeeze_excitation' boolean
                if "se_ratio" in s and s.get("se_ratio") is not None:
                    se_ratio = float(s.get("se_ratio", 0.0))
                elif s.get("squeeze_excitation", False):
                    # default ratio if boolean flag used without explicit ratio
                    se_ratio = float(s.get("se_ratio", 0.25))
                else:
                    se_ratio = 0.0

            # Build 'depth' bottleneck blocks for this stage
            for i in range(depth):
                cur_stride = stride if i == 0 else 1
                out_ch = filters
                downsample = None
                target_ch = out_ch * Bottleneck.expansion
                if cur_stride != 1 or in_ch != target_ch:
                    downsample = nn.Sequential(
                        conv1x1(in_ch, target_ch, stride=cur_stride),
                        nn.BatchNorm2d(target_ch)
                    )
                block = Bottleneck(in_ch, out_ch, stride=cur_stride, se_ratio=se_ratio, downsample=downsample)
                layers.append(block)
                in_ch = target_ch

        self.features = nn.Sequential(*layers) if layers else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, bp.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

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
            # stage may be Stage or dict
            kernel = s.kernel if isinstance(s, Stage) else s.get("kernel", 3)
            depth = s.depth if isinstance(s, Stage) else int(s.get("depth", 1))
            filters = s.filters if isinstance(s, Stage) else int(s.get("filters", 32))
            stride = s.stride if isinstance(s, Stage) else int(s.get("stride", 1))
            se_ratio = getattr(s, "se_ratio", None) if isinstance(s, Stage) else (s.get("se_ratio", 0.0) or (0.25 if s.get("squeeze_excitation", False) else 0.0))
            for i in range(int(depth)):
                st = stride if i == 0 else 1
                layers.append(conv_block(in_ch, int(filters), kernel=int(kernel), stride=st))
                if se_ratio and float(se_ratio) > 0.0:
                    layers.append(SEBlock(int(filters), float(se_ratio)))
                in_ch = int(filters)
        self.features = nn.Sequential(*layers) if layers else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, bp.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

def render_blueprint(bp: Blueprint) -> nn.Module:
    backbone = bp.backbone.lower() if isinstance(bp.backbone, str) else "convnet"
    if backbone in ["convnet", "simple", "conv_mixer"]:
        return SimpleBackbone(bp)
    elif backbone in ["resnet_custom", "resnet"]:
        return ResNetCustomBackbone(bp)
    else:
        raise NotImplementedError(f"Backbone '{bp.backbone}' not supported in MVP")
