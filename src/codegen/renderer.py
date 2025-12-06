import torch
import torch.nn as nn
from typing import Optional, List
from .blueprint import Blueprint, Stage

def get_activation(name: str):
    name = (name or "").lower()
    if name in ("relu",):
        return lambda: nn.ReLU(inplace=True)
    if name in ("leaky_relu", "lrelu"):
        return lambda: nn.LeakyReLU(inplace=True)
    if name in ("silu", "swish"):
        return lambda: nn.SiLU()
    if name == "gelu":
        return lambda: nn.GELU()
    if name == "mish":
        return lambda: nn.Mish()
    if name == "elu":
        return lambda: nn.ELU(inplace=True)
    if name == "prelu":
        return lambda: nn.PReLU()
    if name == "hardswish":
        return lambda: nn.Hardswish()
    if name == "hardsigmoid":
        return lambda: nn.Hardsigmoid()
    if name == "tanh":
        return lambda: nn.Tanh()
    if name == "sigmoid":
        return lambda: nn.Sigmoid()
    if name in ("identity", "none", "linear"):
        return lambda: nn.Identity()
    return lambda: nn.ReLU(inplace=True)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        output = x.div(keep_prob) * random_tensor
        return output

class SEBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        mid = max(1, int(channels * se_ratio))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.fc(x).view(b, c, 1, 1)
        return x * y

class UniversalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, kernel, expansion=1, se_ratio=0.0, 
                 drop_path=0.0, activation="relu"):
        super().__init__()
        mid_ch = int(in_ch * expansion) if expansion > 1 else out_ch
        act_ctor = get_activation(activation)
        layers = []
        if expansion > 1:
            layers.append(nn.Conv2d(in_ch, mid_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(mid_ch))
            layers.append(act_ctor())
        spatial_in = mid_ch if expansion > 1 else in_ch
        layers.append(nn.Conv2d(spatial_in, mid_ch, kernel, stride=stride, padding=kernel//2, bias=False))
        layers.append(nn.BatchNorm2d(mid_ch))
        layers.append(act_ctor())
        if se_ratio and se_ratio > 0:
            layers.append(SEBlock(mid_ch, se_ratio))
        if expansion > 1:
            layers.append(nn.Conv2d(mid_ch, out_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
        self.body = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.body(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class GenericModel(nn.Module):
    def __init__(self, bp: Blueprint):
        super().__init__()
        in_c = bp.input_shape[0]
        stem_f = 32
        stem_cfg = getattr(bp, "stem", None)
        if stem_cfg:
            stem_f = getattr(stem_cfg, "filters", stem_f)
            kernel = getattr(stem_cfg, "kernel", 3)
            stride = getattr(stem_cfg, "stride", 1)
        else:
            kernel, stride = 3, 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, stem_f, kernel, stride=stride, padding=kernel//2, bias=False),
            nn.BatchNorm2d(stem_f),
            get_activation("relu")()
        )
        current_c = stem_f
        blocks = []
        total_depth = sum(int(s.depth) for s in bp.stages) if bp.stages else 1
        global_idx = 0
        for stage in bp.stages:
            is_bottleneck = getattr(stage, "type", "conv_block") == "bottleneck_block" or getattr(stage, "filters", 0) >= 128
            expansion = 4 if is_bottleneck else 1
            for i in range(int(stage.depth)):
                s = int(stage.stride) if i == 0 else 1
                dp_prob = 0.2 * (global_idx / total_depth)
                block = UniversalBlock(
                    in_ch=current_c,
                    out_ch=int(stage.filters),
                    stride=s,
                    kernel=int(stage.kernel),
                    expansion=expansion,
                    se_ratio=float(getattr(stage, "se_ratio", 0.0)),
                    drop_path=float(getattr(stage, "drop_path", dp_prob)),
                    activation=getattr(stage, "activation", "relu")
                )
                blocks.append(block)
                current_c = int(stage.filters)
                global_idx += 1
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        head_cfg = bp.head or {}
        dropout = float(head_cfg.get("dropout", 0.0))
        fc_layers = []
        head_units = head_cfg.get("units", None)
        if head_units and int(head_units) > 0:
            fc_layers.append(nn.Linear(current_c, int(head_units)))
            act_ctor = get_activation(head_cfg.get("activation", "relu"))
            fc_layers.append(act_ctor())
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(int(head_units), bp.num_classes))
        else:
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(current_c, bp.num_classes))
        self.head = nn.Sequential(*fc_layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x

def render_blueprint(bp: Blueprint) -> nn.Module:
    return GenericModel(bp)
