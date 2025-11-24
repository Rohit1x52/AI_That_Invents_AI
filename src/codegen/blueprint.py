import json
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

def safe_init(cls, **kwargs):
    if hasattr(cls, "__dataclass_fields__"):
        names = cls.__dataclass_fields__.keys()
    else:
        names = set()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in names}
    return cls(**filtered_kwargs)


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.0005
    amsgrad: bool = False
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HeadConfig:
    type: str = "linear"
    units: int = 128
    activation: str = "relu"
    dropout: float = 0.0
    
@dataclass
class StemConfig:
    type: str = "conv"
    filters: int = 32
    kernel: int = 3
    stride: int = 1
    activation: str = "relu"

@dataclass
class Stage:
    type: str
    filters: int
    depth: int = 1
    kernel: int = 3
    stride: int = 1
    padding: int = 1
    activation: str = "relu"
    batch_norm: bool = True
    separable: bool = False
    squeeze_excitation: bool = False
    se_ratio: float = 0.0
    drop_path: float = 0.0
    
    def __post_init__(self):
        # Basic validation
        if self.stride < 1:
            raise ValueError(f"Stride must be >= 1, got {self.stride}")

@dataclass
class Blueprint:
    name: str
    input_shape: List[int]      
    num_classes: int
    backbone: str = "convnet"
    
    # Components
    stem: Optional[StemConfig] = None
    stages: List[Stage] = field(default_factory=list)
    head: Dict[str, Any] = field(default_factory=dict) 
    skip_connections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Hyperparameters
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Metadata
    quantize: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.input_shape) != 3:
            raise ValueError(f"input_shape must be [C, H, W], got {self.input_shape}")

    @classmethod
    def from_dict(cls, d: dict) -> "Blueprint":
        # Parse Stages
        stages_data = d.get("stages", [])
        stages = [safe_init(Stage, **s) for s in stages_data]
        
        # Parse Stem
        stem_data = d.get("stem")
        stem = safe_init(StemConfig, **stem_data) if stem_data else None

        # Parse Optimizer
        opt_data = d.get("optimizer", {})
        optimizer = safe_init(OptimizerConfig, **opt_data)

        # Parse Training
        train_data = d.get("training", {})
        training = safe_init(TrainingConfig, **train_data)

        # Main Init
        return cls(
            name=d.get("name", "blueprint"),
            input_shape=d.get("input_shape", [3, 224, 224]),
            num_classes=d.get("num_classes", 1000),
            backbone=d.get("backbone", "convnet"),
            stem=stem,
            stages=stages,
            head=d.get("head", {}),
            skip_connections=d.get("skip_connections", []),
            optimizer=optimizer,
            training=training,
            quantize=d.get("quantize", {"enabled": False}),
            metadata=d.get("metadata", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "Blueprint":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str, indent: int = 2):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

if __name__ == "__main__":
    # Simulating the JSON data input
    raw_data = {
        "name": "conv_pro_test",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "optimizer": {"type": "adamw", "lr": 0.005}, # Partial override
        "stages": [
            {"type": "conv_block", "filters": 32, "kernel": 3, "activation": "silu"},
            {"type": "bottleneck", "filters": 64, "stride": 2, "se_ratio": 0.25}
        ]
    }

    # Load
    bp = Blueprint.from_dict(raw_data)
    
    print(f"Loaded Blueprint: {bp.name}")
    print(f"Optimizer LR: {bp.optimizer.lr}")
    print(f"Stage 1 Activation: {bp.stages[0].activation}")
    print(f"Stage 2 SE Ratio: {bp.stages[1].se_ratio}")