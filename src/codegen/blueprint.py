import json
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Any, Optional, Type

def safe_init(cls: Type, **kwargs):
    valid_fields = {f.name: f for f in fields(cls)}
    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k not in valid_fields:
            continue
        field_info = valid_fields[k]
        field_type = field_info.type
        if isinstance(v, dict) and hasattr(field_type, "__dataclass_fields__"):
            filtered_kwargs[k] = safe_init(field_type, **v)
        else:
            filtered_kwargs[k] = v
    try:
        return cls(**filtered_kwargs)
    except TypeError as e:
        raise TypeError(f"Failed to initialize {cls.__name__}: {e}") from e

@dataclass
class Stage:
    type: str = "conv_block"
    filters: int = 32
    depth: int = 1
    kernel: int = 3
    stride: int = 1
    padding: int = 1      
    dilation: int = 1        
    groups: int = 1          
    se_ratio: float = 0.0    
    activation: str = "relu"
    normalization: str = "bn"
    dropout: float = 0.0    

    def __post_init__(self):
        if self.filters < 1:
            raise ValueError(f"Filters must be > 0, got {self.filters}")
        if self.kernel < 1:
            raise ValueError(f"Kernel must be > 0, got {self.kernel}")
        if self.stride < 1:
            raise ValueError(f"Stride must be > 0, got {self.stride}")
        if not (0.0 <= self.se_ratio <= 1.0):
            raise ValueError(f"se_ratio must be between 0 and 1, got {self.se_ratio}")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")


@dataclass
class SkipConnection:
    from_idx: Optional[int] = None   
    to_idx: Optional[int] = None     
    type: str = "add"             

    def __post_init__(self):
        if self.from_idx is not None and self.from_idx < 0:
            raise ValueError("from_idx must be >= 0")
        if self.to_idx is not None and self.to_idx < 0:
            raise ValueError("to_idx must be >= 0")


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.0001

@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 128
    early_stopping: int = 10

@dataclass
class Blueprint:
    name: str
    input_shape: List[int]     
    num_classes: int

    backbone: str = "convnet"
    stages: List[Stage] = field(default_factory=list)
    head: Dict[str, Any] = field(default_factory=lambda: {"type": "linear", "units": 128})
    skip_connections: List[SkipConnection] = field(default_factory=list)

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    quantize: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.input_shape) != 3:
            raise ValueError(f"Input shape must be [C, H, W], got {self.input_shape}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Blueprint":
        if not isinstance(d, dict) or not d:
            raise ValueError("Blueprint.from_dict expects a non-empty dict")
        stages_raw = d.get("stages", [])
        stages = []
        for s in stages_raw:
            if isinstance(s, dict):
                stages.append(safe_init(Stage, **s))
            elif isinstance(s, Stage):
                stages.append(s)
            else:
                raise TypeError("Each stage must be a dict or Stage instance")
        skips_raw = d.get("skip_connections", [])
        skips = []
        for s in skips_raw:
            if isinstance(s, dict):
                skips.append(safe_init(SkipConnection, **s))
            elif isinstance(s, SkipConnection):
                skips.append(s)
            else:
                raise TypeError("Each skip_connection must be a dict or SkipConnection instance")
        opt_config = safe_init(OptimizerConfig, **d.get("optimizer", {}))
        train_config = safe_init(TrainingConfig, **d.get("training", {}))

        bp = cls(
            name=d.get("name", "unnamed"),
            input_shape=d.get("input_shape", [3, 32, 32]),
            num_classes=d.get("num_classes", 10),
            backbone=d.get("backbone", "convnet"),
            stages=stages,
            head=d.get("head", {"type": "linear", "units": 128}),
            skip_connections=skips,
            optimizer=opt_config,
            training=train_config,
            quantize=d.get("quantize", False),
            metadata=d.get("metadata", {})
        )
        bp._validate()
        return bp

    def _validate(self):
        if not (isinstance(self.input_shape, list) and len(self.input_shape) == 3):
            raise ValueError("input_shape must be a list of three integers [C, H, W]")
        n_stages = len(self.stages)
        for idx, s in enumerate(self.skip_connections):
            if s.from_idx is not None and (s.from_idx < 0 or s.from_idx >= n_stages):
                raise ValueError(f"skip_connections[{idx}].from_idx out of range: {s.from_idx}")
            if s.to_idx is not None and (s.to_idx < 0 or s.to_idx >= n_stages):
                raise ValueError(f"skip_connections[{idx}].to_idx out of range: {s.to_idx}")
        fc_units = None
        if isinstance(self.head, dict):
            fc_units = self.head.get("units")
        if fc_units is not None and fc_units <= 0:
            raise ValueError("head.units must be > 0")

if __name__ == "__main__":
    raw_json = {
        "name": "EvoNet_Gen5_Indiv4",
        "input_shape": [3, 64, 64],
        "num_classes": 10,
        "stages": [
            {"filters": 32, "kernel": 3, "stride": 2},
            {"filters": 64, "kernel": 5, "se_ratio": 0.25, "invalid_key": 999}
        ],
        "optimizer": {"lr": 0.005}
    }

    bp = Blueprint.from_dict(raw_json)
    print(f"Loaded: {bp.name}")
    print(f"Stage 2 Kernel: {bp.stages[1].kernel}")
    print(f"Learning Rate: {bp.optimizer.lr}")

    try:
        Blueprint.from_dict({"name": "Bad", "input_shape": [3, 32], "num_classes": 10})
    except ValueError as e:
        print(f"\nCaught Expected Error: {e}")
