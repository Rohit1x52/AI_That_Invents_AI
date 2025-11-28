from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Stage:
    type: str = "conv_block"       
    filters: int = 32
    depth: int = 1
    kernel: int = 3
    stride: int = 1
    se_ratio: float = 0.0          
    activation: str = "relu"
    normalization: str = "bn"      
    heads: int = None

@dataclass
class Blueprint:
    name: str
    input_shape: List[int]      # e.g., [3,32,32]
    num_classes: int
    backbone: str = "convnet"
    stages: List[Stage] = field(default_factory=list)
    head: Dict[str, Any] = field(default_factory=lambda: {"type": "linear"})
    quantize: bool = False

    @staticmethod
    def from_dict(d: dict) -> "Blueprint":
        raw_stages = d.get("stages", [])
        stages = []
        for s in raw_stages:
            if isinstance(s, Stage):
                stages.append(s)
            elif isinstance(s, dict):
                s_copy = {
                    "type": s.get("type", "conv_block"),
                    "filters": s.get("filters", 32),
                    "depth": s.get("depth", 1),
                    "kernel": s.get("kernel", 3),
                    "stride": s.get("stride", 1),
                    "se_ratio": s.get("se_ratio", 0.0),
                    "activation": s.get("activation", "relu"),
                    "normalization": s.get("normalization", "bn"),
                    "heads": s.get("heads", None)
                }
                stages.append(Stage(**s_copy))
            else:
                continue

        return Blueprint(
            name=d.get("name", "blueprint"),
            input_shape=d.get("input_shape", [3,32,32]),
            num_classes=d.get("num_classes", 10),
            backbone=d.get("backbone", "convnet"),
            stages=stages,
            head=d.get("head", {"type":"linear"}),
            quantize=d.get("quantize", False),
        )
