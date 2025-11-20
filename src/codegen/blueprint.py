from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Stage:
    type: str
    filters: int = None
    depth: int = 1
    kernel: int = 3
    heads: int = None

@dataclass
class Blueprint:
    name: str
    input_shape: List[int]      # e.g., [3,32,32]
    num_classes: int
    backbone: str
    stages: List[Stage]
    head: Dict[str, Any] = field(default_factory=lambda: {"type": "linear"})
    quantize: bool = False

    @staticmethod
    def from_dict(d: dict) -> "Blueprint":
        stages = [Stage(**s) for s in d.get("stages", [])]
        return Blueprint(
            name=d.get("name", "blueprint"),
            input_shape=d["input_shape"],
            num_classes=d["num_classes"],
            backbone=d.get("backbone", "convnet"),
            stages=stages,
            head=d.get("head", {"type":"linear"}),
            quantize=d.get("quantize", False),
        )
