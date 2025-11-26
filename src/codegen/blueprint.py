from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Stage:
    # make 'type' optional with a sensible default
    type: str = "conv_block"
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
        # defensive conversion: allow stages to be list of dicts with missing fields
        raw_stages = d.get("stages", [])
        stages = []
        for s in raw_stages:
            if isinstance(s, Stage):
                stages.append(s)
            elif isinstance(s, dict):
                # ensure default values exist for missing keys
                s_copy = {
                    "type": s.get("type", "conv_block"),
                    "filters": s.get("filters", None),
                    "depth": s.get("depth", 1),
                    "kernel": s.get("kernel", 3),
                    "heads": s.get("heads", None),
                }
                stages.append(Stage(**s_copy))
            else:
                # unexpected type: skip or raise — here we skip
                continue

        return Blueprint(
            name=d.get("name", "blueprint"),
            input_shape=d["input_shape"],
            num_classes=d["num_classes"],
            backbone=d.get("backbone", "convnet"),
            stages=stages,
            head=d.get("head", {"type":"linear"}),
            quantize=d.get("quantize", False),
        )
