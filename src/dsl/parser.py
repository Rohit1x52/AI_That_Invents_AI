import re
from typing import Dict

def parse_simple_dsl(dsl_str: str) -> Dict:
    # Example DSL: "task=img_cls; dataset=cifar10; target_device=mobile_v1; latency<20ms; params<10M; objectives=top1_acc,latency_ms"
    out = {}
    for part in dsl_str.split(";"):
        p = part.strip()
        if not p: 
            continue
        if "=" in p:
            k, v = [x.strip() for x in p.split("=", 1)]
            if k == "objectives":
                out["objectives"] = [x.strip() for x in v.split(",")]
            else:
                out[k] = v
        elif "<" in p:
            k, v = [x.strip() for x in re.split(r"<|>", p)]
            # convert units
            if v.endswith("ms"):
                val = float(v[:-2])
            elif v.endswith("M"):
                val = float(v[:-1]) * 1_000_000
            else:
                try:
                    val = float(v)
                except:
                    val = v
            if "constraints" not in out:
                out["constraints"] = {}
            out["constraints"][k] = val
    return out
