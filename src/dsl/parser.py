import re
from typing import Dict, Any, List, Union

def parse_value_with_units(value_str: str) -> Union[float, str, int]:
    value_str = value_str.strip()
    
    match = re.match(r"^([\d\.]+)\s*([a-zA-Z%]*)$", value_str)
    
    if not match:
        return value_str
        
    num_str, unit = match.groups()
    try:
        val = float(num_str)
    except ValueError:
        return value_str 

    unit = unit.lower()
    
    if unit in ['ms']: return val
    if unit in ['s', 'sec']: return val * 1000.0
    if unit in ['m', 'min']: return val * 60000.0
    
    if unit == 'k': return val * 1e3
    if unit == 'm': return val * 1e6 
    if unit == 'b': return val * 1e9
    
    if unit == 'mb': return val
    if unit == 'gb': return val * 1024.0
    
    if unit == '%': return val / 100.0
    
    return val

def parse_advanced_dsl(dsl_str: str) -> Dict[str, Any]:
    config = {
        "parameters": {},
        "constraints": [],
        "search_hints": []
    }
    
    parts = [p.strip() for p in dsl_str.split(";") if p.strip()]
    
    pattern = re.compile(r"^([\w\.]+)\s*(=|<=|>=|<|>)\s*(.+)$")

    for part in parts:
        match = pattern.match(part)
        if not match:
            print(f" Warning: Could not parse segment '{part}'")
            continue
            
        key, op, raw_val = match.groups()
        parsed_val = parse_value_with_units(raw_val)
        
        if op == "=":
            if isinstance(raw_val, str) and "," in raw_val:
                parsed_val = [x.strip() for x in raw_val.split(",")]
            
            if key in ["mutate", "search"]:
                config["search_hints"].append({key: parsed_val})
            else:
                config["parameters"][key] = parsed_val
                
        else:
            config["constraints"].append({
                "metric": key,
                "operator": op,
                "threshold": parsed_val,
                "raw": f"{key} {op} {raw_val}"
            })

    return config

if __name__ == "__main__":
    query = "task=segmentation; accuracy >= 95%; latency < 20ms; params < 5M; allowed_ops=conv,attn"
    
    result = parse_advanced_dsl(query)
    
    import json
    print(json.dumps(result, indent=2))