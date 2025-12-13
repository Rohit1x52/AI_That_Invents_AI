import json
from pathlib import Path
from typing import Dict, Any, Optional
from jsonschema import Draft7Validator, ValidationError

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
        "dataset": {"type": "string"},
        "objectives": {"type": "array", "default": ["accuracy", "latency"]},
        "constraints": {
            "type": "object", 
            "default": {},
            "properties": {
                "max_params": {"type": "integer", "default": 5000000}
            }
        }
    },
    "required": ["task", "dataset"]
}

class SpecNormalizer:
    def __init__(self, schema_path: Optional[str] = None):
        self.schema = DEFAULT_SCHEMA
        if schema_path:
            p = Path(schema_path)
            if p.exists():
                try:
                    self.schema = json.loads(p.read_text())
                except Exception:
                    pass
        
        self.validator = Draft7Validator(self.schema)

    def validate(self, spec: Dict[str, Any]) -> bool:
        return self.validator.is_valid(spec)

    def normalize(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validator.is_valid(spec):
            self.validator.validate(spec) 
        
        return self._inject_defaults(spec, self.schema)

    def _inject_defaults(self, instance: Dict, schema: Dict) -> Dict:
        if not isinstance(instance, dict):
            return instance
        
        out = instance.copy()
        
        if "properties" in schema:
            for prop, subschema in schema["properties"].items():
                if prop not in out and "default" in subschema:
                    out[prop] = subschema["default"]
                
                if prop in out and subschema.get("type") == "object":
                    out[prop] = self._inject_defaults(out[prop], subschema)
                    
        return out