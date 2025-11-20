import json
from jsonschema import validate, ValidationError
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "spec_schema.json"

def load_schema():
    return json.loads(SCHEMA_PATH.read_text())

def normalize_spec(raw_spec: dict):
    # Basic normalization + validation
    schema = load_schema()
    validate(instance=raw_spec, schema=schema)
    # fill defaults if needed (example)
    spec = raw_spec.copy()
    if "objectives" not in spec:
        spec["objectives"] = ["top1_acc"]
    return spec
