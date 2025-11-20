import pytest
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint
from src.codegen.validator import validate_blueprint_dict

def test_simple_blueprint():
    bp_dict = {
        "name": "test",
        "input_shape": [3,32,32],
        "num_classes": 10,
        "backbone": "convnet",
        "stages": [
            {"type":"conv_block", "filters": 32, "depth": 2, "kernel":3},
            {"type":"conv_block", "filters": 64, "depth": 2, "kernel":3}
        ]
    }
    meta = validate_blueprint_dict(bp_dict, device="cpu")
    assert meta["out_shape"] == (2, 10)
    assert meta["params"] > 0
