import os, json, tempfile
from src.orchestrator.post_metrics import compute_num_params, compute_flops_fallback, measured_latency_ms

def test_compute_params_and_flops():
    # simple param/flop fallback test using a tiny blueprint model created inline
    from src.codegen.blueprint import Blueprint
    bp = {"name":"tst","input_shape":[3,8,8],"num_classes":10,"stages":[{"filters":8,"depth":1,"kernel":3}]}
    # render model
    from src.codegen.renderer import render_blueprint
    model = render_blueprint(Blueprint.from_dict(bp))
    params = compute_num_params(model)
    assert params > 0
    flops = compute_flops_fallback(params)
    assert flops >= params

def test_latency_cpu_short():
    from src.codegen.blueprint import Blueprint
    bp = {"name":"tst","input_shape":[3,8,8],"num_classes":10,"stages":[{"filters":8,"depth":1,"kernel":3}]}
    from src.codegen.renderer import render_blueprint
    model = render_blueprint(Blueprint.from_dict(bp))
    lat = measured_latency_ms(model, tuple(bp["input_shape"]), device="cpu", runs=3, warmup=1)
    assert lat is None or lat > 0
