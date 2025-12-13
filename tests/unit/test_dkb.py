import pytest
import sqlite3
from src.dkb.client_sqlite import DKBClient

@pytest.fixture
def dkb(tmp_path):
    db_path = tmp_path / "test_dkb.sqlite"
    client = DKBClient(str(db_path))
    yield client
    client.close()

def test_architecture_lifecycle(dkb):
    bp = {
        "name": "resnet_tiny", 
        "input_shape": [3, 32, 32],
        "stages": [{"type": "conv", "filters": 64}]
    }
    
    # 1. Create
    arch_id = dkb.add_architecture(
        name="resnet_tiny", 
        blueprint_json=bp, 
        features={"depth": 10, "width": 64}
    )
    assert arch_id > 0

    # 2. Read (Query)
    # Test strict filtering
    results = dkb.query_architectures(min_params=0) 
    assert len(results) == 1
    assert results[0]["name"] == "resnet_tiny"
    
    # Verify JSON roundtrip
    stored_bp = results[0]["blueprint_json"]
    if isinstance(stored_bp, str):
        import json
        stored_bp = json.loads(stored_bp)
    assert stored_bp["stages"][0]["filters"] == 64

def test_trial_execution_flow(dkb):
    # Setup dependencies
    arch_id = dkb.add_architecture("test_arch", {})
    run_id = "run_abc123"
    config = {"lr": 0.01, "batch_size": 32}

    # 1. Start Trial
    trial_id = dkb.add_trial(arch_id, run_id, config)
    assert trial_id > 0
    
    # 2. Log Metrics (Epoch 0)
    dkb.add_metrics(
        trial_id, 
        epoch=0, 
        val_acc=0.50, 
        val_loss=1.2, 
        latency_cpu_ms=15.5
    )
    
    # 3. Update Status (Complete)
    dkb.update_trial_result(
        trial_id, 
        status="COMPLETED", 
        best_acc=0.85, 
        ckpt_path="/tmp/best.pth"
    )

    # 4. Verify History
    metrics = dkb.get_metrics_for_trial(trial_id)
    assert len(metrics) == 1
    assert metrics[0]["val_acc"] == 0.50
    assert metrics[0]["latency_cpu_ms"] == 15.5
    
    # 5. Verify Final State
    # Assuming get_trial or direct query exists
    # We can query trials via arch usually
    trials = dkb.latest_trials_for_arch(arch_id)
    assert trials[0]["status"] == "COMPLETED"
    assert trials[0]["best_acc"] == 0.85