import os
import tempfile
import json
from src.dkb.client_sqlite import DKBClient

def test_dkb_insert_query():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        dkb = DKBClient(path)
        bp = {"name":"t", "stages":[{"type":"conv_block","filters":16,"depth":1}]}
        arch_id = dkb.add_architecture("test", bp, features={"params":123, "flops":456})
        assert isinstance(arch_id, int)

        archs = dkb.query_architectures(min_params=100)
        assert len(archs) >= 1

        trial_id = dkb.add_trial(arch_id, "run1", {"cfg":"x"}, checkpoint_path=None)
        assert isinstance(trial_id, int)

        mk = dkb.add_metrics(trial_id, epoch=0, val_acc=0.1, val_loss=1.0, latency_cpu_ms=5.0, latency_cuda_ms=None)
        assert isinstance(mk, int)
    finally:
        try:
            dkb.close()
        except Exception:
            pass
        os.remove(path)
