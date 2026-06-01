"""End-to-end smoke test for ONE experiment: data -> model -> loss -> metric on a
single real batch (GPU), without launching the trainer/DDP.

    cd /home/jeongtae/bevseg/bevunify
    $PY tests/smoke_forward.py cvt
"""
import sys
import os
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevunify  # noqa: E402  bootstraps GaussianLSS path
from hydra import initialize_config_dir, compose
from bevunify.common import setup_config, setup_model_module, setup_data_module

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "cvt"
    overrides = [
        f"+experiment={exp}",
        "experiment.save_dir=/tmp/bevunify_smoke/",
        "loader.val_batch_size=1",
        "loader.num_workers=2",   # host get_split forces prefetch_factor; needs workers>0
    ]
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
        setup_config(cfg)                      # resolve while hydra context is active
        print(f"[{exp}] building model_module ...")
        mm = setup_model_module(cfg)
        dm = setup_data_module(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mm = mm.to(device).eval()

    loader = dm.val_dataloader()
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(f"[{exp}] batch keys: {sorted(batch.keys())}")

    # Call forward -> loss -> metric directly (no Trainer attached in a bare smoke).
    with torch.no_grad():
        pred = mm.backbone(batch)
        loss, details, _ = mm.loss_func(pred, batch)
        mm.metrics.update(pred, batch)
    print(f"[{exp}] pred keys: { {k: tuple(v.shape) for k, v in pred.items() if torch.is_tensor(v)} }")
    print(f"[{exp}] loss = {loss.item():.4f}  details = { {k: round(float(v),4) for k,v in details.items()} }")
    metrics = mm.metrics.compute()
    print(f"[{exp}] metric = { {k: (round(float(v),4) if v.numel()==1 else v.tolist()) for k,v in metrics.items()} }")
    print(f"[{exp}] SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
