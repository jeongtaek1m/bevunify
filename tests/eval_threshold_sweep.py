"""Post-hoc threshold-sweep IoU eval (the EAFormer/CVT papers' metric = max IoU over
{0.4, 0.45, 0.5} at vis>=2). Training logs the unified IoU@0.5 only; run this on a
checkpoint when a paper-comparable number is needed.

WARNING: the MAX-over-thresholds number is NOT comparable to the unified IoU@0.5
table (it is >= @0.5 by construction). Use it ONLY against the papers' reported
numbers — never mix it into the cross-model @0.5 comparison.

    $PY tests/eval_threshold_sweep.py <experiment> <ckpt> [--dataset_dir /abs/nuscenes]
    e.g. $PY tests/eval_threshold_sweep.py eaformer logs/bevseg-eaformer/<run>/checkpoints/last.ckpt
"""
import argparse, os, torch
ap = argparse.ArgumentParser()
ap.add_argument("experiment")
ap.add_argument("ckpt")
ap.add_argument("--dataset_dir", default=os.environ.get("NUSCENES_DIR", "./data/nuscenes"))
ap.add_argument("--labels_dir", default=None, help="default: <dataset_dir>/labels_aug")
args = ap.parse_args()
exp, ckpt, D = args.experiment, args.ckpt, args.dataset_dir
L = args.labels_dir or f"{D}/labels_aug"
import bevunify
from hydra import initialize_config_dir, compose
from bevunify.common import setup_config, setup_model_module, setup_data_module
CFG=os.path.abspath(os.path.join(os.path.dirname(bevunify.__file__),"..","config"))
thr=torch.tensor([0.30,0.35,0.40,0.45,0.50])
with initialize_config_dir(version_base="1.3", config_dir=CFG):
    cfg=compose(config_name="config", overrides=[f"+experiment={exp}",
        f"data.dataset_dir={D}", f"data.labels_dir={L}",
        "experiment.logger=csv","experiment.save_dir=/tmp/eaf_eval/",
        "loader.val_batch_size=12","loader.num_workers=8","trainer.devices=1"])
    setup_config(cfg); mm=setup_model_module(cfg); dm=setup_data_module(cfg)
sd=torch.load(ckpt,map_location="cpu",weights_only=False)
miss,unexp=mm.load_state_dict(sd.get("state_dict",sd),strict=False)
print(f"[{exp}] ckpt {ckpt} (missing {len(miss)}, unexpected {len(unexp)})")
mm=mm.cuda().eval()
def acc(vis_min):
    tp=torch.zeros(len(thr)); fp=torch.zeros(len(thr)); fn=torch.zeros(len(thr)); n=0
    with torch.no_grad():
        for batch in dm.val_dataloader():
            bc={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in batch.items()}
            p=mm.backbone(bc)["vehicle"].sigmoid().cpu()
            t=batch["vehicle"].bool()
            if vis_min>0:
                m=(batch["vehicle_visibility"]>=vis_min)[:,None].expand_as(p); pv=p[m]; tv=t[m]
            else:
                pv=p.reshape(-1); tv=t.reshape(-1)
            pb=(pv[:,None]>=thr[None]); tv=tv[:,None]
            tp+=(pb&tv).sum(0); fp+=(pb&~tv).sum(0); fn+=(~pb&tv).sum(0); n+=p.shape[0]
    return tp/(tp+fp+fn+1e-7), n
for vm,name in [(2,"vis>=2"),(0,"vis-all")]:
    iou,n=acc(vm)
    s=" ".join(f"@{float(t):.2f}={float(i):.4f}" for t,i in zip(thr,iou))
    print(f"[{exp}] {name} (n={n}): {s} | PAPER-METRIC MAX(0.4-0.5)={float(iou[2:].max()):.4f}")
