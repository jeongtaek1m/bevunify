"""Unified IoU metric: single threshold + explicit visibility setting + distinct label.

The host IoUMetric hardcodes thresholds [0.4,0.45,0.5] (reports the max) and always
returns key "IoU_{key}". To report IoU@0.5 at both vis>=2 and vis-all for every model,
we expose `thresholds` and a per-instance `label` so the two instances don't collide
in the MetricCollection.
"""
import numpy as np
import torch

from GaussianLSS.metrics import IoUMetric, BaseIoUMetric


class IoUMetricCfg(IoUMetric):
    def __init__(self, min_visibility=0, key="vehicle", thresholds=(0.5,), label=None):
        # init BaseIoUMetric with custom thresholds (IoUMetric.__init__ would force defaults)
        BaseIoUMetric.__init__(self, thresholds=np.array(list(thresholds), dtype=np.float32))
        self.min_visibility = min_visibility
        self.key = key
        vis = "all" if not min_visibility else str(min_visibility)
        thr = "_".join(f"{float(t):g}" for t in thresholds)
        self.label = label or f"{key}@{thr}_vis{vis}"

    # update() is inherited from IoUMetric (uses self.key + self.min_visibility mask).

    def compute(self):
        ious = BaseIoUMetric.compute(self)                 # (n_thresholds,)
        return {f"IoU_{self.label}": torch.round(torch.max(ious), decimals=4)}
