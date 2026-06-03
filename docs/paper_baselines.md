# Paper baselines — nuScenes vehicle BEV segmentation

Reference IoU numbers from each model's paper, for comparing the bevunify runs against
the literature. Compiled 2026-06-02.

## Why these numbers are comparable

bevunify trains/evaluates every model on **one** GaussianLSS GT at:

- BEV grid **100 m × 100 m @ 0.5 m/px → 200×200**, ego-centric
- input **224×480**
- metric **IoU@0.5**, reported at **vis≥2** and **vis-all**

This is exactly the protocol the modern BEV-seg papers use for their comparison tables.
The two standard evaluation "settings" map **directly** onto the bevunify metrics:

| bevunify metric | paper setting | meaning |
|---|---|---|
| `IoU_vehicle@0.5_vis2`   | **Setting 2** ⟵ standard headline | only vehicles with visibility > 40% (= nuScenes visibility token ≥ 2) |
| `IoU_vehicle@0.5_visall` | **Setting 1** | all annotated vehicles (no visibility filter) |

So the bevunify validation curves can be read straight against the tables below.

---

## Main comparison — vehicle IoU, 224×480, 100×100 m @ 0.5 m (200×200)

Numbers as re-evaluated under the **unified protocol** in the GaussianLSS paper (Table 1)
and PointBeV paper (Table 1) — the two cross-check identically.

> **Setting 2 (vis ≥ 2) is the standard headline** in this literature — nearly every
> BEV-seg paper reports this visibility-filtered number as its main result. It maps to
> bevunify's `IoU_vehicle@0.5_vis2`, so compare your runs against the **bold** column.

| Model | in bevunify | **Setting 2 (vis≥2)** ⟵ standard | Setting 1 (vis-all) |
|---|:--:|:--:|:--:|
| **GaussianLSS** (host) | ✓ `gaussianlss` | **42.8** | 38.3 |
| **PointBeV** (EfficientNet-b4, single-frame) | ✓ `pointbev` | **44.0** | 38.7 |
| **Simple-BEV** (RGB-only) | ✓ `simplebev` | **43.0** | 36.9 |
| **LaRa** | ✓ `lara` | **38.9** | 35.4 |
| **CVT** (Cross-View Transformers) | ✓ `cvt` | **36.0** | 31.4 |
| **Lift-Splat-Shoot** | ✓ `lss` | — (see note) | — (see note) |
| *FIERY (static)* — reference | – | 39.8 | 35.8 |
| *BEVFormer* — reference | – | 42.0 | 35.8 |
| *BAEFormer* — reference | – | 38.9 | 36.0 |

> **LSS note:** Lift-Splat-Shoot (ECCV 2020) predates this benchmark and is **not**
> listed in the modern 224×480 Setting-1/2 tables. Its own paper reports **vehicle
> IoU 32.07** (official repo: **33.03**) at the LSS-native setting (128×352 input,
> 100×100 m @ 0.5 m). Treat it as an approximate floor; the bevunify run re-trains LSS
> at 224×480, so its result is only loosely comparable to the 32.07 figure.

### Higher resolution (448×800) — for reference

| Model | **Setting 2 (vis≥2)** ⟵ standard | Setting 1 (vis-all) |
|---|:--:|:--:|
| GaussianLSS | **46.1** | 40.6 |
| PointBeV (EN-b4) | **47.6** | 42.1 |
| Simple-BEV | **44.9** | 40.9 |
| CVT | **37.7** | 32.5 |
| BEVFormer | **45.5** | 39.0 |

bevunify currently runs at **224×480**, so compare against the 224×480 table above.

---

## Reproduction status (bevunify runs)

Target = paper **Setting 2 (vis ≥ 2)** at 224×480. "Reproduced" = our best validation
`IoU_vehicle@0.5_vis2` reaches the paper number. Last updated **2026-06-03**.

- [x] **LSS** — target 32.07† · **38.47** (vis-all 33.80) → met/exceeded. *(40/40 epochs, finished cleanly.)*
- [ ] **CVT** — target 36.0 · **35.32** (vis-all 29.55) → ≈ near miss. *(finished, but only **18 epochs**:
      `max_steps=16500` was sized for CARLA's ~17.4k samples; on nuScenes (~917 steps/ep) that is 18 ep.
      A full 30-ep run (`max_steps≈26,400`) should clear 36.0 — last epochs were still ~flat at 35.3.)*
- [x] **LaRa** — target 38.9 · **40.71** (vis-all 35.63) → met/exceeded. *(30/30 epochs, finished cleanly.)*
- [ ] **PointBeV** (single-frame, non-temporal) — target 44.0 · **41.40** (vis-all 35.86) → **under (−2.6)**.
      *(30/30 epochs, finished; batch 8/GPU eff 16 after OOM at 16/GPU. Large batch + unscaled lr → under.)*
- [ ] **GaussianLSS** (host) — target 42.8 · **41.17** (vis-all 34.73) → ≈ near miss (1.6 under). *(40/40 epochs, finished cleanly; best @ep21. batch 8 = eff 16 on 2 GPU.)*
- [ ] **Simple-BEV** — target 43.0 · **40.54** (vis-all 35.02) → ≈ near miss. *(40/40 epochs, finished; best @ep22; refcam=CAM_FRONT / −0.5 / balanced-MSE+footprint-offset fixes; 224×480 vs paper 448×800.)*

| Model | target (S2, vis≥2) | our best vis2 | epochs | status |
|---|:--:|:--:|:--:|:--:|
| **LSS** | 32.07† | **38.47** | 40/40 (finished) | ✅ met |
| **CVT** | 36.0 | **35.32** | 18 (finished; CARLA-sized budget) | ≈ near, under-trained |
| **LaRa** | 38.9 | **40.71** | 30/30 (finished) | ✅ met |
| **PointBeV** | 44.0 (single-frame) | **41.40** | 30/30 (finished) | ≈ near (−2.6 under) |
| **GaussianLSS** (host) | 42.8 | **41.17** | 40/40 (finished) | ≈ near (1.6 under) |
| **Simple-BEV** | 43.0 | **40.54** | 40/40 (finished) | ≈ near (2.5 under) |

† LSS native setting is 128×352; ours is 224×480 `vis2` — a loose comparison, but it clearly
clears the paper figure.

> ⚠️ **Stability note:** the LSS/CVT runs were repeatedly killed mid-training with **no
> Python traceback** (CVT's last death was *during validation*) — a DDP-deadlock signature
> rather than a clear OOM. The updated `bevunify/viz_callback.py` targets exactly this: per
> its own comments, a **rank-0-only forward** in the viz callback desyncs NCCL collectives
> (SyncBatchNorm `all_reduce`) and trips the DDP ~30-min watchdog; it now forwards on **all
> ranks** (only rank 0 emits the viz). Re-run with the updated callback; if kills persist,
> also rule out node OOM.

---

## Per-model detail

### GaussianLSS  *(host model)*
- Paper: *Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting* (arXiv:2504.01957).
- Vehicle IoU @224×480: **38.3** (Setting 1) / **42.8** (Setting 2). @448×800: 40.6 / 46.1.
- "SOTA among 2D-unprojection methods, competitive with 3D-projection methods."

### PointBeV
- Paper: *PointBeV: A Sparse Approach to BeV Predictions* (CVPR 2024, arXiv:2312.00703).
- Vehicle IoU @224×480 (EN-b4, single-frame): **38.7** / **44.0**. Temporal (-T): 39.9 / 44.7.
- bevunify uses the EfficientNet-b4 single-frame variant.

### Simple-BEV
- Paper: *Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?* (ICRA 2023, arXiv:2206.07959).
- Vehicle IoU @224×480 (RGB-only): **36.9** / **43.0**. Own paper highlights **47.4** RGB-only at higher res (448×800+), and 55.7 with radar — not the 224×480 regime bevunify uses.

### LaRa
- Paper: *LaRa: Latents and Rays for Multi-Camera BEV Semantic Segmentation* (CoRL 2022, arXiv:2206.13294).
- Vehicle IoU @224×480: **35.4** (Setting 1) / **38.9** (Setting 2).

### CVT (Cross-View Transformers)
- Paper: *Cross-view Transformers for real-time Map-view Semantic Segmentation* (CVPR 2022 Oral, arXiv:2205.02833).
- Vehicle IoU @224×480: **31.4** / **36.0** (re-evaluated under unified protocol). @448×800: 32.5 / 37.7.
- Training in the official repo: batch 4/GPU × 4 GPU = **effective batch 16**, AdamW + one-cycle (lr 4e-3), 30 epochs.

### Lift-Splat-Shoot (LSS)
- Paper: *Lift, Splat, Shoot* (ECCV 2020, arXiv:2008.05711).
- Vehicle IoU **32.07** (paper) / **33.03** (repo), LSS-native setting (128×352). Not in the 224×480 Setting-1/2 tables.

---

## ⚠️ "Setting 1/2" means different things across papers

The main table above uses the **GaussianLSS / PointBeV convention** (the one bevunify
follows): both settings are **100×100 m @ 0.5 m (200×200), 224×480**, and the split is
**visibility**:
- Setting 1 = no visibility filter (all vehicles) = bevunify `vis-all`
- Setting 2 = visibility > 40 % (token ≥ 2)        = bevunify `vis2`

**CVT's own paper reuses the same words for different GRIDS (not visibility):**
- CVT "Setting 1" = **100 m × 50 m @ 0.25 m** (Roddick et al. / PON grid) → CVT **37.5**
- CVT "Setting 2" = **100 m × 100 m @ 0.5 m** (Philion & Fidler / LSS grid) → CVT **36.0**
- CVT native input = **224×448**.

CVT's "Setting 2" grid *is* the grid the main table uses, and its 36.0 lines up with the
unified Setting-2 (vis > 40 %) CVT = 36.0. So when reading any BEV-seg table, always check
whether "Setting 1/2" refers to **visibility** (GaussianLSS/PointBeV/LaRa/Simple-BEV) or to
**grid range** (CVT/PON/Roddick).

## Native paper headline numbers (NOT the unified 224×480 protocol)

What each paper reports in *its own* setting — useful context, but **not** apples-to-apples
with the main table or with bevunify.

| Model | native vehicle IoU | input res | grid | visibility | note |
|---|:--:|:--:|:--:|:--:|---|
| Simple-BEV | **~47.4** RGB-only (up to 49.3) | 448×800 → 672×1200 | 100×100 @ 0.5 | > 40 % | +1 m centroid offset; higher-res than bevunify (224×480) |
| CVT | **37.5** (S1) / **36.0** (S2) | 224×448 | 100×50@0.25 / 100×100@0.5 | filter | own "Setting 1/2" = **grids** (see above) |
| LSS | **32.07** (paper) / **33.03** (repo) | 128×352 | 100×100 @ 0.5 | — | ECCV'20; predates the 224×480 tables |
| LaRa | **38.9** | 224×480 | 100×100 @ 0.5 | > 40 % | same as unified table |
| PointBeV | **44.0** (EN-b4) | 224×480 | 100×100 @ 0.5 | > 40 % | defines the unified protocol |
| GaussianLSS | **42.8** | 224×480 | 100×100 @ 0.5 | > 40 % | host; defines the unified protocol |

> Takeaway for bevunify: everything is trained/eval'd at **224×480, 100×100 @ 0.5 m**, so
> the **main table** (224×480 Setting 1/2) is the right comparison. Simple-BEV's 47.4 and
> the 448×800 numbers are higher only because of **higher input resolution**, not a better
> setting per se.

---

## Sources
- GaussianLSS — https://arxiv.org/html/2504.01957v1
- PointBeV — https://openaccess.thecvf.com/content/CVPR2024/papers/Chambon_PointBeV_A_Sparse_Approach_for_BeV_Predictions_CVPR_2024_paper.pdf · https://ar5iv.labs.arxiv.org/html/2312.00703
- Simple-BEV — https://simple-bev.github.io/simple_bev_sep30.pdf
- LaRa — https://arxiv.org/pdf/2206.13294
- CVT — https://arxiv.org/pdf/2205.02833 · https://github.com/bradyz/cross_view_transformers
- Lift-Splat-Shoot — https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf · https://github.com/nv-tlabs/lift-splat-shoot

> Caveats: numbers are the **reported/ re-evaluated** values from the papers above, not
> reproduced here. Setting 1 = no visibility filter (all vehicles); Setting 2 = visibility
> > 40% (token ≥ 2). All main-table rows use 224×480, 100×100 m @ 0.5 m (200×200) — the
> same grid bevunify uses, so they line up with `IoU_vehicle@0.5_visall` / `_vis2`.
