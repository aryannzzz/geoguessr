# Geolocation Hackathon — Written Report

IIT Madras AI Guild geolocation competition. This document is the concise,
decision-focused write-up; the fully-executed notebook
(`notebooks/geolocation_pipeline.ipynb`) carries the code-level walkthrough
and EDA detail. Numbers here are pulled directly from files in this repo —
`outputs/`, `checkpoints/`, and `src/` — as noted inline.

## 1. Problem & constraints summary

Predict `pred_lat`, `pred_lon`, `pred_radius_km` for street-level images.
Per-image score combines three terms, and the final leaderboard number is
the **median** across a test set (not the mean) — a model that is
mediocre everywhere beats one that is excellent on 60% of images and
terrible on the rest. That single fact shapes almost every modeling choice
below.

- **Distance term**: smooth exponential decay in Haversine distance
  (no hard cutoff, but decays fast — near-miss precision matters).
- **Calibration term**: signed exponential decay on the claimed radius.
  Inside the radius, reward shrinks as the radius widens; outside it,
  there's a penalty, and a *tight-but-wrong* radius is penalized harder
  than a *wide-but-wrong* one — confidently wrong is punished harder than
  honestly unsure. A fixed "safe" large radius and naive shrink-to-zero
  are both explicitly wrong strategies.
- **Country bonus**: requires the predicted point's country to match the
  ground-truth country *and* a reasonably tight radius — checked against
  an authoritative boundary GeoJSON, not a bounding box.
- **Hard rules**: free-tier compute only, zero network access at inference
  time, no zero-shot foundation models (task-specific heads must be
  trained), no geolocation-pretrained checkpoints (StreetCLIP etc. banned
  outright even as init), external data allowed but never on hidden test
  images, disclosed with a `source` column.
- **Deadline**: Test Set 1 is live now with a **15 submissions/day cap**,
  hard deadline **Aug 19 EOD**. Test Set 2 arrives later; final score is a
  weighted combination of both.

This metric shape is why the architecture is geocell classification (for a
calibratable confidence signal + country-bonus support) plus within-cell
coordinate regression (for a sharp point estimate), with radius set from
held-out error statistics rather than tuned once and forgotten.

## 2. EDA highlights

Source: `outputs/eda/eda_log.txt` (19,002 provided images).

- **Coordinate spread**: lat mean 18.75° (std 35.24°), lon mean 2.75°
  (std 79.31°); 63.6% northern hemisphere, 56.2% eastern hemisphere.
- **Country imbalance is severe**: 171 distinct countries resolved; top 10
  (US, RU, BR, AU, CA, AR, ZA, IN, CL, KZ/MN tied) account for **60.1%** of
  all images. Shannon entropy 3.612 nats vs. 5.142 max → normalized
  evenness 0.702. Many countries have a single image.
- **9.5% of rows (1,804) had no confident country match** within 50 km of
  any boundary polygon — mostly small islands/atolls missing from the
  simplified 298-feature boundary set, or coastal points noised offshore.
  This is fixed later in `build_manifest.py` (Section 3).
- **All training images are a fixed 640×640 square** (sampled 3,000/19,002,
  zero aspect-ratio variance) — this uniformity turned out to matter a lot,
  because Test Set 1 does *not* share it (Section 4).
- **Near-duplicates**: only 11 perceptual-hash collision groups (23 images,
  0.12%) — negligible leakage risk within the provided set.
- **Label noise**: coordinates carry ~10 decimal digits of precision
  (consistent with programmatically added floating-point jitter, not raw
  GPS EXIF) and zero exact-duplicate coordinate pairs — the "noised_dataset"
  name is literal. Radius calibration treats validation residuals as
  already containing this noise rather than modeling it separately.
- **Geocell k sweep**: k=50 gives 0 cells under 20 images; k=100 gives 2;
  k=200 gives 11; k=400 gives 59. This motivated adaptive k-means with
  small-cell merging rather than a fixed k or fixed-level S2 grid.

## 3. Methodology

### Manifest / country lookup (`src/data/build_manifest.py`)
The provided CSV has no country column, so country is derived via
point-in-polygon lookup (`shapely` + `STRtree`) against the same boundary
GeoJSON the competition's country bonus uses. The boundary file's ISO
codes are Natural-Earth-style, and several real countries (UK, Portugal,
Netherlands, Norway, ...) carry the placeholder `"-99"` in `ISO_A2`; a few
polygon fragments (e.g. one of two Korea polygons) still show `"-99"` even
in the patch field `ISO_A2_EH`. Resolution order: `ISO_A2_EH` →
`ISO_A2` → majority non-placeholder code seen elsewhere under the same
`country_name`. What's left after that (Somaliland, Northern Cyprus,
Akrotiri/Dhekelia, Siachen Glacier, Bir Tawil, disputed reefs, ...) are
genuinely disputed territories with no ISO 3166-1 alpha-2 code and are left
as `None` rather than mislabeled. Points outside every polygon fall back to
nearest-polygon search, only accepted within 50 km (handles coastline
noise / small islands missing from the simplified boundary set).

### Geocells (`src/data/geocells.py`)
Adaptive k-means over train (lat, lon), locked design per the EDA sweep:
**k=150 requested → 147 final cells** after merging any cell under
`MIN_CELL_SIZE=15` into its nearest surviving neighbor (great-circle
distance via `BallTree`, haversine metric). Final cell centroids are
recomputed as the mean lat/lon of each cell's actual final membership
(not the raw k-means centroid, since membership shifts after merging).
Every row also gets a **tangent-plane offset** (`dx_km`, `dy_km`) from its
assigned cell's centroid, using a local flat-earth approximation
(`KM_PER_LAT_DEG = 110.574`, longitude scaled by `cos(lat)`) — this is the
regression target for the offset head, and avoids the distortion a raw
degree-based offset would have at high latitudes.

### Model (`src/models/model.py`)
DINOv2-small (`timm` `vit_small_patch14_dinov2.lvd142m`) backbone, a
general-purpose self-supervised ViT — allowed under the no-zero-shot rule
because it's fine-tuned with task-specific heads, and it is not itself a
geolocation-pretrained checkpoint. Partial fine-tune: only the last
`n_unfrozen_blocks` transformer blocks + final norm are unfrozen (2 in the
baseline config), everything before stays frozen to fit the compute
budget. Two heads sit on top of the shared backbone feature:
- **Head A**: linear geocell classifier, softmax over `num_cells` (147).
- **Head B**: linear→GELU→linear tangent-plane offset regressor
  (`dx_km`, `dy_km` normalized by `offset_scale_km=500`).

### Training (`src/train.py`, `src/data/dataset.py`)
Loss = `cls_loss_weight * CrossEntropy(label_smoothing=0.05)` on geocell
logits + `reg_loss_weight * SmoothL1` on the offset, both weighted 1.0 in
the baseline. AdamW with separate backbone/head learning rates
(`lr_backbone=1e-5`, `lr_heads=1e-3`), gradient clipping (norm 1.0), mixed
precision. Split is stratified by geocell (`make_split` in
`dataset.py`) with a random fallback for cells too small to stratify
(<2 members), so no row is dropped. Full seeding (`numpy`, `torch`,
`random`, dataloader `worker_init_fn`) as the guardrails require.
**No horizontal flip** is used, deliberately: driving side / left-vs-right-
hand traffic correlates with country and is a real geographic cue visible
in street imagery — flipping would train the model against a signal it
should be free to use. Color jitter (brightness/contrast/color/sharpness)
and a scale-jitter random-crop (80-100% of the source side, square-to-
square since all training images are already 640×640 squares) are used
instead, adding scale/translation variety without corrupting orientation.

### Calibration (`src/calibrate.py`)
Radius is **not** a fixed hyperparameter. The best checkpoint runs over
the held-out validation split (same seed/split as training), examples are
bucketed by the classifier's own confidence (max softmax probability,
6 quantile bins), and within each bucket the radius is set to the
`q=0.75` quantile of that bucket's actual Haversine error. Low-confidence
predictions get a wider radius, high-confidence predictions get a tighter
one — directly satisfying the metric's "don't default to one fixed radius,
don't naively shrink to zero" guidance. Written to
`outputs/calibration/baseline_dinov2s_calibration.json`.

### Inference (`src/inference/predict.py`)
Offline-only: loads the local checkpoint, calibration table, and test
images from disk, no network calls. Runs every test image through the
model to get a predicted cell (argmax) + offset -> (lat, lon), then assigns
`pred_radius_km` from the confidence bucket the image falls into.

## 4. The aspect-ratio bug

Training/validation images are a uniform 640×640 square (per EDA,
Section 2). Test Set 1's 500 images are **not** — they range up to
1200x486, wide dashcam/street-view frames. The original inference code did
a naive `resize(img_size, img_size)`, which squashes non-square images
non-uniformly (up to ~2.5x more horizontal than vertical compression for
the widest frames), warping every visual cue (building proportions,
road/lane geometry, horizon shape) the model had actually been trained on.

Fix (`src/inference/predict.py`, `TestImageDataset._resize_and_center_crop`):
resize the short side to `img_size`, then center-crop the square out —
matching the square-framing convention the training images already have,
instead of distorting the aspect ratio.

**Verified impact**, computed directly by rerunning the baseline checkpoint
(`checkpoints/baseline_dinov2s_best.pt`, epoch 4) over all 500 Test Set 1
images under both preprocessing paths and comparing the *actual* predicted
geocell (`argmax` over the classifier head) and final (lat, lon) point,
not a proxy:

- **Fraction of images whose predicted geocell (argmax) changed**:
  **47.6%** (238/500)
- **Median Haversine shift** between the buggy and fixed final predicted
  point: **244.6 km**
- **Mean Haversine shift**: **3,686.5 km** (max 18,401.2 km, min 0.0 km) —
  heavily skewed by a minority of images whose predicted cell flipped to a
  distant one; the mean/median gap is itself a live illustration of why the
  competition's median-robustness design matters for this kind of data.

The median (244.6 km) and mean (3,686.5 km) shift match earlier rough
spot-check figures (~245 km / ~3,686 km) almost exactly, which cross-
validates the measurement. An earlier rough estimate had put the
geocell-flip rate at ~88.6%, which looked inconsistent with the 47.6%
measured here by direct argmax comparison — but both numbers are correct;
they measure different things. 47.6% is the fraction whose **discrete**
geocell classification (argmax) flips. 88.6% is the fraction whose final
predicted point moves by **more than 50 km** — verified by applying that
exact threshold to the same buggy/fixed shift array, which reproduces
88.6% precisely. The gap between the two is explained by the continuous
offset-regression head: preprocessing changes shift the offset prediction
for nearly every image even when the discrete cell vote doesn't flip, so a
majority of images move a nontrivial distance (crossing the 50 km bar)
without necessarily crossing a cell boundary. Either way, the conclusion
holds: this was a real, large-magnitude bug — aspect distortion on the
widest test images was large enough to flip the model's coarse geocell
guess on roughly half of all predictions, and to move the final point by
more than 50 km on the large majority of them, including a full
country-scale distance (thousands of km) on a meaningful minority.

## 5. Results so far — the core diagnosis

Two real Kaggle submissions exist so far (`outputs/submissions/`):

| Submission | Description | Kaggle score |
|---|---|---|
| `trivial_centroid_probe_submission.csv` | Zero model — constant prediction = train-set median (lat=32.793180, lon=12.236248), flat 5000 km radius for every image | **2.37907** |
| `baseline_dinov2s_submission.csv` | Real trained DINOv2-small dual-head model, post aspect-ratio fix | **2.50772** |

The trained model beats a constant-guess baseline by only **~5% relative**.
Reported anecdotally, other teams are scoring **~50** on the same metric —
roughly **20x** higher. That gap, next to the tiny edge over a trivial
baseline, is the core diagnosis carried into this session: **this looks
like a model-capacity problem, not a plumbing bug.** The aspect-ratio fix
(Section 4) was necessary and moved real predictions substantially, but a
pipeline-correctness fix alone was never going to close a 20x gap — it
fixed a preprocessing corruption, not the underlying capacity ceiling.
Locally, the baseline's best validation Haversine median was 1462.2 km at
epoch 4 (`outputs/training/baseline_dinov2s_history.csv`), after which
validation error climbed back to 1844.1 km by epoch 12 — clear overfitting
of the small trainable head/tail against a large, imbalanced, noised
target space at limited image resolution.

## 6. Abandoned approaches

An earlier draft notebook (`eda-modeling-baseline-notebook.ipynb`, outside
this repo) was investigated and rejected before this pipeline was built.
It hardcodes Kaggle-native paths (`/kaggle/input/competitions/...`), so it
cannot run on hardware outside a live Kaggle session, violating the
portable-pipeline guardrail. Its training cell does `import trainer` for a
`ModelTrainer` class that is not defined anywhere in the notebook or repo —
the notebook cannot execute end-to-end as committed. Its coordinate-to-
Cartesian helper `find_xyz` computes `x = cos(lat)*cos(lon)`, `y =
cos(lat)*sin(lon)`, `z = cos(lat)` directly from `lat`/`lon` in **degrees**,
without a `np.radians()` conversion — a real numerical bug (`cos`/`sin` of
a degree value is not the geographic angle you'd get from radians), on top
of `z` being defined as `cos(lat)` rather than `sin(lat)`, which is not
the standard spherical-to-Cartesian projection regardless of units. Most
fundamentally, the notebook trains a plain image classifier over binned
targets, evaluated with `accuracy`/`F1`/`multiclass MCC`, and never
produces `pred_lat`/`pred_lon`/`pred_radius_km` at all — there is no
inference or submission-generation step. It was set aside in favor of the
from-scratch pipeline in this repo.

## 7. This session's investigation

**Resolution mismatch discovery.** `timm`'s default config for
`vit_small_patch14_dinov2.lvd142m` reports a native pretrained input size
of **518×518** (verified this session: `model.default_cfg["input_size"]
== (3, 518, 518)`), while training and inference to this point used
**224×224** — a likely capacity-limiting mismatch worth testing directly
rather than assuming.

**336px test.** A short, matched 5-epoch run at `img_size=336`
(`configs/res336_test.yaml`, all other hyperparameters identical to the
baseline; see `outputs/training/res336_test_history.csv`) beat the 224px
baseline at **every single epoch**:

| epoch | 224px val Haversine median (km) | 336px val Haversine median (km) |
|---|---|---|
| 1 | 2450.3 | 2082.7 |
| 2 | 1589.4 | 1356.9 |
| 3 | 1534.4 | **1196.3** (336px best) |
| 4 | **1462.2** (224px best) | 1252.8 |
| 5 | 1483.2 | 1255.3 |

Best val Haversine median: **1196.3 km at epoch 3 (336px)** vs.
**1462.2 km at epoch 4 (224px)** — an **~18.2% relative improvement**
(computed: (1462.2-1196.3)/1462.2). VRAM headroom on the local RTX 3050
(6 GB) is much larger than assumed: even native 518px uses only ~2.75 GB
at batch size 48, leaving room to push resolution further.

**Follow-up 448px/518px checks.** 3-epoch quick checks at 448px and 518px
were launched to see whether the improvement keeps climbing monotonically
with resolution before committing to a specific resolution for a full
retrain. At report-authoring time, `outputs/training/res448_test_history.csv`
and `res518_test_history.csv` **do not exist yet** — this run is still in
progress (a checkpoint for 448px, `checkpoints/res448_test_best.pt`, has
appeared, but no history CSV). **Results pending — to be appended once
available; no numbers are asserted here.**

**Local proxy metric — methodology and a load-bearing caveat.**
`src/proxy_metric.py` builds a local stand-in for the real (undisclosed)
Kaggle scoring formula so candidate configs can be ranked without burning
Kaggle's 15-submissions/day cap. It composes the three documented terms —
distance decay, signed radius-calibration term, country-match bonus — over
the held-out validation split, using the two real anchors above.

The first attempt let all shape parameters (distance-decay scale `D`,
calibration weight `w_cal`, country bonus weight `c_bonus`, amplitude `A`)
float freely in a 4-parameter fit to the 2 known anchors. This is
underdetermined and **degenerately collapsed**: the fitted decay constant
`D` ran to an unbounded/nonsensical value, flattening the distance term to
near-constant for every image — directly contradicting the README's
documented "steep exponential decay." The fix: fix the metric's *shape*
constants as principled judgment calls informed by the README's
description (`D=1500km`, `w_cal=0.2`, `c_bonus=0.15`, `A=1.0` — see
`outputs/calibration/proxy_metric_fit.json`), then solve a well-posed
**closed-form affine rescaling** (`score = a*raw + b`, 2 unknowns, 2
equations) against the two known anchors. This exactly reproduces both
anchors by construction (`anchor_real_reproduced: 2.50772`,
`anchor_trivial_reproduced: 2.37907` in the fit file) — but that exactness
is *definitional*, not validation of the underlying shape.

A sanity check makes the limitation concrete: a hypothetically **perfect**
prediction (zero error, tight correct radius, country match on every
image) scores only **~2.66** under this fitted shape
(`near_perfect_calibrated_score` in the fit file) — far below the ~50
other teams reportedly score. That gap means the proxy's *absolute scale*
is not trustworthy, only its *local ranking* near the current operating
point. Concretely: the proxy is appropriate for small, same-direction
local deltas (e.g., it correctly ranked 336px above 224px, matching the
real validation Haversine trend) but is **not** appropriate for judging
absolute score or extrapolating what a large architectural change (much
higher resolution, unfreezing significantly more of the backbone, external
data) will do to the real Kaggle score. Because of this, the decision of
whether a large change is worth spending a Kaggle submission on is being
driven by real validation Haversine trends, not the proxy score.

## 8. Full retrain: moving to Kaggle, resolution decision, capacity increase

Local training (RTX 3050 laptop, 6GB) was abandoned mid-session: the
machine suffered **repeated full OS reboots** under sustained training
load (confirmed via `uptime -s` showing a new boot time each time,
killing the training process with no error trace of its own). All
training moved to Kaggle Notebooks (free T4/P100) via a new
self-contained notebook, `notebooks/kaggle_train.ipynb`, which clones this
repo, auto-locates the competition's Kaggle-mounted dataset by matching
filenames (no hardcoded dataset path), and runs the same `src/` code
unmodified.

**Resolution**: matched short (3-5 epoch) local tests at 224/336/448/518px
(same architecture, same hyperparameters, only resolution varied) each
beat the previous at every matched epoch, with sharply diminishing
returns past ~336-448px (336 over 224: -22% val median; 448 over 336:
-3.4%; 518 over 448: -2.4%, at ~25% more time/epoch). **448px** was chosen
for the full retrain as the cost/benefit pick.

**Capacity**: unfroze 6 of 12 DINOv2-small blocks (up from the baseline's
2), weight_decay 0.01 -> 0.03, added a scale-jitter crop + wider color
jitter augmentation (still no horizontal flip), cosine LR schedule with
early stopping (patience=4) instead of a hardcoded epoch count.

**Two real bugs found and fixed in this phase** (both in `src/train.py`,
both caught by inspecting real Kaggle training logs, not anticipated in
advance):
1. Cosine schedule horizon (`T_max`) defaulted to the hard epoch cap
   (16), but early stopping routinely fired ~7-8 epochs in — the LR was
   still ~60% of its starting value when training stopped, never reaching
   the low-LR fine-convergence phase cosine annealing is supposed to
   provide. Fixed by adding a separate `lr_schedule_t_max` config key,
   set to 8 (near where val error had empirically started degrading).
2. `CosineAnnealingLR` is periodic: naively stepping it past `T_max`
   doesn't hold the LR at its floor, it continues the cosine curve back
   *upward* (verified: by epoch 12 with `T_max=8`, LR was back to 50% of
   its starting value). Early stopping's patience window runs several
   epochs past `T_max` before firing, so training was silently undoing
   its own fine-convergence phase. Fixed by only stepping the scheduler
   while `epoch <= t_max`, pinning the LR at its floor afterward --
   verified in a later run where epochs 8-12 produced byte-identical
   validation metrics (LR pinned at exactly 0, no further learning, as
   intended once converged).

With both fixes, two independent retrains from scratch (same config,
same seed) converged to essentially the same result: **best val Haversine
median ~901-906km at epoch 8** -- consistent and reproducible, a real
~38% relative improvement over the 224px/2-block baseline's 1462km.

## 9. Calibration quantile/bin-count: three real anchors, and a lesson in not over-trusting a hypothesis

With the model itself validated as solid and reproducible, three real
Kaggle submissions were made varying only the radius-calibration recipe
(`src/calibrate.py`) on top of models in the same ~900-970km val-median
range:

| Run | Model val median | quantile | n_bins | Kaggle score |
|---|---|---|---|---|
| A | 967.6km (T_max bug present, old plain cosine) | 0.75 | 6 | **5.14** |
| B | 901.3km | 0.50 | 10 | 4.04 |
| C | 906.0km | 0.75 | 10 | 4.87 |

For reference: trivial centroid baseline scored 2.37907; the 224px/2-block
baseline (post aspect-ratio fix) scored 2.50772.

**A -> B**: quantile dropped 0.75 -> 0.5 (radius = median error per bucket
instead of the 75th percentile), on a *better* model. Score dropped
5.14 -> 4.04 despite the model improving. Diagnosis: setting radius to
the bucket median means ~50% of that bucket's own validation examples
exceed their assigned radius by construction (vs ~25% at quantile 0.75).
The problem statement is explicit that a tight-but-wrong radius is
penalized *more* than a wide-but-wrong one -- halving the quantile roughly
doubled the tight-miss rate, and that asymmetric penalty apparently
outweighed the extra calibration reward + country-bonus eligibility
gained on the correct half.

**B -> C**: quantile reverted 0.5 -> 0.75 (n_bins still 10, model
unchanged within noise). Score recovered partially, 4.04 -> 4.87 --
confirming the *direction* of the quantile diagnosis was right.

**But C (4.87) still underperforms A (5.14)**, despite A having the
*worse* model and the *old, buggy* LR schedule. With quantile now matched
between A and C, the only remaining deliberate difference is `n_bins`
(6 vs 10) -- an unvalidated change made at the same time as the quantile
change, on the untested assumption that "more bins = finer resolution =
strictly better." Inspecting run C's calibration table shows this
assumption was likely wrong: with only ~190 validation examples per
bucket (vs ~317 at 6 bins), the 75th-percentile estimate is noisy,
especially in the tails. The ratio of each bucket's assigned radius to
its own median error -- which should move smoothly with confidence if the
quantile estimate is stable -- instead swings erratically bucket to
bucket: 1.65x, 3.16x, 4.03x, **4.92x**, then drops back to 3.18x, 3.64x,
2.93x, 2.18x, 2.09x, 1.67x. That's small-sample noise in the calibration
step itself, not signal -- a real bucket landing at an unluckily-wide
radius purely from having too few examples to estimate its 75th
percentile stably.

**Takeaway, stated plainly**: two calibration hypotheses were tried in a
row (tighter-is-better, then more-bins-is-better) and both were wrong, or
at least net-negative, despite each having a plausible-sounding rationale
going in. The empirically best real setting found across all three
Kaggle anchors remains **quantile=0.75, n_bins=6** -- run A's exact
original recipe, now paired with the improved (~906km) model. Given the
Aug 19 EOD deadline left no further submission budget to keep testing
one variable at a time, that recipe was locked in as the final
calibration choice.

## 10. Final state at submission time

- Final model: `kaggle_448_deep` -- DINOv2-small, 448px, 6/12 blocks
  unfrozen, weight_decay=0.03, cosine LR (T_max=8, correctly pinned),
  early stopping patience=4. Best val Haversine median: ~901-906km
  (epoch 8), reproduced across two independent training runs.
- Final calibration: quantile=0.75, n_bins=6 (`src/calibrate.py`
  defaults) -- the empirically best-scoring recipe across three real
  Kaggle anchors (5.14 vs 4.04 vs 4.87), chosen over two plausible-seeming
  but empirically-refuted alternatives (see Section 9).
- Best real Kaggle score obtained this session: **5.14** (up from the
  224px baseline's 2.50772, and the trivial centroid's 2.37907).
- Known open gap: reported scores of ~50 for other teams remain
  ~10x above what was achieved here. The evidence in Section 9 suggests
  the classifier's confidence is not concentrated near 1.0 for most test
  images even on the improved model (e.g. run C's own bucket 4, the
  *middle* of 10 confidence bins, still had a 1091km median validation
  error) -- meaning most of the remaining gap is a genuine model
  capacity/discrimination problem, not a calibration-tuning problem no
  matter how it's sliced. Time did not allow addressing this further
  (more unfrozen capacity, external data for underrepresented regions,
  or a coarse-to-fine geocell hierarchy were all identified earlier as
  candidate next steps but not attempted) before the deadline.
