# Geolocation Hackathon — IIT Madras AI Guild

Predict `pred_lat`, `pred_lon`, `pred_radius_km` for street-level images.
Final score per test set = **median** of per-image scores combining a
distance term, a signed calibration term, and a country-match bonus. See
"Metric design implications" below before touching modeling code.

## ⚠️ Guardrails — read before editing anything in `src/` or `notebooks/`

These are competition rules, not style preferences. Violating any of them
makes the submission worth **zero points regardless of leaderboard rank**.

### Compute
- Train/infer only on free-tier hardware: Colab T4, Kaggle P100/T4, or a
  local machine. No paid compute tiers used for a compute edge.
- The pipeline must run on hardware we don't control at eval time: no
  hardcoded local paths, no assumed caches. Only read from `data/`,
  `checkpoints/`, and the Kaggle-provided input dirs.
- Seed everything (`numpy`, `torch`, `random`, dataloader `worker_init_fn`)
  and log the seeds used for every run that produces a submitted artifact.

### Offline inference (absolute, non-negotiable)
- **Zero network access at inference time.** No API calls, no on-the-fly
  weight downloads, no reverse image search, no geocoding/mapping lookups.
- No hosted commercial models (GPT-4o, Gemini, Claude, etc.) anywhere in the
  inference path — coding help / debugging / write-up drafting only.
- All weights the final submission depends on must be bundled with the
  notebook or loaded from a zero-network-call local source (Kaggle dataset
  input, disk). Never `from_pretrained(...)` hitting the internet at eval
  time — download once during development, then bundle the local checkpoint.
- Before finalizing: **test the inference notebook/script with network
  access disabled** to confirm compliance. This is a required check, not
  optional.

### No zero-shot foundation models
- Pretrained general-purpose backbones are fine (ResNet, ViT, CLIP, DINO,
  DINOv2, SigLIP, ...) but only as a starting point — task-specific heads
  (classification/regression) must be added and trained on top. Partial or
  full fine-tuning is fine and encouraged if compute allows.
- Using a foundation model **purely zero-shot** with no task-specific
  training is an invalid submission.
- **Any checkpoint already pretrained/fine-tuned specifically for
  geolocalization is banned outright**, even as an initialization to
  fine-tune further. Explicitly includes StreetCLIP and any other
  geo-specific CLIP variant or pretrained geocell classifier. If a
  checkpoint's stated training objective is "predict location," it cannot
  be used at all. When a backbone's provenance is ambiguous, prefer a
  clearly general-purpose one (OpenCLIP ViT-B/L, DINOv2, SigLIP-base) over
  anything with "geo" in its name or paper.

### VLM-specific rule
- OK: take a local open-weight VLM (Qwen-VL, LLaVA, PaliGemma, ...), strip
  the language-generation head, freeze or fine-tune only the vision encoder,
  train a new regression/classification head on the visual embeddings.
- Not OK: prompting a VLM in native form to output a location guess in
  text — a zero-shot violation regardless of what else is fine-tuned.

### Data rules
- Provided: 19,002 geotagged images + `image_id, lat, lon, iso_country_code`.
- External data allowed and encouraged, with two conditions:
  1. Never train on the actual hidden test images or any external source of
     their ground truth, even inadvertently.
  2. Disclose every external dataset in the write-up; be ready to justify
     usage rights/license.
- Incidental overlap between an external dataset and test images by chance
  is fine — just don't deliberately seek or exploit it.
- Never train on the provided test images.
- Track a `source` column in the merged manifest for every external dataset
  so it can be disclosed/ablated later.

### Submission mechanics
- Test Set 1: live now, 15 submissions/day cap, deadline **Aug 19 EOD**.
- Test Set 2: released later as a separate hidden competition — the frozen
  pipeline must reproduce a CSV for it with minimal changes.
- Final score = weighted combination of both test sets.

## Metric design implications (shapes modeling choices — don't ignore)

1. **Distance term**: smooth exponential decay in Haversine distance, no
   hard cutoff but decays fast → near-miss precision matters.
2. **Calibration term**: signed exponential decay on claimed radius — inside
   radius: reward shrinks as radius widens; outside radius: penalty, and a
   *tight-but-wrong* radius is penalized more than a *wide-but-wrong* one.
   **Confidently wrong is punished harder than honestly unsure.**
   - Never default to one fixed "safe" large radius for everything.
   - Never naively shrink radius toward 0.
   - Treat radius as a conformal-prediction / calibration problem, not a
     hyperparameter tuned once and forgotten.
3. **Country bonus**: binary-ish, requires predicted point's country to
   match ground truth AND a reasonably tight radius. Use an authoritative
   country-boundaries GeoJSON (`geopandas`/`shapely`) — never a bounding box.
4. **Median, not mean**, of per-image scores → a model mediocre everywhere
   beats one excellent on 60% and terrible on 40%. Favors being honest about
   uncertainty (wide radius when unsure) over average-error optimization.

**Architecture takeaway**: coarse geocell classification (S2 cells / equal-
area grid / k-means over train coords) for a calibratable confidence signal
+ country-bonus support, plus within-cell coordinate regression (offset from
centroid, ideally in a local tangent-plane projection near poles) for a
sharp point estimate. Radius derived from held-out validation error
statistics conditioned on confidence (quantile binning / split-conformal),
not a constant or raw regression output.

## Repo layout

```
data/
  raw/         # provided competition CSV + images. Only the two small
               # reference files (country_boundaries.geojson,
               # sample_submission.csv) are committed; the image
               # directories (training_dataset/, test_images_sampled/)
               # are gitignored -- too large, pull them from the
               # Kaggle-mounted dataset instead (see notebooks/kaggle_train.ipynb)
  external/    # any external datasets, each in its own subdir + LICENSE note
  processed/   # merged manifests, geocell assignments, splits -- committed,
               # small (~4MB), so Kaggle runs don't need to rebuild geocells
notebooks/
  geolocation_pipeline.ipynb  # the fully-executed deliverable notebook
                               # (EDA, abandoned approaches, modeling,
                               # calibration, inference, bug writeups)
  kaggle_train.ipynb          # self-contained: clones this repo, locates
                               # the Kaggle-mounted dataset, builds/loads
                               # geocells, trains, calibrates, runs
                               # inference -> submission CSV. Training now
                               # happens here (Kaggle T4/P100), not locally.
src/
  data/        # dataset/dataloader, geocell assignment, manifest merging
  models/      # backbone + dual-head architecture, baselines
  inference/   # offline-only inference script -> submission CSV
  calibrate.py # radius calibration (confidence-bucketed quantiles)
  proxy_metric.py # local proxy for the Kaggle scoring formula, fit against
               # known anchor submissions -- see REPORT.md for methodology
               # and its documented limitations
configs/       # yaml configs per run (backbone, geocell scheme, hparams, seed)
outputs/       # EDA/training/calibration logs, plots, submission CSVs --
               # committed (small); raw checkpoints are not
checkpoints/   # local model weights (gitignored -- too large for git;
               # Kaggle runs write these to /kaggle/working/ for download)
```

## Status

Phases 0-5 complete locally (scaffold, EDA, geocell/model design, baseline
training, calibration, offline inference) -- see REPORT.md and
notebooks/geolocation_pipeline.ipynb for full writeups, results, and a
critical aspect-ratio preprocessing bug found and fixed post-baseline.

Training moved off local hardware (an RTX 3050 laptop that kept rebooting
under sustained load) onto Kaggle Notebooks (free T4/P100). Use
notebooks/kaggle_train.ipynb going forward for anything training-related;
local `src/train.py` still works for quick smoke tests but is no longer
the primary training path. A resolution/capacity investigation (this
session) found DINOv2-small's native 518px resolution outperforms the
224px baseline at every tested checkpoint (224 -> 336 -> 448 -> 518, each
step better, diminishing returns after ~448px) -- the Kaggle notebook
picks up that investigation with a full run at 448px + more unfrozen
backbone blocks + regularization, parameterized at the top of the notebook
so resolution/unfrozen-blocks/epochs are easy to change.
