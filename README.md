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
  raw/         # provided competition CSV + images (gitignored, not committed)
  external/    # any external datasets, each in its own subdir + LICENSE note
  processed/   # merged manifests, geocell assignments, splits
notebooks/     # the single fully-executed deliverable notebook lives here
src/
  data/        # dataset/dataloader, geocell assignment, manifest merging
  models/      # backbone + dual-head architecture, baselines
  calibration/ # radius calibration (conformal / quantile binning)
  inference/   # offline-only inference script -> submission CSV
configs/       # yaml configs per run (backbone, geocell scheme, hparams, seed)
outputs/       # generated submission CSVs, plots, metrics
checkpoints/   # local model weights bundled for offline inference
```

## Status

Phase 0 (this scaffold) complete. Phase 1 (data acquisition + EDA) blocked
on Kaggle competition slug + credentials — see chat for what's needed.
