"""Local proxy for the (undisclosed) Kaggle scoring formula.

Fits a parametrized approximation of the documented metric shape --
    score = distance_decay(err_km) + calibration_term(err_km, radius_km)
            + country_match_bonus
    (median over images)
-- against two known anchors (real Kaggle submission scores) using the
held-out validation split as a stand-in for Test Set 1. This is NOT a
recovery of the real formula (2 anchors can't pin down 3+ free constants);
it is a directionally-consistent local ranking tool so we don't have to
burn Kaggle submissions to compare candidate configs.

Anchors (real Kaggle scores, see project notes):
    trivial_centroid_probe_submission.csv -> 2.37907
    baseline_dinov2s_submission.csv (post aspect-fix)  -> 2.50772

Run: python3 src/proxy_metric.py [--config configs/baseline.yaml]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GeoImageDataset, make_split
from src.data.geocells import haversine_km
from src.models.model import GeoModel
from src.train import set_seed, worker_init_fn, offsets_to_latlon
from src.data.build_manifest import load_country_index, point_country

PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUTS = ROOT / "outputs"

TRIVIAL_LAT = 32.793179905420004
TRIVIAL_LON = 12.236247882004932
TRIVIAL_RADIUS_KM = 5000.0

ANCHOR_TRIVIAL_SCORE = 2.37907
ANCHOR_REAL_SCORE = 2.50772

R_MAX_KM = 20015.0  # half Earth's circumference: max possible radius claim


def score_components(err_km, radius_km, country_match, A, D, w_cal, c_bonus):
    dist_term = A * np.exp(-err_km / D)
    radial_frac = np.minimum(radius_km / R_MAX_KM, 1.0)
    sign = np.where(err_km <= radius_km, 1.0, -1.0)
    cal_term = sign * w_cal * (1.0 - radial_frac)
    country_term = np.where(country_match, c_bonus, 0.0)
    return dist_term, cal_term, country_term


def median_score(err_km, radius_km, country_match, A, D, w_cal, c_bonus):
    d, c, k = score_components(err_km, radius_km, country_match, A, D, w_cal, c_bonus)
    return float(np.median(d + c + k))


def get_real_model_val_predictions(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    df = pd.read_csv(PROCESSED / "train_manifest_geocells.csv")
    cell_table = pd.read_csv(PROCESSED / "geocell_centroids.csv").set_index("cell_id")
    num_cells = df["cell_id"].nunique()

    _, val_df = make_split(df, val_frac=cfg["val_frac"], seed=cfg["seed"])
    val_ds = GeoImageDataset(val_df, img_size=cfg["img_size"], train=False,
                              offset_scale_km=cfg["offset_scale_km"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], worker_init_fn=worker_init_fn,
                             pin_memory=True)

    ckpt = torch.load(CHECKPOINTS / f"{cfg['run_name']}_best.pt", map_location=device, weights_only=False)
    model = GeoModel(num_cells=num_cells, backbone_name=cfg["backbone_name"],
                      img_size=cfg["img_size"], n_unfrozen_blocks=cfg["n_unfrozen_blocks"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[proxy] loaded {cfg['run_name']} checkpoint from epoch {ckpt['epoch']}")

    with open(OUTPUTS / "calibration" / f"{cfg['run_name']}_calibration.json") as f:
        calib = json.load(f)
    bin_edges = np.array(calib["bin_edges"])
    n_bins = calib["n_bins"]
    radii_table = np.array([calib["table"][i]["radius_km"] for i in range(n_bins)])

    all_pred_lat, all_pred_lon, all_true_lat, all_true_lon, all_conf = [], [], [], [], []
    with torch.no_grad():
        for imgs, cell_ids, offsets, lats, lons in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits, offset_pred = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values.cpu().numpy()
            pred_cell = logits.argmax(dim=1).cpu().numpy()
            pred_offset_km = offset_pred.cpu().numpy() * cfg["offset_scale_km"]
            pred_lat, pred_lon = offsets_to_latlon(pred_cell, pred_offset_km, cell_table)
            all_pred_lat.append(pred_lat)
            all_pred_lon.append(pred_lon)
            all_true_lat.append(lats.numpy())
            all_true_lon.append(lons.numpy())
            all_conf.append(conf)

    pred_lat = np.concatenate(all_pred_lat)
    pred_lon = np.concatenate(all_pred_lon)
    true_lat = np.concatenate(all_true_lat)
    true_lon = np.concatenate(all_true_lon)
    conf = np.concatenate(all_conf)

    idx = np.clip(np.digitize(conf, bin_edges[1:-1], right=True), 0, n_bins - 1)
    radius_km = radii_table[idx]
    err_km = haversine_km(pred_lat, pred_lon, true_lat, true_lon)

    return {
        "pred_lat": pred_lat, "pred_lon": pred_lon,
        "true_lat": true_lat, "true_lon": true_lon,
        "err_km": err_km, "radius_km": radius_km,
        "val_df": val_df.reset_index(drop=True),
    }


def country_match_array(pred_lat, pred_lon, true_iso_codes, tree, geoms, iso_codes, names):
    matches = np.zeros(len(pred_lat), dtype=bool)
    for i in range(len(pred_lat)):
        code, _, _ = point_country(pred_lon[i], pred_lat[i], tree, geoms, iso_codes, names)
        matches[i] = (code is not None) and (code == true_iso_codes[i])
    return matches


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("[proxy] loading country boundary index ...")
    tree, geoms, iso_codes, names = load_country_index()

    print("[proxy] running real model over val split ...")
    real = get_real_model_val_predictions(cfg)
    val_df = real["val_df"]
    true_iso = val_df["iso_country_code"].values

    print("[proxy] computing country match (real model predictions) ...")
    real_match = country_match_array(real["pred_lat"], real["pred_lon"], true_iso, tree, geoms, iso_codes, names)

    n = len(val_df)
    trivial_lat = np.full(n, TRIVIAL_LAT)
    trivial_lon = np.full(n, TRIVIAL_LON)
    trivial_err = haversine_km(trivial_lat, trivial_lon, real["true_lat"], real["true_lon"])
    trivial_radius = np.full(n, TRIVIAL_RADIUS_KM)

    print("[proxy] computing country match (trivial predictor, single fixed point) ...")
    trivial_code, trivial_name, _ = point_country(TRIVIAL_LON, TRIVIAL_LAT, tree, geoms, iso_codes, names)
    print(f"[proxy] trivial centroid ({TRIVIAL_LAT:.3f},{TRIVIAL_LON:.3f}) resolves to country={trivial_code} ({trivial_name})")
    trivial_match = (true_iso == trivial_code)

    real_hit_rate = float(np.mean(real["err_km"] <= real["radius_km"]))
    trivial_hit_rate = float(np.mean(trivial_err <= trivial_radius))

    print(f"\n[proxy] val n={n}")
    print(f"[proxy] real model: median err={np.median(real['err_km']):.1f}km  "
          f"country_match_rate={real_match.mean():.3f}  radius-hit-rate={real_hit_rate:.3f}")
    print(f"[proxy] trivial:    median err={np.median(trivial_err):.1f}km  "
          f"country_match_rate={trivial_match.mean():.3f}  radius-hit-rate={trivial_hit_rate:.3f}  "
          f"(fixed point country={trivial_code})")

    # A 4-parameter fit-to-2-anchors is underdetermined; letting the
    # optimizer pick all of A, D, w_cal, c_bonus freely degenerately
    # collapsed to D -> unbounded (distance term saturating near-constant for
    # every image, residual ~4.2 -- an obviously wrong fit that contradicts
    # the documented "steep exponential decay"). Instead: fix the *shape*
    # (D, w_cal, c_bonus) from the problem description as principled,
    # judgment-call constants -- D=1500km is "steep" in the sense that it
    # separates our model's median error (1462km -> raw dist 0.38) from the
    # trivial baseline's (7571km -> raw dist 0.006) by ~60x, and w_cal/
    # c_bonus are kept secondary to the dominant distance term -- then solve
    # only for an affine RESCALING (a, b) of the resulting raw composite onto
    # the two known Kaggle scores. This is exactly determined (2 unknowns, 2
    # equations, closed-form), so it can't degenerate, and it's honest about
    # what's actually known: we know the correct *shape* of the metric from
    # the problem statement, not its absolute scale -- the anchors fix scale
    # and offset only. Ranking between candidate configs is preserved
    # regardless of (a, b) as long as a > 0, since it's a monotonic map.
    D = 1500.0
    w_cal = 0.2
    c_bonus = 0.15
    A = 1.0  # raw distance-term amplitude; scale is absorbed into (a, b) below

    raw_real = median_score(real["err_km"], real["radius_km"], real_match, A, D, w_cal, c_bonus)
    raw_triv = median_score(trivial_err, trivial_radius, trivial_match, A, D, w_cal, c_bonus)

    # solve a*raw_real + b = ANCHOR_REAL_SCORE ; a*raw_triv + b = ANCHOR_TRIVIAL_SCORE
    a_scale = (ANCHOR_REAL_SCORE - ANCHOR_TRIVIAL_SCORE) / (raw_real - raw_triv)
    b_offset = ANCHOR_REAL_SCORE - a_scale * raw_real

    s_real = a_scale * raw_real + b_offset
    s_triv = a_scale * raw_triv + b_offset

    print(f"\n[proxy] fixed shape constants: A={A}  D={D}km  w_cal={w_cal}  c_bonus={c_bonus}")
    print(f"[proxy] raw composite medians: real={raw_real:.4f}  trivial={raw_triv:.4f}")
    print(f"[proxy] affine calibration: a_scale={a_scale:.4f}  b_offset={b_offset:.4f}")
    print(f"[proxy] reproduced scores (exact by construction): real={s_real:.5f} (target {ANCHOR_REAL_SCORE})  "
          f"trivial={s_triv:.5f} (target {ANCHOR_TRIVIAL_SCORE})")

    # sanity check: raw composite for a near-perfect prediction (err~0,
    # tight-correct radius, country match) -- what would the calibrated
    # score look like if we nailed every image? Useful to compare against
    # the "other teams scoring ~50" data point (not itself an anchor).
    raw_near_perfect = A * 1.0 + w_cal * 1.0 * (1.0 - 0.01) + c_bonus
    s_near_perfect = a_scale * raw_near_perfect + b_offset
    print(f"[proxy] sanity check -- near-perfect-prediction raw={raw_near_perfect:.3f} -> "
          f"calibrated score={s_near_perfect:.2f} (reported other-team scores are ~50, for context only)")

    out = {
        "A": A, "D_km": D, "w_cal": w_cal, "c_bonus": c_bonus, "R_MAX_KM": R_MAX_KM,
        "a_scale": a_scale, "b_offset": b_offset,
        "anchor_real_target": ANCHOR_REAL_SCORE, "anchor_real_reproduced": s_real,
        "anchor_trivial_target": ANCHOR_TRIVIAL_SCORE, "anchor_trivial_reproduced": s_triv,
        "near_perfect_raw": raw_near_perfect, "near_perfect_calibrated_score": s_near_perfect,
        "val_n": n,
        "real_median_err_km": float(np.median(real["err_km"])),
        "real_country_match_rate": float(real_match.mean()),
        "real_radius_hit_rate": real_hit_rate,
        "trivial_median_err_km": float(np.median(trivial_err)),
        "trivial_country_match_rate": float(trivial_match.mean()),
        "trivial_radius_hit_rate": trivial_hit_rate,
        "trivial_fixed_country": trivial_code,
    }
    out_dir = OUTPUTS / "calibration"
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / "proxy_metric_fit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[proxy] wrote {out_dir}/proxy_metric_fit.json")


def score_config(config_path, fit_path=None):
    """Score any trained checkpoint against the fitted proxy constants,
    without re-fitting. Use this to rank candidate configs before deciding
    whether a real Kaggle submission is worth spending."""
    fit_path = fit_path or (OUTPUTS / "calibration" / "proxy_metric_fit.json")
    with open(fit_path) as f:
        fit = json.load(f)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("[proxy] loading country boundary index ...")
    tree, geoms, iso_codes, names = load_country_index()
    print(f"[proxy] running {cfg['run_name']} over val split ...")
    real = get_real_model_val_predictions(cfg)
    val_df = real["val_df"]
    true_iso = val_df["iso_country_code"].values
    match = country_match_array(real["pred_lat"], real["pred_lon"], true_iso, tree, geoms, iso_codes, names)

    raw = median_score(real["err_km"], real["radius_km"], match,
                        fit["A"], fit["D_km"], fit["w_cal"], fit["c_bonus"])
    score = fit["a_scale"] * raw + fit["b_offset"]
    hit_rate = float(np.mean(real["err_km"] <= real["radius_km"]))

    print(f"\n[proxy] {cfg['run_name']}: median_err={np.median(real['err_km']):.1f}km  "
          f"country_match_rate={match.mean():.3f}  radius_hit_rate={hit_rate:.3f}")
    print(f"[proxy] {cfg['run_name']}: raw={raw:.4f}  proxy_score={score:.5f}  "
          f"(baseline anchor for comparison: {ANCHOR_REAL_SCORE})")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--fit-only", action="store_true",
                         help="(re)fit the proxy's affine calibration from the two Kaggle anchors")
    parser.add_argument("--score-only", action="store_true",
                         help="score --config against the existing fit without re-fitting "
                              "(requires a checkpoint + calibration.json for that run_name)")
    args = parser.parse_args()
    if args.score_only:
        score_config(args.config)
    else:
        main(args.config)
