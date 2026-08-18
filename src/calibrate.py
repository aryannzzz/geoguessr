"""Phase 4: radius calibration from validation residuals.

Runs the trained model on the held-out val split (same seed/split as
training, so this is honest held-out data) and buckets examples by the
model's own classifier confidence (max softmax prob over geocells). Within
each confidence bucket, the predicted radius is a quantile of that bucket's
actual Haversine error -- so low-confidence predictions get a wider radius
and high-confidence predictions get a tighter one, rather than one fixed
global radius. This lets validation residuals (which already contain the
label noise found in EDA) set the radius directly instead of modeling that
noise separately.

quantile=0.5 (down from an earlier 0.75), n_bins=10 (up from 6): a real
Kaggle submission showed the wider/coarser settings pushed the *median*
claimed radius to ~5,530km -- the competition's country bonus requires a
"reasonably tight" radius (see README/problem statement), and the
calibration term's reward also shrinks as radius widens, so a wide median
radius forfeits both on the majority of predictions (scoring is
median-based) regardless of how accurate the underlying point is. Setting
radius = the *median* (not p75) error per bucket pairs naturally with the
competition's own median-based scoring: on the accurate half of a bucket
you're tight-and-correct, on the inaccurate half you're tight-and-wrong --
a deliberate, informed risk trade rather than defaulting to "wide and
safe." More bins gives finer confidence resolution so genuinely confident
predictions aren't lumped in with mediocre ones.

Run: python3 src/calibrate.py [--config configs/baseline.yaml] [--quantile 0.5] [--n-bins 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GeoImageDataset, make_split
from src.data.geocells import haversine_km, km_per_lon_deg, KM_PER_LAT_DEG
from src.models.model import GeoModel
from src.train import set_seed, worker_init_fn, offsets_to_latlon

PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUTS = ROOT / "outputs"


def main(config_path, quantile=0.5, n_bins=10):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"loaded checkpoint from epoch {ckpt['epoch']}")

    all_conf, all_err = [], []
    with torch.no_grad():
        for imgs, cell_ids, offsets, lats, lons in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits, offset_pred = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values.cpu().numpy()
            pred_cell = logits.argmax(dim=1).cpu().numpy()
            pred_offset_km = offset_pred.cpu().numpy() * cfg["offset_scale_km"]
            pred_lat, pred_lon = offsets_to_latlon(pred_cell, pred_offset_km, cell_table)
            err = haversine_km(pred_lat, pred_lon, lats.numpy(), lons.numpy())
            all_conf.append(conf)
            all_err.append(err)

    conf = np.concatenate(all_conf)
    err = np.concatenate(all_err)
    print(f"val n={len(conf)}  overall median err={np.median(err):.1f}km  "
          f"mean err={np.mean(err):.1f}km")

    bin_edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    bin_idx = np.clip(np.digitize(conf, bin_edges[1:-1], right=True), 0, actual_n_bins - 1)

    table = []
    for b in range(actual_n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            radius = float(np.quantile(err, quantile))
        else:
            radius = float(np.quantile(err[mask], quantile))
        table.append({
            "bin": b,
            "conf_lo": float(bin_edges[b]),
            "conf_hi": float(bin_edges[b + 1]),
            "n": n,
            "median_err_km": float(np.median(err[mask])) if n else None,
            "radius_km": radius,
        })
        print(f"  bin {b}: conf [{bin_edges[b]:.3f}, {bin_edges[b+1]:.3f})  n={n:4d}  "
              f"median_err={table[-1]['median_err_km']}  radius(q{quantile})={radius:.1f}km")

    out = {"quantile": quantile, "n_bins": actual_n_bins, "bin_edges": bin_edges.tolist(), "table": table}
    calib_dir = OUTPUTS / "calibration"
    calib_dir.mkdir(exist_ok=True, parents=True)
    with open(calib_dir / f"{cfg['run_name']}_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {calib_dir}/{cfg['run_name']}_calibration.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(conf, err, s=4, alpha=0.3)
    axes[0].set_xlabel("classifier confidence (max softmax prob)")
    axes[0].set_ylabel("Haversine error (km)")
    axes[0].set_yscale("log")
    axes[0].set_title("val error vs. confidence")

    centers = [(t["conf_lo"] + t["conf_hi"]) / 2 for t in table]
    radii = [t["radius_km"] for t in table]
    axes[1].plot(centers, radii, marker="o")
    axes[1].set_xlabel("confidence bin center")
    axes[1].set_ylabel(f"calibrated radius (q{quantile}) km")
    axes[1].set_title("confidence -> radius mapping")
    fig.tight_layout()
    fig.savefig(calib_dir / f"{cfg['run_name']}_calibration_curve.png", dpi=130)
    print(f"wrote {calib_dir}/{cfg['run_name']}_calibration_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()
    main(args.config, quantile=args.quantile, n_bins=args.n_bins)
