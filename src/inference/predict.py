"""Phase 5: offline inference on Test Set 1 -> submission CSV.

Loads the best checkpoint, runs every test image through the model to get
a predicted geocell + tangent-plane offset -> (lat, lon), then assigns a
per-example radius from the confidence-bucketed calibration table written
by src/calibrate.py. No network calls; everything needed (checkpoint,
calibration table, images) is local.

Run: python3 src/inference/predict.py [--config configs/baseline.yaml]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import IMG_MEAN, IMG_STD
from src.data.geocells import km_per_lon_deg, KM_PER_LAT_DEG
from src.models.model import GeoModel
from src.train import set_seed, offsets_to_latlon

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUTS = ROOT / "outputs"


class TestImageDataset(Dataset):
    """Unlabeled test images, same preprocessing as val (no augmentation)."""

    def __init__(self, image_ids, img_dir, img_size=224):
        self.image_ids = list(image_ids)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMG_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        im = Image.open(self.img_dir / image_id).convert("RGB")
        if im.size != (self.img_size, self.img_size):
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        arr = (arr - self.mean) / self.std
        return arr, image_id


def radius_for_confidence(conf, calib):
    bin_edges = np.array(calib["bin_edges"])
    n_bins = calib["n_bins"]
    idx = np.clip(np.digitize(conf, bin_edges[1:-1], right=True), 0, n_bins - 1)
    radii = np.array([calib["table"][i]["radius_km"] for i in idx])
    return radii


def main(config_path, submission_template=None, out_path=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_sub = pd.read_csv(submission_template or (RAW / "sample_submission.csv"))
    image_ids = sample_sub["image_id"].tolist()
    test_dir = RAW / "test_images_sampled"

    cell_table = pd.read_csv(PROCESSED / "geocell_centroids.csv").set_index("cell_id")
    num_cells = len(cell_table)

    with open(OUTPUTS / "calibration" / f"{cfg['run_name']}_calibration.json") as f:
        calib = json.load(f)

    ckpt = torch.load(CHECKPOINTS / f"{cfg['run_name']}_best.pt", map_location=device, weights_only=False)
    model = GeoModel(num_cells=num_cells, backbone_name=cfg["backbone_name"],
                      img_size=cfg["img_size"], n_unfrozen_blocks=cfg["n_unfrozen_blocks"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']}, {num_cells} cells, "
          f"{len(image_ids)} test images")

    test_ds = TestImageDataset(image_ids, test_dir, img_size=cfg["img_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    all_ids, all_lat, all_lon, all_conf = [], [], [], []
    with torch.no_grad():
        for imgs, ids in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits, offset_pred = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1).values.cpu().numpy()
            pred_cell = logits.argmax(dim=1).cpu().numpy()
            pred_offset_km = offset_pred.cpu().numpy() * cfg["offset_scale_km"]
            pred_lat, pred_lon = offsets_to_latlon(pred_cell, pred_offset_km, cell_table)

            all_ids.extend(ids)
            all_lat.append(pred_lat)
            all_lon.append(pred_lon)
            all_conf.append(conf)

    pred_lat = np.concatenate(all_lat)
    pred_lon = np.concatenate(all_lon)
    conf = np.concatenate(all_conf)
    radius = radius_for_confidence(conf, calib)

    out_df = pd.DataFrame({
        "image_id": all_ids,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "pred_radius_km": radius,
    })
    # keep the exact row order of the sample submission
    out_df = out_df.set_index("image_id").loc[image_ids].reset_index()

    assert list(out_df.columns) == ["image_id", "pred_lat", "pred_lon", "pred_radius_km"]
    assert len(out_df) == len(sample_sub), "row count mismatch vs. sample_submission.csv"
    assert out_df["pred_lat"].between(-90, 90).all()
    assert out_df["pred_lon"].between(-180, 180).all()

    out_path = Path(out_path) if out_path else OUTPUTS / "submissions" / f"{cfg['run_name']}_submission.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  (n={len(out_df)})")
    print(f"pred_radius_km: min={radius.min():.1f} median={np.median(radius):.1f} max={radius.max():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--submission-template", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.config, submission_template=args.submission_template, out_path=args.out)
