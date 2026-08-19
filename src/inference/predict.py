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
    """Unlabeled test images, same preprocessing as val (no augmentation
    beyond TTA crops -- no flip, ever: it breaks the driving-side/left-right
    geographic cue the same way it would in training).

    Train/val images are all pre-cropped to 640x640 square, so a plain
    resize-to-square never distorts them. Test images are NOT square (500/500
    sampled are e.g. 647x486 or up to 1200x486 -- wide dashcam/street-view
    frames) -- a plain resize-to-square would squash them non-uniformly
    (up to ~2.5x more horizontal than vertical compression for the widest
    ones), warping every visual cue the model was trained on. Instead we
    resize the short side to img_size, then take `tta_crops` square crops
    spread along the long axis (e.g. left/center/right for a wide frame)
    instead of only the center -- a wide dashcam frame often has useful
    cues (signage, roadside features) outside the center crop.
    `tta_crops=1` reproduces the original single-center-crop behavior
    exactly (crop offset = max_offset // 2, same as before).
    """

    def __init__(self, image_ids, img_dir, img_size=224, tta_crops=3):
        self.image_ids = list(image_ids)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.tta_crops = max(1, tta_crops)
        self.mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMG_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.image_ids)

    def _resize_short_side(self, im):
        w, h = im.size
        if min(w, h) == self.img_size:
            return im
        short = min(w, h)
        scale = self.img_size / short
        new_w, new_h = round(w * scale), round(h * scale)
        return im.resize((new_w, new_h), Image.BILINEAR)

    def _tta_crop_boxes(self, w, h):
        is_wide = w >= h
        long_dim = w if is_wide else h
        max_offset = max(0, long_dim - self.img_size)
        if self.tta_crops == 1 or max_offset == 0:
            offsets = [max_offset // 2]
        else:
            offsets = np.linspace(0, max_offset, self.tta_crops).round().astype(int).tolist()
        boxes = []
        for off in offsets:
            left, top = (off, 0) if is_wide else (0, off)
            boxes.append((left, top, left + self.img_size, top + self.img_size))
        return boxes

    def _to_tensor(self, im):
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return (arr - self.mean) / self.std

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        im = Image.open(self.img_dir / image_id).convert("RGB")
        im = self._resize_short_side(im)
        w, h = im.size
        boxes = self._tta_crop_boxes(w, h)
        # always exactly self.tta_crops boxes (single-box case repeats the
        # center crop) so every item stacks into a fixed-size tensor
        while len(boxes) < self.tta_crops:
            boxes.append(boxes[-1])
        tensors = torch.stack([self._to_tensor(im.crop(b)) for b in boxes])
        return tensors, image_id


def radius_for_confidence(conf, calib):
    bin_edges = np.array(calib["bin_edges"])
    n_bins = calib["n_bins"]
    idx = np.clip(np.digitize(conf, bin_edges[1:-1], right=True), 0, n_bins - 1)
    radii = np.array([calib["table"][i]["radius_km"] for i in idx])
    return radii


def main(config_path, submission_template=None, out_path=None, tta_crops=None):
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

    tta_crops = tta_crops if tta_crops is not None else cfg.get("tta_crops", 3)
    print(f"loaded checkpoint from epoch {ckpt['epoch']}, {num_cells} cells, "
          f"{len(image_ids)} test images, tta_crops={tta_crops}")

    test_ds = TestImageDataset(image_ids, test_dir, img_size=cfg["img_size"], tta_crops=tta_crops)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    all_ids, all_lat, all_lon, all_conf = [], [], [], []
    with torch.no_grad():
        for imgs, ids in test_loader:
            # imgs: [B, tta_crops, 3, H, W] -- flatten crops into the batch
            # dim for one forward pass, then average back per image. Average
            # softmax probs (not logits) across crops -- the standard,
            # better-calibrated way to ensemble a classifier -- then argmax
            # the averaged distribution for the final cell. Offsets are
            # averaged directly too; a minor approximation when crops
            # disagree on the winning cell (their offsets are then relative
            # to different centroids), acceptable since offsets are small
            # relative to cell size and TTA crops of the same image mostly
            # agree in practice.
            B, T = imgs.shape[0], imgs.shape[1]
            imgs_flat = imgs.view(B * T, *imgs.shape[2:]).to(device, non_blocking=True)
            logits, offset_pred = model(imgs_flat)
            logits = logits.view(B, T, -1)
            offset_pred = offset_pred.view(B, T, 2)

            probs = F.softmax(logits, dim=-1).mean(dim=1)
            conf = probs.max(dim=1).values.cpu().numpy()
            pred_cell = probs.argmax(dim=1).cpu().numpy()
            pred_offset_km = offset_pred.mean(dim=1).cpu().numpy() * cfg["offset_scale_km"]
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
    parser.add_argument("--tta-crops", type=int, default=None,
                         help="crops per image, averaged (softmax probs + offsets); "
                              "1 reproduces the original single-center-crop behavior exactly; "
                              "defaults to cfg['tta_crops'] or 3")
    args = parser.parse_args()
    main(args.config, submission_template=args.submission_template, out_path=args.out,
         tta_crops=args.tta_crops)
