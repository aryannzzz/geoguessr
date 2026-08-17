"""One-off verification: rerun the baseline checkpoint over all 500 Test Set 1
images under both the buggy (naive square resize) and fixed (short-side
resize + center-crop) preprocessing paths, and compare the model's actual
classifier argmax geocell -- not a proxy inferred from final lat/lon -- to
quantify the aspect-ratio bug's real impact.
"""
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
from src.data.geocells import haversine_km
from src.models.model import GeoModel
from src.train import set_seed, offsets_to_latlon

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"


class DualPreprocessTestDataset(Dataset):
    def __init__(self, image_ids, img_dir, img_size=224):
        self.image_ids = list(image_ids)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMG_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.image_ids)

    def _buggy(self, im):
        return im.resize((self.img_size, self.img_size), Image.BILINEAR)

    def _fixed(self, im):
        w, h = im.size
        if (w, h) == (self.img_size, self.img_size):
            return im
        short = min(w, h)
        scale = self.img_size / short
        new_w, new_h = round(w * scale), round(h * scale)
        im = im.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - self.img_size) // 2
        top = (new_h - self.img_size) // 2
        return im.crop((left, top, left + self.img_size, top + self.img_size))

    def _to_tensor(self, im):
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return (arr - self.mean) / self.std

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        im = Image.open(self.img_dir / image_id).convert("RGB")
        return self._to_tensor(self._buggy(im)), self._to_tensor(self._fixed(im)), image_id


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "baseline.yaml"))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_sub = pd.read_csv(RAW / "sample_submission.csv")
    image_ids = sample_sub["image_id"].tolist()
    test_dir = RAW / "test_images_sampled"

    cell_table = pd.read_csv(PROCESSED / "geocell_centroids.csv").set_index("cell_id")
    num_cells = len(cell_table)

    ckpt = torch.load(CHECKPOINTS / f"{cfg['run_name']}_best.pt", map_location=device, weights_only=False)
    model = GeoModel(num_cells=num_cells, backbone_name=cfg["backbone_name"],
                      img_size=cfg["img_size"], n_unfrozen_blocks=cfg["n_unfrozen_blocks"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']}, {num_cells} cells, {len(image_ids)} test images")

    ds = DualPreprocessTestDataset(image_ids, test_dir, img_size=cfg["img_size"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    all_ids = []
    buggy_cell, fixed_cell = [], []
    buggy_lat, buggy_lon, fixed_lat, fixed_lon = [], [], [], []
    with torch.no_grad():
        for buggy_imgs, fixed_imgs, ids in loader:
            for imgs, cell_out, lat_out, lon_out in [
                (buggy_imgs, buggy_cell, buggy_lat, buggy_lon),
                (fixed_imgs, fixed_cell, fixed_lat, fixed_lon),
            ]:
                imgs = imgs.to(device, non_blocking=True)
                logits, offset_pred = model(imgs)
                pred_cell = logits.argmax(dim=1).cpu().numpy()
                pred_offset_km = offset_pred.cpu().numpy() * cfg["offset_scale_km"]
                pred_lat, pred_lon = offsets_to_latlon(pred_cell, pred_offset_km, cell_table)
                cell_out.append(pred_cell)
                lat_out.append(pred_lat)
                lon_out.append(pred_lon)
            all_ids.extend(ids)

    buggy_cell = np.concatenate(buggy_cell)
    fixed_cell = np.concatenate(fixed_cell)
    buggy_lat, buggy_lon = np.concatenate(buggy_lat), np.concatenate(buggy_lon)
    fixed_lat, fixed_lon = np.concatenate(fixed_lat), np.concatenate(fixed_lon)

    moved_frac = float((buggy_cell != fixed_cell).mean())
    shift_km = haversine_km(buggy_lat, buggy_lon, fixed_lat, fixed_lon)

    print(f"\nn={len(all_ids)}")
    print(f"fraction with different classifier argmax geocell: {moved_frac:.3f} ({moved_frac*100:.1f}%)")
    print(f"median haversine shift (final point): {np.median(shift_km):.1f} km")
    print(f"mean haversine shift (final point): {np.mean(shift_km):.1f} km")


if __name__ == "__main__":
    main()
