"""Train/val Dataset for the geocell-classification + tangent-plane-offset
regression targets, plus the stratified-by-geocell split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "raw" / "training_dataset" / "noised_dataset" / "images"

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def make_split(df, val_frac=0.1, seed=42):
    """Stratify by geocell where possible; cells too small to stratify (<2
    members) fall back to a random assignment so no row is dropped."""
    counts = df["cell_id"].value_counts()
    stratifiable = df["cell_id"].isin(counts[counts >= 2].index)

    df_strat = df[stratifiable]
    df_rest = df[~stratifiable]

    train_s, val_s = train_test_split(
        df_strat, test_size=val_frac, random_state=seed, stratify=df_strat["cell_id"])

    if len(df_rest) > 0:
        rng = np.random.default_rng(seed)
        rest_val_mask = rng.random(len(df_rest)) < val_frac
        train_r = df_rest[~rest_val_mask]
        val_r = df_rest[rest_val_mask]
        train_df = pd.concat([train_s, train_r]).sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df = pd.concat([val_s, val_r]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        train_df = train_s.sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df = val_s.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df


class GeoImageDataset(Dataset):
    def __init__(self, df, img_size=224, train=True, offset_scale_km=500.0):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train
        self.offset_scale_km = offset_scale_km
        self.mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMG_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_id):
        stem = image_id[:-4] if image_id.endswith(".jpg") else image_id
        path = IMG_DIR / f"{stem}.jpg"
        im = Image.open(path).convert("RGB")
        if self.train:
            im = self._scale_jitter_crop(im)
        if im.size != (self.img_size, self.img_size):
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        return im

    def _scale_jitter_crop(self, im, scale_range=(0.8, 1.0)):
        # Mild zoom-in jitter: crop a random square sub-region (80-100% of
        # the source side) before resizing to img_size. Train images are all
        # 640x640, so cropping square-to-square never distorts aspect ratio
        # the way the test-time resize bug did. Adds scale/translation
        # variety without touching orientation (no flip, no rotation), so it
        # can't corrupt the driving-side/left-right geographic cue.
        w, h = im.size
        scale = np.random.uniform(*scale_range)
        crop_size = int(round(min(w, h) * scale))
        max_off_x = w - crop_size
        max_off_y = h - crop_size
        left = np.random.randint(0, max_off_x + 1) if max_off_x > 0 else 0
        top = np.random.randint(0, max_off_y + 1) if max_off_y > 0 else 0
        return im.crop((left, top, left + crop_size, top + crop_size))

    def _augment(self, im):
        # No horizontal flip: it destroys a real geographic cue (driving
        # side / left-vs-right-hand traffic correlates with country), so
        # flipping would train against a signal the model should be free to
        # use. Color jitter -- robustness to lighting/exposure without
        # touching spatial/geographic content.
        from PIL import ImageEnhance
        for enhancer_cls, lo, hi in [
            (ImageEnhance.Brightness, 0.8, 1.2),
            (ImageEnhance.Contrast, 0.8, 1.2),
            (ImageEnhance.Color, 0.8, 1.2),
            (ImageEnhance.Sharpness, 0.7, 1.3),
        ]:
            factor = np.random.uniform(lo, hi)
            im = enhancer_cls(im).enhance(factor)
        return im

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = self._load_image(row["image_id"])
        if self.train:
            im = self._augment(im)
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        arr = (arr - self.mean) / self.std

        cell_id = torch.tensor(int(row["cell_id"]), dtype=torch.long)
        offset = torch.tensor(
            [row["dx_km"] / self.offset_scale_km, row["dy_km"] / self.offset_scale_km],
            dtype=torch.float32,
        )
        lat = torch.tensor(row["lat"], dtype=torch.float64)
        lon = torch.tensor(row["lon"], dtype=torch.float64)
        return arr, cell_id, offset, lat, lon
