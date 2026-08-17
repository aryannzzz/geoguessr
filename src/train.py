"""Phase 3 training: geocell classifier + tangent-plane offset regressor on
a DINOv2-small backbone (last N blocks unfrozen). Mixed precision, seeded,
checkpointed every epoch, logs a loss curve + validation Haversine error.

Run: python3 src/train.py [--config configs/baseline.yaml]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUTS = ROOT / "outputs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def offsets_to_latlon(cell_ids, offsets_km, cell_table):
    centroid_lat = cell_table.loc[cell_ids, "centroid_lat"].values
    centroid_lon = cell_table.loc[cell_ids, "centroid_lon"].values
    dx_km = offsets_km[:, 0]
    dy_km = offsets_km[:, 1]
    lat = centroid_lat + dy_km / KM_PER_LAT_DEG
    lon = centroid_lon + dx_km / km_per_lon_deg(centroid_lat)
    return lat, lon


def run_epoch(model, loader, optimizer, scaler, cfg, device, cell_table, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    smoothl1 = nn.SmoothL1Loss()

    total_loss = total_cls = total_reg = 0.0
    n_batches = 0
    all_pred_lat, all_pred_lon, all_true_lat, all_true_lon = [], [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, cell_ids, offsets, lats, lons in loader:
            imgs = imgs.to(device, non_blocking=True)
            cell_ids = cell_ids.to(device, non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=cfg["amp"] and device.type == "cuda"):
                logits, offset_pred = model(imgs)
                loss_cls = ce(logits, cell_ids)
                loss_reg = smoothl1(offset_pred, offsets)
                loss = cfg["cls_loss_weight"] * loss_cls + cfg["reg_loss_weight"] * loss_reg

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_reg += loss_reg.item()
            n_batches += 1

            pred_cell = logits.argmax(dim=1).detach().cpu().numpy()
            pred_offset_km = (offset_pred.detach().cpu().numpy()) * cfg["offset_scale_km"]
            pred_lat, pred_lon = offsets_to_latlon(pred_cell, pred_offset_km, cell_table)
            all_pred_lat.append(pred_lat)
            all_pred_lon.append(pred_lon)
            all_true_lat.append(lats.numpy())
            all_true_lon.append(lons.numpy())

    pred_lat = np.concatenate(all_pred_lat)
    pred_lon = np.concatenate(all_pred_lon)
    true_lat = np.concatenate(all_true_lat)
    true_lon = np.concatenate(all_true_lon)
    err_km = haversine_km(pred_lat, pred_lon, true_lat, true_lon)

    metrics = {
        "loss": total_loss / n_batches,
        "loss_cls": total_cls / n_batches,
        "loss_reg": total_reg / n_batches,
        "haversine_median_km": float(np.median(err_km)),
        "haversine_mean_km": float(np.mean(err_km)),
        "haversine_p90_km": float(np.quantile(err_km, 0.9)),
    }
    return metrics, err_km


def main(config_path, debug_limit=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, seed: {cfg['seed']}")

    df = pd.read_csv(PROCESSED / "train_manifest_geocells.csv")
    cell_table = pd.read_csv(PROCESSED / "geocell_centroids.csv").set_index("cell_id")
    num_cells = df["cell_id"].nunique()
    cfg["num_cells"] = int(num_cells)

    train_df, val_df = make_split(df, val_frac=cfg["val_frac"], seed=cfg["seed"])
    if debug_limit:
        train_df = train_df.iloc[:debug_limit].reset_index(drop=True)
        val_df = val_df.iloc[:max(debug_limit // 4, 8)].reset_index(drop=True)
    print(f"train: {len(train_df)}  val: {len(val_df)}  num_cells: {num_cells}")

    train_ds = GeoImageDataset(train_df, img_size=cfg["img_size"], train=True,
                                offset_scale_km=cfg["offset_scale_km"])
    val_ds = GeoImageDataset(val_df, img_size=cfg["img_size"], train=False,
                              offset_scale_km=cfg["offset_scale_km"])

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], worker_init_fn=worker_init_fn,
                               generator=g, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], worker_init_fn=worker_init_fn,
                             pin_memory=True, persistent_workers=True)

    model = GeoModel(num_cells=num_cells, backbone_name=cfg["backbone_name"],
                      img_size=cfg["img_size"], n_unfrozen_blocks=cfg["n_unfrozen_blocks"]).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.cell_head.parameters()) + list(model.offset_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr_backbone"]},
        {"params": head_params, "lr": cfg["lr_heads"]},
    ], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    use_cosine = cfg.get("lr_schedule", "constant") == "cosine"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]) if use_cosine else None
    patience = cfg.get("patience")  # None disables early stopping

    CHECKPOINTS.mkdir(exist_ok=True, parents=True)
    (OUTPUTS / "training").mkdir(exist_ok=True, parents=True)

    history = []
    best_val_median = float("inf")
    epochs_no_improve = 0
    min_delta = cfg.get("early_stop_min_delta", 1.0)  # km
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        ep_t0 = time.time()
        train_metrics, _ = run_epoch(model, train_loader, optimizer, scaler, cfg, device, cell_table, train=True)
        val_metrics, _ = run_epoch(model, val_loader, optimizer, scaler, cfg, device, cell_table, train=False)
        if scheduler is not None:
            scheduler.step()
        ep_time = time.time() - ep_t0

        row = {"epoch": epoch, "epoch_time_s": ep_time}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        print(f"epoch {epoch:2d}/{cfg['epochs']}  ({ep_time:5.1f}s)  "
              f"train_loss={train_metrics['loss']:.4f}  val_loss={val_metrics['loss']:.4f}  "
              f"val_haversine_median={val_metrics['haversine_median_km']:.1f}km  "
              f"val_haversine_mean={val_metrics['haversine_mean_km']:.1f}km"
              + (f"  lr_backbone={scheduler.get_last_lr()[0]:.2e}" if scheduler is not None else ""))

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                   CHECKPOINTS / f"{cfg['run_name']}_last.pt")
        if val_metrics["haversine_median_km"] < best_val_median - min_delta:
            best_val_median = val_metrics["haversine_median_km"]
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       CHECKPOINTS / f"{cfg['run_name']}_best.pt")
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"\nearly stopping: no val improvement > {min_delta}km for {patience} epochs "
                      f"(best {best_val_median:.1f}km)")
                break

    total_time = time.time() - t0
    print(f"\ntotal training time: {total_time/60:.1f} min, best val haversine median: {best_val_median:.1f}km")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUTS / "training" / f"{cfg['run_name']}_history.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    axes[0].set_title("total loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(hist_df["epoch"], hist_df["train_loss_cls"], label="train cls")
    axes[1].plot(hist_df["epoch"], hist_df["val_loss_cls"], label="val cls")
    axes[1].plot(hist_df["epoch"], hist_df["train_loss_reg"], label="train reg", linestyle="--")
    axes[1].plot(hist_df["epoch"], hist_df["val_loss_reg"], label="val reg", linestyle="--")
    axes[1].set_title("cls (CE) vs reg (SmoothL1) loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    axes[2].plot(hist_df["epoch"], hist_df["val_haversine_median_km"], label="median", marker="o")
    axes[2].plot(hist_df["epoch"], hist_df["val_haversine_mean_km"], label="mean", marker="o")
    axes[2].plot(hist_df["epoch"], hist_df["val_haversine_p90_km"], label="p90", marker="o")
    axes[2].set_title("validation Haversine error (km)")
    axes[2].set_xlabel("epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUTPUTS / "training" / f"{cfg['run_name']}_curves.png", dpi=130)
    print(f"wrote {OUTPUTS/'training'}/{cfg['run_name']}_history.csv and _curves.png")

    with open(OUTPUTS / "training" / f"{cfg['run_name']}_config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--debug-limit", type=int, default=None,
                         help="cap train/val rows for a fast smoke test")
    args = parser.parse_args()
    main(args.config, debug_limit=args.debug_limit)
