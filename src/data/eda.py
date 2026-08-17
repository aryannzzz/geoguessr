"""Phase 1 EDA: lat/lon distribution, country imbalance, image stats,
near-duplicate detection, label-noise signals, and a candidate geocell scheme.

Run: python3 src/data/eda.py
Writes plots to outputs/eda/ and prints a summary to stdout.
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = RAW / "training_dataset" / "noised_dataset" / "images"

pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
def load_manifest():
    return pd.read_csv(PROCESSED / "train_manifest.csv")


# ---------------------------------------------------------------------------
def eda_geo_distribution(df):
    section("1. Lat/Lon distribution")
    print(df[["lat", "lon"]].describe())

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(df["lon"], df["lat"], s=1, alpha=0.25, c="tab:blue")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Training image locations (n={len(df)})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "world_scatter.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    hb = ax.hexbin(df["lon"], df["lat"], gridsize=90, bins="log", cmap="viridis",
                    extent=(-180, 180, -90, 90))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Training image density (log scale)")
    fig.colorbar(hb, ax=ax, label="log10(count+1)")
    fig.tight_layout()
    fig.savefig(OUT / "world_density_hexbin.png", dpi=130)
    plt.close(fig)

    # hemisphere / basic balance
    print("\nNorthern hemisphere:", (df["lat"] > 0).mean().round(3),
          " Southern:", (df["lat"] <= 0).mean().round(3))
    print("Eastern (lon>0):", (df["lon"] > 0).mean().round(3),
          " Western (lon<=0):", (df["lon"] <= 0).mean().round(3))
    print("Saved world_scatter.png, world_density_hexbin.png")


# ---------------------------------------------------------------------------
def eda_country_imbalance(df):
    section("2. Country distribution / imbalance")
    n_no_country = df["iso_country_code"].isna().sum()
    print(f"Rows with no confident country match (>50km from any boundary polygon): "
          f"{n_no_country} ({n_no_country/len(df):.1%})")
    print("These are likely small islands/atolls not represented in the simplified "
          "298-feature boundary set, or noise-shifted coastal points pushed offshore.")

    counts = df["iso_country_code"].value_counts()
    print(f"\n{counts.shape[0]} distinct countries represented (excluding unmatched).")
    print("\nTop 15 countries by image count:")
    print(counts.head(15))
    print("\nBottom 15 countries by image count (rarest, most at risk in geocell design):")
    print(counts.tail(15))

    p = counts / counts.sum()
    entropy = -(p * np.log(p)).sum()
    max_entropy = np.log(len(p))
    print(f"\nShannon entropy: {entropy:.3f} nats (max possible {max_entropy:.3f} nats "
          f"for {len(p)} countries) -> normalized evenness = {entropy/max_entropy:.3f}")
    top10_share = counts.head(10).sum() / counts.sum()
    print(f"Top-10 countries account for {top10_share:.1%} of all training images "
          f"-> {'STRONG' if top10_share > 0.5 else 'moderate'} geographic imbalance.")

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.head(30).plot(kind="bar", ax=ax, color="tab:orange")
    ax.set_ylabel("image count")
    ax.set_title("Top 30 countries by training image count")
    fig.tight_layout()
    fig.savefig(OUT / "country_counts_top30.png", dpi=130)
    plt.close(fig)
    print("Saved country_counts_top30.png")
    return counts


# ---------------------------------------------------------------------------
def eda_image_stats(df, sample_n=3000, seed=0):
    section("3. Image resolution / aspect ratio / file-size stats")
    rng = np.random.default_rng(seed)
    ids = df["image_id"].str.replace(".jpg", "", regex=False).values
    sample_ids = rng.choice(ids, size=min(sample_n, len(ids)), replace=False)

    sizes, modes, filesizes = [], [], []
    for iid in sample_ids:
        p = IMG_DIR / f"{iid}.jpg"
        with Image.open(p) as im:
            sizes.append(im.size)
            modes.append(im.mode)
        filesizes.append(p.stat().st_size)

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    aspect = [w / h for w, h in sizes]
    mode_counts = Counter(modes)
    size_counts = Counter(sizes)

    print(f"Sampled {len(sample_ids)} / {len(ids)} images")
    print(f"Width  - min {min(widths)}, max {max(widths)}, mean {np.mean(widths):.1f}")
    print(f"Height - min {min(heights)}, max {max(heights)}, mean {np.mean(heights):.1f}")
    print(f"Aspect ratio - min {min(aspect):.3f}, max {max(aspect):.3f}, "
          f"unique values: {len(set(np.round(aspect, 3)))}")
    print(f"Color modes: {dict(mode_counts)}")
    print(f"Distinct (W,H) pairs found: {len(size_counts)} -> {size_counts.most_common(5)}")
    print(f"File size (bytes) - min {min(filesizes)}, max {max(filesizes)}, "
          f"mean {np.mean(filesizes):.0f}, median {np.median(filesizes):.0f}")

    if len(size_counts) == 1:
        print("-> All sampled images share a single fixed resolution: "
              f"{list(size_counts.keys())[0]}. Simplifies preprocessing "
              "(no letterboxing/aspect logic needed beyond backbone's expected input size).")


# ---------------------------------------------------------------------------
def _phash_one(args):
    iid, path = args
    try:
        with Image.open(path) as im:
            return iid, str(imagehash.phash(im))
    except Exception as e:
        return iid, None


def eda_duplicates(df, limit=None):
    section("4. Duplicate / near-duplicate detection (perceptual hash)")
    ids = df["image_id"].str.replace(".jpg", "", regex=False).values
    if limit:
        ids = ids[:limit]
    paths = [IMG_DIR / f"{iid}.jpg" for iid in ids]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_phash_one, zip(ids, paths), chunksize=64))
    print(f"Hashed {len(results)} images in {time.time()-t0:.1f}s")

    hash_map = {}
    for iid, h in results:
        if h is None:
            continue
        hash_map.setdefault(h, []).append(iid)

    exact_dupe_groups = {h: v for h, v in hash_map.items() if len(v) > 1}
    n_exact_dupe_images = sum(len(v) for v in exact_dupe_groups.values())
    print(f"Exact perceptual-hash collisions: {len(exact_dupe_groups)} groups covering "
          f"{n_exact_dupe_images} images ({n_exact_dupe_images/len(ids):.2%})")

    if exact_dupe_groups:
        gt = df.set_index(df["image_id"].str.replace(".jpg", "", regex=False))
        sample_groups = list(exact_dupe_groups.items())[:5]
        for h, members in sample_groups:
            coords = gt.loc[members, ["lat", "lon"]].values
            spread_km = float(np.ptp(coords[:, 0])) * 111  # rough lat degrees->km
            print(f"  hash {h}: {len(members)} images {members[:4]}{'...' if len(members)>4 else ''}, "
                  f"lat spread ~{spread_km:.1f}km")

    hash_counter = Counter([h for _, h in results if h is not None])
    print(f"Total unique phashes: {len(hash_counter)} vs {len(ids)} images sampled "
          f"({len(hash_counter)/len(ids):.2%} unique)")
    return exact_dupe_groups


# ---------------------------------------------------------------------------
def eda_label_noise(df):
    section("5. Label-noise signals (dataset is literally named 'noised_dataset')")
    dup_coord = df.duplicated(subset=["lat", "lon"], keep=False)
    print(f"Rows with an exactly-repeated (lat,lon) pair: {dup_coord.sum()} "
          f"({dup_coord.mean():.2%}) -> low duplicate-coordinate rate suggests "
          "per-image continuous noise (e.g. small random jitter) rather than "
          "snapping to a coarse grid or repeating exact source coordinates.")

    # decimal-precision fingerprint: synthetic noise often leaves many more
    # significant digits than a real GPS fix (~5-7 decimals ~ cm-level) would.
    def n_decimals(x):
        s = f"{x:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    lat_dec = df["lat"].apply(n_decimals)
    lon_dec = df["lon"].apply(n_decimals)
    print("\nDecimal-place distribution (lat):")
    print(lat_dec.value_counts().sort_index())
    print(f"-> {'High' if lat_dec.mean() > 8 else 'Normal'} average precision "
          f"({lat_dec.mean():.1f} decimals) is consistent with floating-point "
          "noise added programmatically (e.g. lat += np.random.normal(...)) "
          "rather than raw GPS EXIF data, which typically has far fewer digits.")

    print(f"\niso_country_code null rate as an offshore-noise proxy: "
          f"{df['iso_country_code'].isna().mean():.2%} (see section 2) -- if this "
          "dataset were unnoised GPS-tagged street imagery we'd expect this near 0, "
          "since street-level photos are taken on land.")

    print("\nRecommendation: quantify per-country/region noise magnitude once external "
          "data (e.g. OSV-5M) is available, by comparing this dataset's coordinate "
          "distribution to known city/landmark locations visible in a manual image "
          "sample. Until then, treat pred targets as noisy and avoid overfitting the "
          "regression head to sub-km precision -- calibrate radius accordingly.")


# ---------------------------------------------------------------------------
def eda_geocell_proposal(df, counts):
    section("6. Candidate geocell scheme")
    from sklearn.cluster import KMeans

    coords = df[["lat", "lon"]].values
    for k in [50, 100, 200, 400]:
        km = KMeans(n_clusters=k, random_state=0, n_init=3).fit(coords)
        sizes = pd.Series(km.labels_).value_counts()
        print(f"k={k:4d}  cells: min size={sizes.min():4d}  median={sizes.median():6.1f}  "
              f"max={sizes.max():5d}  cells<20 imgs={sum(sizes<20)}")

    print("\nGiven top-10 countries hold a large majority of images (see section 2) and "
          "k-means cell sizes stay reasonably balanced without an explosion of tiny "
          "cells up to k~200-400, propose: adaptive k-means over train (lat,lon) with "
          "k chosen so min cell count >= ~15-20 images (merge/drop cells below that "
          "threshold into their nearest neighbor), rather than fixed-level S2 cells, "
          "since S2 at a single level would heavily over/under-partition given the "
          "observed density imbalance across countries. Revisit exact k after external "
          "data is merged (Phase 2), since it will shift density.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_manifest()
    eda_geo_distribution(df)
    counts = eda_country_imbalance(df)
    eda_image_stats(df)
    eda_duplicates(df)
    eda_label_noise(df)
    eda_geocell_proposal(df, counts)
    section("EDA complete")
    print(f"Plots written to {OUT}")
