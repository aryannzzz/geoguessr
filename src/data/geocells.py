"""Geocell assignment: adaptive k-means over train (lat, lon), small cells
merged into their nearest surviving neighbor, plus per-row tangent-plane
offsets (dx_km, dy_km) from each row's assigned cell centroid.

Locked design (see EDA + user sign-off, 2026-08-17):
- k=150 starting point, cells under MIN_CELL_SIZE=15 merged into nearest
  neighbor centroid (by haversine distance between centroids).
- Rows with no resolved iso_country_code (~9.5% of train, see EDA) are
  NOT excluded here -- geocell assignment and the regression target only
  depend on (lat, lon), which every row has regardless of country match.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

K = 150
MIN_CELL_SIZE = 15
SEED = 42

DEG2RAD = np.pi / 180.0
KM_PER_LAT_DEG = 110.574


def km_per_lon_deg(lat_deg):
    return 111.320 * np.cos(lat_deg * DEG2RAD)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_geocells(df, k=K, min_size=MIN_CELL_SIZE, seed=SEED):
    coords = df[["lat", "lon"]].values
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(coords)
    labels = km.labels_.copy()
    centroids = km.cluster_centers_.copy()

    sizes = pd.Series(labels).value_counts()
    small = sizes[sizes < min_size].index.tolist()
    survivors = sizes[sizes >= min_size].index.tolist()

    if small:
        surv_centroids = centroids[survivors]
        # BallTree over surviving centroids for a proper great-circle nearest
        tree = BallTree(np.radians(surv_centroids), metric="haversine")
        for s in small:
            member_mask = labels == s
            member_coords = np.radians(coords[member_mask])
            _, nn_idx = tree.query(member_coords, k=1)
            new_labels = np.array(survivors)[nn_idx[:, 0]]
            labels[member_mask] = new_labels

    # renumber surviving labels to a dense 0..n_cells-1 range
    final_ids = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(final_ids)}
    labels = np.array([remap[l] for l in labels])
    n_cells = len(final_ids)

    # recompute centroids as the mean lat/lon of each final cell's members
    # (not the original k-means centroid, since membership shifted after merge)
    df = df.copy()
    df["cell_id"] = labels
    cell_centroids = df.groupby("cell_id")[["lat", "lon"]].mean().rename(
        columns={"lat": "centroid_lat", "lon": "centroid_lon"})
    cell_sizes = df["cell_id"].value_counts().rename("cell_size")

    df = df.merge(cell_centroids, on="cell_id", how="left")
    dlat = df["lat"] - df["centroid_lat"]
    dlon = df["lon"] - df["centroid_lon"]
    df["dy_km"] = dlat * KM_PER_LAT_DEG
    df["dx_km"] = dlon * km_per_lon_deg(df["centroid_lat"])

    cell_table = cell_centroids.join(cell_sizes).reset_index()
    return df, cell_table, n_cells


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED / "train_manifest.csv")
    df, cell_table, n_cells = build_geocells(df)

    print(f"k requested: {K}, final cell count after merging <{MIN_CELL_SIZE}-image cells: {n_cells}")
    sizes = cell_table["cell_size"]
    print(f"cell size: min={sizes.min()} p25={sizes.quantile(.25):.0f} "
          f"median={sizes.median():.0f} p75={sizes.quantile(.75):.0f} max={sizes.max()}")
    print(f"cells still <{MIN_CELL_SIZE}: {(sizes < MIN_CELL_SIZE).sum()} (should be 0)")

    offset_dist = np.sqrt(df["dx_km"] ** 2 + df["dy_km"] ** 2)
    print(f"\ntangent-plane offset magnitude (row -> its cell centroid), km: "
          f"median={offset_dist.median():.1f} p90={offset_dist.quantile(.9):.1f} max={offset_dist.max():.1f}")

    df.to_csv(PROCESSED / "train_manifest_geocells.csv", index=False)
    cell_table.to_csv(PROCESSED / "geocell_centroids.csv", index=False)
    print(f"\nwrote {PROCESSED/'train_manifest_geocells.csv'} and {PROCESSED/'geocell_centroids.csv'}")
