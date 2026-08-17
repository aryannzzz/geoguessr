"""Build the unified training manifest: image_id, lat, lon, iso_country_code, source.

The provided CSV has no iso_country_code column, so country is derived here
via point-in-polygon lookup against data/raw/country_boundaries.geojson (the
same authoritative boundaries the competition metric's country bonus uses).
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_country_index():
    """Load boundary polygons with resolved ISO codes.

    The provided boundary file uses Natural-Earth-style properties: ISO_A2 is
    "-99" (a documented placeholder for "no code") for several real countries
    (UK, Portugal, Netherlands, Norway, ...), and even ISO_A2_EH — the field
    meant to patch that — is still "-99" for a few polygon fragments (e.g. one
    of the two South Korea / North Korea polygons) while the country's *other*
    polygon fragment has the correct code. We resolve by: ISO_A2_EH -> ISO_A2
    -> majority code seen elsewhere under the same country_name. What's left
    after that (Somaliland, Northern Cyprus, Dhekelia/Akrotiri Sovereign Base
    Areas, Siachen Glacier, Bir Tawil, disputed reefs, ...) are genuinely
    disputed/unrecognized territories with no ISO 3166-1 alpha-2 code -- left
    as None rather than mislabeled.
    """
    with open(RAW / "country_boundaries.geojson") as f:
        gj = json.load(f)

    raw_names, raw_a2, raw_eh = [], [], []
    for feat in gj["features"]:
        p = feat["properties"]
        raw_names.append(p["country_name"])
        raw_a2.append(p["ISO_A2"])
        raw_eh.append(p["ISO_A2_EH"])

    # majority non-placeholder code per country name, for the remaining gaps
    name_to_code = {}
    for name, a2, eh in zip(raw_names, raw_a2, raw_eh):
        code = eh if eh != "-99" else (a2 if a2 != "-99" else None)
        if code:
            name_to_code.setdefault(name, Counter()).update([code])
    name_to_code = {n: c.most_common(1)[0][0] for n, c in name_to_code.items()}

    geoms, iso_codes, names = [], [], []
    for feat, name, a2, eh in zip(gj["features"], raw_names, raw_a2, raw_eh):
        code = eh if eh != "-99" else (a2 if a2 != "-99" else name_to_code.get(name))
        geoms.append(shape(feat["geometry"]))
        iso_codes.append(code)
        names.append(name)
    tree = STRtree(geoms)
    return tree, geoms, iso_codes, names


def point_country(lon, lat, tree, geoms, iso_codes, names, max_nearest_km=50):
    pt = Point(lon, lat)
    idx = tree.query(pt, predicate="intersects")
    if len(idx) > 0:
        i = int(idx[0])
        return iso_codes[i], names[i], 0.0
    # fall back to true nearest-polygon search (handles coastline/small-island
    # points whose exact coordinate falls just outside the polygon boundary)
    i = int(tree.nearest(pt))
    dist_deg = geoms[i].distance(pt)
    dist_km = dist_deg * 111.0  # rough deg->km, fine for a small offshore gap
    if dist_km > max_nearest_km:
        return None, None, dist_km
    return iso_codes[i], names[i], dist_km


def build():
    df = pd.read_csv(RAW / "training_dataset" / "noised_dataset" / "ground_truth_coordinates.csv")
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df["image_id"] = df["image_id"] + ".jpg"
    df["source"] = "provided"

    tree, geoms, iso_codes, names = load_country_index()
    countries, cnames, dists = [], [], []
    for lon, lat in zip(df["lon"], df["lat"]):
        iso, name, dist_km = point_country(lon, lat, tree, geoms, iso_codes, names)
        countries.append(iso)
        cnames.append(name)
        dists.append(dist_km)
    df["iso_country_code"] = countries
    df["country_name"] = cnames
    df["country_match_dist_km"] = dists

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "train_manifest.csv"
    df.to_csv(out_path, index=False)
    n_unmatched = df["iso_country_code"].isna().sum()
    n_nearest = (df["country_match_dist_km"] > 0).sum()
    print(f"wrote {out_path} ({len(df)} rows, {n_unmatched} still unmatched (>50km from any polygon), "
          f"{n_nearest} resolved via nearest-polygon fallback)")
    return df


if __name__ == "__main__":
    build()
