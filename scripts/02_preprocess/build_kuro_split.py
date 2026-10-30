#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split the Kuro Siwo subset for v5:
  train    = the old 25 events retrieved in the first 5 batches
  held-out = new events retrieved afterwards (for cross-event generalisation evaluation)

Leak prevention:
  - of the old 25, act205 (Sri-Lanka 2017) is the same flood as Sen1Floods11, so tiles
    whose bbox intersects the S1F11 test/bolivia chips are excluded from training
    (bbox taken from grid_id -> envelope in catalogue.gpkg)

Output (processed/):
  kuro_trn_X.npy (N,3,224,224) float32 linear power
  kuro_trn_W.npy (N,224,224)   uint8  0=land, 1=water (permanent+flood), 255=invalid
  kuro_split.json  split breakdown
Run: WSL native python.
"""
import os, json, csv, glob, sqlite3, struct
import numpy as np
import rasterio as rio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

SUB = "/home/kk/jp_claude/RMSD/data/KuroSiwo_subset"
P = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD/processed"
S1F_S1HAND = "/home/kk/jp_claude/RMSD/data/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S1Hand"
DB = "/home/kk/jp_claude/RMSD/data/KuroSiwo/catalogue.gpkg"

TRAIN_ACTIDS = {118, 130, 147, 174, 205, 267, 273, 275, 277, 279, 321, 324, 411, 417,
                1111002, 1111003, 1111004, 1111005, 1111006, 1111007, 1111008,
                1111009, 1111010, 1111011, 1111013}

# ---- bbox of the S1F11 test/bolivia chips ----
import pandas as pd
chips = pd.read_csv(P + "/s1f11_chips.csv")
test_chips = chips[chips["split"].isin(["test", "bolivia"])]["chip"]
test_boxes = []
for ch in test_chips:
    with rio.open(f"{S1F_S1HAND}/{ch}_S1Hand.tif") as src:
        test_boxes.append(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
print(f"S1F11 held-out chip boxes: {len(test_boxes)}")

def inter(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

# ---- catalogue: grid_id -> bbox(4326) (used to exclude overlaps on the training side) ----
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("SELECT srs_id FROM gpkg_geometry_columns WHERE table_name='catalogue'")
srs_id = cur.fetchone()[0]
cur.execute("SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs_id,))
org, oid = cur.fetchone()
src_crs = CRS.from_string(f"{org}:{oid}")

def envelope(blob):
    if blob is None or blob[:2] != b"GP":
        return None
    flags = blob[3]
    if ((flags >> 1) & 0x07) == 0:
        return None
    end = "<" if flags & 1 else ">"
    v = struct.unpack(f"{end}4d", blob[8:40])
    return v[0], v[2], v[1], v[3]

grid_box = {}
cur.execute("SELECT grid_id, geom FROM catalogue")
for gid, blob in cur:
    g = str(gid).replace("-", "")
    if g in grid_box:
        continue
    e = envelope(blob)
    if e:
        grid_box[g] = transform_bounds(src_crs, "EPSG:4326", *e)
print(f"catalogue grids with bbox: {len(grid_box)}")

# ---- scan and split the subset ----
Xs, Ws = [], []
split = {"train_events": [], "test_events": [], "excluded_leak_tiles": 0,
         "train_tiles": 0, "test_tiles_by_event": {}}
for xf in sorted(glob.glob(os.path.join(SUB, "*_X.npy"))):
    actid = int(os.path.basename(xf).split("_")[0])
    X = np.load(xf)
    MLU = np.load(xf.replace("_X.npy", "_MLU.npy"))
    meta = list(csv.DictReader(open(xf.replace("_X.npy", "_meta.csv"))))
    if actid in TRAIN_ACTIDS:
        keep = np.ones(len(X), bool)
        if actid == 205:   # Sri-Lanka: exclude tiles intersecting the S1F11 held-out area
            for i, m in enumerate(meta):
                bb = grid_box.get(m["grid_id"])
                if bb is not None and any(inter(bb, tb) for tb in test_boxes):
                    keep[i] = False
            split["excluded_leak_tiles"] += int((~keep).sum())
        W = np.where(MLU == 3, 255, ((MLU == 1) | (MLU == 2)).astype(np.uint8))
        Xs.append(X[keep]); Ws.append(W[keep].astype(np.uint8))
        split["train_events"].append(actid)
        split["train_tiles"] += int(keep.sum())
    else:
        split["test_events"].append(actid)
        split["test_tiles_by_event"][str(actid)] = len(X)

X_out = np.concatenate(Xs, 0)
W_out = np.concatenate(Ws, 0)
np.save(P + "/kuro_trn_X.npy", X_out)
np.save(P + "/kuro_trn_W.npy", W_out)
with open(P + "/kuro_split.json", "w") as f:
    json.dump(split, f, indent=2)
print(f"train: {len(split['train_events'])} events, {X_out.shape[0]} tiles "
      f"(leak-excluded {split['excluded_leak_tiles']})")
print(f"held-out: {len(split['test_events'])} events, "
      f"{sum(split['test_tiles_by_event'].values())} tiles: {sorted(split['test_events'])}")
print(f"W stats: water {(W_out==1).mean():.3f} invalid {(W_out==255).mean():.3f}")
