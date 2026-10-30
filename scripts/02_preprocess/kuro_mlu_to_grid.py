#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9: reproject the Kuro Siwo MLU flood labels (2=flood) onto the GEE export rectangle grids.

MLU placement information:
  - KuroSiwo_subset/<actid>_MLU.npy : (n_tiles, 224, 224) uint8, 0/1/2 (2=flood)
    row i corresponds to the grid_id in row i of <actid>_meta.csv
  - catalogue.gpkg : grid_id (with hyphens) -> geom bbox (EPSG:3857), 2240m square = 224px@10m
  -> the subset grid_id has hyphens removed; strip the catalogue side to match.

Processing: for the grid of each export rectangle KURO_<region>_S2.tif, place that actid's
      MLU==2 tiles in EPSG:3857, reproject, and build the flood mask.
Output: data/KuroSiwo/mlu_on_grid/<region>_flood.tif (uint8: 1=flood, 0=other)
"""
import os, glob, re, sqlite3, struct
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

R = "/home/kk/jp_claude/RMSD"
GEE = "/mnt/d/gdrive/Google Earth Engine"
SUB = R + "/data/KuroSiwo_subset"
OUT = R + "/data/KuroSiwo/mlu_on_grid"
os.makedirs(OUT, exist_ok=True)
MERC = "EPSG:3857"

def env(b):
    f = b[3]; e = "<" if f & 1 else ">"
    v = struct.unpack(f"{e}4d", b[8:40])
    return v[0], v[2], v[1], v[3]   # minx,miny,maxx,maxy

# --- catalogue: grid_id(strip) -> bbox_3857, per actid ---
con = sqlite3.connect(R + "/data/KuroSiwo/catalogue.gpkg"); cur = con.cursor()

regions = sorted(glob.glob(GEE + "/KURO_*_S2.tif"))
by_act = {}
for p in regions:
    reg = os.path.basename(p).replace("KURO_", "").replace("_S2.tif", "")
    act = int(reg.split("_")[0])
    by_act.setdefault(act, []).append((reg, p))

for act, items in by_act.items():
    mlu_p = f"{SUB}/{act}_MLU.npy"
    meta_p = f"{SUB}/{act}_meta.csv"
    if not os.path.exists(mlu_p):
        print(f"act {act}: no subset MLU, skip {len(items)} regions", flush=True)
        continue
    MLU = np.load(mlu_p, mmap_mode="r")
    meta = pd.read_csv(meta_p)
    # grid_id(strip) -> bbox
    cur.execute("SELECT grid_id, geom FROM catalogue WHERE actid=?", (act,))
    gid2bbox = {}
    for gid, blob in cur:
        k = gid.replace("-", "")
        if k not in gid2bbox:
            gid2bbox[k] = env(blob)
    # place only flood tiles (rows whose MLU contains 2)
    flood_rows = [(i, meta.grid_id.iloc[i]) for i in range(len(meta))
                  if meta.pflood.iloc[i] > 0]
    print(f"act {act}: {len(flood_rows)} flood-relevant tiles, {len(items)} regions", flush=True)

    for reg, s2p in items:
        with rio.open(s2p) as ref:
            H, W, tr, crs = ref.height, ref.width, ref.transform, ref.crs
            bl, bb, br, bt = ref.bounds
        # convert ref bounds to 3857 and place only overlapping tiles (speed-up)
        from rasterio.warp import transform_bounds
        rb = transform_bounds(crs, MERC, bl, bb, br, bt)
        acc = np.zeros((H, W), np.uint8)
        placed = 0
        for i, gid in flood_rows:
            bx = gid2bbox.get(gid)
            if bx is None:
                continue
            x0, y0, x1, y1 = bx
            if x1 < rb[0] or rb[2] < x0 or y1 < rb[1] or rb[3] < y0:
                continue
            tile = np.asarray(MLU[i])
            flood = (tile == 2).astype(np.uint8)
            if flood.sum() == 0:
                continue
            src_tr = from_bounds(x0, y0, x1, y1, tile.shape[1], tile.shape[0])
            tmp = np.zeros((H, W), np.uint8)
            reproject(flood, tmp, src_transform=src_tr, src_crs=MERC,
                      dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
            acc = np.maximum(acc, tmp)
            placed += 1
        prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                    crs=crs, transform=tr, nodata=0, compress="lzw")
        with rio.open(f"{OUT}/{reg}_flood.tif", "w", **prof) as o:
            o.write(acc, 1)
        print(f"  {reg}: placed {placed} tiles, flood px {int(acc.sum())}", flush=True)

con.close()
print("DONE", flush=True)
