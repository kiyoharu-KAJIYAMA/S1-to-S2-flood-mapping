#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosaic the CEMS flood masks bundled with STURM (128px tiles) per AOI and reproject
them onto the export grid fetched from GEE (10m, UTM).

STURM tile values are 0/1/2 with undocumented meaning. Analysis of the S1-side masks gave
  0 = land (70%), 1 = water (lowest VV, 19%), 2 = minority class (4%)
so this script **verifies the class meaning empirically with our own S2 (MNDWI)**.

Output: data/STURMFlood/masks_on_grid/<AOI>_mask.tif  (uint8: 0=land, 1,2 kept as is)
        + processed/sturm_mask_class_check.csv (per-class MNDWI statistics)
"""
import os, glob, re
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RS

R = "/home/kk/jp_claude/RMSD"
D = R + "/data/STURMFlood/raw/Dataset"
GEE = "/mnt/d/gdrive/Google Earth Engine"
OUT = R + "/data/STURMFlood/masks_on_grid"
os.makedirs(OUT, exist_ok=True)

aois = sorted({os.path.basename(p).replace("_S2.tif", "")
               for p in glob.glob(GEE + "/*_S2.tif")})
print("AOIs:", len(aois))

rows = []
for a in aois:
    s2p = f"{GEE}/{a}_S2.tif"
    with rio.open(s2p) as ref:
        prof = ref.profile.copy()
        H, W, tr, crs = ref.height, ref.width, ref.transform, ref.crs

    dst = np.zeros((H, W), np.uint8)
    n_src = 0
    for sub in ("Sentinel1/Floodmaps", "Sentinel2/Floodmaps"):
        fs = sorted(glob.glob(f"{D}/{sub}/{a}_*.tif"))
        if not fs:
            continue
        srcs = [rio.open(f) for f in fs]
        try:
            mos, mos_tr = merge(srcs, nodata=0)
        finally:
            for s in srcs:
                s.close()
        tmp = np.zeros((H, W), np.uint8)
        reproject(mos[0], tmp, src_transform=mos_tr, src_crs=srcs[0].crs if False else rio.open(fs[0]).crs,
                  dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest,
                  src_nodata=0, dst_nodata=0)
        dst = np.maximum(dst, tmp)          # union of the S1-side and S2-side masks
        n_src += len(fs)

    prof.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    op = f"{OUT}/{a}_mask.tif"
    with rio.open(op, "w", **prof) as o:
        o.write(dst, 1)

    # --- verify class meaning with our own S2 MNDWI (evaluated on a downsampled grid) ---
    with rio.open(s2p) as s2:
        f = max(1, int(max(s2.shape) / 1200))
        h, w = max(1, H // f), max(1, W // f)
        b = s2.read([2, 5], out_shape=(2, h, w), resampling=RS.average).astype(np.float32)
    mk = np.zeros((h, w), np.uint8)
    reproject(dst, mk, src_transform=tr, src_crs=crs,
              dst_transform=tr * rio.Affine.scale(W / w, H / h), dst_crs=crs,
              resampling=Resampling.nearest)
    G, SW = b[0], b[1]
    ok = (G > 0) & (SW > 0)
    mndwi = np.where(ok, (G - SW) / np.clip(G + SW, 1e-6, None), np.nan)
    st = {}
    for v in (0, 1, 2):
        sel = (mk == v) & ok
        st[v] = (float(sel.mean()), float(np.nanmedian(mndwi[sel])) if sel.sum() > 50 else np.nan)
    rows.append(dict(aoi=a, n_src_tiles=n_src,
                     frac0=round(st[0][0], 3), mndwi0=round(st[0][1], 3),
                     frac1=round(st[1][0], 3), mndwi1=round(st[1][1], 3),
                     frac2=round(st[2][0], 3), mndwi2=round(st[2][1], 3)))
    print(f"[{a}] tiles={n_src}  class1 {st[1][0]*100:.1f}% MNDWI {st[1][1]:.2f} | "
          f"class2 {st[2][0]*100:.1f}% MNDWI {st[2][1]:.2f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(R + "/processed/sturm_mask_class_check.csv", index=False)
print()
print(df.to_string(index=False))
print("\nVerdict: the class with high MNDWI (>0) is water. Check the table above.")
