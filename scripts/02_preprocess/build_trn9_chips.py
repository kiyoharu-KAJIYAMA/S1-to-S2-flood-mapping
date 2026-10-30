#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the v9 Kuro-S2 chips (no concat; only writes _trn9new_{X,Y,F}.npy).

Input: /mnt/d/gdrive/Google Earth Engine/KURO_<region>_S1.tif, _S2.tif
       data/KuroSiwo/mlu_on_grid/<region>_flood.tif  (1=flood)
Same convention as build_trn8 (X=linear power, Y=optical indices, cloud NaN, NaN padding of partial chips).
Acceptance: valid>=80%, cloud<=10%, (MNDWI>0 fraction>=0.5% or Kuro flood fraction>=0.5%), cover>=35%.
Output: processed/_trn9new_{X,Y,F}.npy, processed/_trn9new_meta.csv
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.enums import Resampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RMSD_preprocessing_v2 import cloud_like_mask, norm_diff  # noqa: E402

R = "/home/kk/jp_claude/RMSD"
GEE = "/mnt/d/gdrive/Google Earth Engine"
MSK = R + "/data/KuroSiwo/mlu_on_grid"
P = R + "/processed"
SZ, MIN_VALID, MAX_CLOUD, MIN_WATER, MIN_COVER = 512, 0.80, 0.10, 0.005, 0.35

regions = sorted(os.path.basename(p).replace("KURO_", "").replace("_S2.tif", "")
                 for p in glob.glob(GEE + "/KURO_*_S2.tif"))
print("regions:", len(regions), flush=True)


def scan():
    keep = []
    for reg in regions:
        s2p, s1p, mp = f"{GEE}/KURO_{reg}_S2.tif", f"{GEE}/KURO_{reg}_S1.tif", f"{MSK}/{reg}_flood.tif"
        if not (os.path.exists(s1p) and os.path.exists(mp)):
            print(f"  {reg}: missing S1/mask, skip", flush=True); continue
        with rio.open(s2p) as s2, rio.open(mp) as mk:
            H, W = s2.height, s2.width
            n = 0
            for r0 in range(0, H, SZ):
                for c0 in range(0, W, SZ):
                    h, w = min(SZ, H - r0), min(SZ, W - c0)
                    if (h * w) / float(SZ * SZ) < MIN_COVER:
                        continue
                    win = Window(c0, r0, w, h)
                    b = s2.read([1, 2, 3, 4, 5], window=win).astype(np.float32) / 10000.0
                    B, G, Rd, Nir, Sw = b
                    valid = (G > 0) & (Sw > 0) & (Nir > 0)
                    if valid.mean() < MIN_VALID:
                        continue
                    cloud, vpx = cloud_like_mask(Rd, G, B, Nir)
                    cf = float(cloud.sum() / max(vpx.sum(), 1))
                    if cf > MAX_CLOUD:
                        continue
                    mndwi = norm_diff(G, Sw); mndwi[~valid] = np.nan
                    wf = float(np.nanmean(mndwi > 0))
                    ff = float((mk.read(1, window=win) == 1).mean())
                    if wf < MIN_WATER and ff < MIN_WATER:
                        continue
                    keep.append((reg, r0, c0, valid.mean(), cf, wf, ff))
                    n += 1
            print(f"  {reg}: {n} chips", flush=True)
    return keep


def build(keep):
    n = len(keep)
    X = np.lib.format.open_memmap(P + "/_trn9new_X.npy", mode="w+", dtype=np.float32, shape=(n, 3, SZ, SZ))
    Y = np.lib.format.open_memmap(P + "/_trn9new_Y.npy", mode="w+", dtype=np.float32, shape=(n, 3, SZ, SZ))
    F = np.lib.format.open_memmap(P + "/_trn9new_F.npy", mode="w+", dtype=np.uint8, shape=(n, SZ, SZ))
    by = {}
    for i, k in enumerate(keep):
        by.setdefault(k[0], []).append((i,) + k[1:])
    rows = []
    for reg, items in by.items():
        s2p, s1p, mp = f"{GEE}/KURO_{reg}_S2.tif", f"{GEE}/KURO_{reg}_S1.tif", f"{MSK}/{reg}_flood.tif"
        act = int(reg.split("_")[0])
        with rio.open(s2p) as s2, rio.open(mp) as mk, rio.open(s1p) as s1src:
            H, W = s2.height, s2.width
            kw = dict(crs=s2.crs, transform=s2.transform, width=W, height=H)
            with WarpedVRT(s1src, resampling=Resampling.bilinear, **kw) as s1:
                for (i, r0, c0, vf, cf, wf, ff) in items:
                    h, w = min(SZ, H - r0), min(SZ, W - c0)
                    win = Window(c0, r0, w, h)
                    b = s2.read([1, 2, 3, 4, 5], window=win).astype(np.float32) / 10000.0
                    B, G, Rd, Nir, Sw = b
                    valid = (G > 0) & (Sw > 0) & (Nir > 0)
                    d = s1.read([1, 2], window=win).astype(np.float32)     # dB
                    ok = np.isfinite(d).all(0) & (d[0] != 0)
                    vv = np.where(ok, 10 ** (d[0] / 10.0), np.nan)
                    vh = np.where(ok, 10 ** (d[1] / 10.0), np.nan)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rvi = (4.0 * vh) / (vv + vh)
                    rvi[~np.isfinite(rvi)] = np.nan
                    cloud, _ = cloud_like_mask(Rd, G, B, Nir)
                    bad = cloud | (~valid)
                    mndwi, ndvi, ndwi = norm_diff(G, Sw), norm_diff(Nir, Rd), norm_diff(G, Nir)
                    for arr in (mndwi, ndvi, ndwi):
                        arr[bad] = np.nan
                    xf = np.full((3, SZ, SZ), np.nan, np.float32)
                    yf = np.full((3, SZ, SZ), np.nan, np.float32)
                    ff_arr = np.zeros((SZ, SZ), np.uint8)
                    xf[:, :h, :w] = np.stack([vv, vh, rvi], 0)
                    yf[:, :h, :w] = np.stack([mndwi, ndvi, ndwi], 0)
                    ff_arr[:h, :w] = (mk.read(1, window=win) == 1).astype(np.uint8)
                    X[i], Y[i], F[i] = xf, yf, ff_arr
                    rows.append(dict(idx=i, folder=f"kuro_{reg}", date=f"act{act}", region=reg,
                                     actid=act, row=r0, col=c0,
                                     cover=round(h * w / float(SZ * SZ), 3),
                                     valid_frac=round(float(vf), 3), cloud_fraction=round(cf, 4),
                                     water_frac=round(wf, 4), kuro_flood_frac=round(ff, 4),
                                     flood=True))
        print(f"  {reg}: written {len(items)}", flush=True)
    X.flush(); Y.flush(); F.flush()
    pd.DataFrame(rows).sort_values("idx").to_csv(P + "/_trn9new_meta.csv", index=False)
    return n


if __name__ == "__main__":
    keep = scan()
    print("total chips:", len(keep), flush=True)
    build(keep)
    print("DONE _trn9new", flush=True)
