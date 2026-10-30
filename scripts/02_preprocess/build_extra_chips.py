#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chip the export tiffs (Drive) of measures #1/#2 into 512px chips (mask-free).
Same convention as build_trn9_chips, but accepts by MNDWI water fraction instead of the
Kuro flood mask (the trainer does not read F, so no flood mask is needed).

Usage: build_extra_chips.py <PREFIX> <OUTNAME>
  e.g. build_extra_chips.py GFD  _gfdnew
       build_extra_chips.py KURR _kurrnew

Input: /mnt/d/gdrive/Google Earth Engine/<PREFIX>_<region>_S1.tif, _S2.tif
Acceptance: valid>=80%, cloud<=10%, MNDWI>0 fraction>=0.5%, cover>=35% (same thresholds as build_trn9_chips)
Output: processed/<OUTNAME>_{X,Y}.npy, processed/<OUTNAME>_meta.csv
"""
import os, sys, glob
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
P = R + "/processed"
SZ, MIN_VALID, MAX_CLOUD, MIN_WATER, MIN_COVER = 512, 0.80, 0.10, 0.005, 0.35

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "GFD"
OUT = sys.argv[2] if len(sys.argv) > 2 else "_gfdnew"

regions = sorted(os.path.basename(p).replace(f"{PREFIX}_", "").replace("_S2.tif", "")
                 for p in glob.glob(f"{GEE}/{PREFIX}_*_S2.tif"))
print(f"{PREFIX} regions with S2: {len(regions)}", flush=True)


def scan():
    keep = []
    for reg in regions:
        s2p, s1p = f"{GEE}/{PREFIX}_{reg}_S2.tif", f"{GEE}/{PREFIX}_{reg}_S1.tif"
        if not os.path.exists(s1p):
            print(f"  {reg}: missing S1, skip", flush=True); continue
        try:
            with rio.open(s2p) as s2:
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
                        if wf < MIN_WATER:
                            continue
                        keep.append((reg, r0, c0, float(valid.mean()), cf, wf))
                        n += 1
                print(f"  {reg}: {n} chips", flush=True)
        except Exception as ex:
            print(f"  {reg}: READ FAIL {str(ex)[:60]}", flush=True)
    return keep


def build(keep):
    n = len(keep)
    if n == 0:
        print("no chips; abort", flush=True); return 0
    X = np.lib.format.open_memmap(f"{P}/{OUT}_X.npy", mode="w+", dtype=np.float32, shape=(n, 3, SZ, SZ))
    Y = np.lib.format.open_memmap(f"{P}/{OUT}_Y.npy", mode="w+", dtype=np.float32, shape=(n, 3, SZ, SZ))
    by = {}
    for i, k in enumerate(keep):
        by.setdefault(k[0], []).append((i,) + k[1:])
    rows = []
    for reg, items in by.items():
        s2p, s1p = f"{GEE}/{PREFIX}_{reg}_S2.tif", f"{GEE}/{PREFIX}_{reg}_S1.tif"
        evid = reg.split("_")[0]
        with rio.open(s2p) as s2, rio.open(s1p) as s1src:
            H, W = s2.height, s2.width
            kw = dict(crs=s2.crs, transform=s2.transform, width=W, height=H)
            with WarpedVRT(s1src, resampling=Resampling.bilinear, **kw) as s1:
                for (i, r0, c0, vf, cf, wf) in items:
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
                    xf[:, :h, :w] = np.stack([vv, vh, rvi], 0)
                    yf[:, :h, :w] = np.stack([mndwi, ndvi, ndwi], 0)
                    X[i], Y[i] = xf, yf
                    rows.append(dict(idx=i, folder=f"{OUT.strip('_')}_{reg}", date=f"ev{evid}",
                                     region=reg, event=evid, row=r0, col=c0,
                                     cover=round(h * w / float(SZ * SZ), 3),
                                     valid_frac=round(vf, 3), cloud_fraction=round(cf, 4),
                                     water_frac=round(wf, 4), flood=True))
        print(f"  {reg}: written {len(items)}", flush=True)
    X.flush(); Y.flush()
    pd.DataFrame(rows).sort_values("idx").to_csv(f"{P}/{OUT}_meta.csv", index=False)
    return n


if __name__ == "__main__":
    keep = scan()
    print(f"total {PREFIX} chips: {len(keep)}", flush=True)
    n = build(keep)
    print(f"DONE {OUT}: {n} chips", flush=True)
