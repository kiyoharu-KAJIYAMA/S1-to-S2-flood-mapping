#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v14 countermeasure chipping: same as build_extra_chips, but
  (1) optionally takes a region allowlist (keep only GFD tiles with dt<=1)
  (2) warps the JRC permanent-water mask <PRE>JRC_<region>.tif onto the S2 grid and sets
      permanent-water pixels of teacher Y to NaN (excludes permanent water from training ->
      focus on the flood signal, limits Kuro dilution)
Acceptance uses the MNDWI water fraction after the JRC mask (= flood water), so chips with
permanent water only are dropped automatically.

Usage: build_extra_chips_v14.py <PREFIX> <OUTNAME> [allowlist_csv]
  e.g. build_extra_chips_v14.py GFD  _gfd1new  processed/gfd_strict_regions.csv
       build_extra_chips_v14.py KURR _kurr2new
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
OUT = sys.argv[2] if len(sys.argv) > 2 else "_gfd1new"
ALLOW = sys.argv[3] if len(sys.argv) > 3 else None

allow = None
if ALLOW and os.path.exists(ALLOW):
    allow = set(pd.read_csv(ALLOW).region.astype(str).tolist())
    print(f"allowlist: {len(allow)} regions", flush=True)

regions = sorted(os.path.basename(p).replace(f"{PREFIX}_", "").replace("_S2.tif", "")
                 for p in glob.glob(f"{GEE}/{PREFIX}_*_S2.tif"))
if allow is not None:
    regions = [r for r in regions if r in allow]
print(f"{PREFIX} regions to chip: {len(regions)}", flush=True)


def load_jrc(reg, s2):
    """Return the permanent-water bool from <PRE>JRC_<reg>.tif warped onto the S2 grid; all False if absent."""
    jp = f"{GEE}/{PREFIX}JRC_{reg}.tif"
    if not os.path.exists(jp):
        return None
    return jp


def scan():
    keep = []
    for reg in regions:
        s2p, s1p = f"{GEE}/{PREFIX}_{reg}_S2.tif", f"{GEE}/{PREFIX}_{reg}_S1.tif"
        if not os.path.exists(s1p):
            print(f"  {reg}: missing S1, skip", flush=True); continue
        jp = f"{GEE}/{PREFIX}JRC_{reg}.tif"
        has_jrc = os.path.exists(jp)
        try:
            with rio.open(s2p) as s2:
                H, W = s2.height, s2.width
                perm_full = None
                if has_jrc:
                    with rio.open(jp) as jsrc, WarpedVRT(
                            jsrc, crs=s2.crs, transform=s2.transform, width=W, height=H,
                            resampling=Resampling.nearest) as jvrt:
                        perm_full = jvrt.read(1).astype(np.uint8)
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
                        perm = (perm_full[r0:r0 + h, c0:c0 + w] > 0) if perm_full is not None \
                            else np.zeros((h, w), bool)
                        mndwi = norm_diff(G, Sw)
                        mndwi[~valid | perm] = np.nan   # exclude permanent water too, measure flood water
                        wf = float(np.nanmean(mndwi > 0)) if np.isfinite(mndwi).any() else 0.0
                        if wf < MIN_WATER:
                            continue
                        keep.append((reg, r0, c0, float(valid.mean()), cf, wf, has_jrc))
                        n += 1
                print(f"  {reg}: {n} chips (jrc={has_jrc})", flush=True)
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
        jp = f"{GEE}/{PREFIX}JRC_{reg}.tif"
        evid = reg.split("_")[0]
        with rio.open(s2p) as s2, rio.open(s1p) as s1src:
            H, W = s2.height, s2.width
            kw = dict(crs=s2.crs, transform=s2.transform, width=W, height=H)
            perm_full = None
            if os.path.exists(jp):
                with rio.open(jp) as jsrc, WarpedVRT(jsrc, resampling=Resampling.nearest, **kw) as jv:
                    perm_full = jv.read(1).astype(np.uint8)
            with WarpedVRT(s1src, resampling=Resampling.bilinear, **kw) as s1:
                for (i, r0, c0, vf, cf, wf, hj) in items:
                    h, w = min(SZ, H - r0), min(SZ, W - c0)
                    win = Window(c0, r0, w, h)
                    b = s2.read([1, 2, 3, 4, 5], window=win).astype(np.float32) / 10000.0
                    B, G, Rd, Nir, Sw = b
                    valid = (G > 0) & (Sw > 0) & (Nir > 0)
                    perm = (perm_full[r0:r0 + h, c0:c0 + w] > 0) if perm_full is not None \
                        else np.zeros((h, w), bool)
                    d = s1.read([1, 2], window=win).astype(np.float32)
                    ok = np.isfinite(d).all(0) & (d[0] != 0)
                    vv = np.where(ok, 10 ** (d[0] / 10.0), np.nan)
                    vh = np.where(ok, 10 ** (d[1] / 10.0), np.nan)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rvi = (4.0 * vh) / (vv + vh)
                    rvi[~np.isfinite(rvi)] = np.nan
                    cloud, _ = cloud_like_mask(Rd, G, B, Nir)
                    bad = cloud | (~valid) | perm       # exclude permanent water from the teacher
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
                                     water_frac=round(wf, 4), jrc=bool(hj), flood=True))
        print(f"  {reg}: written {len(items)}", flush=True)
    X.flush(); Y.flush()
    pd.DataFrame(rows).sort_values("idx").to_csv(f"{P}/{OUT}_meta.csv", index=False)
    return n


if __name__ == "__main__":
    keep = scan()
    print(f"total {PREFIX} chips (v14): {len(keep)}", flush=True)
    n = build(keep)
    print(f"DONE {OUT}: {n} chips", flush=True)
