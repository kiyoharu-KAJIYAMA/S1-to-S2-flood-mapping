#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v16: apply JRC permanent-water exclusion to the teacher Y of base (trn6, 1500) -> _basejrc component.

Warp BASEJRC_<tile>.tif onto each chip geo_tif in trn6_geo_index.csv and set trn6_Y to NaN
at permanent-water pixels. X is unchanged, so _basejrc_X.npy is a symlink to trn6_X.
Output: processed/_basejrc_{X,Y}.npy, _basejrc_meta.csv
Run with: /usr/bin/python3.10 (rasterio).
"""
import os
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

R = "/home/kk/jp_claude/RMSD"
P = R + "/processed"
GEE = "/mnt/d/gdrive/Google Earth Engine"
SZ = 512

trn6_X = P + "/trn6_X_VV_VH_RVI_raw.npy"
trn6_Y = P + "/trn6_Y_MNDWI_NDVI_NDWI_raw.npy"


def tilekey(lon, lat):
    tx = int(np.floor(lon / 10.0) * 10)
    ty = int(np.ceil(lat / 10.0) * 10)
    ew = f"{abs(tx)}{'E' if tx >= 0 else 'W'}"
    ns = f"{abs(ty)}{'N' if ty >= 0 else 'S'}"
    return f"{ew}_{ns}"


def main():
    gi = pd.read_csv(P + "/trn6_geo_index.csv")
    m6 = pd.read_csv(P + "/trn6_matched_samples_good.csv")
    Ysrc = np.load(trn6_Y, mmap_mode="r")
    N = Ysrc.shape[0]
    assert len(gi) == N == len(m6), f"len mismatch gi={len(gi)} Y={N} m6={len(m6)}"

    Yout = np.lib.format.open_memmap(P + "/_basejrc_Y.npy", mode="w+",
                                     dtype=np.float32, shape=(N, 3, SZ, SZ))
    # copy then mask
    for s in range(0, N, 64):
        e = min(s + 64, N)
        Yout[s:e] = Ysrc[s:e]

    miss_tile, applied, perm_px, tot_valid = 0, 0, 0, 0
    for i in range(N):
        r = gi.iloc[i]
        if not bool(r.ok) or not r.geo_tif or not os.path.exists(str(r.geo_tif)):
            continue
        clon = (r.lon_min + r.lon_max) / 2.0
        clat = (r.lat_min + r.lat_max) / 2.0
        jp = f"{GEE}/BASEJRC_{tilekey(clon, clat)}.tif"
        if not os.path.exists(jp):
            miss_tile += 1
            continue
        with rio.open(str(r.geo_tif)) as g:
            H, W = min(g.height, SZ), min(g.width, SZ)
            with rio.open(jp) as jsrc, WarpedVRT(
                    jsrc, crs=g.crs, transform=g.transform, width=g.width, height=g.height,
                    resampling=Resampling.nearest) as jv:
                perm_full = jv.read(1)
        perm = perm_full[:H, :W] > 0
        # set permanent water to NaN among valid (non-NaN) pixels
        yi = Yout[i]
        valid = np.isfinite(yi[0, :H, :W])
        tot_valid += int(valid.sum())
        pm = perm & valid
        perm_px += int(pm.sum())
        yi[:, :H, :W][:, perm] = np.nan
        Yout[i] = yi
        applied += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N} (applied={applied}, miss_tile={miss_tile})", flush=True)
    Yout.flush()

    # X is unchanged -> symlink
    dstx = P + "/_basejrc_X.npy"
    if os.path.lexists(dstx):
        os.remove(dstx)
    os.symlink("trn6_X_VV_VH_RVI_raw.npy", dstx)

    meta = m6[["folder", "date", "flood"]].copy()
    meta["source"] = "base_jrc"
    meta.to_csv(P + "/_basejrc_meta.csv", index=False)

    print(f"\nDONE _basejrc: N={N}, applied JRC to {applied}, missing tile {miss_tile}", flush=True)
    print(f"permanent-water px masked: {perm_px:,} of {tot_valid:,} valid "
          f"({100.0*perm_px/max(tot_valid,1):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
