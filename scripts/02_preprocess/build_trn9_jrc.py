#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TODO #12: retrofit JRC permanent-water removal onto the teacher Y of _trn9new (05_KuroSiwo_S2).

KUROJRC_<region>.tif was exported by gee_export_kuro_jrc.py with the same region/crs/scale as the
original export. "Should be identical" is not trusted: grid agreement is verified mechanically before applying.

  check 1: the cover (= h*w/512^2) of edge chips computed from each region's tif
           matches the cover column of _trn9new_meta.csv
  check 2: consistency of max(row)+512 <= height + SZ, max(col)+512 <= width + SZ
  -> any region failing either check is excluded from application and named (never proceed silently)

Output: processed/_trn9jrc_{X,Y}.npy, _trn9jrc_meta.csv (with jrc column and per-chip bbox)
        X is unchanged, so it is a symlink.
Run: /usr/bin/python3.10 (rasterio)
Usage: --check          ... verify only and exit
       --drop_unmasked  ... drop chips that could not be masked from the output
                            (to make jrc all True in 05; default keeps them with jrc=False)
       (no flag)        ... verify then apply, keeping chips that could not be masked
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_bounds
from rasterio.windows import Window

R = "/home/kk/jp_claude/RMSD"
P = R + "/processed"
GEE = "/mnt/d/gdrive/Google Earth Engine"
SZ = 512
SRC, OUT = "_trn9new", "_trn9jrc"


def grid_of(region):
    p = f"{GEE}/KUROJRC_{region}.tif"
    if not os.path.exists(p):
        return None
    with rio.open(p) as s:
        return dict(path=p, w=s.width, h=s.height, crs=s.crs, tr=s.transform)


def check(meta):
    """Verify grid agreement per region and return (ok_regions, report)."""
    rows = []
    for reg, g in meta.groupby("region"):
        gi = grid_of(reg)
        if gi is None:
            rows.append(dict(region=reg, n=len(g), status="MISSING_TIF",
                             detail="KUROJRC tif missing"))
            continue
        bad_cover, worst = 0, 0.0
        for _, r in g.iterrows():
            r0, c0 = int(r["row"]), int(r["col"])
            h = min(SZ, gi["h"] - r0)
            w = min(SZ, gi["w"] - c0)
            if h <= 0 or w <= 0:
                bad_cover += 1
                worst = 1.0
                continue
            cov = round(h * w / float(SZ * SZ), 3)
            d = abs(cov - float(r["cover"]))
            worst = max(worst, d)
            if d > 0.002:                     # cover is recorded to 3 decimal places
                bad_cover += 1
        status = "OK" if bad_cover == 0 else "GRID_MISMATCH"
        rows.append(dict(region=reg, n=len(g), status=status,
                         detail=f"w={gi['w']} h={gi['h']} bad_cover={bad_cover} "
                                f"max|dcover|={worst:.4f}"))
    rep = pd.DataFrame(rows).sort_values(["status", "region"])
    ok = set(rep[rep.status == "OK"].region)
    return ok, rep


def main():
    meta = pd.read_csv(f"{P}/{SRC}_meta.csv")
    Ysrc = np.load(f"{P}/{SRC}_Y.npy", mmap_mode="r")
    assert len(meta) == Ysrc.shape[0], f"{len(meta)} != {Ysrc.shape[0]}"

    ok, rep = check(meta)
    print(rep.to_string(index=False), flush=True)
    n_ok = int(meta.region.isin(ok).sum())
    print(f"\ngrid OK regions: {len(ok)}/{meta.region.nunique()}  "
          f"chips {n_ok}/{len(meta)}", flush=True)
    rep.to_csv(f"{P}/{OUT}_gridcheck.csv", index=False)
    if len(ok) < meta.region.nunique():
        print("!! some regions have grid mismatch/missing tif. Check the table above.", flush=True)
    if "--check" in sys.argv:
        return
    if not ok:
        sys.exit("No region can be applied. Aborting.")

    # ---------------- apply ----------------
    drop = "--drop_unmasked" in sys.argv
    keep_rows = meta.index[meta.region.isin(ok)] if drop else meta.index
    if drop and len(keep_rows) < len(meta):
        print(f"\n--drop_unmasked: dropping {len(meta) - len(keep_rows)} chips, "
              f"writing {len(keep_rows)} chips", flush=True)

    N = Ysrc.shape[0]
    Yout = np.lib.format.open_memmap(f"{P}/{OUT}_Y.npy", mode="w+",
                                     dtype=np.float32, shape=Ysrc.shape)
    for s in range(0, N, 64):
        e = min(s + 64, N)
        Yout[s:e] = Ysrc[s:e]

    bbox = [None] * N
    applied, perm_px, tot_valid, skipped = 0, 0, 0, 0
    cache = {}
    for i, r in meta.iterrows():
        reg = r["region"]
        if reg not in ok:
            skipped += 1
            continue
        if reg not in cache:
            cache[reg] = grid_of(reg)
        gi = cache[reg]
        r0, c0 = int(r["row"]), int(r["col"])
        h, w = min(SZ, gi["h"] - r0), min(SZ, gi["w"] - c0)
        win = Window(c0, r0, w, h)
        with rio.open(gi["path"]) as s:
            pm = s.read(1, window=win) > 0
        yi = Yout[i]
        valid = np.isfinite(yi[0, :h, :w])
        tot_valid += int(valid.sum())
        perm_px += int((pm & valid).sum())
        yi[:, :h, :w][:, pm] = np.nan
        Yout[i] = yi
        # per-chip bbox (input for upgrading the 05 bbox_level from region -> chip)
        b = rio.windows.bounds(win, gi["tr"])
        lo, la, hi, ha = transform_bounds(gi["crs"], "EPSG:4326", *b, densify_pts=21)
        bbox[i] = (round(lo, 6), round(la, 6), round(hi, 6), round(ha, 6), str(gi["crs"]))
        applied += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N} applied={applied}", flush=True)
    Yout.flush()

    m = meta.copy()
    m["jrc"] = m.region.isin(ok)
    for k, col in enumerate(["lon_min", "lat_min", "lon_max", "lat_max", "crs_native"]):
        m[col] = [b[k] if b else "" for b in bbox]

    if drop and len(keep_rows) < N:
        # rows are dropped, so X must be materialised too (a symlink would have the wrong row count)
        Xsrc = np.load(f"{P}/{SRC}_X.npy", mmap_mode="r")
        ki = np.asarray(keep_rows, dtype=np.int64)
        Xo = np.lib.format.open_memmap(f"{P}/{OUT}_X.npy", mode="w+", dtype=np.float32,
                                       shape=(len(ki),) + Xsrc.shape[1:])
        Yo = np.lib.format.open_memmap(f"{P}/{OUT}_Y2.npy", mode="w+", dtype=np.float32,
                                       shape=(len(ki),) + Ysrc.shape[1:])
        for s in range(0, len(ki), 64):
            e = min(s + 64, len(ki))
            Xo[s:e] = Xsrc[ki[s:e]]
            Yo[s:e] = Yout[ki[s:e]]
        Xo.flush(); Yo.flush()
        del Xo, Yo, Yout
        os.replace(f"{P}/{OUT}_Y2.npy", f"{P}/{OUT}_Y.npy")
        m = m.loc[keep_rows].reset_index(drop=True)
        m["idx"] = np.arange(len(m))
    else:
        dstx = f"{P}/{OUT}_X.npy"
        if os.path.lexists(dstx):
            os.remove(dstx)
        os.symlink(f"{SRC}_X.npy", dstx)

    m.to_csv(f"{P}/{OUT}_meta.csv", index=False)

    print(f"\nDONE {OUT}: applied {applied}/{N} chips (skipped {skipped})", flush=True)
    print(f"permanent-water px masked: {perm_px:,} / {tot_valid:,} valid "
          f"({100.0*perm_px/max(tot_valid,1):.2f}%)", flush=True)
    if perm_px == 0:
        print("!! not a single pixel was masked. Suspect the occurrence threshold or the grid.", flush=True)


if __name__ == "__main__":
    main()
