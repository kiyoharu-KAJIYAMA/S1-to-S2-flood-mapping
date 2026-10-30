#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the "observed Sentinel-2 optical indices" for evaluation axes A / B.

Why
  Of the 3 axes, model-predicted MNDWI and Lee (SAR threshold) could be compared, but
  no axis had the third reference, "what does thresholding the **observed** MNDWI of the
  same period give?". Without it, "is SAR->MSI translation a substitute for observed
  optical?" cannot be answered. The source data of axes A/B are local, so no download is needed.

  Axis A: Sen1Floods11 S2Hand (13 band, L1C TOA)  — already aligned to the same 512x512 grid as X
  Axis B: FloodPlanet <region>/S2/<chip>.tif (10 band, L1C TOA) — needs reprojection to the S1 grid
        Band order is B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (determined from the spectral shape,
        same indices inspect_c2s_fp2.py used for the MNDWI check)

  Axis C (Kuro Siwo) has no S2, so it is fetched by a separate script (aws_s2_kuro_*.py).

Cloud handling
  Build a per-pixel cloud_like with this collection's default heuristic (same rule as
  RMSD_preprocessing.estimate_cloud_fraction) and remove it from valid. Both chip and
  pixel counts before/after masking are kept in the CSV, so the "clouds not removed" case
  can be reproduced downstream.

Output (processed/):
  s1f11_Yobs_MNDWI_NDVI_NDWI.npy (446,3,512,512) float32  NaN=invalid
  s1f11_Yobs_valid.npy           (446,512,512)   uint8
  s1f11_Yobs_chips.csv
  fp_eval_Yobs_MNDWI_NDVI_NDWI.npy (54,3,512,512) float32
  fp_eval_Yobs_valid.npy           (54,512,512)   uint8
  fp_eval_Yobs_chips.csv
Run: /usr/bin/python3 (rasterio)
"""
import argparse
import csv
import os

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling, reproject

R = "/home/kk/jp_claude/RMSD"
P = R + "/processed"
S1F_HL = R + "/data/Sen1Floods11/v1.1/data/flood_events/HandLabeled"
FP = R + "/data/FloodPlanet/FloodPlanet"
SZ = 512
SCALE = 10000.0        # S2 DN -> reflectance

# 0-based band index
S1F_IDX = dict(blue=1, green=2, red=3, nir=7, swir=11)      # 13 band: B1..B12
FP_IDX = dict(blue=0, green=1, red=2, nir=6, swir=8)        # 10 band: B2..B12


def cloud_like(red, green, blue, nir,
               brightness=0.28, whiteness=0.16, ndvi_th=0.35):
    """Same per-pixel rule as RMSD_preprocessing.estimate_cloud_fraction (0-1 reflectance)."""
    vis = (red + green + blue) / 3.0
    rng = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    return (vis > brightness) & (rng < whiteness) & (ndvi < ndvi_th)


def indices(green, swir, nir, red):
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - swir) / (green + swir)
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - nir) / (green + nir)
    return np.stack([mndwi, ndvi, ndwi], 0).astype(np.float32)


def finish(Y, ok):
    """Set non-valid pixels to NaN."""
    Y = np.where(ok[None], Y, np.nan).astype(np.float32)
    Y[~np.isfinite(Y)] = np.nan
    return Y


def build_s1f11():
    chips = list(csv.DictReader(open(P + "/s1f11_chips.csv")))
    N = len(chips)
    print("axis A (Sen1Floods11): %d chips" % N, flush=True)
    Y = np.full((N, 3, SZ, SZ), np.nan, np.float32)
    V = np.zeros((N, SZ, SZ), np.uint8)
    rows = []
    for i, c in enumerate(chips):
        p = "%s/S2Hand/%s_S2Hand.tif" % (S1F_HL, c["chip"])
        if not os.path.exists(p):
            print("  !! missing %s" % p, flush=True)
            rows.append(dict(idx=i, chip=c["chip"], event=c["event"],
                             n_valid=0, cloud_frac=1.0, mndwi_median=""))
            continue
        with rio.open(p) as s:
            a = s.read().astype(np.float32) / SCALE
        b = {k: a[v] for k, v in S1F_IDX.items()}
        fin = np.isfinite(a).all(0) & (a[S1F_IDX["green"]] > 0)
        cl = cloud_like(b["red"], b["green"], b["blue"], b["nir"]) & fin
        ok = fin & ~cl
        Y[i] = finish(indices(b["green"], b["swir"], b["nir"], b["red"]), ok)
        V[i] = ok.astype(np.uint8)
        med = float(np.nanmedian(Y[i, 0])) if ok.any() else float("nan")
        rows.append(dict(idx=i, chip=c["chip"], event=c["event"],
                         n_valid=int(ok.sum()),
                         cloud_frac=round(float(cl.sum()) / max(int(fin.sum()), 1), 4),
                         mndwi_median=round(med, 4) if np.isfinite(med) else ""))
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, N), flush=True)

    np.save(P + "/s1f11_Yobs_MNDWI_NDVI_NDWI.npy", Y)
    np.save(P + "/s1f11_Yobs_valid.npy", V)
    with open(P + "/s1f11_Yobs_chips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hold = {"India", "Somalia", "Spain", "Bolivia"}
    hr = [r for r in rows if r["event"] in hold]
    print("  saved. held-out(139) chips=%d, median cloud_frac=%.3f, "
          "chips with >=90%% valid: %d"
          % (len(hr), float(np.median([r["cloud_frac"] for r in hr])),
             sum(1 for r in hr if r["n_valid"] >= 0.9 * SZ * SZ)), flush=True)


def build_fp():
    chips = list(csv.DictReader(open(P + "/fp_eval_chips.csv")))
    N = len(chips)
    print("axis B (FloodPlanet): %d chips" % N, flush=True)
    Y = np.full((N, 3, SZ, SZ), np.nan, np.float32)
    V = np.zeros((N, SZ, SZ), np.uint8)
    rows = []
    for i, c in enumerate(chips):
        reg, cid = c["region"], c["chip"]
        s1p = "%s/%s/S1/%s.tif" % (FP, reg, cid)
        s2p = "%s/%s/S2/%s.tif" % (FP, reg, cid)
        if not (os.path.exists(s1p) and os.path.exists(s2p)):
            print("  !! missing %s/%s" % (reg, cid), flush=True)
            rows.append(dict(idx=i, chip=cid, region=reg, n_valid=0,
                             cloud_frac=1.0, mndwi_median="",
                             s1_date=c["s1_date"], s2_date=c["s2_date"]))
            continue
        # put on the same S1 grid used when X was built
        with rio.open(s1p) as s1:
            H, W = min(s1.height, SZ), min(s1.width, SZ)
            dst_tr, dst_crs = s1.transform, s1.crs
            dh, dw = s1.height, s1.width
        with rio.open(s2p) as s2:
            nb = s2.count
            A = np.full((nb, dh, dw), np.nan, np.float32)
            for k in range(nb):
                reproject(source=rio.band(s2, k + 1), destination=A[k],
                          dst_transform=dst_tr, dst_crs=dst_crs,
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
        A = A[:, :H, :W] / SCALE
        idx = FP_IDX if nb == 10 else dict(blue=1, green=2, red=3, nir=7, swir=11)
        b = {k: A[v] for k, v in idx.items()}
        fin = np.isfinite(A).all(0) & (A[idx["green"]] > 0)
        cl = cloud_like(b["red"], b["green"], b["blue"], b["nir"]) & fin
        ok = fin & ~cl
        yy = finish(indices(b["green"], b["swir"], b["nir"], b["red"]), ok)
        Y[i, :, :H, :W] = yy
        V[i, :H, :W] = ok.astype(np.uint8)
        med = float(np.nanmedian(yy[0])) if ok.any() else float("nan")
        rows.append(dict(idx=i, chip=cid, region=reg, n_valid=int(ok.sum()),
                         cloud_frac=round(float(cl.sum()) / max(int(fin.sum()), 1), 4),
                         mndwi_median=round(med, 4) if np.isfinite(med) else "",
                         s1_date=c["s1_date"], s2_date=c["s2_date"]))

    np.save(P + "/fp_eval_Yobs_MNDWI_NDVI_NDWI.npy", Y)
    np.save(P + "/fp_eval_Yobs_valid.npy", V)
    with open(P + "/fp_eval_Yobs_chips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hw = {r["chip"]: float(r["h"]) * float(r["w"]) for r in chips}
    print("  saved. median cloud_frac=%.3f, chips with >=90%% valid: %d/%d"
          % (float(np.median([r["cloud_frac"] for r in rows])),
             sum(1 for r in rows
                 if r["n_valid"] >= 0.9 * hw.get(r["chip"], SZ * SZ)), N), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["s1f11", "fp"])
    a = ap.parse_args()
    if "s1f11" in a.axis:
        build_s1f11()
    if "fp" in a.axis:
        build_fp()
