#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the v5 combined training data (trn5 prefix, fast2-compatible format).

trn5 = trn4 (2403: SEN12FLOOD trn2 + S1F11 train/valid + C2S-MS)
     + FloodPlanet (chips with an S1∩S2 pair; those whose bbox intersects S1F11 test/bolivia are excluded)

FloodPlanet spec (confirmed in the 2026-07-20 feasibility study):
  S1 = [VV, VH] dB (float32, ~10m, e.g. 311x311)   -> linearized, RVI added
  S2 = 10 bands [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] reflectance x1e4
     -> Y=[MNDWI(B3,B11), NDVI(B8,B4), NDWI(B3,B8)]; cloud/nodata set to NaN
  Chips are smaller than 512, so they are placed on a 512x512 canvas with NaN padding
  (fast2's masked loss ignores invalid pixels; even with crop=256 every crop contains valid pixels)

tst5 = symlink to tst4 (the monitoring set stays S1F11 test+bolivia, unchanged)
Run: WSL native python (rasterio).
"""
import os, sys
import glob as g
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_bounds, reproject, Resampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RMSD_preprocessing_v2 import cloud_like_mask, norm_diff

P = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD/processed"
FP = "/home/kk/jp_claude/RMSD/data/FloodPlanet/FloodPlanet"
HL = "/home/kk/jp_claude/RMSD/data/Sen1Floods11/v1.1/data/flood_events/HandLabeled"
CANVAS = 512
# 10-band S2 indices: [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
IB2, IB3, IB4, IB8, IB11 = 0, 1, 2, 6, 8

def b4326(path):
    with rio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)

def inter(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

# ---------- bbox of S1F11 held-out (test/bolivia) — leak prevention ----------
df_s1f = pd.read_csv(P + "/s1f11_chips.csv")
held = df_s1f[df_s1f["split"].isin(["test", "bolivia"])]
test_boxes = [b4326(f"{HL}/S1Hand/{c}_S1Hand.tif") for c in held["chip"]]
print(f"S1F11 held-out boxes: {len(test_boxes)}", flush=True)

# ---------- FloodPlanet chip processing ----------
X_parts, Y_parts, rows = [], [], []
n_excl = n_skip = 0
events = sorted([e for e in os.listdir(FP) if os.path.isdir(os.path.join(FP, e))])
for ev in events:
    s1s = {os.path.basename(p) for p in g.glob(os.path.join(FP, ev, "S1", "*.tif"))}
    s2s = {os.path.basename(p) for p in g.glob(os.path.join(FP, ev, "S2", "*.tif"))}
    for chip in sorted(s1s & s2s):
        p1 = os.path.join(FP, ev, "S1", chip)
        p2 = os.path.join(FP, ev, "S2", chip)
        bb = b4326(p1)
        if any(inter(bb, tb) for tb in test_boxes):
            n_excl += 1
            continue
        with rio.open(p1) as src:
            s1 = src.read().astype(np.float64)     # (2,H,W) dB
            crs1, tr1, sh1 = src.crs, src.transform, (src.height, src.width)
        H, W = sh1
        if H > CANVAS or W > CANVAS or s1.shape[0] != 2:
            n_skip += 1
            continue
        with rio.open(p2) as src:
            same = (src.crs == crs1 and src.transform == tr1
                    and (src.height, src.width) == sh1)
            if same:
                s2 = src.read().astype(np.float32)
            else:
                s2 = np.full((src.count, H, W), np.nan, np.float32)
                for b in range(src.count):
                    reproject(rio.band(src, b + 1), s2[b],
                              dst_transform=tr1, dst_crs=crs1,
                              dst_nodata=np.nan, resampling=Resampling.bilinear)
        if s2.shape[0] < 10:
            n_skip += 1
            continue
        # X: dB -> linear, RVI
        s1[~np.isfinite(s1)] = np.nan
        vv = np.power(10.0, s1[0] / 10.0)
        vh = np.power(10.0, s1[1] / 10.0)
        with np.errstate(over="ignore", invalid="ignore"):
            rvi = (4.0 * vh) / (vv + vh)
        rvi[~np.isfinite(rvi)] = np.nan
        # Y: indices + cloud/nodata NaN
        nod = (s2[IB3] == 0) | (s2[IB11] == 0) | (s2[IB8] == 0) | (s2[IB4] == 0) | ~np.isfinite(s2[IB3])
        R, G, B, Nir, Swir = s2[IB4]/1e4, s2[IB3]/1e4, s2[IB2]/1e4, s2[IB8]/1e4, s2[IB11]/1e4
        cloud, _ = cloud_like_mask(R, G, B, Nir)
        bad = cloud | nod
        Y = np.stack([norm_diff(G, Swir), norm_diff(Nir, R), norm_diff(G, Nir)], 0)
        Y[:, bad] = np.nan
        # NaN-pad onto the 512 canvas (top-left placement)
        Xc = np.full((3, CANVAS, CANVAS), np.nan, np.float32)
        Yc = np.full((3, CANVAS, CANVAS), np.nan, np.float32)
        Xc[0, :H, :W], Xc[1, :H, :W], Xc[2, :H, :W] = vv, vh, rvi
        Yc[:, :H, :W] = Y
        X_parts.append(Xc); Y_parts.append(Yc)
        rows.append(dict(folder=f"fp_{ev}", date=os.path.splitext(chip)[0], flood=True))
    print(f"  {ev}: cumulative kept={len(rows)} excluded={n_excl} skipped={n_skip}", flush=True)

Nfp = len(X_parts)
print(f"FloodPlanet: kept {Nfp}, excluded(held-out overlap) {n_excl}, skipped {n_skip}", flush=True)
X_fp = np.stack(X_parts).astype(np.float32); del X_parts
Y_fp = np.stack(Y_parts).astype(np.float32); del Y_parts

# pre-check: how many chips pass fast2's degenerate filter (valid>=0.2)
xv = np.array([float(np.mean((X_fp[i, 0] > 0) & np.isfinite(X_fp[i, 0]))) for i in range(Nfp)])
yv = np.array([float(np.mean(np.isfinite(Y_fp[i, 0]))) for i in range(Nfp)])
print(f"sanity: FP x_valid mean={xv.mean():.3f} | y_valid mean={yv.mean():.3f} | "
      f"survive fast2 filter (>=0.2 both): {int(((xv>=0.2)&(yv>=0.2)).sum())}/{Nfp}", flush=True)

# ---------- concatenate with trn4 (memmap streaming to save memory) ----------
df4 = pd.read_csv(P + "/trn4_matched_samples_good.csv")
X4 = np.load(P + "/trn4_X_VV_VH_RVI_raw.npy", mmap_mode="r")
Y4 = np.load(P + "/trn4_Y_MNDWI_NDVI_NDWI_raw.npy", mmap_mode="r")
N4 = X4.shape[0]
assert len(df4) == N4
N5 = N4 + Nfp
print(f"concat: trn4 {N4} + fp {Nfp} = trn5 {N5}", flush=True)

for src_mm, add, name in [(X4, X_fp, "X_VV_VH_RVI_raw"), (Y4, Y_fp, "Y_MNDWI_NDVI_NDWI_raw")]:
    out = np.lib.format.open_memmap(P + f"/trn5_{name}.npy", mode="w+",
                                    dtype=np.float32, shape=(N5, 3, CANVAS, CANVAS))
    for s in range(0, N4, 100):
        e = min(s + 100, N4)
        out[s:e] = src_mm[s:e]
    out[N4:] = add
    out.flush(); del out
    print(f"saved trn5_{name}.npy", flush=True)

df5 = pd.concat([df4[["folder", "date", "flood"]], pd.DataFrame(rows)], ignore_index=True)
assert len(df5) == N5
df5.to_csv(P + "/trn5_matched_samples_good.csv", index=False)
print(f"saved trn5_matched_samples_good.csv ({len(df5)} rows, "
      f"{df5['folder'].map(os.path.basename).nunique()} scenes)", flush=True)

# ---------- tst5 = symlink to tst4 (monitoring set unchanged) ----------
for name in ["X_VV_VH_RVI_raw", "Y_MNDWI_NDVI_NDWI_raw"]:
    dst = P + f"/tst5_{name}.npy"
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(f"tst4_{name}.npy", dst)
    print(f"symlink tst5_{name}.npy -> tst4_{name}.npy", flush=True)
print("DONE", flush=True)
