#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the v4 combined training data (trn4/tst4 prefixes, fast2-compatible format).

trn4 = SEN12FLOOD trn2 (913) + Sen1Floods11 train/valid (341) + C2S-MS (test-area intersections excluded)
tst4 = Sen1Floods11 test + bolivia (105)   not used for training (monitoring in fast2)

X: [VV, VH, RVI] linear power / Y: [MNDWI, NDVI, NDWI] (cloud and nodata are NaN)
Leak prevention: C2S chips whose bbox intersects the S1F11 test/bolivia chips are excluded.
C2S events identical to S1F11 ones share the folder ID (they move together in scene splits).
Run: WSL native python (rasterio).
"""
import os, csv
import glob as g
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_bounds, reproject, Resampling

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RMSD_preprocessing_v2 import cloud_like_mask, norm_diff

P = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD/processed"
S1F = "/home/kk/jp_claude/RMSD/data/Sen1Floods11/v1.1"
HL = S1F + "/data/flood_events/HandLabeled"
C2S = "/home/kk/jp_claude/RMSD/data/C2SMSFloods/chips"

# C2S UUID -> event name (events identical to S1F11 ones get the same name)
C2S_EVENT = {
    "0c7daa97": "s1f11_Somalia", "3e71513a": "s1f11_Mekong", "4567c2a2": "c2s_harvey",
    "55d43c77": "c2s_nebraska", "5ef4fb15": "s1f11_Ghana", "749aca10": "c2s_australia",
    "89b504aa": "s1f11_Bolivia", "8fcb3e59": "c2s_congo", "95eff095": "s1f11_Sri-Lanka",
    "975ecc1f": "s1f11_Nigeria", "befccce4": "s1f11_Paraguay", "c19f5f27": "s1f11_Pakistan",
    "d89b6d2c": "s1f11_India", "dc70be79": "c2s_kenya", "e7d1917e": "c2s_slovakia",
    "e7e6e693": "s1f11_Spain", "e92c811f": "c2s_niger", "f59da1a3": "s1f11_USA",
}

def b4326(path):
    with rio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)

def inter(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

# ---------- Sen1Floods11: X from the existing npy, Y from S2Hand ----------
df = pd.read_csv(P + "/s1f11_chips.csv")
X_all = np.load(P + "/s1f11_X_VV_VH_RVI_raw.npy")   # linear power (446,3,512,512)

# S2Hand 13 bands: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
IB, IG, IR, INIR, ISWIR = 1, 2, 3, 7, 11

def s2hand_to_Y(chip):
    with rio.open(f"{HL}/S2Hand/{chip}_S2Hand.tif") as src:
        s2 = src.read().astype(np.float32)
    nod = (s2[IG] == 0) | (s2[ISWIR] == 0) | (s2[INIR] == 0) | (s2[IR] == 0)
    s2 /= 10000.0
    R, G, B, Nir, Swir = s2[IR], s2[IG], s2[IB], s2[INIR], s2[ISWIR]
    cloud, _ = cloud_like_mask(R, G, B, Nir)
    bad = cloud | nod
    Y = np.stack([norm_diff(G, Swir), norm_diff(Nir, R), norm_diff(G, Nir)], 0)
    Y[:, bad] = np.nan
    return Y.astype(np.float32)

X_parts, Y_parts, rows = [], [], []
is_trainpool = df["split"].isin(["train", "valid"]).to_numpy()
is_testpool = df["split"].isin(["test", "bolivia"]).to_numpy()

Xt_parts, Yt_parts = [], []
test_boxes = []
print("building Sen1Floods11 part...", flush=True)
for i in range(len(df)):
    chip, ev = df.loc[i, "chip"], df.loc[i, "event"]
    Y = s2hand_to_Y(chip)
    if is_trainpool[i]:
        X_parts.append(X_all[i]); Y_parts.append(Y)
        rows.append(dict(folder=f"s1f11_{ev}", date=chip, flood=True))
    else:
        Xt_parts.append(X_all[i]); Yt_parts.append(Y)
        test_boxes.append(b4326(f"{HL}/S1Hand/{chip}_S1Hand.tif"))
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(df)}", flush=True)
print(f"s1f11: train-pool {len(X_parts)}, test-pool {len(Xt_parts)}", flush=True)

# ---------- C2S-MS ----------
def read_match(path, ref_path):
    """Reproject path onto the ref grid (as is if the grid is identical). float32."""
    with rio.open(ref_path) as rf, rio.open(path) as src:
        same = (src.crs == rf.crs and src.transform == rf.transform
                and src.width == rf.width and src.height == rf.height)
        a = src.read(1).astype(np.float32)
        if same:
            return a
        out = np.full((rf.height, rf.width), np.nan, np.float32)
        reproject(a, out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=None, dst_transform=rf.transform, dst_crs=rf.crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
        return out

n_excl = n_skip = 0
print("building C2S-MS part...", flush=True)
for ev in sorted(os.listdir(C2S)):
    evname = C2S_EVENT[ev[:8]]
    s1d, s2d = os.path.join(C2S, ev, "s1"), os.path.join(C2S, ev, "s2")
    s2_by_cid = {c.rsplit("_", 1)[-1]: os.path.join(s2d, c) for c in os.listdir(s2d)}
    for c1 in sorted(os.listdir(s1d)):
        cid = c1.rsplit("_", 1)[-1]
        cd1 = os.path.join(s1d, c1)
        cd2 = s2_by_cid.get(cid)
        vvp = os.path.join(cd1, "VV.tif")
        if cd2 is None or not os.path.exists(vvp):
            n_skip += 1
            continue
        bb = b4326(vvp)
        if any(inter(bb, tb) for tb in test_boxes):
            n_excl += 1
            continue
        vv_db = read_match(vvp, vvp)
        vh_db = read_match(os.path.join(cd1, "VH.tif"), vvp)
        with np.errstate(over="ignore", invalid="ignore"):
            vv = np.power(10.0, vv_db / 10.0); vh = np.power(10.0, vh_db / 10.0)
            rvi = (4.0 * vh) / (vv + vh)
        rvi[~np.isfinite(rvi)] = np.nan
        try:
            Gd = read_match(os.path.join(cd2, "B3.tif"), vvp)
            Rd = read_match(os.path.join(cd2, "B4.tif"), vvp)
            Bd = read_match(os.path.join(cd2, "B2.tif"), vvp)
            Nd = read_match(os.path.join(cd2, "B8.tif"), vvp)
            Sd = read_match(os.path.join(cd2, "B11.tif"), vvp)
            Cl = read_match(os.path.join(cd2, "LabelCloud.tif"), vvp)
        except Exception as e:
            n_skip += 1
            continue
        nod = (Gd == 0) | (Sd == 0) | (Nd == 0) | (Rd == 0) | ~np.isfinite(Gd)
        R, G, B, Nir, Swir = Rd/1e4, Gd/1e4, Bd/1e4, Nd/1e4, Sd/1e4
        cloud, _ = cloud_like_mask(R, G, B, Nir)
        bad = cloud | nod | (Cl > 0)
        Y = np.stack([norm_diff(G, Swir), norm_diff(Nir, R), norm_diff(G, Nir)], 0)
        Y[:, bad] = np.nan
        if Y.shape[1:] != (512, 512) or vv.shape != (512, 512):
            n_skip += 1
            continue
        X_parts.append(np.stack([vv, vh, rvi], 0).astype(np.float32))
        Y_parts.append(Y.astype(np.float32))
        rows.append(dict(folder=evname, date=cid, flood=True))
print(f"c2s: added {sum(r['folder'].startswith(('c2s', 's1f11')) for r in rows) - 341}, "
      f"excluded(test-overlap) {n_excl}, skipped {n_skip}", flush=True)

# ---------- concatenate with SEN12FLOOD trn2 ----------
X_sf = np.load(P + "/trn2_X_VV_VH_RVI_raw.npy")
Y_sf = np.load(P + "/trn2_Y_MNDWI_NDVI_NDWI_raw.npy")
df_sf = pd.read_csv(P + "/trn2_matched_samples_good.csv")
print(f"sen12flood trn2: {len(df_sf)}", flush=True)

X_new = np.stack(X_parts).astype(np.float32)
Y_new = np.stack(Y_parts).astype(np.float32)
X4 = np.concatenate([X_sf, X_new], 0)
Y4 = np.concatenate([Y_sf, Y_new], 0)
df4 = pd.concat([df_sf[["folder", "date", "flood"]],
                 pd.DataFrame(rows)], ignore_index=True)
assert len(df4) == X4.shape[0]
np.save(P + "/trn4_X_VV_VH_RVI_raw.npy", X4)
np.save(P + "/trn4_Y_MNDWI_NDVI_NDWI_raw.npy", Y4)
df4.to_csv(P + "/trn4_matched_samples_good.csv", index=False)
print(f"trn4: {X4.shape[0]} pairs, {df4['folder'].map(os.path.basename).nunique()} scenes", flush=True)

X4t = np.stack(Xt_parts).astype(np.float32)
Y4t = np.stack(Yt_parts).astype(np.float32)
np.save(P + "/tst4_X_VV_VH_RVI_raw.npy", X4t)
np.save(P + "/tst4_Y_MNDWI_NDVI_NDWI_raw.npy", Y4t)
print(f"tst4: {X4t.shape[0]} pairs (s1f11 test+bolivia)", flush=True)

fin = float(np.mean(np.isfinite(Y_new[:, 0])))
print(f"sanity: new-part Y finite frac={fin:.3f}", flush=True)
