#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the 446 hand-labeled Sen1Floods11 chips into RMSD-pipeline npy format.

S1Hand is in dB, so convert back to linear power before saving (same convention as the
SEN12FLOOD raw X: linear power, the model converts to dB itself). RVI is computed from linear power.

Output (SEN12FLOOD/processed/ -- visible from the container):
  s1f11_X_VV_VH_RVI_raw.npy  (446,3,512,512) float32 linear power
  s1f11_label.npy            (446,512,512) int8   (-1 nodata / 0 land / 1 water)
  s1f11_jrc.npy              (446,512,512) uint8  (1=JRC permanent water)
  s1f11_chips.csv            chip name, event, official split
Run with: WSL native python (rasterio).
"""
import os, csv
import glob as g
import numpy as np
import rasterio as rio

S1F = "/home/kk/jp_claude/RMSD/data/Sen1Floods11/v1.1"
HL = os.path.join(S1F, "data/flood_events/HandLabeled")
OUT = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD/processed"

# official split
split_of = {}
for split in ["train", "valid", "test", "bolivia"]:
    with open(os.path.join(S1F, f"splits/flood_handlabeled/flood_{split}_data.csv")) as f:
        for row in csv.reader(f):
            if row:
                split_of[os.path.basename(row[0]).replace("_S1Hand.tif", "")] = split

s1_files = sorted(g.glob(os.path.join(HL, "S1Hand/*.tif")))
N = len(s1_files)
print(f"chips: {N}", flush=True)

X = np.full((N, 3, 512, 512), np.nan, np.float32)
LB = np.full((N, 512, 512), -1, np.int8)
JR = np.zeros((N, 512, 512), np.uint8)
rows = []
for i, p in enumerate(s1_files):
    chip = os.path.basename(p).replace("_S1Hand.tif", "")
    event = chip.split("_")[0]
    with rio.open(p) as src:
        s1 = src.read().astype(np.float32)          # (2,512,512) dB
    vv_db, vh_db = s1[0], s1[1]
    with np.errstate(over="ignore", invalid="ignore"):
        vv = np.power(10.0, vv_db / 10.0)            # -> linear power
        vh = np.power(10.0, vh_db / 10.0)
    vv[~np.isfinite(vv_db)] = np.nan
    vh[~np.isfinite(vh_db)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = (4.0 * vh) / (vv + vh)
    rvi[~np.isfinite(rvi)] = np.nan
    X[i, 0], X[i, 1], X[i, 2] = vv, vh, rvi

    with rio.open(os.path.join(HL, f"LabelHand/{chip}_LabelHand.tif")) as src:
        LB[i] = src.read(1).astype(np.int8)
    with rio.open(os.path.join(HL, f"JRCWaterHand/{chip}_JRCWaterHand.tif")) as src:
        JR[i] = (src.read(1) > 0).astype(np.uint8)

    rows.append(dict(idx=i, chip=chip, event=event, split=split_of.get(chip, "?")))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N}", flush=True)

np.save(os.path.join(OUT, "s1f11_X_VV_VH_RVI_raw.npy"), X)
np.save(os.path.join(OUT, "s1f11_label.npy"), LB)
np.save(os.path.join(OUT, "s1f11_jrc.npy"), JR)
with open(os.path.join(OUT, "s1f11_chips.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "chip", "event", "split"])
    w.writeheader(); w.writerows(rows)

lb_valid = LB >= 0
print(f"saved. label valid px: {lb_valid.mean():.3f}, water (of valid): "
      f"{(LB == 1).sum() / max(lb_valid.sum(), 1):.3f}, JRC perm px: {JR.mean():.4f}", flush=True)
from collections import Counter
print("splits:", Counter(r["split"] for r in rows), flush=True)
