#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the v6 training data (trn6 prefix, fast2-compatible format).

trn6 = only the flood==True pairs of trn5 (flood-only corpus, no non-flood):
  SEN12FLOOD flood-day pairs + S1F11 train/valid + C2S-MS + FloodPlanet
The only difference from trn5 is dropping the SEN12FLOOD non-flood-day pairs (about 87% of trn5).
train/val is an 80/20 scene-level split done at fast2 run time; test is tst6 (= tst4 symlink,
S1F11 test+bolivia -- all acquired during flood events, so the corpus character is preserved).
Run with: WSL native python.
"""
import os
import numpy as np
import pandas as pd

P = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD/processed"

df5 = pd.read_csv(P + "/trn5_matched_samples_good.csv")
keep = df5["flood"].astype(bool).to_numpy()
idx = np.flatnonzero(keep)
df6 = df5.iloc[idx].reset_index(drop=True)

def src_of(f):
    b = os.path.basename(str(f))
    if b.startswith("fp_"):
        return "FloodPlanet"
    if b.startswith("c2s_"):
        return "C2S-MS"
    if b.startswith("s1f11_"):
        return "S1F11(+C2S)"
    return "SEN12FLOOD"

comp = df6["folder"].map(src_of).value_counts()
print(f"trn6 = {len(df6)} flood-only pairs (from trn5 {len(df5)}):")
print(comp.to_string())
print(f"scenes: {df6['folder'].map(os.path.basename).nunique()}")

for name in ["X_VV_VH_RVI_raw", "Y_MNDWI_NDVI_NDWI_raw"]:
    src = np.load(P + f"/trn5_{name}.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(P + f"/trn6_{name}.npy", mode="w+",
                                    dtype=np.float32, shape=(len(idx), 3, 512, 512))
    for k in range(0, len(idx), 100):
        sel = idx[k:k+100]
        out[k:k+len(sel)] = src[sel]
    out.flush(); del out, src
    print(f"saved trn6_{name}.npy")

df6.to_csv(P + "/trn6_matched_samples_good.csv", index=False)
print("saved trn6_matched_samples_good.csv")

for name in ["X_VV_VH_RVI_raw", "Y_MNDWI_NDVI_NDWI_raw"]:
    dst = P + f"/tst6_{name}.npy"
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(f"tst4_{name}.npy", dst)
    print(f"symlink tst6_{name}.npy -> tst4_{name}.npy")
print("DONE")
