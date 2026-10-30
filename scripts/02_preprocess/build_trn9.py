#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the v9 training arrays: trn9 = trn6 (1500) + _trn9new (Kuro-S2 11 train events, 1055) = 2555.

Convention unified with v1-v12 (X=linear power, Y=optical indices, cloud NaN, 512px).
The trainer reads {prefix}_X_VV_VH_RVI_raw.npy / _Y_..._raw.npy and
{prefix}_matched_samples_good.csv (folder column required, rows=N).
tst9 is a symlink to tst4.
Output: processed/trn9_{X,Y}_*.npy, trn9_matched_samples_good.csv, tst9_* symlink
"""
import os
import numpy as np
import pandas as pd

P = "/home/kk/jp_claude/RMSD/processed"
SZ = 512


def load(name):
    xp = f"{P}/{name}_X.npy" if os.path.exists(f"{P}/{name}_X.npy") else f"{P}/{name}_X_VV_VH_RVI_raw.npy"
    yp = f"{P}/{name}_Y.npy" if os.path.exists(f"{P}/{name}_Y.npy") else f"{P}/{name}_Y_MNDWI_NDVI_NDWI_raw.npy"
    return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")


x6, y6 = load("trn6")
x9, y9 = load("_trn9new")
m6 = pd.read_csv(f"{P}/trn6_matched_samples_good.csv")
m9 = pd.read_csv(f"{P}/_trn9new_meta.csv")
assert x6.shape[0] == len(m6), f"trn6 X {x6.shape[0]} != csv {len(m6)}"
assert x9.shape[0] == len(m9), f"_trn9new X {x9.shape[0]} != csv {len(m9)}"

N = x6.shape[0] + x9.shape[0]
print(f"trn9: {x6.shape[0]} (trn6) + {x9.shape[0]} (kuro_s2) = {N}", flush=True)

X = np.lib.format.open_memmap(f"{P}/trn9_X_VV_VH_RVI_raw.npy", mode="w+",
                              dtype=np.float32, shape=(N, 3, SZ, SZ))
Y = np.lib.format.open_memmap(f"{P}/trn9_Y_MNDWI_NDVI_NDWI_raw.npy", mode="w+",
                              dtype=np.float32, shape=(N, 3, SZ, SZ))
off = 0
CH = 64
for (x, y) in [(x6, y6), (x9, y9)]:
    n = x.shape[0]
    for s in range(0, n, CH):
        e = min(s + CH, n)
        X[off + s:off + e] = x[s:e]
        Y[off + s:off + e] = y[s:e]
    off += n
X.flush(); Y.flush()

# meta csv: always keep the folder column; also add a source tag.
d6 = m6[["folder", "date", "flood"]].copy(); d6["source"] = "trn6"
d9 = m9[["folder", "date", "flood"]].copy(); d9["source"] = "kuro_s2"
out = pd.concat([d6, d9], ignore_index=True, sort=False)
out["idx"] = np.arange(len(out))
assert len(out) == N
out.to_csv(f"{P}/trn9_matched_samples_good.csv", index=False)

# tst9 -> tst4 symlink
for s in ("X_VV_VH_RVI_raw", "Y_MNDWI_NDVI_NDWI_raw"):
    dst = f"{P}/tst9_{s}.npy"
    if not os.path.exists(dst):
        os.symlink(f"{P}/tst4_{s}.npy", dst)

print(f"trn9 saved X={X.shape}, csv rows={len(out)}, unique folders={out.folder.map(os.path.basename).nunique()}", flush=True)
print("DONE trn9", flush=True)
