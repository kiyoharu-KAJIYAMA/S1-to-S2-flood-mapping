#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trn16 = _basejrc (JRC-applied base 1500) + _trn9new (Kuro-S2) + _gfd1new + _kurr2new + _sturmnew.

Same composition as trn15, but the "all-JRC" version: JRC permanent-water exclusion is also
applied to the teacher of the base 1500. Real-flood, same-time pairs only; permanent water is
fully excluded from the teacher. Convention unified with v1-v15 (512px).
tst16 is a symlink to tst4.
"""
import os
import numpy as np
import pandas as pd

P = "/home/kk/jp_claude/RMSD/processed"
SZ = 512
# _trn9jrc = _trn9new with JRC permanent-water removal applied afterwards (TODO #12).
# Falls back to _trn9new where it has not been built.
_KURO = "_trn9jrc" if os.path.exists(f"{P}/_trn9jrc_Y.npy") else "_trn9new"
PARTS = [("_basejrc", "base_jrc"), (_KURO, "kuro_s2_jrc" if _KURO == "_trn9jrc" else "kuro_s2"),
         ("_gfd1new", "gfd_dt1_jrc"), ("_kurr2new", "kuro_recover_jrc"),
         ("_sturmnew", "sturm_jrc")]


def load(name):
    xp = f"{P}/{name}_X.npy" if os.path.exists(f"{P}/{name}_X.npy") else f"{P}/{name}_X_VV_VH_RVI_raw.npy"
    yp = f"{P}/{name}_Y.npy" if os.path.exists(f"{P}/{name}_Y.npy") else f"{P}/{name}_Y_MNDWI_NDVI_NDWI_raw.npy"
    return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")


def meta(name):
    for c in (f"{P}/{name}_matched_samples_good.csv", f"{P}/{name}_meta.csv"):
        if os.path.exists(c):
            return pd.read_csv(c)
    raise FileNotFoundError(name)


arrays, metas, N = [], [], 0
for name, tag in PARTS:
    if not os.path.exists(f"{P}/{name}_X.npy") and not os.path.exists(f"{P}/{name}_X_VV_VH_RVI_raw.npy"):
        print(f"  skip {name} (missing)", flush=True); continue
    x, y = load(name); mm = meta(name)
    assert x.shape[0] == len(mm), f"{name} X {x.shape[0]} != csv {len(mm)}"
    arrays.append((x, y, tag)); metas.append((mm, tag)); N += x.shape[0]
    print(f"  {name}: {x.shape[0]} chips", flush=True)

print(f"trn16 total: {N}", flush=True)
X = np.lib.format.open_memmap(f"{P}/trn16_X_VV_VH_RVI_raw.npy", mode="w+", dtype=np.float32, shape=(N, 3, SZ, SZ))
Y = np.lib.format.open_memmap(f"{P}/trn16_Y_MNDWI_NDVI_NDWI_raw.npy", mode="w+", dtype=np.float32, shape=(N, 3, SZ, SZ))
off, CH = 0, 64
for (x, y, tag) in arrays:
    n = x.shape[0]
    for s in range(0, n, CH):
        e = min(s + CH, n)
        X[off + s:off + e] = x[s:e]; Y[off + s:off + e] = y[s:e]
    off += n
X.flush(); Y.flush()

parts_csv = []
for (mm, tag) in metas:
    d = mm.copy()
    if "date" not in d.columns:
        d["date"] = tag
    if "flood" not in d.columns:
        d["flood"] = True
    d = d[["folder", "date", "flood"]].copy()
    d["source"] = tag
    parts_csv.append(d)
out = pd.concat(parts_csv, ignore_index=True, sort=False)
out["idx"] = np.arange(len(out))
assert len(out) == N
out.to_csv(f"{P}/trn16_matched_samples_good.csv", index=False)

for s in ("X_VV_VH_RVI_raw", "Y_MNDWI_NDVI_NDWI_raw"):
    dst = f"{P}/tst16_{s}.npy"
    if not os.path.exists(dst):
        os.symlink(f"{P}/tst4_{s}.npy", dst)

print(f"trn16 saved X={X.shape}, csv rows={len(out)}", flush=True)
print("source breakdown:", out.source.value_counts().to_dict(), flush=True)
print("DONE trn16", flush=True)
