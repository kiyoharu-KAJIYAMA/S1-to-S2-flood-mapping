#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract only "all flood tiles + a balanced set of negatives" from Kuro Siwo tiles and save as npy.
Core piece for streaming the batches so everything fits within 1TB.

Each tile (folder containing info.json):
  input SAR = MS1_IVV, MS1_IVH (flood time, float32 linear intensity)
  GT        = MK0_MLU (0=no water,1=permanent water,2=flood,3=invalid)
  pflood, actid, grid_id from info.json

Selection: pflood >= min_pflood are flood positives (all kept); pflood==0 are negatives
      (per event, randomly keep n_positives x neg_ratio). Deterministic RNG (grid_id hash) for reproducibility.

Output (--out):
  {actid}_X.npy   (N,3,H,W) float32 [VV,VH,RVI] linear intensity
  {actid}_MLU.npy (N,H,W)   uint8   raw labels
  {actid}_meta.csv  grid_id,pflood,pwater,is_flood
  _manifest.csv     appended: processed grid_id (prevents reprocessing)
Run: WSL native python (rasterio). No GPU needed; can run alongside training.
"""
import os, sys, json, csv, glob, argparse, hashlib
import numpy as np
import rasterio as rio

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="scan root (extracted Kuro Siwo)")
ap.add_argument("--out", default="/home/kk/jp_claude/RMSD/data/KuroSiwo_subset")
ap.add_argument("--min_pflood", type=float, default=1.0, help="pflood(%) at or above which a tile counts as a flood positive")
ap.add_argument("--neg_ratio", type=float, default=1.0, help="number of negatives kept, as a multiple of the positives")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

manifest_path = os.path.join(a.out, "_manifest.csv")
done = set()
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        done = {r[0] for r in csv.reader(f) if r}
print(f"already-harvested grids: {len(done)}", flush=True)

infos = glob.glob(os.path.join(a.src, "**", "info.json"), recursive=True)
print(f"tiles found under {a.src}: {len(infos)}", flush=True)

by_event = {}
for ip in infos:
    d = os.path.dirname(ip)
    try:
        with open(ip) as f:
            m = json.load(f)
    except Exception:
        continue
    gid = str(m.get("grid_id", "")).replace("-", "")
    if not gid or gid in done:
        continue
    actid = m.get("actid")
    pf = float(m.get("pflood", 0) or 0)
    pw = float(m.get("pwater", 0) or 0)
    by_event.setdefault(actid, []).append((d, gid, pf, pw, pf >= a.min_pflood))

def h01(s):
    return int(hashlib.md5((str(a.seed) + s).encode()).hexdigest(), 16) % 10000 / 10000.0

def read1(pat, d):
    fs = glob.glob(os.path.join(d, pat))
    return rio.open(fs[0]).read(1).astype(np.float32) if fs else None

n_events = n_flood = n_neg = 0
for actid, tiles in sorted(by_event.items()):
    flood = [t for t in tiles if t[4]]
    negs = [t for t in tiles if not t[4] and t[2] == 0.0]
    k = int(round(len(flood) * a.neg_ratio))
    sel = flood + sorted(negs, key=lambda t: h01(t[1]))[:k]
    if not sel:
        continue
    Xs, Ys, rows = [], [], []
    for d, gid, pf, pw, isf in sel:
        vv = read1("MS1_IVV_*.tif", d); vh = read1("MS1_IVH_*.tif", d)
        mlu = read1("MK0_MLU_*.tif", d)
        if vv is None or vh is None or mlu is None:
            continue
        if vv.shape != vh.shape or vv.shape != mlu.shape:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            rvi = (4.0 * vh) / (vv + vh)
        rvi[~np.isfinite(rvi)] = np.nan
        Xs.append(np.stack([vv, vh, rvi], 0).astype(np.float32))
        Ys.append(mlu.astype(np.uint8))
        rows.append((gid, pf, pw, int(isf)))
    if not Xs:
        continue
    shp = max({x.shape for x in Xs}, key=lambda s: sum(1 for x in Xs if x.shape == s))
    keep = [i for i, x in enumerate(Xs) if x.shape == shp]
    X = np.stack([Xs[i] for i in keep]); Y = np.stack([Ys[i] for i in keep])
    rows = [rows[i] for i in keep]
    np.save(os.path.join(a.out, f"{actid}_X.npy"), X)
    np.save(os.path.join(a.out, f"{actid}_MLU.npy"), Y)
    with open(os.path.join(a.out, f"{actid}_meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["grid_id", "pflood", "pwater", "is_flood"]); w.writerows(rows)
    with open(manifest_path, "a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r[0], actid])
    nf = sum(r[3] for r in rows)
    n_events += 1; n_flood += nf; n_neg += len(rows) - nf
    print(f"  act{actid}: {len(rows)} tiles ({nf} flood + {len(rows)-nf} neg) "
          f"X{X.shape} MLUuniq={np.unique(Y)[:6]}", flush=True)

print(f"\nDONE: {n_events} events, {n_flood} flood + {n_neg} neg -> {a.out}", flush=True)
