#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve the geo (source tiff + canvas placement) of every base (trn6, 1500) chip.

For each row (folder,date,flood) of trn6_matched_samples_good.csv, locate the
geo-referenced source tiff and return CRS/transform/H/W/canvas placement.
Output: processed/trn6_geo_index.csv  (idx, source, geo_tif, H, W, lon/lat bounds, ok)

If all 1500/1500 resolve, v16 (base-JRC) via JRC warp becomes possible.
Run: not with /opt/miniconda3/envs/gam/bin/python (pandas); rasterio is required, so
      run with /usr/bin/python3.10.
"""
import os, glob, sys
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_bounds

R = "/home/kk/jp_claude/RMSD"
P = R + "/processed"
HL = R + "/data/Sen1Floods11/v1.1/data/flood_events/HandLabeled"
C2S = R + "/data/C2SMSFloods/chips"
FP = R + "/data/FloodPlanet/FloodPlanet"

# evname -> C2S uuid8 (reverse lookup of build_trn4.C2S_EVENT)
C2S_EVENT = {
    "0c7daa97": "s1f11_Somalia", "3e71513a": "s1f11_Mekong", "4567c2a2": "c2s_harvey",
    "55d43c77": "c2s_nebraska", "5ef4fb15": "s1f11_Ghana", "749aca10": "c2s_australia",
    "89b504aa": "s1f11_Bolivia", "8fcb3e59": "c2s_congo", "95eff095": "s1f11_Sri-Lanka",
    "975ecc1f": "s1f11_Nigeria", "befccce4": "s1f11_Paraguay", "c19f5f27": "s1f11_Pakistan",
    "d89b6d2c": "s1f11_India", "dc70be79": "c2s_kenya", "e7d1917e": "c2s_slovakia",
    "e7e6e693": "s1f11_Spain", "e92c811f": "c2s_niger", "f59da1a3": "s1f11_USA",
}
EV2UUID = {}
for u, e in C2S_EVENT.items():
    EV2UUID.setdefault(e, []).append(u)

# C2S directories (uuid8 -> full directory name)
c2s_dirs = {}
if os.path.isdir(C2S):
    for d in os.listdir(C2S):
        c2s_dirs[d[:8]] = os.path.join(C2S, d)


def resolve_c2s(evname, cid):
    for u in EV2UUID.get(evname, []):
        base = c2s_dirs.get(u)
        if not base:
            continue
        s1d = os.path.join(base, "s1")
        if not os.path.isdir(s1d):
            continue
        for c1 in os.listdir(s1d):
            if c1.rsplit("_", 1)[-1] == str(cid):
                vv = os.path.join(s1d, c1, "VV.tif")
                if os.path.exists(vv):
                    return vv
    return None


def resolve(folder, date):
    """Return (source, geo_tif, placement). placement is 'full' or 'topleft'."""
    f = str(folder)
    d = str(date)
    # SEN12FLOOD: absolute path
    if "SEN12FLOOD" in f and os.path.isdir(f):
        for band in ("B04", "B03", "B02"):
            t = os.path.join(f, f"S2_{d}_{band}.tif")
            if os.path.exists(t):
                return "SEN12", t, "full"
        return "SEN12", None, "full"
    # Sen1Floods11: date = long chip name, S1Hand tiff exists
    t = os.path.join(HL, "S1Hand", f"{d}_S1Hand.tif")
    if os.path.exists(t):
        return "s1f11", t, "full"
    # C2S: folder is the C2S event name, date=cid
    if f in EV2UUID:
        t = resolve_c2s(f, d)
        if t:
            return "c2s", t, "full"
    # FloodPlanet: fp_{ev}
    if f.startswith("fp_"):
        ev = f[3:]
        for sub in ("S1", "S2"):
            t = os.path.join(FP, ev, sub, f"{d}.tif")
            if os.path.exists(t):
                return "fp", t, "topleft"
    return "UNRESOLVED", None, "full"


def main():
    df = pd.read_csv(P + "/trn6_matched_samples_good.csv")
    rows = []
    for i, r in df.iterrows():
        src, tif, place = resolve(r["folder"], r["date"])
        rec = dict(idx=i, folder=r["folder"], date=r["date"], source=src,
                   geo_tif=tif or "", place=place, H=0, W=0,
                   lon_min=np.nan, lat_min=np.nan, lon_max=np.nan, lat_max=np.nan, ok=False)
        if tif and os.path.exists(tif):
            try:
                with rio.open(tif) as s:
                    rec["H"], rec["W"] = s.height, s.width
                    lo = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
                    rec.update(lon_min=lo[0], lat_min=lo[1], lon_max=lo[2], lat_max=lo[3], ok=True)
            except Exception as ex:
                rec["source"] = f"READFAIL:{str(ex)[:30]}"
        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)} resolved", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(P + "/trn6_geo_index.csv", index=False)
    print("\n=== resolution by source ===", flush=True)
    print(out.groupby("source").agg(n=("ok", "size"), ok=("ok", "sum")).to_string(), flush=True)
    print(f"\nTOTAL ok: {out.ok.sum()}/{len(out)}", flush=True)
    # JRC 10deg tiles needed (occurrence naming: lon in 10deg steps [180W..170E], lat in 10deg steps [80N..50S])
    ok = out[out.ok]
    tiles = set()
    for _, r in ok.iterrows():
        for lon in (r.lon_min, r.lon_max):
            for lat in (r.lat_min, r.lat_max):
                tx = int(np.floor(lon / 10.0) * 10)
                ty = int(np.ceil(lat / 10.0) * 10)
                ew = f"{abs(tx)}{'E' if tx >= 0 else 'W'}"
                ns = f"{abs(ty)}{'N' if ty >= 0 else 'S'}"
                tiles.add(f"{ew}_{ns}")
    print(f"\nJRC 10deg tiles needed: {len(tiles)}", flush=True)
    print(sorted(tiles), flush=True)


if __name__ == "__main__":
    main()
