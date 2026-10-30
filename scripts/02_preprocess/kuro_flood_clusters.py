#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 preparation: design export rectangles that cut out only the surroundings of Kuro Siwo flood areas.

Approach: taking the full AOIs (50,000-200,000 km2) would make the export tens of GB, so
  1) extract from the catalogue the **tiles with pflood >= MIN_PFLOOD** (= tiles showing flood)
  2) quantise tile centres onto a coarse grid (CELL_DEG) and cluster them
  3) take bbox + buffer per cluster, split at the maximum size -> list of export rectangles

Target is **only the train_events with usable S2 (usable>=0.30)**.
test_events (18) are held out for zero-shot evaluation and are never included.

Output: processed/kuro_v9_export_regions.csv
"""
import sqlite3, struct, json
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

R = "/home/kk/jp_claude/RMSD"
MIN_PFLOOD = 5.0        # % — 1% gives a total area of 105,000 km2 (19GB), so restrict to substantially flooded tiles
CELL_DEG = 0.25         # cluster quantisation (~25 km)
BUF_DEG = 0.02          # surrounding context (~2 km)
MAX_SPAN_DEG = 0.6      # maximum side of one rectangle (~60 km)
CAP_KM2_PER_EVENT = 8000  # cap to prevent one event dominating (Pakistan 1111009 alone takes 60%)

split = json.load(open(R + "/processed/kuro_split.json"))
train_ev, test_ev = set(split["train_events"]), set(split["test_events"])
av = pd.read_csv(R + "/processed/gee_kuro_s2_availability2.csv")
use_ev = set(av[(av.role == "TRAIN") & (av.usable_frac >= 0.30)].actid.tolist())
assert not (use_ev & test_ev), "held-out event leaked into v9 target!"
print(f"target events ({len(use_ev)}): {sorted(use_ev)}", flush=True)

con = sqlite3.connect(R + "/data/KuroSiwo/catalogue.gpkg")
cur = con.cursor()
cur.execute("SELECT srs_id FROM gpkg_geometry_columns WHERE table_name='catalogue'")
srs = cur.fetchone()[0]
cur.execute("SELECT definition, organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs,))
de, org, oid = cur.fetchone()
crs = CRS.from_string(f"{org}:{oid}") if oid and oid > 0 else CRS.from_wkt(de)

def env(b):
    if b is None or b[:2] != b"GP":
        return None
    f = b[3]
    if ((f >> 1) & 7) == 0:
        return None
    e = "<" if f & 1 else ">"
    v = struct.unpack(f"{e}4d", b[8:40])
    return v[0], v[2], v[1], v[3]

q = ",".join(str(int(a)) for a in sorted(use_ev))
cur.execute(f"SELECT actid, grid_id, pflood, pwater, flood_date, geom FROM catalogue "
            f"WHERE actid IN ({q}) AND pflood >= ?", (MIN_PFLOOD,))
seen, rows = set(), []
for actid, gid, pf, pw, fd, blob in cur:
    key = (int(actid), gid)
    if key in seen:
        continue
    e = env(blob)
    if e is None:
        continue
    seen.add(key)
    rows.append((int(actid), gid, float(pf), float(pw), str(fd)[:10]) + e)
con.close()
df = pd.DataFrame(rows, columns=["actid", "grid_id", "pflood", "pwater", "date",
                                 "x0", "y0", "x1", "y1"])
print(f"flood tiles (pflood>={MIN_PFLOOD}%, deduped): {len(df)}", flush=True)

# 3857 -> 4326 (Web Mercator, so convert in closed form in one go)
RE = 6378137.0
def m2lon(x): return np.degrees(x / RE)
def m2lat(y): return np.degrees(2 * np.arctan(np.exp(y / RE)) - np.pi / 2)
df["lon0"], df["lon1"] = m2lon(df.x0.values), m2lon(df.x1.values)
df["lat0"], df["lat1"] = m2lat(df.y0.values), m2lat(df.y1.values)
df["clon"], df["clat"] = (df.lon0 + df.lon1) / 2, (df.lat0 + df.lat1) / 2
tile_km = ((df.x1 - df.x0).median() / 1000.0)
print(f"tile size (3857 m): {tile_km*1000:.0f} m", flush=True)

regs = []
for actid, g in df.groupby("actid"):
    g = g.copy()
    g["ci"] = np.floor(g.clon / CELL_DEG).astype(int)
    g["cj"] = np.floor(g.clat / CELL_DEG).astype(int)
    for (ci, cj), gg in g.groupby(["ci", "cj"]):
        w, s = gg.lon0.min() - BUF_DEG, gg.lat0.min() - BUF_DEG
        e, n = gg.lon1.max() + BUF_DEG, gg.lat1.max() + BUF_DEG
        # split at the maximum size
        nx = max(1, int(np.ceil((e - w) / MAX_SPAN_DEG)))
        ny = max(1, int(np.ceil((n - s) / MAX_SPAN_DEG)))
        for ix in range(nx):
            for iy in range(ny):
                ww = w + (e - w) * ix / nx
                ee_ = w + (e - w) * (ix + 1) / nx
                ss = s + (n - s) * iy / ny
                nn = s + (n - s) * (iy + 1) / ny
                area = abs((ee_ - ww) * 111 * np.cos(np.radians((ss + nn) / 2)) * (nn - ss) * 111)
                regs.append(dict(actid=actid, date=gg.date.iloc[0],
                                 region=f"{actid}_c{ci}_{cj}_{ix}{iy}",
                                 lon_min=round(ww, 4), lat_min=round(ss, 4),
                                 lon_max=round(ee_, 4), lat_max=round(nn, 4),
                                 n_flood_tiles=int(len(gg)), area_km2=round(area, 1)))

out = pd.DataFrame(regs)
# per event, take rectangles in order of flood-tile count and stop at CAP_KM2_PER_EVENT
kept = []
for actid, g in out.groupby("actid"):
    g = g.sort_values("n_flood_tiles", ascending=False)
    cum, sel = 0.0, []
    for _, r in g.iterrows():
        if cum >= CAP_KM2_PER_EVENT:
            break
        sel.append(r); cum += r.area_km2
    kept.append(pd.DataFrame(sel))
    if len(sel) < len(g):
        print(f"  actid {actid}: capped {len(g)}->{len(sel)} regions "
              f"({g.area_km2.sum():.0f} -> {cum:.0f} km2)", flush=True)
out = pd.concat(kept, ignore_index=True).sort_values(["actid", "region"])
out.to_csv(R + "/processed/kuro_v9_export_regions.csv", index=False)
print(f"\nexport regions: {len(out)}  total area {out.area_km2.sum():,.0f} km2", flush=True)
print(out.groupby("actid").agg(regions=("region", "size"), tiles=("n_flood_tiles", "sum"),
                               area_km2=("area_km2", "sum")).to_string())
px = out.area_km2.sum() * 1e6 / 100.0
print(f"\nestimated pixels @10m: {px/1e6:,.0f} Mpx  -> S1(2b f32) {px*8/1e9:.1f} GB, "
      f"S2(5b u16) {px*10/1e9:.1f} GB")
