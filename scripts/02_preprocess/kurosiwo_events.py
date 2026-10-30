#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kuro Siwo catalogue: print the bbox (EPSG:4326) and date per actid, and
cross-check against the event bboxes of SEN12FLOOD / Sen1Floods11 / FloodPlanet / C2S-MS."""
import sqlite3, struct, json, os, re
import glob as g
import numpy as np
import rasterio as rio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

DB = "/home/kk/jp_claude/RMSD/data/KuroSiwo/catalogue.gpkg"
con = sqlite3.connect(DB)
cur = con.cursor()

cur.execute("SELECT srs_id FROM gpkg_geometry_columns WHERE table_name='catalogue'")
srs_id = cur.fetchone()[0]
cur.execute("SELECT definition, organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs_id,))
definition, org, org_id = cur.fetchone()
print(f"layer srs: {org}:{org_id}")
src_crs = CRS.from_string(f"{org}:{org_id}") if org_id > 0 else CRS.from_wkt(definition)

def envelope(blob):
    if blob is None or blob[:2] != b"GP":
        return None
    flags = blob[3]
    if ((flags >> 1) & 0x07) == 0:
        return None
    end = "<" if flags & 1 else ">"
    v = struct.unpack(f"{end}4d", blob[8:40])
    return v[0], v[2], v[1], v[3]   # minx,miny,maxx,maxy

cur.execute("SELECT actid, flood_date, geom FROM catalogue")
agg = {}
for actid, fdate, blob in cur:
    e = envelope(blob)
    if e is None:
        continue
    a = agg.setdefault(actid, [1e18, 1e18, -1e18, -1e18, fdate])
    a[0] = min(a[0], e[0]); a[1] = min(a[1], e[1])
    a[2] = max(a[2], e[2]); a[3] = max(a[3], e[3])

ks_events = []
for actid, (x0, y0, x1, y1, fdate) in sorted(agg.items()):
    b = transform_bounds(src_crs, "EPSG:4326", x0, y0, x1, y1)
    ks_events.append(dict(name=f"act{actid}", bbox=b, date=str(fdate)[:10]))
    print(f"act{actid:4d} center=({(b[0]+b[2])/2:8.2f},{(b[1]+b[3])/2:7.2f}) "
          f"bbox=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}) flood_date={str(fdate)[:10]}")
print(f"total Kuro Siwo events: {len(ks_events)}")

# ---- event bboxes of the other datasets (condensed version of the event_overlap_audit logic) ----
def b4326(path):
    with rio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds) if src.crs else None

def union(bs):
    bs = [b for b in bs if b]
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))

others = []
S1F = "/home/kk/jp_claude/RMSD/data/Sen1Floods11/v1.1"
by_ev = {}
for p in g.glob(S1F + "/data/flood_events/HandLabeled/S1Hand/*.tif"):
    by_ev.setdefault(os.path.basename(p).split("_")[0], []).append(b4326(p))
for ev, bs in by_ev.items():
    others.append(("Sen1Floods11", ev, union(bs)))
FP = "/home/kk/jp_claude/RMSD/data/FloodPlanet/FloodPlanet"
for ev in sorted(os.listdir(FP)):
    fs = g.glob(os.path.join(FP, ev, "S1", "*.tif"))
    if fs:
        others.append(("FloodPlanet", ev, union([b4326(p) for p in fs])))
C = "/home/kk/jp_claude/RMSD/data/C2SMSFloods/chips"
for ev in sorted(os.listdir(C)):
    s1d = os.path.join(C, ev, "s1")
    if not os.path.isdir(s1d):
        continue
    bs = [b4326(os.path.join(s1d, ch, "VV.tif")) for ch in os.listdir(s1d)
          if os.path.exists(os.path.join(s1d, ch, "VV.tif"))]
    others.append(("C2SMS", ev[:8], union(bs)))
ROOT = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD"
for f in sorted(g.glob(ROOT + "/trn/[0-9]*") + g.glob(ROOT + "/tst/[0-9]*")):
    tifs = g.glob(os.path.join(f, "S2_*_B04.tif")) or g.glob(os.path.join(f, "S1*_VV.tif"))
    if tifs:
        b = b4326(tifs[0])
        if b:
            others.append(("SEN12FLOOD", os.path.relpath(f, ROOT), b))

def inter(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

print("\n=== Kuro Siwo vs others: spatial overlaps ===")
n = 0
hit_ev = set()
for ke in ks_events:
    hits = [(ds, name) for ds, name, bb in others if inter(ke["bbox"], bb)]
    if hits:
        n += len(hits)
        hit_ev.add(ke["name"])
        uniq = sorted(set((d, re.sub(r'(trn|tst)/[0-9]+', 'scenes', nm)) for d, nm in hits))
        print(f"{ke['name']} ({ke['date']}): {len(hits)} hits -> {uniq[:6]}")
print(f"Kuro Siwo events overlapping others: {len(hit_ev)}/{len(ks_events)} ({n} pairs)")
