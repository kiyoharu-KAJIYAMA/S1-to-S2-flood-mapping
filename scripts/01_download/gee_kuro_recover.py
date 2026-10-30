#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure #1: recover Kuro-S2 for all 25 training events.
Currently usable>=0.30 holds for 11/25. Recover the remaining 14 with 3 approaches:
  (1) wider date window (WIN in [7,12,20,30] days)
  (2) multi-day cloud-free composite (cloud-mask every S2 scene in the window with
      s2cloudless, then mosaic pixels that were clear at least once; its coverage =
      clear_cover, the effective coverage of a "multi-day composite")
  (3) nearest clear day (single-day best within the window, reported alongside)
test_events (18) are held-out. TRAIN only.

Output: processed/gee_kuro_recover.csv  (clear_cover and single-day best per event x window)
This gives the yield: which window recovers how many events to usable>=0.30.
"""
import sqlite3, struct, json
import datetime as dt
import numpy as np
import pandas as pd
import ee
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

R = "/home/kk/jp_claude/RMSD"
ee.Initialize(project="rmsd-503302")
SCALE, CLOUD_PROB = 300, 40
WINDOWS = [7, 12, 20, 30]

split = json.load(open(R + "/processed/kuro_split.json"))
train_ev, test_ev = set(split["train_events"]), set(split["test_events"])
av = pd.read_csv(R + "/processed/gee_kuro_s2_availability2.csv")
# recovery targets = TRAIN and usable_frac < 0.30
targets = set(av[(av.role == "TRAIN") & (av.usable_frac < 0.30)].actid.tolist())
print(f"recover targets ({len(targets)}): {sorted(targets)}", flush=True)

con = sqlite3.connect(R + "/data/KuroSiwo/catalogue.gpkg"); cur = con.cursor()
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

cur.execute("SELECT actid, flood_date, geom FROM catalogue")
agg = {}
for actid, fd, blob in cur:
    if int(actid) not in targets:
        continue
    e = env(blob)
    if e is None:
        continue
    a = agg.setdefault(int(actid), [1e18, 1e18, -1e18, -1e18, fd])
    a[0], a[1] = min(a[0], e[0]), min(a[1], e[1])
    a[2], a[3] = max(a[2], e[2]), max(a[3], e[3])
con.close()


def clear_cover(aoi, t0, win):
    """AOI coverage of pixels clear at least once within +-win days, plus single-day best coverage."""
    d0 = ee.Date((t0 - dt.timedelta(days=win)).isoformat())
    d1 = ee.Date((t0 + dt.timedelta(days=win + 1)).isoformat())
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(aoi).filterDate(d0, d1)
    pr = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(d0, d1)
    n = s2.size().getInfo()
    if n == 0:
        return 0, 0.0, None, 0.0
    j = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
        s2, pr, ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def m(img):
        cp = ee.Image(img.get("cp")).select("probability")
        return ee.Image(img).updateMask(cp.lt(CLOUD_PROB))
    masked = j.map(m)
    comp = masked.select("B3").mosaic()
    cover = comp.mask().reduceRegion(ee.Reducer.mean(), aoi, SCALE,
                                     maxPixels=1e9, bestEffort=True).get("B3").getInfo()
    # single-day best (nearest clear day)
    days = sorted({dt.datetime.utcfromtimestamp(t / 1000).date()
                   for t in j.aggregate_array("system:time_start").getInfo()})
    best_d, best_c = None, 0.0
    for dd in days:
        a0 = ee.Date(dd.isoformat())
        sub = masked.filterDate(a0, a0.advance(1, "day")).select("B3").mosaic()
        c = sub.mask().reduceRegion(ee.Reducer.mean(), aoi, SCALE,
                                    maxPixels=1e9, bestEffort=True).get("B3").getInfo()
        if c and c > best_c:
            best_c, best_d = float(c), str(dd)
    return int(n), float(cover or 0), best_d, best_c


rows = []
for actid, (x0, y0, x1, y1, fd) in sorted(agg.items()):
    w, s, e, n = transform_bounds(crs, "EPSG:4326", x0, y0, x1, y1)
    aoi = ee.Geometry.Rectangle([w, s, e, n], None, False)
    t0 = dt.datetime.fromisoformat(str(fd)[:19])
    rec = dict(actid=actid, date=str(fd)[:10])
    for win in WINDOWS:
        try:
            ns, cover, bd, bc = clear_cover(aoi, t0, win)
        except Exception as ex:
            print(f"  {actid} win{win} FAIL {str(ex)[:60]}", flush=True)
            ns, cover, bd, bc = -1, 0.0, None, 0.0
        rec[f"n{win}"] = ns
        rec[f"cover{win}"] = round(cover, 3)
        rec[f"best{win}"] = bd
        rec[f"bestcov{win}"] = round(bc, 3)
    rows.append(rec)
    print(f"{actid} {rec['date']}  " +
          "  ".join(f"±{w}d cov={rec[f'cover{w}']:.2f}(1d {rec[f'bestcov{w}']:.2f})"
                    for w in WINDOWS), flush=True)

df = pd.DataFrame(rows)
df.to_csv(R + "/processed/gee_kuro_recover.csv", index=False)
print("\n== recovery yield (composite clear_cover) ==", flush=True)
for w in WINDOWS:
    for th in (0.30, 0.50):
        k = int((df[f"cover{w}"] >= th).sum())
        print(f"  ±{w}d: cover>={th:.0%} -> {k}/{len(df)} recovered", flush=True)
print("\nper-event best window to reach cover>=0.30:")
for _, r in df.iterrows():
    hit = [w for w in WINDOWS if r[f"cover{w}"] >= 0.30]
    print(f"  {int(r.actid)} {r.date}: " +
          (f"±{min(hit)}d (cover {r[f'cover{min(hit)}']:.2f})" if hit
           else f"UNRECOVERABLE (max cover {max(r[f'cover{w}'] for w in WINDOWS):.2f})"))
