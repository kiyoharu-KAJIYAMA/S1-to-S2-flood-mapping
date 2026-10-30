#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure #1 export: export the 12 recoverable Kuro training events as multi-day cloud-free S2 composites + co-temporal S1.

Recovery methods (all three implemented at once):
  (1) window widening : per event, the smallest window with cover>=0.30 (from gee_kuro_recover.csv)
  (2) multi-day composite : mask all S2 in the window with s2cloudless (prob<40), order by distance
               from the flood date (farthest first) and mosaic -> per pixel, the "clear observation
               closest to the flood date" is adopted
  (3) nearest clear day : exactly the per-pixel selection of (2) (recovers per pixel even when no single day is clear)
  S1        : as in Kuro, IW/VV+VH within flood_date -18h..+42h, single-orbit mosaic (co-temporal SAR)

Region generation: cluster the pflood>=5% tiles of catalogue.gpkg (same logic as kuro_flood_clusters).
Excluded: 118 (2015-02, before S2A) and 130 (2015-08, max cover 0.06) are unrecoverable.
Output: Drive 'Google Earth Engine', KURR_<region>_S1/S2.tif
Usage: --submit / --status / --cancel
"""
import sys, sqlite3, struct, time
import datetime as dt
import numpy as np
import pandas as pd
import ee
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"
R = "/home/kk/jp_claude/RMSD"
CLOUD_PROB = 40
MIN_PFLOOD = 5.0
CELL_DEG, BUF_DEG, MAX_SPAN_DEG = 0.25, 0.02, 0.6
CAP_KM2_PER_EVENT = 8000
WINDOWS = [7, 12, 20, 30]
TASKS_CSV = R + "/processed/gee_kuro_recover_tasks.csv"
SKIP_CSV = R + "/processed/gee_kuro_recover_skipped.csv"
ee.Initialize(project=PROJECT)

# --- recoverable events + smallest window ---
rec = pd.read_csv(R + "/processed/gee_kuro_recover.csv")
ev_win = {}
for _, r in rec.iterrows():
    hit = [w for w in WINDOWS if r[f"cover{w}"] >= 0.30]
    if hit:
        ev_win[int(r.actid)] = min(hit)
print(f"recoverable events ({len(ev_win)}): {ev_win}", flush=True)

# --- pflood tiles from the gpkg -> rectangles (same as clusters) ---
con = sqlite3.connect(R + "/data/KuroSiwo/catalogue.gpkg"); cur = con.cursor()
cur.execute("SELECT srs_id FROM gpkg_geometry_columns WHERE table_name='catalogue'")
srs = cur.fetchone()[0]
cur.execute("SELECT definition, organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs,))
de, org, oid = cur.fetchone()
gcrs = CRS.from_string(f"{org}:{oid}") if oid and oid > 0 else CRS.from_wkt(de)


def env(b):
    if b is None or b[:2] != b"GP":
        return None
    f = b[3]
    if ((f >> 1) & 7) == 0:
        return None
    e = "<" if f & 1 else ">"
    v = struct.unpack(f"{e}4d", b[8:40])
    return v[0], v[2], v[1], v[3]


q = ",".join(str(a) for a in sorted(ev_win))
cur.execute(f"SELECT actid, grid_id, pflood, flood_date, geom FROM catalogue "
            f"WHERE actid IN ({q}) AND pflood >= ?", (MIN_PFLOOD,))
seen, rows = set(), []
for actid, gid, pf, fd, blob in cur:
    key = (int(actid), gid)
    if key in seen:
        continue
    e = env(blob)
    if e is None:
        continue
    seen.add(key)
    rows.append((int(actid), gid, float(pf), str(fd)[:10]) + e)
con.close()
df = pd.DataFrame(rows, columns=["actid", "grid_id", "pflood", "date", "x0", "y0", "x1", "y1"])
print(f"flood tiles (pflood>={MIN_PFLOOD}%): {len(df)}", flush=True)

RE = 6378137.0
m2lon = lambda x: np.degrees(x / RE)
m2lat = lambda y: np.degrees(2 * np.arctan(np.exp(y / RE)) - np.pi / 2)
df["lon0"], df["lon1"] = m2lon(df.x0.values), m2lon(df.x1.values)
df["lat0"], df["lat1"] = m2lat(df.y0.values), m2lat(df.y1.values)
df["clon"], df["clat"] = (df.lon0 + df.lon1) / 2, (df.lat0 + df.lat1) / 2

regs = []
for actid, g in df.groupby("actid"):
    g = g.copy()
    g["ci"] = np.floor(g.clon / CELL_DEG).astype(int)
    g["cj"] = np.floor(g.clat / CELL_DEG).astype(int)
    for (ci, cj), gg in g.groupby(["ci", "cj"]):
        w, s = gg.lon0.min() - BUF_DEG, gg.lat0.min() - BUF_DEG
        e, n = gg.lon1.max() + BUF_DEG, gg.lat1.max() + BUF_DEG
        nx = max(1, int(np.ceil((e - w) / MAX_SPAN_DEG)))
        ny = max(1, int(np.ceil((n - s) / MAX_SPAN_DEG)))
        for ix in range(nx):
            for iy in range(ny):
                ww = w + (e - w) * ix / nx; ee_ = w + (e - w) * (ix + 1) / nx
                ss = s + (n - s) * iy / ny; nn = s + (n - s) * (iy + 1) / ny
                area = abs((ee_ - ww) * 111 * np.cos(np.radians((ss + nn) / 2)) * (nn - ss) * 111)
                regs.append(dict(actid=actid, date=gg.date.iloc[0],
                                 region=f"{actid}_c{ci}_{cj}_{ix}{iy}",
                                 lon_min=round(ww, 4), lat_min=round(ss, 4),
                                 lon_max=round(ee_, 4), lat_max=round(nn, 4),
                                 n_flood_tiles=int(len(gg)), area_km2=round(area, 1)))
out = pd.DataFrame(regs)
kept = []
for actid, g in out.groupby("actid"):
    g = g.sort_values("n_flood_tiles", ascending=False)
    cum, sel = 0.0, []
    for _, r in g.iterrows():
        if cum >= CAP_KM2_PER_EVENT:
            break
        sel.append(r); cum += r.area_km2
    kept.append(pd.DataFrame(sel))
REGS = pd.concat(kept, ignore_index=True).sort_values(["actid", "region"])
REGS.to_csv(R + "/processed/kuro_recover_regions.csv", index=False)
print(f"export regions: {len(REGS)}  total area {REGS.area_km2.sum():,.0f} km2", flush=True)


def composite_s2(aoi, t_flood, win):
    """Cloud-free composite within the window: mosaic farthest-from-flood-date first -> per-pixel nearest clear observation."""
    d0 = ee.Date((t_flood - dt.timedelta(days=win)).isoformat())
    d1 = ee.Date((t_flood + dt.timedelta(days=win + 1)).isoformat())
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(aoi).filterDate(d0, d1)
    pr = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(d0, d1)
    j = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
        s2, pr, ee.Filter.equals(leftField="system:index", rightField="system:index")))
    anchor = ee.Date(t_flood.isoformat())

    def prep(img):
        img = ee.Image(img)
        cp = ee.Image(img.get("cp")).select("probability")
        dd = ee.Number(img.date().difference(anchor, "day")).abs()
        return img.updateMask(cp.lt(CLOUD_PROB)).set("dd", dd)
    # sort farthest first (descending) -> mosaic prefers the last = nearest
    coll = j.map(prep).sort("dd", False)
    return coll.mosaic().select(["B2", "B3", "B4", "B8", "B11"]).toUint16(), j


def build(r):
    aoi = ee.Geometry.Rectangle([r.lon_min, r.lat_min, r.lon_max, r.lat_max], None, False)
    win = ev_win[int(r.actid)]
    t_flood = dt.datetime.fromisoformat(str(r.date)[:10] + "T00:00:00")
    s2, j = composite_s2(aoi, t_flood, win)
    # co-temporal S1 (around flood_date)
    w0 = ee.Date((t_flood - dt.timedelta(hours=18)).isoformat())
    w1 = ee.Date((t_flood + dt.timedelta(hours=42)).isoformat())
    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(w0, w1)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    if s1c.size().getInfo() == 0:
        return None, None, None, None, "no S1 at flood_date"
    ref = ee.Image(s1c.sort("system:time_start").first())
    s1c = (s1c.filter(ee.Filter.eq("relativeOrbitNumber_start", ref.get("relativeOrbitNumber_start")))
              .filter(ee.Filter.eq("orbitProperties_pass", ref.get("orbitProperties_pass"))))
    s1 = s1c.mosaic().select(["VV", "VH"]).toFloat()
    region = aoi.intersection(s1c.geometry(), ee.ErrorMargin(100))
    crs = ee.Image(j.first()).select("B3").projection().crs()
    return s1, s2, region, crs, f"ok (win +-{win}d)"


def op_states():
    return {str(o.get("name", "")).rsplit("/", 1)[-1]: o for o in ee.data.listOperations()}


def submit():
    out_tasks, skipped = [], []
    for _, r in REGS.iterrows():
        try:
            s1, s2, region, crs, msg = build(r)
        except Exception as ex:
            skipped.append(dict(region=r.region, reason=str(ex)[:80]))
            print(f"  {r.region}: FAIL {str(ex)[:60]}", flush=True); continue
        if s1 is None:
            skipped.append(dict(region=r.region, reason=msg))
            print(f"  {r.region}: SKIP {msg}", flush=True); continue
        for img, tag in [(s1, "S1"), (s2, "S2")]:
            t = ee.batch.Export.image.toDrive(
                image=img, description=f"KURR_{r.region}_{tag}", folder=FOLDER,
                fileNamePrefix=f"KURR_{r.region}_{tag}", region=region, scale=10, crs=crs,
                maxPixels=int(2e9), fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})
            t.start()
            out_tasks.append((f"KURR_{r.region}_{tag}", t.id))
        print(f"  {r.region}: {msg}", flush=True)
        time.sleep(0.4)
    pd.DataFrame(out_tasks, columns=["name", "task_id"]).to_csv(TASKS_CSV, index=False)
    pd.DataFrame(skipped).to_csv(SKIP_CSV, index=False)
    print(f"\nsubmitted {len(out_tasks)} tasks ({len(out_tasks)//2} tiles), "
          f"skipped {len(skipped)}", flush=True)


def status():
    df = pd.read_csv(TASKS_CSV); st = op_states()
    s = pd.DataFrame([dict(name=r["name"],
                           state=(st.get(r.task_id) or {}).get("metadata", {}).get("state", "?"))
                      for _, r in df.iterrows()])
    print(s.state.value_counts().to_dict(), flush=True)
    bad = s[~s.state.isin(["SUCCEEDED", "RUNNING", "PENDING"])]
    if len(bad):
        print(bad.to_string(index=False), flush=True)


def cancel():
    df = pd.read_csv(TASKS_CSV); st = op_states(); n = 0
    for _, r in df.iterrows():
        o = st.get(r.task_id)
        if o and o.get("metadata", {}).get("state") in ("PENDING", "RUNNING"):
            try:
                ee.data.cancelOperation(o["name"]); n += 1
            except Exception:
                pass
    print("cancelled", n, flush=True)


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--cancel" in sys.argv:
        cancel()
    elif "--submit" in sys.argv:
        submit()
    else:
        print("use --submit | --status | --cancel  (regions written on import)")
