#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9: export raw S1/S2 bands from GEE for the 77 rectangles around Kuro Siwo flood areas.
Recipe identical to v8 (gee_export_sturm_aois.py).

  S1: COPERNICUS/S1_GRD (dB) VV,VH float32. flood_date +-18h; mosaic only scenes with the
      same relativeOrbit / orbitPass as the scene closest to the record time (avoids seams)
  S2: COPERNICUS/S2_HARMONIZED (L1C) B2,B3,B4,B8,B11 uint16. All scenes of the best day
      per event, masked with s2cloudless (prob>40), then mosaicked
  Region: rectangle from kuro_v9_export_regions.csv intersected with the S1 footprint;
      projection = S2 UTM, 10 m

train_events only (none of the 18 held-out events).
Drive output folder: 'Google Earth Engine', file names KURO_<region>_S1/S2.tif
Usage: --submit / --status / --cancel
"""
import sys, time
import datetime as dt
import pandas as pd
import ee

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"
R = "/home/kk/jp_claude/RMSD"
CLOUD_PROB = 40
TASKS_CSV = R + "/processed/gee_kuro_v9_tasks.csv"
ee.Initialize(project=PROJECT)

regs = pd.read_csv(R + "/processed/kuro_v9_export_regions.csv")
av = pd.read_csv(R + "/processed/gee_kuro_s2_availability2.csv").set_index("actid")
print(f"regions: {len(regs)}  events: {regs.actid.nunique()}  "
      f"area {regs.area_km2.sum():,.0f} km2", flush=True)


def build(r):
    aoi = ee.Geometry.Rectangle([r.lon_min, r.lat_min, r.lon_max, r.lat_max], None, False)
    t1 = dt.datetime.fromisoformat(str(r.date)[:10] + "T00:00:00")
    w0 = ee.Date((t1 - dt.timedelta(hours=18)).isoformat())
    w1 = ee.Date((t1 + dt.timedelta(hours=42)).isoformat())   # make sure the flood day itself is included
    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi).filterDate(w0, w1)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    ref = ee.Image(s1c.sort("system:time_start").first())
    s1c = (s1c.filter(ee.Filter.eq("relativeOrbitNumber_start",
                                   ref.get("relativeOrbitNumber_start")))
              .filter(ee.Filter.eq("orbitProperties_pass", ref.get("orbitProperties_pass"))))
    s1 = s1c.mosaic().select(["VV", "VH"]).toFloat()

    bd = str(av.loc[r.actid, "best_date"])
    d0 = ee.Date(bd)
    s2c = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterBounds(aoi).filterDate(d0, d0.advance(1, "day")))
    prob = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(aoi).filterDate(d0, d0.advance(1, "day")))
    joined = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
        s2c, prob, ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def mask_cloud(img):
        cp = ee.Image(img.get("cp")).select("probability")
        return img.updateMask(cp.lt(CLOUD_PROB))

    s2 = joined.map(mask_cloud).mosaic().select(["B2", "B3", "B4", "B8", "B11"]).toUint16()
    region = aoi.intersection(s1c.geometry(), ee.ErrorMargin(100))
    crs = ee.Image(s2c.first()).select("B3").projection().crs()
    return s1, s2, region, crs


def submit():
    tasks = []
    for _, r in regs.iterrows():
        try:
            s1, s2, region, crs = build(r)
        except Exception as ex:
            print("BUILD FAIL", r.region, str(ex)[:80], flush=True); continue
        for img, tag in [(s1, "S1"), (s2, "S2")]:
            t = ee.batch.Export.image.toDrive(
                image=img, description=f"KURO_{r.region}_{tag}", folder=FOLDER,
                fileNamePrefix=f"KURO_{r.region}_{tag}", region=region, scale=10, crs=crs,
                maxPixels=int(2e9), fileFormat="GeoTIFF",
                formatOptions={"cloudOptimized": True})
            t.start()
            tasks.append((f"KURO_{r.region}_{tag}", t.id))
        print("submitted", r.region, flush=True)
        time.sleep(0.5)
    pd.DataFrame(tasks, columns=["name", "task_id"]).to_csv(TASKS_CSV, index=False)
    print(f"\n{len(tasks)} tasks submitted", flush=True)


def op_states():
    return {str(o.get("name", "")).rsplit("/", 1)[-1]: o for o in ee.data.listOperations()}


def status():
    df = pd.read_csv(TASKS_CSV); st = op_states()
    rows = []
    for _, r in df.iterrows():
        md = (st.get(r.task_id) or {}).get("metadata", {})
        rows.append(dict(name=r["name"], state=md.get("state", "?"),
                         progress=round(float(md.get("progress", 0) or 0) * 100)))
    s = pd.DataFrame(rows)
    print(s.state.value_counts().to_dict())
    bad = s[~s.state.isin(["SUCCEEDED", "RUNNING", "PENDING"])]
    if len(bad):
        print(bad.to_string(index=False))


def cancel():
    df = pd.read_csv(TASKS_CSV); st = op_states(); n = 0
    for _, r in df.iterrows():
        o = st.get(r.task_id)
        if o and o.get("metadata", {}).get("state") in ("PENDING", "RUNNING"):
            try:
                ee.data.cancelOperation(o["name"]); n += 1
            except Exception:
                pass
    print("cancelled", n)


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--cancel" in sys.argv:
        cancel()
    elif "--submit" in sys.argv:
        submit()
    else:
        print("use --submit | --status | --cancel")
