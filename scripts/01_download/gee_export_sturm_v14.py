#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v15 preparation: re-fetch the 16 STURM AOIs from GEE and export the JRC permanent-water mask
in addition to S1/S2. Naming follows GFD/KURR (STURM_<region>_S1/S2, STURMJRC_<region>) so
build_extra_chips_v14 can chip them directly with JRC permanent-water exclusion.

Recipe identical to gee_export_sturm_aois.py (S1: flood time +-18h single-orbit mosaic / S2:
best-day cloud-masked mosaic). JRC: JRC/GSW1_4/GlobalSurfaceWater occurrence>=80% = permanent water.
Usage: --submit / --status
"""
import sys, time, re
import datetime as dt
import pandas as pd
import ee

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"
R = "/home/kk/jp_claude/RMSD"
CLOUD_PROB, OCC = 40, 80
TASKS_CSV = R + "/processed/gee_sturm_v14_tasks.csv"
ee.Initialize(project=PROJECT)

perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(OCC).unmask(0).toUint8()

cloud = pd.read_csv(R + "/processed/gee_aoi_cloud.csv")
scope = pd.read_csv(R + "/processed/sturm_redownload_scope.csv")
m = cloud.merge(scope[["event", "aoi", "s1_datetime", "s2_datetime",
                       "lon_min", "lat_min", "lon_max", "lat_max"]], on=["event", "aoi"])
use = m[(m.n_s1 > 0) & (m.aoi_cloud.notna()) & (m.aoi_cloud <= 0.30)].copy()
print(f"STURM AOIs to export: {len(use)}", flush=True)


def sanit(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")


def build(row):
    aoi = ee.Geometry.Rectangle([row.lon_min, row.lat_min, row.lon_max, row.lat_max])
    t1 = dt.datetime.fromisoformat(str(row.s1_datetime))
    w0 = ee.Date((t1 - dt.timedelta(hours=18)).isoformat())
    w1 = ee.Date((t1 + dt.timedelta(hours=18)).isoformat())
    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(w0, w1)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    ref = ee.Image(s1c.sort("system:time_start").first())
    s1c = (s1c.filter(ee.Filter.eq("relativeOrbitNumber_start", ref.get("relativeOrbitNumber_start")))
              .filter(ee.Filter.eq("orbitProperties_pass", ref.get("orbitProperties_pass"))))
    s1 = s1c.mosaic().select(["VV", "VH"]).toFloat()

    best_day = ee.Date(str(row.s2_time).replace(" ", "T"))
    d0 = ee.Date.fromYMD(best_day.get("year"), best_day.get("month"), best_day.get("day"))
    s2c = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(aoi).filterDate(d0, d0.advance(1, "day"))
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(d0, d0.advance(1, "day"))
    joined = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
        s2c, prob, ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def mc(img):
        cp = ee.Image(img.get("cp")).select("probability")
        return img.updateMask(cp.lt(CLOUD_PROB))
    s2 = joined.map(mc).mosaic().select(["B2", "B3", "B4", "B8", "B11"]).toUint16()
    region = aoi.intersection(s1c.geometry(), ee.ErrorMargin(100))
    crs = ee.Image(s2c.first()).select("B3").projection().crs()
    return s1, s2, region, crs


def op_states():
    return {str(o.get("name", "")).rsplit("/", 1)[-1]: o for o in ee.data.listOperations()}


def submit():
    tasks = []
    for _, r in use.iterrows():
        reg = sanit(f"{r.event}_{r.aoi}")   # event+aoi for uniqueness (aoi alone collides: AOI01/02)
        try:
            s1, s2, region, crs = build(r)
        except Exception as ex:
            print("BUILD FAIL", reg, str(ex)[:70], flush=True); continue
        exports = [(s1, f"STB_{reg}_S1", 10, crs),
                   (s2, f"STB_{reg}_S2", 10, crs),
                   (perm.clip(region), f"STBJRC_{reg}", 30, "EPSG:3857")]
        for img, name, scale, ecrs in exports:
            t = ee.batch.Export.image.toDrive(
                image=img, description=name, folder=FOLDER, fileNamePrefix=name,
                region=region, scale=scale, crs=ecrs, maxPixels=int(2e9),
                fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})
            t.start()
            tasks.append((name, t.id))
        print("submitted", reg, flush=True)
        time.sleep(0.5)
    pd.DataFrame(tasks, columns=["name", "task_id"]).to_csv(TASKS_CSV, index=False)
    print(f"\n{len(tasks)} tasks ({len(tasks)//3} AOIs x S1+S2+JRC) submitted", flush=True)


def status():
    df = pd.read_csv(TASKS_CSV); st = op_states()
    s = pd.Series([(st.get(r.task_id) or {}).get("metadata", {}).get("state", "?")
                   for _, r in df.iterrows()])
    print(s.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--submit" in sys.argv:
        submit()
    else:
        print("use --submit | --status")
