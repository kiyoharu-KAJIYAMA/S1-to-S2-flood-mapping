#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export raw bands from GEE to Drive for the 16 STURM-derived AOIs.

Policy: index computation and cloud masking should use the same code as the existing
      RMSD_preprocessing_v2, so GEE outputs the "raw bands as is".
  S1: VV,VH of COPERNICUS/S1_GRD (calibrated + terrain-corrected, dB) as float32
      -> mosaic only scenes with the same relativeOrbit/orbitPass as the scene closest to the recorded time (avoids seams)
  S2: B2,B3,B4,B8,B11 of COPERNICUS/S2_HARMONIZED (L1C) as uint16 (reflectance x10000)
      -> mosaic all scenes of the best day, remove s2cloudless prob>40 (removed pixels = 0 = existing nodata convention)
  Region: AOI rectangle ∩ S1 footprint (areas without S1 are unusable)
  CRS : UTM of S2, 10 m

Destination: Google Drive folder 'RMSD_GEE'
Usage: python3 gee_export_sturm_aois.py [--submit] [--status]
"""
import sys, time
import datetime as dt
import pandas as pd
import ee

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"   # existing folder on Drive (local sync target D:\gdrive\Google Earth Engine)
R = "/home/kk/jp_claude/RMSD"
CLOUD_PROB = 40
ee.Initialize(project=PROJECT)

cloud = pd.read_csv(R + "/processed/gee_aoi_cloud.csv")
scope = pd.read_csv(R + "/processed/sturm_redownload_scope.csv")
m = cloud.merge(scope[["event", "aoi", "s1_datetime", "s2_datetime",
                       "lon_min", "lat_min", "lon_max", "lat_max"]], on=["event", "aoi"])
use = m[(m.n_s1 > 0) & (m.aoi_cloud.notna()) & (m.aoi_cloud <= 0.30)].copy()
print(f"AOIs to export: {len(use)}")


def build(row):
    aoi = ee.Geometry.Rectangle([row.lon_min, row.lat_min, row.lon_max, row.lat_max])
    t1 = dt.datetime.fromisoformat(str(row.s1_datetime))
    w1 = (ee.Date((t1 - dt.timedelta(hours=18)).isoformat()),
          ee.Date((t1 + dt.timedelta(hours=18)).isoformat()))

    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi).filterDate(w1[0], w1[1])
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    # align to the orbit of the scene closest to the recorded time (avoids seams from mixed asc/desc or other orbits)
    ref = ee.Image(s1c.sort("system:time_start").first())
    s1c = (s1c.filter(ee.Filter.eq("relativeOrbitNumber_start",
                                   ref.get("relativeOrbitNumber_start")))
              .filter(ee.Filter.eq("orbitProperties_pass", ref.get("orbitProperties_pass"))))
    s1 = s1c.mosaic().select(["VV", "VH"]).toFloat()          # dB

    # S2: cloud-mask and mosaic all scenes from the same day as the best image
    best_day = ee.Date(str(row.s2_time).replace(" ", "T"))
    d0 = ee.Date.fromYMD(best_day.get("year"), best_day.get("month"), best_day.get("day"))
    s2c = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterBounds(aoi).filterDate(d0, d0.advance(1, "day")))
    prob = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(aoi).filterDate(d0, d0.advance(1, "day")))
    joined = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
        s2c, prob, ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def mask_cloud(img):
        cp = ee.Image(img.get("cp")).select("probability")
        return img.updateMask(cp.lt(CLOUD_PROB))

    s2 = (joined.map(mask_cloud).mosaic()
          .select(["B2", "B3", "B4", "B8", "B11"]).toUint16())

    region = aoi.intersection(s1c.geometry(), ee.ErrorMargin(100))
    crs = ee.Image(s2c.first()).select("B3").projection().crs()
    return s1, s2, region, crs


def op_states():
    st = {}
    for op in ee.data.listOperations():
        st[str(op.get("name", "")).rsplit("/", 1)[-1]] = op
    return st


def cancel():
    """Cancel unfinished (PENDING/RUNNING) tasks."""
    df = pd.read_csv(R + "/processed/gee_export_tasks.csv")
    st = op_states()
    n = 0
    for _, r in df.iterrows():
        op = st.get(r.task_id)
        if not op:
            continue
        state = op.get("metadata", {}).get("state", "?")
        if state in ("PENDING", "RUNNING"):
            try:
                ee.data.cancelOperation(op["name"])
                print("cancelled", r["name"], state, flush=True)
                n += 1
            except Exception as e:
                print("cancel failed", r["name"], str(e)[:80], flush=True)
        else:
            print("keep", r["name"], state, flush=True)
    print(f"cancelled {n} tasks")


def submit(skip=()):
    tasks = []
    for _, r in use.iterrows():
        name = f"{r.event}_{r.aoi}".replace("/", "-")
        if all(f"{name}_{g}" in skip for g in ("S1", "S2")):
            print("skip (already exported)", name, flush=True)
            continue
        s1, s2, region, crs = build(r)
        for img, tag, dtype in [(s1, "S1", "float"), (s2, "S2", "uint16")]:
            if f"{name}_{tag}" in skip:
                print("skip", name, tag, flush=True)
                continue
            t = ee.batch.Export.image.toDrive(
                image=img, description=f"RMSD_{name}_{tag}", folder=FOLDER,
                fileNamePrefix=f"{name}_{tag}", region=region, scale=10, crs=crs,
                maxPixels=int(2e9), fileFormat="GeoTIFF",
                formatOptions={"cloudOptimized": True})
            t.start()
            tasks.append((f"{name}_{tag}", t.id))
            print("submitted", name, tag, t.id, flush=True)
        time.sleep(1)
    pd.DataFrame(tasks, columns=["name", "task_id"]).to_csv(
        R + "/processed/gee_export_tasks.csv", index=False)
    print(f"\n{len(tasks)} tasks submitted to Drive folder '{FOLDER}'")


def status():
    df = pd.read_csv(R + "/processed/gee_export_tasks.csv")
    st = {}
    for op in ee.data.listOperations():
        # name has the form "projects/<p>/operations/<TASK_ID>"
        st[str(op.get("name", "")).rsplit("/", 1)[-1]] = op
    rows = []
    for _, r in df.iterrows():
        op = st.get(r.task_id) or {}
        md = op.get("metadata", {})
        err = op.get("error", {}).get("message", "")
        rows.append(dict(name=r["name"], state=md.get("state", "?"),
                         progress=round(float(md.get("progress", 0) or 0) * 100),
                         err=str(err)[:70]))
    s = pd.DataFrame(rows)
    print(s.to_string(index=False))
    print("\n", s.state.value_counts().to_dict())


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--cancel" in sys.argv:
        cancel()
    elif "--submit" in sys.argv:
        skip = set()
        if "--skip-done" in sys.argv:
            df = pd.read_csv(R + "/processed/gee_export_tasks.csv")
            st = op_states()
            for _, r in df.iterrows():
                op = st.get(r.task_id) or {}
                if op.get("metadata", {}).get("state") == "SUCCEEDED":
                    skip.add(r["name"])
            print("skipping already-succeeded:", sorted(skip))
        submit(skip)
    else:
        print("use --submit [--skip-done] | --status | --cancel")
