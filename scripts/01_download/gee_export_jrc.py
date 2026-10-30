#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v14 fix: export JRC permanent-water masks.
occurrence>=OCC% of JRC/GSW1_4/GlobalSurfaceWater is treated as permanent water; for each tile region
a uint8 mask with 1=permanent water is exported to Drive. Re-chipping sets the permanent-water pixels of teacher Y to NaN.

Target tiles:
  - GFD: gfd_regions_candidates.csv with keep==1 and dt_s1s2<=DT_MAX (strict Δt)  -> GFDJRC_<region>
  - KURR: all regions of kuro_recover_regions.csv                                  -> KURRJRC_<region>
Output: Drive 'Google Earth Engine', <PRE>JRC_<region>.tif
Usage: --submit / --status
"""
import sys, time
import pandas as pd
import ee

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"
R = "/home/kk/jp_claude/RMSD"
OCC = 80          # occurrence>=80% is treated as permanent water
DT_MAX = 1        # strict GFD Δt
TASKS_CSV = R + "/processed/gee_jrc_tasks.csv"
ee.Initialize(project=PROJECT)

gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
perm = gsw.gte(OCC).unmask(0).toUint8()   # 1=permanent water

# --- target region list ---
gfd = pd.read_csv(R + "/processed/gfd_regions_candidates.csv")
gfd = gfd[(gfd.keep == 1) & (gfd.dt_s1s2 <= DT_MAX)][
    ["region", "lon_min", "lat_min", "lon_max", "lat_max"]].copy()
gfd["pre"] = "GFDJRC"
kur = pd.read_csv(R + "/processed/kuro_recover_regions.csv")[
    ["region", "lon_min", "lat_min", "lon_max", "lat_max"]].copy()
kur["pre"] = "KURRJRC"
tiles = pd.concat([gfd, kur], ignore_index=True)
print(f"JRC targets: {len(gfd)} GFD(dt<={DT_MAX}) + {len(kur)} KURR = {len(tiles)}", flush=True)


def op_states():
    return {str(o.get("name", "")).rsplit("/", 1)[-1]: o for o in ee.data.listOperations()}


def submit():
    tasks = []
    for _, r in tiles.iterrows():
        region = ee.Geometry.Rectangle([r.lon_min, r.lat_min, r.lon_max, r.lat_max], None, False)
        name = f"{r.pre}_{r.region}"
        t = ee.batch.Export.image.toDrive(
            image=perm.clip(region), description=name, folder=FOLDER,
            fileNamePrefix=name, region=region, scale=30, crs="EPSG:3857",
            maxPixels=int(2e9), fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})
        t.start()
        tasks.append((name, t.id))
        if len(tasks) % 40 == 0:
            print(f"  submitted {len(tasks)}", flush=True)
        time.sleep(0.15)
    pd.DataFrame(tasks, columns=["name", "task_id"]).to_csv(TASKS_CSV, index=False)
    print(f"submitted {len(tasks)} JRC masks", flush=True)


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
