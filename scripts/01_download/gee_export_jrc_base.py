#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v16: export the JRC permanent-water mask for base (trn6, 1500).

Group the chip footprints in trn6_geo_index.csv (output of base_chip_geo.py) into 10deg tiles;
per tile, clip JRC/GSW1_4 occurrence>=80 to the union bbox (+buffer) of the member chips and
export it to Drive as BASEJRC_<tilekey>.tif. Same JRC criterion as v14/v15.
The warp is crs-independent (WarpedVRT reprojects onto the chip grid), so export in EPSG:4326.

Usage: --submit / --status
"""
import sys, time
import numpy as np
import pandas as pd
import ee

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"
R = "/home/kk/jp_claude/RMSD"
OCC = 80
BUF = 0.05  # deg, footprint margin
TASKS_CSV = R + "/processed/gee_jrc_base_tasks.csv"

ee.Initialize(project=PROJECT)
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
perm = gsw.gte(OCC).unmask(0).toUint8()


def tilekey(lon, lat):
    tx = int(np.floor(lon / 10.0) * 10)
    ty = int(np.ceil(lat / 10.0) * 10)
    ew = f"{abs(tx)}{'E' if tx >= 0 else 'W'}"
    ns = f"{abs(ty)}{'N' if ty >= 0 else 'S'}"
    return f"{ew}_{ns}"


def groups():
    g = pd.read_csv(R + "/processed/trn6_geo_index.csv")
    g = g[g.ok].copy()
    g["clon"] = (g.lon_min + g.lon_max) / 2
    g["clat"] = (g.lat_min + g.lat_max) / 2
    g["tile"] = [tilekey(lo, la) for lo, la in zip(g.clon, g.clat)]
    agg = g.groupby("tile").agg(lon_min=("lon_min", "min"), lat_min=("lat_min", "min"),
                                lon_max=("lon_max", "max"), lat_max=("lat_max", "max"),
                                n=("idx", "size")).reset_index()
    return agg


def submit():
    agg = groups()
    print(f"{len(agg)} JRC-base tiles (from {agg.n.sum()} chips)", flush=True)
    tasks = []
    for _, r in agg.iterrows():
        rect = ee.Geometry.Rectangle(
            [r.lon_min - BUF, r.lat_min - BUF, r.lon_max + BUF, r.lat_max + BUF], None, False)
        name = f"BASEJRC_{r.tile}"
        t = ee.batch.Export.image.toDrive(
            image=perm.clip(rect), description=name, folder=FOLDER,
            fileNamePrefix=name, region=rect, scale=30, crs="EPSG:4326",
            maxPixels=int(3e9), fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})
        t.start()
        tasks.append((name, t.id, int(r.n)))
        print(f"  submit {name} ({int(r.n)} chips)", flush=True)
        time.sleep(0.2)
    pd.DataFrame(tasks, columns=["name", "task_id", "n_chips"]).to_csv(TASKS_CSV, index=False)
    print(f"submitted {len(tasks)} BASEJRC tiles", flush=True)


def status():
    df = pd.read_csv(TASKS_CSV)
    ops = {str(o.get("name", "")).rsplit("/", 1)[-1]: o for o in ee.data.listOperations()}
    st = [(ops.get(r.task_id) or {}).get("metadata", {}).get("state", "?") for _, r in df.iterrows()]
    print(pd.Series(st).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--submit" in sys.argv:
        submit()
    else:
        print("use --submit | --status")
        print(groups().to_string())
