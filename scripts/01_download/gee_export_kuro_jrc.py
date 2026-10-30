#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO #12: export to retro-fit a JRC permanent-water mask onto _trn9new (release 05_KuroSiwo_S2).

Background: _trn9new was built in the v9 round and build_trn9_chips.py has no JRC handling.
      The other 7 components remove permanent water from teacher Y, so only 05 is defined differently.
      The KURO_<region>_S2.tif used for chipping were deleted, so only the JRC mask is
      re-exported on the identical grid.

Do NOT guess the grid:
  import build() from gee_export_kuro_v9.py as is and reuse the region and crs it returns,
  plus scale=10. Only the image is swapped for JRC/GSW1_4 occurrence>=80, so the
  output raster lands on the same grid as the original KURO_<region>_S2.tif (by construction).
  The S2 image itself is not downloaded; a single uint8 band only.

Output: KUROJRC_<region>.tif (54 regions) in Drive 'Google Earth Engine'
Usage: --submit / --status / --cancel
Run:   /usr/bin/python3.10 (the interpreter with earthengine-api installed)
"""
import sys
import time

import pandas as pd
import ee

sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/01_download")

PROJECT = "rmsd-503302"
FOLDER = "Google Earth Engine"
R = "/home/kk/jp_claude/RMSD"
OCC = 80                       # same permanent-water definition as the other components (gee_export_jrc.py)
TASKS_CSV = R + "/processed/gee_kuro_jrc_tasks.csv"

ee.Initialize(project=PROJECT)

# Reuse build(). gee_export_kuro_v9 runs ee.Initialize and loads the regions
# at import time, but the only side effect is print, so this is safe.
import gee_export_kuro_v9 as V9          # noqa: E402

regs = V9.regs
perm = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        .gte(OCC).unmask(0).toUint8().rename("perm"))
print(f"KUROJRC targets: {len(regs)} regions (OCC>={OCC})", flush=True)


def submit():
    tasks, fails = [], []
    for _, r in regs.iterrows():
        try:
            _s1, _s2, region, crs = V9.build(r)     # same region / crs as the original export
        except Exception as ex:
            print("BUILD FAIL", r.region, str(ex)[:90], flush=True)
            fails.append(r.region)
            continue
        t = ee.batch.Export.image.toDrive(
            image=perm, description=f"KUROJRC_{r.region}", folder=FOLDER,
            fileNamePrefix=f"KUROJRC_{r.region}", region=region, scale=10, crs=crs,
            maxPixels=int(2e9), fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True})
        t.start()
        tasks.append((f"KUROJRC_{r.region}", t.id))
        print("submitted", r.region, flush=True)
        time.sleep(0.4)
    pd.DataFrame(tasks, columns=["name", "task_id"]).to_csv(TASKS_CSV, index=False)
    print(f"\n{len(tasks)} tasks submitted, {len(fails)} build failures", flush=True)
    if fails:
        print("FAILED regions:", fails, flush=True)


def op_states():
    return {str(o.get("name", "")).rsplit("/", 1)[-1]: o for o in ee.data.listOperations()}


def status():
    df = pd.read_csv(TASKS_CSV)
    st = op_states()
    rows = []
    for _, r in df.iterrows():
        md = (st.get(r.task_id) or {}).get("metadata", {})
        rows.append(dict(name=r["name"], state=md.get("state", "?"),
                         progress=round(float(md.get("progress", 0) or 0) * 100)))
    s = pd.DataFrame(rows)
    print(s.state.value_counts().to_dict(), flush=True)
    bad = s[~s.state.isin(["SUCCEEDED", "RUNNING", "PENDING"])]
    if len(bad):
        print(bad.to_string(index=False), flush=True)


def cancel():
    df = pd.read_csv(TASKS_CSV)
    st = op_states()
    n = 0
    for _, r in df.iterrows():
        o = st.get(r.task_id)
        if o and o.get("metadata", {}).get("state") in ("PENDING", "RUNNING"):
            try:
                ee.data.cancelOperation(o["name"])
                n += 1
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
        print("use --submit | --status | --cancel")
