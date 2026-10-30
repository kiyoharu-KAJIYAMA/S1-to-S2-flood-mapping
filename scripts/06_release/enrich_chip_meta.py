#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the per-chip metadata for release -> processed/trn16_chip_meta.csv (4875 rows).

For every trn16 chip, collect as far as possible
  - S1 / S2 acquisition dates (and their difference dt_days)
  - WGS84 bbox (chip-level where available, otherwise region-level)
  - source-dataset ID / event / country
from the respective primary sources and merge them into one CSV.

Sources:
  sen12       trn6_geo_index.csv date/bbox (chip-level)
  s1f11       Sen1Floods11_Metadata.geojson s1_date/s2_date per location + geo_index bbox
  c2s         acquisition datetime embedded in the C2S chip S1/S2 product names + geo_index bbox
  fp          FloodPlanet STAC item (<ID>.json) datetime/bbox
  kuros2      kuro_v9_export_regions.csv (region-level bbox, flood date); source tifs deleted
  gfd         gfd_regions_candidates.csv (date/country) + chip-level bbox from the GEE source tif
  kurr        kuro_recover_regions.csv    (date)         + chip-level bbox from the GEE source tif
  sturm       sturm_cdse_availability.csv (datetime/country/type) + chip-level bbox from the GEE source tif

Run: python3 scripts/06_release/enrich_chip_meta.py   (requires rasterio)
"""
import csv
import glob
import json
import os
import re
import collections

import rasterio as rio
from rasterio.warp import transform_bounds
from rasterio.windows import Window

R = os.environ.get("RMSD_ROOT", "/home/kk/jp_claude/RMSD")
P = f"{R}/processed"
D = f"{R}/data"
GEE = os.environ.get("RMSD_GEE", "/mnt/d/gdrive/Google Earth Engine")
SZ = 512

COLS = ["chip_index", "dataset", "scene_id", "chip_id", "region", "row", "col",
        "s1_date", "s2_date", "dt_days", "lon_min", "lat_min", "lon_max", "lat_max",
        "bbox_level", "crs_native", "country", "event",
        "valid_frac", "cloud_fraction", "water_frac",
        "y_valid_frac_measured", "water_frac_measured",
        "jrc_permanent_masked", "flood", "source_ref"]

# Whether JRC permanent water has been removed from teacher Y.
# kuros2 was built in the v9 round (build_trn9_chips.py) without JRC handling; TODO #12
# (build_trn9_jrc.py) added it afterwards. A jrc column in the component meta takes
# precedence; this table is used only when absent. If this disagrees with the code, the release lies.
JRC_APPLIED = {"sen12": True, "s1f11": True, "c2s": True, "fp": True,
               "kuros2": True, "gfd": True, "kurr": True, "sturm": True}

BASE_KEYS = {"SEN12": "sen12", "s1f11": "s1f11", "c2s": "c2s", "fp": "fp"}
EXT = {  # component -> (component meta, region csv, GEE prefix or None)
    # after TODO #12 use _trn9jrc_meta.csv (has chip-level bbox and a jrc column)
    "kuros2": ("_trn9jrc_meta.csv" if os.path.exists(f"{P}/_trn9jrc_meta.csv")
               else "_trn9new_meta.csv", "kuro_v9_export_regions.csv", None),
    "gfd": ("_gfd1new_meta.csv", "gfd_regions_candidates.csv", "GFD"),
    "kurr": ("_kurr2new_meta.csv", "kuro_recover_regions.csv", "KURR"),
    "sturm": ("_sturmnew_meta.csv", "sturm_cdse_availability.csv", "STB"),
}
DT_RE = re.compile(r"_(\d{8})T(\d{6})_")


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def blank():
    return {c: "" for c in COLS}


def days_between(a, b):
    """Difference in days between two 'YYYY-MM-DD' strings; '' if either is missing."""
    from datetime import date
    try:
        y1, m1, d1 = (int(x) for x in a[:10].split("-"))
        y2, m2, d2 = (int(x) for x in b[:10].split("-"))
        return abs((date(y1, m1, d1) - date(y2, m2, d2)).days)
    except Exception:
        return ""


# ------------------------------------------------------------------ primary-source loaders
def s1f11_dates():
    """location -> (s1_date, s2_date). Sen1Floods11_Metadata.geojson is per location."""
    p = f"{D}/Sen1Floods11/v1.1/Sen1Floods11_Metadata.geojson"
    out = {}
    if not os.path.exists(p):
        return out
    for f in json.load(open(p)).get("features", []):
        pr = f.get("properties", {})
        loc = str(pr.get("location", "")).strip()
        if loc:
            out[loc] = (str(pr.get("s1_date", "")).replace("/", "-"),
                        str(pr.get("s2_date", "")).replace("/", "-"))
    return out


def c2s_s2_date_by_uuid():
    """C2S uuid -> S2 acquisition date (one S2 product per uuid)."""
    out = {}
    base = f"{D}/C2SMSFloods/chips"
    if not os.path.isdir(base):
        return out
    for u in os.listdir(base):
        s2 = f"{base}/{u}/s2"
        if not os.path.isdir(s2):
            continue
        for name in os.listdir(s2):
            m = DT_RE.search(name)
            if m:
                out[u] = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
                break
    return out


def fp_items():
    """FloodPlanet: chip ID -> dict(s1_date, s2_date, bbox)."""
    out = collections.defaultdict(dict)
    for sensor in ("S1", "S2"):
        for p in glob.glob(f"{D}/FloodPlanet/stac_catalog/*/{sensor}/*/*.json"):
            try:
                j = json.load(open(p))
            except Exception:
                continue
            i = j.get("id")
            if not i:
                continue
            dt = str((j.get("properties") or {}).get("datetime", ""))[:10]
            out[i][f"{sensor.lower()}_date"] = dt
            if j.get("bbox") and "bbox" not in out[i]:
                out[i]["bbox"] = j["bbox"]
    return out


def sen12_pairs():
    """SEN12FLOOD: (folder, S2 date) -> paired S1 date, day_diff, etc."""
    p = f"{P}/trn2_matched_samples_good.csv"
    out = {}
    if not os.path.exists(p):
        return out
    for r in rd(p):
        out[(r["folder"], r["date"])] = r
    return out


def kuro_s2_dates():
    """Kuro: actid -> date of the S2 actually obtained ('' if none)."""
    out = {}
    p = f"{P}/gee_kuro_s2_availability2.csv"
    if os.path.exists(p):
        for r in rd(p):
            if r.get("best_date"):
                out[str(r["actid"])] = r["best_date"][:10]
    p = f"{P}/gee_kuro_recover.csv"
    if os.path.exists(p):
        for r in rd(p):
            a = str(r["actid"])
            if a in out:
                continue
            for c in ("best7", "best12", "best20", "best30"):
                if r.get(c):
                    out[a] = r[c][:10]
                    break
    return out


class TifGeo:
    """Chip-level bbox from the GEE source tif (opened once per region)."""

    def __init__(self):
        self.cache = {}

    def get(self, prefix, region):
        key = (prefix, region)
        if key not in self.cache:
            hit = glob.glob(f"{GEE}/{prefix}_{region}_S2.tif") or \
                  glob.glob(f"{GEE}/*_{region}_S2.tif")
            if not hit:
                self.cache[key] = None
            else:
                with rio.open(hit[0]) as s:
                    self.cache[key] = (s.transform, s.crs, s.width, s.height)
        return self.cache[key]

    def chip_bbox(self, prefix, region, row, col):
        g = self.get(prefix, region)
        if g is None:
            return None
        tr, crs, W, H = g
        try:
            r0, c0 = int(row), int(col)
        except (TypeError, ValueError):
            return None
        w, h = min(SZ, W - c0), min(SZ, H - r0)
        if w <= 0 or h <= 0:
            return None
        b = rio.windows.bounds(Window(c0, r0, w, h), tr)
        try:
            return transform_bounds(crs, "EPSG:4326", *b, densify_pts=21), str(crs)
        except Exception:
            return None


# ------------------------------------------------------------------ main
def main():
    comp = rd(f"{P}/trn16_component.csv")
    m16 = rd(f"{P}/trn16_matched_samples_good.csv")
    geo = rd(f"{P}/trn6_geo_index.csv")
    assert len(comp) == len(m16), f"{len(comp)} != {len(m16)}"

    s1f11d = s1f11_dates()
    c2sd = c2s_s2_date_by_uuid()
    fpd = fp_items()
    s12p = sen12_pairs()
    kd = kuro_s2_dates()
    tg = TifGeo()
    out = [None] * len(comp)

    # ---------- 4 base datasets ----------
    base_pos = [i for i, r in enumerate(m16) if r["source"] == "base_jrc"]
    assert len(base_pos) == len(geo)
    for k, i in enumerate(base_pos):
        g = geo[k]
        ds = BASE_KEYS[g["source"]]
        r = blank()
        r.update(chip_index=i, dataset=ds, scene_id=os.path.basename(g["folder"]),
                 chip_id=g["date"], lon_min=g["lon_min"], lat_min=g["lat_min"],
                 lon_max=g["lon_max"], lat_max=g["lat_max"], bbox_level="chip",
                 crs_native="EPSG:4326", jrc_permanent_masked="", flood="True",
                 source_ref=os.path.basename(g["geo_tif"]))
        if ds == "sen12":
            r["s2_date"] = g["date"]          # SEN12FLOOD: geo_index date is the actual date
            r["chip_id"] = os.path.basename(g["folder"])
            pr = s12p.get((g["folder"], g["date"]))
            if pr:                            # paired S1 already recorded (chosen from the time series)
                r["s1_date"] = pr.get("s1_date", "")
                r["cloud_fraction"] = pr.get("cloud_fraction", "")
                r["valid_frac"] = pr.get("optical_valid_fraction", "")
                r["flood"] = pr.get("flood", "True")
                r["source_ref"] = os.path.basename(pr.get("VV", "")) or r["source_ref"]
        elif ds == "s1f11":
            loc = str(g["folder"]).replace("s1f11_", "")
            d1, d2 = s1f11d.get(loc, ("", ""))
            r["s1_date"], r["s2_date"], r["event"] = d1, d2, loc
        elif ds == "c2s":
            m = DT_RE.search(str(g["geo_tif"]))
            if m:
                r["s1_date"] = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
            parts = str(g["geo_tif"]).split("/chips/")
            if len(parts) > 1:
                r["s2_date"] = c2sd.get(parts[1].split("/")[0], "")
            r["event"] = str(g["folder"]).replace("s1f11_", "")
        elif ds == "fp":
            it = fpd.get(g["date"], {})
            r["s1_date"], r["s2_date"] = it.get("s1_date", ""), it.get("s2_date", "")
            if it.get("bbox"):
                b = it["bbox"]
                r.update(lon_min=b[0], lat_min=b[1], lon_max=b[2], lat_max=b[3])
            r["event"] = str(g["folder"]).replace("fp_", "")
        r["dt_days"] = days_between(r["s1_date"], r["s2_date"])
        out[i] = r

    # ---------- 4 external datasets ----------
    for key, (mcsv, rcsv, pre) in EXT.items():
        pos = [i for i, c in enumerate(comp) if c["component"] == key]
        mm = rd(f"{P}/{mcsv}")
        assert len(pos) == len(mm), f"{key}: {len(pos)} != {len(mm)}"
        reg = rd(f"{P}/{rcsv}")
        if key == "sturm":
            rmap = {f"{x['event']}-{x['aoi']}": x for x in reg}
        else:
            rmap = {x["region"]: x for x in reg}
        for k, i in enumerate(pos):
            src = mm[k]
            g = rmap.get(src.get("region", ""), {})
            r = blank()
            r.update(chip_index=i, dataset=key, scene_id=src.get("folder", ""),
                     chip_id=src.get("folder", ""), region=src.get("region", ""),
                     row=src.get("row", ""), col=src.get("col", ""),
                     valid_frac=src.get("valid_frac", ""),
                     cloud_fraction=src.get("cloud_fraction", ""),
                     water_frac=src.get("water_frac", ""),
                     jrc_permanent_masked=src.get("jrc", ""),   # if empty, decided by JRC_APPLIED table
                     flood=src.get("flood", "True"),
                     event=src.get("event", src.get("actid", "")),
                     country=g.get("country", ""),
                     source_ref=f"{pre}_{src.get('region','')}_S2.tif" if pre else rcsv)
            if key in ("kuros2", "kurr"):
                # Kuro Siwo is SAR-based: region CSV date = flood-time S1 acquisition date.
                # The co-temporal S2 was searched in this project, so take it from the availability CSV.
                r["s1_date"] = g.get("date", "")
                r["s2_date"] = kd.get(str(r["event"]), "")
            elif key == "gfd":
                r["s2_date"] = g.get("best_s2_date", "")
                r["s1_date"] = g.get("s1_date", "")
            elif key == "sturm":
                r["s2_date"] = str(g.get("best_s2_time", ""))[:10]
                m = DT_RE.search(str(g.get("s1_example", "")))
                if m:
                    r["s1_date"] = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
                r["event"] = g.get("event", r["event"])
            r["dt_days"] = days_between(r["s1_date"], r["s2_date"])

            bb = tg.chip_bbox(pre, r["region"], r["row"], r["col"]) if pre else None
            if bb:
                (lo, la, hi, ha), crs = bb
                r.update(lon_min=round(lo, 6), lat_min=round(la, 6),
                         lon_max=round(hi, 6), lat_max=round(ha, 6),
                         bbox_level="chip", crs_native=crs)
            elif str(src.get("lon_min", "")).strip() != "":
                # component meta itself carries a chip-level bbox (output of build_trn9_jrc.py)
                r.update(lon_min=src["lon_min"], lat_min=src["lat_min"],
                         lon_max=src["lon_max"], lat_max=src["lat_max"],
                         bbox_level="chip", crs_native=src.get("crs_native", ""))
            elif g.get("lon_min", "") != "":
                r.update(lon_min=g["lon_min"], lat_min=g["lat_min"],
                         lon_max=g["lon_max"], lat_max=g["lat_max"],
                         bbox_level="region", crs_native="EPSG:4326")
            out[i] = r

    assert all(o is not None for o in out)

    # ---------- align the JRC flag with what was actually implemented ----------
    # Prefer the per-chip jrc column of the component meta when present (chips whose
    # region skipped JRC may be mixed in). Otherwise use the JRC_APPLIED table.
    for r in out:
        v = str(r.get("jrc_permanent_masked", "")).strip()
        if v in ("True", "False"):
            continue
        r["jrc_permanent_masked"] = str(JRC_APPLIED[r["dataset"]])

    # ---------- measured: valid-pixel and water fractions of teacher Y (all chips) ----------
    # build-time valid_frac / water_frac are not defined consistently across components,
    # so the release also carries values re-measured from the arrays, so recipients can filter on a common basis.
    import numpy as np
    Y = np.load(f"{P}/trn16_Y_MNDWI_NDVI_NDWI_raw.npy", mmap_mode="r")
    assert Y.shape[0] == len(out)
    print(f"measuring water fraction on {len(out)} chips ...", flush=True)
    for i in range(len(out)):
        m = np.isfinite(Y[i, 0])
        v = float(m.mean())
        w = float((Y[i, 0][m] > 0).mean()) if m.any() else 0.0
        out[i]["y_valid_frac_measured"] = round(v, 4)
        out[i]["water_frac_measured"] = round(w, 4)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(out)}", flush=True)

    dst = f"{P}/trn16_chip_meta.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)

    # ---------- coverage report ----------
    by = collections.defaultdict(list)
    for r in out:
        by[r["dataset"]].append(r)
    print(f"wrote {dst}  ({len(out)} rows)\n")
    print(f"{'dataset':9} {'chips':>5} {'s1_date':>8} {'s2_date':>8} {'bbox':>6} {'level':>7}")
    for k in ["sen12", "s1f11", "c2s", "fp", "kuros2", "gfd", "kurr", "sturm"]:
        rows = by[k]
        n = len(rows)
        f1 = sum(1 for r in rows if r["s1_date"])
        f2 = sum(1 for r in rows if r["s2_date"])
        bb = sum(1 for r in rows if str(r["lon_min"]) != "")
        lv = collections.Counter(r["bbox_level"] for r in rows).most_common(1)
        print(f"{k:9} {n:5} {f1:8} {f2:8} {bb:6} {(lv[0][0] if lv else '-'):>7}")
    dts = [int(r["dt_days"]) for r in out if str(r["dt_days"]).isdigit()]
    if dts:
        print(f"\nS1-S2 day gap dt_days: n={len(dts)}  median={sorted(dts)[len(dts)//2]}  max={max(dts)}")

    print(f"\n{'dataset':9} {'chips':>5} {'JRC':>6} {'water<1%':>9} {'water median':>11}")
    for k in ["sen12", "s1f11", "c2s", "fp", "kuros2", "gfd", "kurr", "sturm"]:
        rows = by[k]
        w = sorted(float(r["water_frac_measured"]) for r in rows)
        lo = sum(1 for x in w if x < 0.01)
        # report the values actually written, not the fixed table (a mismatch would misreport)
        jv = collections.Counter(r["jrc_permanent_masked"] for r in rows)
        js = "True" if list(jv) == ["True"] else ("False" if list(jv) == ["False"] else str(dict(jv)))
        print(f"{k:9} {len(rows):5} {js:>6} {lo:9} {100*w[len(w)//2]:10.2f}%")


if __name__ == "__main__":
    main()
