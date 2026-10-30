"""Test A (Sen1Floods11 held-out events) by ESA WorldCover 2020 class, as landcover_iou.py does for Test B:
IoU of the threshold and the translation, share of the missed water in each class and how much of it is recovered.
WorldCover is sampled onto the chip grids from the public COGs and cached. Run with /opt/miniconda3/bin/python."""
import csv, json, math, os
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

R = "/home/kk/jp_claude/RMSD/"; P = R + "processed/"; O = R + "output/water_states/"
HL = R + "data/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S1Hand/"
URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_%s_Map.tif"
ch = list(csv.DictReader(open(P + "s1f11_chips.csv")))
rows = [i for i, r in enumerate(ch) if r["event"] in {"India", "Somalia", "Spain", "Bolivia"}]
CACHE = P + "s1f11_testA_worldcover_2020.npy"


def tile(lon, lat):
    a, b = int(math.floor(lat / 3) * 3), int(math.floor(lon / 3) * 3)
    return ("N%02d" % a if a >= 0 else "S%02d" % -a) + ("E%03d" % b if b >= 0 else "W%03d" % -b)


if not os.path.exists(CACHE):
    WC = np.zeros((len(rows), 512, 512), np.uint8)
    for k, i in enumerate(rows):
        with rasterio.open(HL + ch[i]["chip"] + "_S1Hand.tif") as ref:
            t, crs, (h, w) = ref.transform, ref.crs, ref.shape
            w_, s_, e_, n_ = rasterio.warp.transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
        for tl in sorted({tile(lo, la) for lo in (w_, e_) for la in (s_, n_)}):
            part = np.zeros((h, w), np.uint8)
            with rasterio.open("/vsicurl/" + URL % tl) as src:
                reproject(rasterio.band(src, 1), part, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=t, dst_crs=crs, resampling=Resampling.nearest)
            WC[k, :h, :w] = np.where(WC[k, :h, :w] == 0, part, WC[k, :h, :w])
        if k % 20 == 0:
            print("worldcover %d/%d" % (k + 1, len(rows)), flush=True)
    np.save(CACHE, WC)
WC = np.load(CACHE)

lab = np.asarray(np.load(P + "s1f11_label.npy", mmap_mode="r")[rows])
jrc = np.asarray(np.load(P + "s1f11_jrc.npy", mmap_mode="r")[rows]) > 0
YO = np.asarray(np.load(P + "s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])
OV = np.asarray(np.load(P + "s1f11_Yobs_valid.npy", mmap_mode="r")[rows], bool)
z = np.load(P + "s1f11_lee_db.npz"); VV = z["VV"][rows]; VH = z["VH"][rows]
U = np.load(O + "testA_translation_votes.npy").astype(bool)
valid = (lab >= 0) & ~jrc & np.isfinite(VH) & np.isfinite(VV)          # SAR-only pixel set, as in landcover_iou.py
gt = (lab == 1) & valid; rule = (VV < -5) & (VH < -22) & valid; Uv = U & valid
miss = gt & ~rule
NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse",
         90: "Herbaceous wetland", 95: "Mangroves"}


def iou(p, m):
    tp = (p & gt & m).sum(); return tp / max((p & m).sum() + (gt & m & ~p).sum(), 1)


print("\n%-19s %9s %6s | %6s %6s %6s | %7s %9s %9s" % ("WorldCover 2020", "px", "water", "rule", "transl", "gain",
                                                    "missed", "recovered", "unrec.sh"))
out = []
for k, nm in NAMES.items():
    m = (WC == k) & valid
    if (gt & m).sum() < 2000:
        continue
    r = dict(cls=nm, px=int(m.sum()), water=float(gt[m].mean()), iou_rule=float(iou(rule, m)), iou_transl=float(iou(Uv, m)),
             missed_share=float((miss & m).sum() / miss.sum()), recovered=float(Uv[miss & m].mean()),
             unrec_share=float((miss & ~Uv & m).sum() / (miss & ~Uv).sum()))
    out.append(r)
    print("%-19s %9d %5.1f%% | %6.3f %6.3f %+6.3f | %6.1f%% %8.1f%% %8.1f%%" % (
        nm, r["px"], 100 * r["water"], r["iou_rule"], r["iou_transl"], r["iou_transl"] - r["iou_rule"],
        100 * r["missed_share"], 100 * r["recovered"], 100 * r["unrec_share"]))
print("%-19s %9d %5.1f%% | %6.3f %6.3f %+6.3f" % ("All", valid.sum(), 100 * gt[valid].mean(), iou(rule, valid), iou(Uv, valid),
                                                 iou(Uv, valid) - iou(rule, valid)))
json.dump(out, open(O + "testA_landcover.json", "w"), indent=1)
