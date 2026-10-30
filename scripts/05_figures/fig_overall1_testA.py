"""Results Fig. 2 with one chip: Sentinel-2 true colour, Sentinel-2 false colour (B12, B8, B4), Lee VV, Lee VH and the
outcome in the four classes of fig_overall4_testA.py. Run with /usr/bin/python3 (rasterio).
Usage: fig_overall1_testA.py [chip] [--mndwi]   (default: India_747992; --mndwi shows observed MNDWI instead)"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio as rio
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import TestA, true_colour, R, O
S2 = R + "data/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
META = R + "data/Sen1Floods11/v1.1/Sen1Floods11_Metadata.geojson"
T = TestA()
MNDWI = "--mndwi" in sys.argv
THREE = "--three" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
k = T.names.index(args[0] if args else "India_747992")
CLS = np.full(T.lab.shape[1:], -1, np.int8)
CLS[T.land[k]] = 0; CLS[T.water[k] & T.rule[k]] = 1; CLS[T.rec[k]] = 2; CLS[T.miss[k] & ~T.U[k]] = 3
DATES = {}
for f in json.load(open(META))["features"]:
    p = f["properties"]
    if p.get("location") and p["location"] not in DATES:
        DATES[p["location"]] = (p["s2_date"].replace("/", "-"), p["s1_date"].replace("/", "-"))

C_RULE, C_TR, C_MISS, C_LAND = "#2a78d6", "#1baf7a", "#8c8c8c", "#dcc9ad"
COLOR = [C_LAND, C_RULE, C_TR, C_MISS]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 36, "pdf.fonttype": 42})
FS, FL, FD = 38, 40, 30                                       # titles, panel letters, dates


def false_colour(path, bands=(13, 8, 4)):
    with rio.open(path) as src:
        x = np.dstack([src.read(b).astype(np.float32) for b in bands])
    ok = np.all(x > 0, axis=2)
    out = np.ones_like(x)
    for c in range(3):
        lo, hi = np.percentile(x[..., c][ok], (2, 98))
        out[..., c] = np.clip((x[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
    out[~ok] = 1.0
    return out


def tag(ax, text):
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=FD, ha="right", va="bottom",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))


img = np.ones(CLS.shape + (3,))
for j in range(4):
    img[CLS == j] = matplotlib.colors.to_rgb(COLOR[j])
path = S2 + T.names[k] + "_S2Hand.tif"
if MNDWI:
    import csv
    idx = [int(r["idx"]) for r in csv.DictReader(open(R + "processed/s1f11_chips.csv")) if r["chip"] == T.names[k]][0]
    mn = np.asarray(np.load(R + "processed/s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[idx, 0], np.float32)
    second = (np.ma.masked_invalid(mn), dict(cmap=matplotlib.colormaps["RdBu"].with_extremes(bad="white"), vmin=-0.6, vmax=0.6))
else:
    second = (false_colour(path), {})
panels = [(true_colour(path), {}), second,
          (T.VV[k], dict(cmap="gray", vmin=-20, vmax=-2)), (T.VH[k], dict(cmap="gray", vmin=-28, vmax=-10)), (img, {})]
titles = ["Sentinel-2 (R, G, B)", "Sentinel-2 MNDWI" if MNDWI else "Sentinel-2 (B12, B8, B4)", "Sentinel-1 VV",
          "Sentinel-1 VH", "Detection vs. label"]
d2, d1 = DATES[T.ev[k]]
src = [0, 3, 4] if THREE else range(5)                      # --three: true colour, VH and outcome only
panels = [panels[i] for i in src]; titles = [titles[i] for i in src]
n_s2 = 1 if THREE else 2

fig = plt.figure(figsize=(20, 8.6) if THREE else (33, 8.6))
gs = fig.add_gridspec(1, len(panels), wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.17)
for j, (a_img, kw) in enumerate(panels):
    a = fig.add_subplot(gs[0, j])
    a.imshow(a_img, interpolation="nearest", **kw)
    a.set_title(titles[j], fontsize=FS, pad=8)
    a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values():
        s_.set_visible(False)
    a.text(0.02, 0.98, "abcde"[j], transform=a.transAxes, fontsize=FL, fontweight="bold", ha="left", va="top",
           bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
    if j < n_s2:
        tag(a, d2)
    elif j < len(panels) - 1:
        tag(a, d1)
h = [plt.Line2D([], [], marker="s", ls="", ms=22, mfc=c, mec="none") for c in (C_RULE, C_TR, C_MISS, C_LAND)]
fig.legend(h, ["VH-22", "Recovered", "Unrecovered", "Land"], loc="lower center", ncol=4, frameon=False,
           fontsize=FS, handletextpad=0.4, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
for ext in ("png", "pdf"):
    tail = ("" if T.names[k] == "India_747992" else "_" + T.names[k]) + ("_mndwi" if MNDWI else "") + ("_three" if THREE else "")
    fig.savefig(O + "fig_overall1_testA%s.%s" % (tail, ext), dpi=150, bbox_inches="tight")
n = [(CLS == j).sum() for j in range(4)]
print(T.names[k], (d2, d1), "land %d, threshold %d, recovered %d, unrecovered %d" % tuple(n))
