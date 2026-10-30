"""Leave-one-country-out (LOCO) folds over the 11 Sen1Floods11 events.

Candidate pool = every trn16 chip except GFD (all splits: train, holdout_A, holdout_B, excluded) plus the 105
Sen1Floods11 hand-labeled chips that are not in trn16 (the 90 official-test chips and the 15 Bolivia chips), taken from
the s1f11_* evaluation arrays. Chips are linked into flood clusters with the rule of flood_event_clusters.py (centres
within 50 km and Sentinel-1 dates within 15 days). For each fold, the held-out country's chips are the test set and
every chip in a cluster that contains one of them is excluded from training.

Writes experiments/loco_pool.csv (one row per candidate chip with its role in each fold) and prints fold summaries."""
import collections, csv, os, sys
from datetime import date
import numpy as np
import rasterio as rio

R = "/home/kk/jp_claude/RMSD"; P = R + "/processed"
sys.path.insert(0, R + "/scripts/04_eval")
from flood_event_clusters import cluster, parse_date, MERGE  # noqa: E402

S2 = R + "/data/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
COUNTRIES = ["Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"]

cm = {r["chip_index"]: r for r in csv.DictReader(open(P + "/trn16_chip_meta.csv"))}
split = list(csv.DictReader(open(P + "/trn16_split.csv")))
nodes = []
for r in split:
    m = cm[r["idx"]]
    if m["dataset"] == "gfd":
        continue
    lo, la, hi, ha = (float(m[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max"))
    nodes.append(dict(source="trn16", index=int(r["idx"]), ds=MERGE.get(m["dataset"], m["dataset"]), dataset=m["dataset"],
                      event=m["event"], split=r["split"], lon=(lo + hi) / 2, lat=(la + ha) / 2,
                      date=parse_date(m["s1_date"]), water=float(m["water_frac_measured"] or 0),
                      yvalid=float(m["y_valid_frac_measured"] or 0), chip=m["chip_id"]))
in_pool = {n["chip"] for n in nodes if n["dataset"] == "s1f11"}
ev_date = collections.defaultdict(list)
for n in nodes:
    if n["dataset"] == "s1f11" and n["date"]:
        ev_date[n["event"]].append(n["date"])
ev_date = {e: sorted(d)[len(d) // 2] for e, d in ev_date.items()}
ev_date.setdefault("Bolivia", date(2018, 2, 15))     # the FloodPlanet Bolivia chips linked to this event carry this date

ch = list(csv.DictReader(open(P + "/s1f11_chips.csv")))
Y = np.load(P + "/s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")
J = np.load(P + "/s1f11_jrc.npy", mmap_mode="r")
X = np.load(P + "/s1f11_X_VV_VH_RVI_raw.npy", mmap_mode="r")
extra = 0
for i, r in enumerate(ch):
    if r["chip"] in in_pool:
        continue
    with rio.open(S2 + r["chip"] + "_S2Hand.tif") as src:
        b = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            b = rio.coords.BoundingBox(*transform_bounds(src.crs, "EPSG:4326", *b))
    y = np.asarray(Y[i, 0], np.float64).copy(); y[np.asarray(J[i]) > 0] = np.nan
    fin = np.isfinite(y)
    vv = np.asarray(X[i, 0])
    nodes.append(dict(source="s1f11", index=i, ds="s1f11", dataset="s1f11", event=r["event"], split="official_" + r["split"],
                      lon=(b.left + b.right) / 2, lat=(b.bottom + b.top) / 2, date=ev_date.get(r["event"]),
                      water=float((y[fin] > 0).mean()) if fin.any() else 0.0,
                      yvalid=min(float(fin.mean()), float((np.isfinite(vv) & (vv > 0)).mean())), chip=r["chip"]))
    extra += 1
print("trn16 non-GFD chips:", len(nodes) - extra, "| extra Sen1Floods11 chips:", extra, flush=True)

root = cluster(nodes, 50.0, 15)
by_cluster = collections.defaultdict(list)
for i, rt in enumerate(root):
    by_cluster[rt].append(i)
country_clusters = {c: {root[i] for i, n in enumerate(nodes) if n["dataset"] == "s1f11" and n["event"] == c} for c in COUNTRIES}

passes = lambda n: n["water"] >= 0.005 and n["yvalid"] >= 0.2
rows = []
for i, n in enumerate(nodes):
    row = dict(node=i, source=n["source"], source_index=n["index"], dataset=n["dataset"], event=n["event"], chip=n["chip"],
               trn16_split=n["split"], cluster=root[i], water_frac=round(n["water"], 4), passes_filters=int(passes(n)))
    for c in COUNTRIES:
        if n["dataset"] == "s1f11" and n["event"] == c:
            row["fold_" + c] = "test"
        elif root[i] in country_clusters[c]:
            row["fold_" + c] = "excluded"
        else:
            row["fold_" + c] = "train"
    rows.append(row)
os.makedirs(R + "/experiments", exist_ok=True)
with open(R + "/experiments/loco_pool.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

DS = ["kuros2", "kurr", "c2s", "sturm", "s1f11", "fp", "sen12"]
print("\ncandidate pool by dataset (all / passing filters):")
for d in DS:
    a = [r for r in rows if r["dataset"] == d]
    print("  %-7s %5d / %5d" % (d, len(a), sum(r["passes_filters"] for r in a)))
print("\n%-10s %5s %6s %6s | %s" % ("fold", "test", "excl", "train", "excluded by dataset (passing filters)"))
for c in COUNTRIES:
    te = [r for r in rows if r["fold_" + c] == "test"]
    ex = [r for r in rows if r["fold_" + c] == "excluded" and r["passes_filters"]]
    tr = [r for r in rows if r["fold_" + c] == "train" and r["passes_filters"]]
    exd = collections.Counter(r["dataset"] for r in ex)
    print("%-10s %5d %6d %6d | %s" % (c, len(te), len(ex), len(tr), dict(exd)))
