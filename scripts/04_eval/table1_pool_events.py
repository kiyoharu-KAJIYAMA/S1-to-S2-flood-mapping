"""Table I events column: flood events of the 2,735-chip paper pool as each dataset defines them.
Pool = loco_pool.csv rows with fold_Paper == train and passes_filters == 1, minus the trainer's SAR-validity drop
(fewer than 20% finite positive VV pixels), which reproduces gate3_paper.json exactly.
Event ids: Kuro Siwo actid (Copernicus EMS activation number, or Kuro Siwo's own 1111xxx id), STURM-Flood EMSR code,
Sen1Floods11 / C2S-MS / FloodPlanet event name, SEN12-FLOOD flooded region. Floods present in several datasets are
listed by place and date so that the total counts each flood once."""
import csv, datetime as dt
from collections import Counter, defaultdict
import numpy as np

R = "/home/kk/jp_claude/RMSD/"
meta = {int(r["chip_index"]): r for r in csv.DictReader(open(R + "processed/trn16_chip_meta.csv"))}
rows = [r for r in csv.DictReader(open(R + "experiments/loco_pool.csv")) if r["fold_Paper"] == "train" and r["passes_filters"] == "1"]
X = np.load(R + "processed/trn16_X_VV_VH_RVI_raw.npy", mmap_mode="r")
keep = []
for r in rows:
    if r["source"] == "trn16":
        vv = np.asarray(X[int(r["source_index"]), 0])
        if np.mean(np.isfinite(vv) & (vv > 0)) < 0.2:
            continue
    keep.append(r)
GROUP = {"kuros2": "Kuro Siwo", "kurr": "Kuro Siwo", "c2s": "C2S-MS Floods", "sturm": "STURM-Flood",
         "s1f11": "Sen1Floods11", "fp": "FloodPlanet", "sen12": "SEN12-FLOOD"}


def native(r):
    if r["source"] != "trn16":
        return r["event"]
    m = meta[int(r["source_index"])]
    if r["dataset"] == "sen12":
        return "Australia 2019-02" if float(m["lon_min"]) > 100 else "East Africa 2019-03"
    return (m["event"] or "").replace("c2s_", "").replace("fp_", "").replace("s1f11_", "")


per = defaultdict(Counter); box = {}
for r in keep:
    g = GROUP[r["dataset"]]; e = native(r)
    per[g][e] += 1
    if r["source"] == "trn16":
        m = meta[int(r["source_index"])]
        b = box.setdefault((g, e), dict(lon=[1e9, -1e9], lat=[1e9, -1e9], dates=set()))
        b["lon"] = [min(b["lon"][0], float(m["lon_min"])), max(b["lon"][1], float(m["lon_max"]))]
        b["lat"] = [min(b["lat"][0], float(m["lat_min"])), max(b["lat"][1], float(m["lat_max"]))]
        if m["s1_date"]:
            b["dates"].add(m["s1_date"])
print("pool chips", len(keep), Counter(GROUP[r["dataset"]] for r in keep))
for g in ("Kuro Siwo", "C2S-MS Floods", "STURM-Flood", "Sen1Floods11", "FloodPlanet", "SEN12-FLOOD"):
    print("%-14s events %2d  %s" % (g, len(per[g]), ", ".join("%s(%d)" % kv for kv in per[g].most_common())))


def same(a, b, pad=1.0, days=45):
    A, B = box[a], box[b]
    if A["lon"][0] - pad > B["lon"][1] or B["lon"][0] - pad > A["lon"][1]:
        return False
    if A["lat"][0] - pad > B["lat"][1] or B["lat"][0] - pad > A["lat"][1]:
        return False
    if not A["dates"] or not B["dates"]:
        return True
    da = [dt.date.fromisoformat(d) for d in A["dates"]]; db = [dt.date.fromisoformat(d) for d in B["dates"]]
    return min(abs((x - y).days) for x in da for y in db) <= days


# union-find over events that coincide in place and date across datasets
keys = sorted(box); parent = {k: k for k in keys}
find = lambda k: k if parent[k] == k else find(parent[k])
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        if a[0] != b[0] and same(a, b):
            parent[find(a)] = find(b)
groups = defaultdict(list)
for k in keys:
    groups[find(k)].append(k)
shared = [v for v in groups.values() if len(v) > 1]
total = sum(len(per[g]) for g in per); distinct = len(groups)
print("\nfloods present in several datasets (%d):" % len(shared))
for v in shared:
    print("  ", " = ".join("%s %s" % k for k in v))
print("per-dataset sum %d, distinct floods %d" % (total, distinct))
