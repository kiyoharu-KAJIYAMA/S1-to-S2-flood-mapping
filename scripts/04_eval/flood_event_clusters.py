#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recount the 7 datasets at a common granularity — define "one flood" by spatial cluster.

Why
  Each dataset counts a different unit, so they cannot be compared.
    Kuro Siwo   … activation (actid). One activation splits into 1-7 AOIs
    STURM       … activation x AOI
    SEN12-FLOOD … AOI only (no activation information)
    FloodPlanet … region; Sen1Floods11 / C2S-MS / GFD … their own events
  Summing as is would make SEN12-FLOOD's 267 AOIs look like "267 floods".

Definition
  Take every ingested chip (trn16 pool 4,875 + axis C Kuro tiles) as a vertex, and
    (a) bbox centers within LINK_KM, and
    (b) S1 acquisition dates within LINK_DAYS
  link any pair satisfying both as an edge; connected components via Union-Find.
  One connected component = one flood event. Linking crosses datasets, so a flood
  held by several datasets collapses to one.

Output: output/flood_event_clusters/{clusters.csv,summary.md}
Run: /usr/bin/python3 (numpy only)
"""
import argparse
import collections
import csv
import json
import os
from datetime import date

import numpy as np

R = "/home/kk/jp_claude/RMSD"
P = f"{R}/processed"
OUT = f"{R}/output/flood_event_clusters"
MERGE = {"kuros2": "kuro", "kurr": "kuro"}
NAME = {"fp": "FloodPlanet", "s1f11": "Sen1Floods11", "kuro": "Kuro Siwo",
        "sturm": "STURM-Flood", "c2s": "C2S-MS", "sen12": "SEN12-FLOOD", "gfd": "GFD"}
ORDER = ["fp", "s1f11", "kuro", "sturm", "c2s", "sen12", "gfd"]
NATIVE_UNIT = {"fp": "regions 19", "s1f11": "events 11", "kuro": "activations 43",
               "sturm": "activations 60 / AOI 143", "c2s": "events 18",
               "sen12": "AOI 267", "gfd": "events 913"}


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def parse_date(s):
    try:
        y, m, d = s.split("-"); return date(int(y), int(m), int(d))
    except Exception:
        return None


def load_nodes():
    """List of (dataset, split, lon_c, lat_c, date, native_key)."""
    cm = {r["chip_index"]: r for r in csv.DictReader(open(f"{P}/trn16_chip_meta.csv"))}
    nodes = []
    for r in csv.DictReader(open(f"{P}/trn16_split.csv")):
        m = cm.get(r["idx"])
        if not m:
            continue
        try:
            lo, la = float(m["lon_min"]), float(m["lat_min"])
            hi, ha = float(m["lon_max"]), float(m["lat_max"])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite([lo, la, hi, ha])):
            continue
        ds = MERGE.get(m["dataset"], m["dataset"])
        nodes.append(dict(ds=ds, split=r["split"], lon=(lo + hi) / 2, lat=(la + ha) / 2,
                          date=parse_date(m["s1_date"]),
                          native=m["event"] or m["scene_id"]))
    # axis C Kuro tiles
    p = f"{P}/kuro_tiles_geo.csv"
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r["role"] != "test":
                continue
            nodes.append(dict(ds="kuro", split="axisC",
                              lon=(float(r["lon_min"]) + float(r["lon_max"])) / 2,
                              lat=(float(r["lat_min"]) + float(r["lat_max"])) / 2,
                              date=parse_date(r["flood_date"]), native=r["event"]))
    return nodes


def cluster(nodes, link_km, link_days):
    n = len(nodes)
    uf = UF(n)
    deg = link_km / 111.0
    grid = collections.defaultdict(list)
    for i, v in enumerate(nodes):
        grid[(int(v["lon"] / deg), int(v["lat"] / deg))].append(i)
    for (gx, gy), ix in grid.items():
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near += grid.get((gx + dx, gy + dy), [])
        for i in ix:
            a = nodes[i]
            for j in near:
                if j <= i:
                    continue
                b = nodes[j]
                if abs(a["lon"] - b["lon"]) > deg or abs(a["lat"] - b["lat"]) > deg:
                    continue
                dlon = (a["lon"] - b["lon"]) * np.cos(np.radians((a["lat"] + b["lat"]) / 2))
                if (dlon ** 2 + (a["lat"] - b["lat"]) ** 2) ** 0.5 > deg:
                    continue
                if a["date"] and b["date"] and abs((a["date"] - b["date"]).days) > link_days:
                    continue
                uf.union(i, j)
    return [uf.find(i) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link_km", type=float, default=50.0)
    ap.add_argument("--link_days", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    nodes = load_nodes()
    print("chips (incl. axis C tiles): %d" % len(nodes), flush=True)
    root = cluster(nodes, a.link_km, a.link_days)

    cl = collections.defaultdict(lambda: dict(ds=collections.Counter(),
                                              split=collections.Counter(),
                                              dates=[], native=collections.defaultdict(set)))
    for i, v in enumerate(nodes):
        c = cl[root[i]]
        c["ds"][v["ds"]] += 1
        c["split"][v["split"]] += 1
        c["native"][v["ds"]].add(v["native"])
        if v["date"]:
            c["dates"].append(v["date"])
    print("spatial clusters (= flood events): %d" % len(cl), flush=True)

    # per dataset: number of clusters it touches and number it holds alone
    touch = collections.Counter()
    solo = collections.Counter()
    shared = collections.Counter()
    for c in cl.values():
        ds = list(c["ds"])
        for d in ds:
            touch[d] += 1
            if len(ds) == 1:
                solo[d] += 1
            else:
                shared[d] += 1

    # clusters per dataset x split. A cluster containing both train and eval
    # is a leak candidate, so count it separately.
    ds_split = collections.defaultdict(collections.Counter)
    mixed = []            # clusters where train and eval coexist
    for k, c in cl.items():
        for i_ds in c["ds"]:
            pass
        # rebuild the (dataset, split) pairs contained in this cluster
        pass
    per = collections.defaultdict(lambda: collections.defaultdict(set))
    for i, v in enumerate(nodes):
        per[v["ds"]][v["split"]].add(root[i])
    TEST = ("holdout_A", "holdout_B", "axisC")
    for k, c in cl.items():
        s = set(c["split"])
        if "train" in s and (s & set(TEST)):
            mixed.append((k, dict(c["ds"]), dict(c["split"])))

    # "net gain" in priority order
    net = collections.Counter()
    taken = set()
    for d in ORDER:
        for k, c in cl.items():
            if d in c["ds"] and k not in taken:
                net[d] += 1
                taken.add(k)

    lines = ["# Flood event counts unified by spatial cluster", "",
             "Link condition: bbox centers within %.0f km and S1 acquisition dates within %d days."
             % (a.link_km, a.link_days),
             "One connected component is taken as one flood event. Linking crosses datasets,",
             "so a flood held by several datasets collapses to one.", "",
             "Total chips %d (incl. axis C tiles) -> **%d flood events**"
             % (len(nodes), len(cl)), "",
             "| Dataset | Native unit | Train | Axis A | Axis B | Axis C | Excluded | Touched | Solo | Net |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in ORDER:
        p = per[d]
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %d | %d | +%d |"
                     % (NAME[d], NATIVE_UNIT[d], len(p["train"]),
                        len(p["holdout_A"]) or "—", len(p["holdout_B"]) or "—",
                        len(p["axisC"]) or "—", len(p["excluded"]) or "—",
                        touch[d], solo[d], net[d]))
    lines.append("| **Total** | | %d | %d | %d | %d | %d | **%d** | %d | **%d** |"
                 % (sum(len(per[d]["train"]) for d in ORDER),
                    sum(len(per[d]["holdout_A"]) for d in ORDER),
                    sum(len(per[d]["holdout_B"]) for d in ORDER),
                    sum(len(per[d]["axisC"]) for d in ORDER),
                    sum(len(per[d]["excluded"]) for d in ORDER),
                    sum(touch.values()), sum(solo.values()), sum(net.values())))
    tr_cl = {k for k, c in cl.items() if c["split"]["train"]}
    te_cl = {k for k, c in cl.items()
             if any(c["split"][s] for s in ("holdout_A", "holdout_B", "axisC"))}
    lines += ["",
              "Events used for training %d / used for evaluation %d / appearing in both %d."
              % (len(tr_cl), len(te_cl), len(tr_cl & te_cl)),
              "",
              "\"Touched\" = number of clusters containing chips of that dataset (train and eval may double count).",
              "\"Solo\" = clusters held by that dataset only.",
              "\"Net\" = clusters not already taken earlier in priority order (%s). The total equals the number of events."
              % " > ".join(NAME[d] for d in ORDER), "",
              "## ★ Events where train and eval coexist (leak candidates) ★", ""]
    if not mixed:
        lines.append("None. Evaluation events are separated from training events both spatially and temporally.")
    else:
        lines.append("| # | Dataset | Split breakdown |")
        lines.append("|---|---|---|")
        for i, (k, dsc, spc) in enumerate(mixed, 1):
            lines.append("| %d | %s | %s |"
                         % (i, ", ".join("%s(%d)" % (NAME[d], n) for d, n in dsc.items()),
                            ", ".join("%s %d" % (s, n) for s, n in spc.items())))
    lines += ["", "## Events held by multiple datasets", ""]
    multi = [(k, c) for k, c in cl.items() if len(c["ds"]) > 1]
    multi.sort(key=lambda x: -sum(x[1]["ds"].values()))
    lines.append("| # | Dataset | Chips | Median date | Use |")
    lines.append("|---|---|---:|---|---|")
    for i, (k, c) in enumerate(multi, 1):
        dts = sorted(c["dates"])
        lines.append("| %d | %s | %d | %s | %s |"
                     % (i, ", ".join("%s(%d)" % (NAME[d], n) for d, n in c["ds"].most_common()),
                        sum(c["ds"].values()),
                        dts[len(dts) // 2].isoformat() if dts else "-",
                        ", ".join("%s %d" % (s, n) for s, n in c["split"].most_common())))
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")

    with open(f"{OUT}/clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n_chips", "datasets", "splits", "date_med", "native_keys"])
        for k, c in sorted(cl.items(), key=lambda x: -sum(x[1]["ds"].values())):
            dts = sorted(c["dates"])
            w.writerow([k, sum(c["ds"].values()),
                        "|".join("%s:%d" % (d, n) for d, n in c["ds"].most_common()),
                        "|".join("%s:%d" % (s, n) for s, n in c["split"].most_common()),
                        dts[len(dts) // 2].isoformat() if dts else "",
                        "|".join("%s=%s" % (d, ",".join(sorted(v))[:60])
                                 for d, v in c["native"].items())])
    print("\n".join(lines[:22]))
    print("\nsaved %s/summary.md" % OUT)


if __name__ == "__main__":
    main()
