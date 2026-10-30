#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""For the sensitivity (ablation) experiments: build an index assigning each trn16 row to one of 8 independent dataset components.

trn16 already exists on both local and server as the concatenated array
(base_jrc 1500 + kuro_s2 1055 + gfd 1343 + kurr 619 + sturm 358) = 4875. The base 1500 contains
4 datasets internally, so it is split further into 4 by the source column of trn6_geo_index.csv.

This way no huge per-component npy is needed (disk +0, transfer +0).
The trainer reads this CSV via --components and pulls only the matching rows of the trn16 array into RAM.

Output: processed/trn16_component.csv  (idx, component, scene)
  scene = "<component>/<folder basename>" — keeps the scene-level train/val split from
  mixing even if folder names collide between components.
Standard library only (no pandas).
"""
import csv
import os
import sys
import collections

P = os.environ.get("RMSD_PROCESSED", "/home/kk/jp_claude/RMSD/processed")

# trn16 source column -> component name (the base is split further below)
SRC2CMP = {
    "kuro_s2": "kuros2",
    "kuro_s2_jrc": "kuros2",          # tag after JRC application in TODO #12
    "gfd_dt1_jrc": "gfd",
    "kuro_recover_jrc": "kurr",
    "sturm_jrc": "sturm",
}
# trn6_geo_index.csv source column -> component name inside the base
BASE2CMP = {"SEN12": "sen12", "s1f11": "s1f11", "c2s": "c2s", "fp": "fp"}


def rd(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    m = rd(f"{P}/trn16_matched_samples_good.csv")
    g = rd(f"{P}/trn6_geo_index.csv")
    t6 = rd(f"{P}/trn6_matched_samples_good.csv")

    base_rows = [i for i, r in enumerate(m) if r["source"] == "base_jrc"]
    assert len(base_rows) == len(g) == len(t6), \
        f"base {len(base_rows)} != geo {len(g)} != trn6 {len(t6)}"
    # confirm via the folder column that the base is in the same order as trn6 (the split is meaningless otherwise)
    for k, i in enumerate(base_rows):
        assert m[i]["folder"] == t6[k]["folder"] == g[k]["folder"], f"row {i} folder mismatch"

    out = []
    bi = 0
    for i, r in enumerate(m):
        if r["source"] == "base_jrc":
            cmp_ = BASE2CMP[g[bi]["source"]]
            bi += 1
        else:
            cmp_ = SRC2CMP[r["source"]]
        scene = f"{cmp_}/{os.path.basename(str(r['folder']))}"
        out.append({"idx": i, "component": cmp_, "scene": scene})

    dst = f"{P}/trn16_component.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "component", "scene"])
        w.writeheader()
        w.writerows(out)

    cnt = collections.Counter(r["component"] for r in out)
    sc = collections.defaultdict(set)
    for r in out:
        sc[r["component"]].add(r["scene"])
    print(f"wrote {dst}: {len(out)} rows", flush=True)
    print(f"{'component':10} {'chips':>6} {'scenes':>7}")
    for c in ["sen12", "s1f11", "c2s", "fp", "kuros2", "gfd", "kurr", "sturm"]:
        print(f"{c:10} {cnt[c]:6} {len(sc[c]):7}")
    print(f"{'TOTAL':10} {sum(cnt.values()):6} {sum(len(v) for v in sc.values()):7}")
    assert sum(cnt.values()) == len(m) == 4875


if __name__ == "__main__":
    sys.exit(main())
