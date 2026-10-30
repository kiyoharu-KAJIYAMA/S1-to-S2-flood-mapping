"""Build the training pools for REQUEST_terra_20261022_sameday_l1c_tau.md (§1-2).

train_variant.py selects training chips with
    component in <components> & split == "train" & water_frac_measured >= drop_low_water
and has no per-chip option, so each arm gets its own data directory
    output/sameday/pools/<arm>/trn16_{split}.csv          (rewritten: non-selected rows get split
                                                           "sameday_excluded", everything else kept)
    output/sameday/pools/<arm>/trn16_{component,chip_meta}.csv, trn16_{X,Y}_*.npy   (symlinks to the
                                                           originals -- row order is untouched)
    output/sameday/pools/<arm>/rows.csv                   (original trn16 indices of the selected chips)
Row alignment is therefore identical to the original trn16, and --datapath pointing here makes the
unmodified training script train on exactly the selected chips. Nothing under processed/ is touched.

Arms (counted before the script's degenerate-chip filter, which is applied at training time):
    SD  same-day chips (dt_days == 0)                                     848
    MX  per-component random sample of the SD counts from the full pool   848  (rng seed 0, fp = all)
    NC  full pool minus s1f11                                             2,599
    L1  full pool minus fp and sen12 (L2A)                                2,636
    R0  full pool                                                         2,795
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
P = f"{ROOT}/processed"
OUT = f"{ROOT}/output/sameday/pools"
COMP = ["sen12", "s1f11", "c2s", "fp", "kuros2", "kurr", "sturm"]
FILES = ["trn16_component.csv", "trn16_chip_meta.csv", "trn16_X_VV_VH_RVI_raw.npy", "trn16_Y_MNDWI_NDVI_NDWI_raw.npy"]
EXPECT = {"SD": 848, "MX": 848, "NC": 2599, "L1": 2636, "R0": 2795}


def main():
    ci = pd.read_csv(f"{P}/trn16_component.csv")
    sp = pd.read_csv(f"{P}/trn16_split.csv")
    cm = pd.read_csv(f"{P}/trn16_chip_meta.csv", low_memory=False)
    assert len(ci) == len(sp) == len(cm) and (ci["idx"].to_numpy() == np.arange(len(ci))).all()
    pool = (ci["component"].isin(COMP).to_numpy() & (sp["split"] == "train").to_numpy()
            & ~(cm["water_frac_measured"].to_numpy() < 0.005))
    dt = pd.to_numeric(cm["dt_days"], errors="coerce").to_numpy()
    comp = ci["component"].to_numpy()
    idx = np.arange(len(ci))

    sel = {}
    sel["R0"] = idx[pool]
    sel["SD"] = idx[pool & (dt == 0)]
    sel["NC"] = idx[pool & (comp != "s1f11")]
    sel["L1"] = idx[pool & ~np.isin(comp, ["fp", "sen12"])]
    rng = np.random.default_rng(0)
    mx = []
    for c in COMP:
        n = int(((comp == c) & pool & (dt == 0)).sum())
        cand = idx[pool & (comp == c)]
        if n == 0:
            continue
        mx.append(np.sort(rng.choice(cand, size=n, replace=False)))
    sel["MX"] = np.sort(np.concatenate(mx))

    summary = {}
    for arm in ["SD", "MX", "NC", "L1", "R0"]:
        rows = sel[arm]
        assert len(rows) == EXPECT[arm], (arm, len(rows), EXPECT[arm])
        d = f"{OUT}/{arm}"
        os.makedirs(d, exist_ok=True)
        for f in FILES:
            dst = f"{d}/{f}"
            if not os.path.lexists(dst):
                os.symlink(f"{P}/{f}", dst)
        s2 = sp.copy()
        keep = np.zeros(len(s2), bool)
        keep[rows] = True
        s2.loc[~keep, "split"] = "sameday_excluded"
        s2.to_csv(f"{d}/trn16_split.csv", index=False)
        pd.DataFrame({"row": rows, "component": comp[rows], "scene": ci["scene"].to_numpy()[rows],
                      "dt_days": dt[rows]}).to_csv(f"{d}/rows.csv", index=False)
        by = pd.Series(comp[rows]).value_counts().to_dict()
        chk = pd.read_csv(f"{d}/trn16_split.csv")
        n_train = int(((chk["split"] == "train") & ci["component"].isin(COMP) & ~(cm["water_frac_measured"] < 0.005)).sum())
        assert n_train == len(rows), (arm, n_train, len(rows))
        summary[arm] = {"n": int(len(rows)), "by_component": {k: int(v) for k, v in sorted(by.items())},
                        "scenes": int(pd.Series(ci["scene"].to_numpy()[rows]).nunique()),
                        "dt_days": {k: int(v) for k, v in pd.Series(np.where(np.isnan(dt[rows]), -1, dt[rows]))
                                    .map(lambda x: "0" if x == 0 else "1" if x == 1 else "2-3" if 0 < x <= 3 else ">3/NA")
                                    .value_counts().sort_index().items()}}
        print(f"{arm}: {len(rows)} chips, {summary[arm]['scenes']} scenes, {summary[arm]['by_component']}, dt {summary[arm]['dt_days']}")
    assert set(sel["MX"]) & set(sel["SD"]) >= set(idx[pool & (comp == "fp")])  # fp is all same-day
    summary["MX_overlap_with_SD"] = int(len(set(sel["MX"]) & set(sel["SD"])))
    json.dump(summary, open(f"{OUT}/pools_summary.json", "w"), indent=2)
    print("MX ∩ SD:", summary["MX_overlap_with_SD"], "->", f"{OUT}/pools_summary.json")


if __name__ == "__main__":
    main()
