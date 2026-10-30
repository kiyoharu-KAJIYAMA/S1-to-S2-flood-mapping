#!/usr/bin/env python3
"""Per-fold training-pool breakdown and dt_days distribution (request section 1,
last bullet, and section 7). Uses the pool table's passes_filters column, which
applies the same two training-time filters as the trainer.

Output: output/loco/pool_stats.json
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
OUT = f"{ROOT}/output/loco"
FOLDS = ["Official64", "USA", "India", "Paraguay", "Ghana", "Sri-Lanka", "Mekong", "Spain",
         "Pakistan", "Somalia", "Nigeria", "Bolivia"]


def q(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return dict(n=int(a.size), p25=round(float(np.percentile(a, 25)), 2),
                median=round(float(np.median(a)), 2), p75=round(float(np.percentile(a, 75)), 2),
                max=round(float(a.max()), 2), share_le_1d=round(float((a <= 1).mean()), 4),
                share_le_3d=round(float((a <= 3).mean()), 4))


def main():
    pool = pd.read_csv(f"{ROOT}/experiments/loco_pool.csv")
    meta = pd.read_csv(f"{OUT}/root/processed/trn18_chip_meta.csv", low_memory=False)
    dt = np.abs(meta["dt_days"].to_numpy(np.float64))
    res = {"note": "train rows passing both training filters (passes_filters); dt_days is "
                   "|S1 date - S2 date| from trn16_chip_meta; the 105 chips added from the "
                   "s1f11_* arrays carry no dates and are counted under dt_missing",
           "folds": {}}
    for f in FOLDS:
        keep = (pool[f"fold_{f}"] == "train").to_numpy() & (pool["passes_filters"] == 1).to_numpy()
        test = (pool[f"fold_{f}"] == "test").to_numpy()
        excl = (pool[f"fold_{f}"] == "excluded").to_numpy()
        by_ds = pool["dataset"][keep].value_counts().to_dict()
        d = dt[keep]
        res["folds"][f] = dict(
            train=int(keep.sum()), test=int(test.sum()), excluded=int(excl.sum()),
            excluded_by_dataset=pool["dataset"][excl].value_counts().to_dict(),
            train_by_dataset={k: int(v) for k, v in by_ds.items()},
            dt_days=q(d), dt_missing=int((~np.isfinite(d)).sum()),
            dt_days_by_dataset={ds: q(dt[keep & (pool["dataset"] == ds).to_numpy()])
                                for ds in sorted(by_ds)})
        s = res["folds"][f]
        print(f"{f:11} train {s['train']:>5} test {s['test']:>3} excl {s['excluded']:>3}  "
              f"dt median {s['dt_days']['median']} p75 {s['dt_days']['p75']} "
              f"(<=1d {100 * s['dt_days']['share_le_1d']:.0f}%)  dt missing {s['dt_missing']}",
              flush=True)
    json.dump(res, open(f"{OUT}/pool_stats.json", "w"), indent=2)
    print("saved", f"{OUT}/pool_stats.json", flush=True)


if __name__ == "__main__":
    main()
