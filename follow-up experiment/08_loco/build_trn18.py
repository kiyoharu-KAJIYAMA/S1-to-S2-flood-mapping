#!/usr/bin/env python3
"""Build the trn18 pool and the per-fold split directories (request 4.1, 4.2).

Row i of trn18 is row i of experiments/loco_pool.csv (node == i), so the node
table is the identity and every fold column maps straight onto trn18 rows.

Method A (gate 2 passed): every Sen1Floods11 chip comes from the s1f11_* arrays,
    with JRC > 0 and invalid observation pixels set to NaN in the teacher.
Method B: rows with source == trn16 are copied from trn16, only the 105 rows
    with source == s1f11 are taken from the s1f11_* arrays.

The arrays are written as new files (23 GB for X + Y): trn16 rows have to be
re-indexed and npy files cannot be appended, so links cannot express the pool.

Per fold, root_<Fold>/processed/ holds symlinks to the shared trn18 files plus
its own trn18_split.csv, because train_variant.py reads <prefix>_split.csv from
the data directory and must not be edited.

Output: output/loco/root/processed/trn18_*, output/loco/root_<Fold>/processed/
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
P = f"{ROOT}/processed"
OUT = f"{ROOT}/output/loco"
D = f"{OUT}/root/processed"
FOLDS = ["Official64", "USA", "India", "Paraguay", "Ghana", "Sri-Lanka", "Mekong", "Spain",
         "Pakistan", "Somalia", "Nigeria", "Bolivia"]
COMPS = ["sen12", "s1f11", "c2s", "fp", "kuros2", "kurr", "sturm"]


def water_frac(mndwi):
    f = np.isfinite(mndwi)
    return float((mndwi[f] > 0).mean()) if f.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="", choices=["", "A", "B"],
                    help="empty: take the verdict of gate.json")
    ap.add_argument("--scene_style", default="", help="scene for s1f11 rows added from the "
                    "s1f11_* arrays; empty copies the trn16 convention")
    args = ap.parse_args()
    method = args.method or json.load(open(f"{OUT}/gate.json"))["gate2"]["method"]
    print(f"[trn18] method {method}", flush=True)

    pool = pd.read_csv(f"{ROOT}/experiments/loco_pool.csv")
    assert (pool["node"].to_numpy() == np.arange(len(pool))).all(), "node must be 0..N-1"
    s1 = pd.read_csv(f"{P}/s1f11_chips.csv")
    s1_idx = {c: i for i, c in enumerate(s1["chip"].astype(str))}
    cm16 = pd.read_csv(f"{P}/trn16_chip_meta.csv", low_memory=False)
    ci16 = pd.read_csv(f"{P}/trn16_component.csv")
    scene16 = {str(cm16.chip_id[i]): ci16.scene[i] for i in range(len(ci16))
               if ci16.component[i] == "s1f11"}

    X16 = np.load(f"{P}/trn16_X_VV_VH_RVI_raw.npy", mmap_mode="r")
    Y16 = np.load(f"{P}/trn16_Y_MNDWI_NDVI_NDWI_raw.npy", mmap_mode="r")
    Xs = np.load(f"{P}/s1f11_X_VV_VH_RVI_raw.npy", mmap_mode="r")
    Ys = np.load(f"{P}/s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")
    Vs = np.load(f"{P}/s1f11_Yobs_valid.npy", mmap_mode="r")
    Js = np.load(f"{P}/s1f11_jrc.npy", mmap_mode="r")

    N = len(pool)
    os.makedirs(D, exist_ok=True)
    X18 = np.lib.format.open_memmap(f"{D}/trn18_X_VV_VH_RVI_raw.npy", mode="w+",
                                    dtype=np.float32, shape=(N,) + X16.shape[1:])
    Y18 = np.lib.format.open_memmap(f"{D}/trn18_Y_MNDWI_NDVI_NDWI_raw.npy", mode="w+",
                                    dtype=np.float32, shape=(N,) + Y16.shape[1:])
    meta_cols = list(cm16.columns)
    comp, meta, n_from_s1 = [], [], 0
    for i, r in pool.iterrows():
        from_s1 = (r["source"] == "s1f11") or (method == "A" and r["dataset"] == "s1f11")
        if from_s1:
            j = s1_idx[str(r["chip"])]
            X18[i] = Xs[j]
            y = np.asarray(Ys[j], np.float32).copy()
            bad = (np.asarray(Js[j]) > 0) | ~np.asarray(Vs[j], bool)
            y[:, bad] = np.nan
            Y18[i] = y
            wf = water_frac(y[0])
            n_from_s1 += 1
        else:
            k = int(r["source_index"])
            X18[i] = X16[k]
            Y18[i] = Y16[k]
            wf = None
        if r["source"] == "trn16":
            k = int(r["source_index"])
            row = cm16.iloc[k].to_dict()
            scene = ci16.scene[k]
        else:
            row = {c: np.nan for c in meta_cols}
            row.update(dataset=r["dataset"], event=r["event"], chip_id=r["chip"],
                       scene_id=f"s1f11_{r['event']}", flood=True)
            # trn16 names Sen1Floods11 scenes "s1f11/s1f11_<Event>"
            scene = args.scene_style.format(event=r["event"]) if args.scene_style \
                else scene16.get(str(r["chip"]), f"s1f11/s1f11_{r['event']}")
        row["chip_index"] = i
        if wf is not None:
            row["water_frac_measured"] = round(wf, 6)
        meta.append(row)
        comp.append(dict(idx=i, component=r["dataset"], scene=scene))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{N}", flush=True)
    X18.flush()
    Y18.flush()
    del X18, Y18
    pd.DataFrame(comp).to_csv(f"{D}/trn18_component.csv", index=False)
    pd.DataFrame(meta, columns=meta_cols).to_csv(f"{D}/trn18_chip_meta.csv", index=False)
    pool[["node"]].assign(trn18_row=np.arange(N)).to_csv(f"{D}/trn18_nodes.csv", index=False)
    # the trainer's default split file: everything train, folds override it below
    pd.DataFrame(dict(idx=np.arange(N), dataset=pool["dataset"], event=pool["event"],
                      split="train", reason="")).to_csv(f"{D}/trn18_split.csv", index=False)
    for f in ("tst4_X_VV_VH_RVI_raw.npy", "tst4_Y_MNDWI_NDVI_NDWI_raw.npy"):
        if os.path.exists(f"{P}/{f}") and not os.path.exists(f"{D}/{f}"):
            os.symlink(f"{P}/{f}", f"{D}/{f}")
    print(f"[trn18] {N} rows written, {n_from_s1} from s1f11_* arrays; "
          f"components {pd.Series([c['component'] for c in comp]).value_counts().to_dict()}",
          flush=True)

    # per-fold split files and data directories
    summary = {"method": method, "n_rows": N, "n_from_s1f11_arrays": n_from_s1, "folds": {}}
    low = pd.DataFrame(meta)["water_frac_measured"].to_numpy() < 0.005
    for fold in FOLDS:
        col = pool[f"fold_{fold}"].astype(str)
        split = col.map({"train": "train", "excluded": "excluded_loco", "test": "holdout_loco"})
        reason = col.map({"train": "", "excluded": f"shares a flood with the {fold} test chips",
                          "test": f"{fold} test chips"})
        sp = pd.DataFrame(dict(idx=np.arange(N), dataset=pool["dataset"], event=pool["event"],
                               split=split, reason=reason))
        sp.to_csv(f"{D}/trn18_split_{fold}.csv", index=False)
        fd = f"{OUT}/root_{fold}/processed"
        os.makedirs(fd, exist_ok=True)
        for f in ("trn18_X_VV_VH_RVI_raw.npy", "trn18_Y_MNDWI_NDVI_NDWI_raw.npy",
                  "trn18_component.csv", "trn18_chip_meta.csv", "tst4_X_VV_VH_RVI_raw.npy",
                  "tst4_Y_MNDWI_NDVI_NDWI_raw.npy"):
            dst = f"{fd}/{f}"
            if os.path.lexists(dst):
                os.remove(dst)
            if os.path.exists(f"{D}/{f}"):
                os.symlink(f"{D}/{f}", dst)
        sp.to_csv(f"{fd}/trn18_split.csv", index=False)
        keep = (split == "train").to_numpy() & pool["dataset"].isin(COMPS).to_numpy() & ~low
        summary["folds"][fold] = dict(
            test=int((split == "holdout_loco").sum()), excluded=int((split == "excluded_loco").sum()),
            train_after_filters=int(keep.sum()),
            by_component={c: int(v) for c, v in pool["dataset"][keep].value_counts().items()})
        print(f"  fold {fold:11} test {summary['folds'][fold]['test']:>3}  excluded "
              f"{summary['folds'][fold]['excluded']:>3}  train(after filters) {int(keep.sum())}",
              flush=True)
    json.dump(summary, open(f"{OUT}/trn18_summary.json", "w"), indent=2)
    print("saved", f"{OUT}/trn18_summary.json", flush=True)


if __name__ == "__main__":
    main()
