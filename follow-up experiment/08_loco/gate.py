#!/usr/bin/env python3
"""Gates 1 and 2 of the leave-one-country-out request (section 3).

Gate 1: the evaluation code still reproduces the paper on Test A (139 chips):
        rule IoU 0.4017, translation IoU 0.4949, recovery 0.286 (+-0.003), using
        cur-s1..s5 with their stored tau and a >=3-of-5 majority.
Gate 2: trn16 and the s1f11_* arrays carry the same MNDWI teacher. Twenty random
        chips present in both are compared after masking JRC > 0 to NaN on the
        s1f11 side; report the share of pixels whose finiteness agrees and the
        maximum absolute difference where both are finite. Within 0.01 and 1%
        means "method A" for building trn18, otherwise "method B".

Output: output/loco/gate.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
sys.path.insert(0, f"{ROOT}/scripts/06_ablation")
import common as C  # noqa: E402

OUT = f"{ROOT}/output/loco"
TARGET = dict(rule_iou=0.4017, trans_iou=0.4949, recovery=0.286, tol=0.003)


def gate1(dev):
    T = C.load("A")
    votes = np.zeros(T.gt.shape, np.int8)
    taus = {}
    for run in C.RUNS:
        ck = C.ckpt(run)
        tau = C.tau_of(run, "A")
        taus[run] = tau
        pred = C.predict(C.build_model(ck, dev), C.preprocess(T.X, ck), dev, bs=8)
        votes += (pred > tau)
        del pred
    maj = votes >= 3
    r_rule = C.evaluate(T.rule, T, per_group=False)
    r_maj = C.evaluate(maj & T.valid, T, per_group=False)
    got = dict(rule_iou=r_rule["overall"]["iou"], trans_iou=r_maj["overall"]["iou"],
               recovery=r_maj["recovery"]["rate"])
    ok = all(abs(got[k] - TARGET[k]) <= TARGET["tol"] for k in ("rule_iou", "trans_iou", "recovery"))
    out = dict(n_chips=len(T.names), taus=taus, target={k: TARGET[k] for k in got}, got=got,
               diff={k: round(got[k] - TARGET[k], 4) for k in got},
               precision=r_maj["overall"]["precision"], recall=r_maj["overall"]["recall"],
               passed=bool(ok))
    print(f"gate1: rule IoU {got['rule_iou']:.4f} (target 0.4017)  trans IoU "
          f"{got['trans_iou']:.4f} (0.4949)  recovery {got['recovery']:.4f} (0.286)  "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return out


def gate2(n=20, seed=0):
    pool = pd.read_csv(f"{ROOT}/experiments/loco_pool.csv")
    s1 = pd.read_csv(f"{C.P}/s1f11_chips.csv")
    s1_idx = {c: i for i, c in enumerate(s1["chip"].astype(str))}
    both = pool[(pool["source"] == "trn16") & (pool["dataset"] == "s1f11")]
    rng = np.random.default_rng(seed)
    pick = both.iloc[rng.choice(len(both), size=min(n, len(both)), replace=False)]
    Yt = np.load(f"{C.P}/trn16_Y_MNDWI_NDVI_NDWI_raw.npy", mmap_mode="r")
    Ys = np.load(f"{C.P}/s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")
    Js = np.load(f"{C.P}/s1f11_jrc.npy", mmap_mode="r")
    rows = []
    for _, r in pick.iterrows():
        j = s1_idx.get(str(r["chip"]))
        if j is None:
            rows.append(dict(chip=r["chip"], error="not in s1f11_chips.csv"))
            continue
        a = np.asarray(Yt[int(r["source_index"]), 0], np.float32)
        b = np.asarray(Ys[j, 0], np.float32).copy()
        b[np.asarray(Js[j]) > 0] = np.nan
        fa, fb = np.isfinite(a), np.isfinite(b)
        agree = float((fa == fb).mean())
        bothf = fa & fb
        mx = float(np.abs(a[bothf] - b[bothf]).max()) if bothf.any() else float("nan")
        rows.append(dict(chip=r["chip"], trn16_row=int(r["source_index"]), s1f11_row=int(j),
                         finite_agree=round(agree, 5), max_abs_diff=round(mx, 6),
                         finite_trn16=int(fa.sum()), finite_s1f11=int(fb.sum())))
    ok_rows = [x for x in rows if "error" not in x]
    worst_agree = min(x["finite_agree"] for x in ok_rows)
    worst_diff = max(x["max_abs_diff"] for x in ok_rows)
    method = "A" if (worst_diff <= 0.01 and worst_agree >= 0.99) else "B"
    print(f"gate2: {len(ok_rows)} chips, finiteness agreement min {worst_agree:.5f}, "
          f"max |diff| {worst_diff:.6f} -> method {method}", flush=True)
    return dict(n=len(ok_rows), worst_finite_agree=worst_agree, worst_max_abs_diff=worst_diff,
                method=method, chips=rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = C.device()
    res = {"gate2": gate2(), "gate1": gate1(dev)}
    res["pass"] = bool(res["gate1"]["passed"])
    C.save_json(res, f"{OUT}/gate.json")
    print("saved", f"{OUT}/gate.json", flush=True)


if __name__ == "__main__":
    main()
