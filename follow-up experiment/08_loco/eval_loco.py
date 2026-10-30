#!/usr/bin/env python3
"""Calibration (section 5) and evaluation (section 6) of one leave-one-country-out
fold.

Calibration set: every hand-labelled Sen1Floods11 chip outside the test country
(446 minus the test chips; for Official64 the 382 chips that are not the test
64), no water-fraction filter, pixel range lab_ok as in eval_optical_axis.py.
Each run's tau and the re-calibrated rule (b) come from the same F1 search grids
as eval_optical_axis.py (unet 49 points, Lee 21 x 23).

Test set: the hand-labelled chips of the test country, pixel range as in
common.load (label >= 0, Yobs_valid, finite MNDWI, not JRC). The distance D is
always measured from the fixed rule (a): VV < -5 dB and VH < -22 dB.

Detectors: rule (a), rule (b), the t18 majority (>= 2 of 3 seeds, >= 3 of 5)
plus every seed alone, the cur-s1..s5 majority with their stored tau, and
rule (a) dilated by 30 m.

Output: output/loco/eval/<fold>.json and the majority masks
        output/loco/masks/<fold>_{t18,cur}.npy (np.packbits, test-chip order)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
sys.path.insert(0, f"{ROOT}/scripts/07_unseen")
sys.path.insert(0, f"{ROOT}/scripts/06_ablation")
import common as C  # noqa: E402
import unseen_common as U  # noqa: E402

OUT = f"{ROOT}/output/loco"
BANDS = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 100), (100, 200), (200, np.inf)]
Q = (25, 50, 75)


# ------------------------------------------------------------------ data
def load_s1f11(rows):
    """Sen1Floods11 chips at the given rows of s1f11_chips.csv, laid out like
    common.load so that common.evaluate accepts it."""
    ch = list(csv.DictReader(open(f"{C.P}/s1f11_chips.csv")))
    rows = np.asarray(rows)
    T = C.TestSet()
    T.name = "s1f11_subset"
    T.rows = rows
    T.names = [ch[i]["chip"] for i in rows]
    T.groups = np.array([ch[i]["event"] for i in rows])
    T.X = np.asarray(np.load(f"{C.P}/s1f11_X_VV_VH_RVI_raw.npy", mmap_mode="r")[rows], np.float32)
    lab = np.asarray(np.load(f"{C.P}/s1f11_label.npy", mmap_mode="r")[rows])
    jrc = np.asarray(np.load(f"{C.P}/s1f11_jrc.npy", mmap_mode="r")[rows]) > 0
    YO = np.asarray(np.load(f"{C.P}/s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])
    OV = np.asarray(np.load(f"{C.P}/s1f11_Yobs_valid.npy", mmap_mode="r")[rows], bool)
    z = np.load(f"{C.P}/s1f11_lee_db.npz")
    T.VV = np.asarray(z["VV"][rows], np.float32)
    T.VH = np.asarray(z["VH"][rows], np.float32)
    T.MNDWI = YO[:, 0].astype(np.float32)
    T.NDVI = YO[:, 1].astype(np.float32)
    T.valid = (lab >= 0) & OV & np.isfinite(T.MNDWI) & ~jrc
    T.gt = (lab == 1) & T.valid
    T.lab_ok = np.isfinite(T.X[:, 0]) & (T.X[:, 0] > 0) & (lab >= 0) & ~jrc
    T.gt_lab = (lab == 1) & T.lab_ok
    T.rule = (T.VV < C.RULE_VV) & (T.VH < C.RULE_VH)
    T.M = (T.MNDWI > 0.14) & T.valid
    T.D = C._rule_distance(T.rule)
    with np.errstate(invalid="ignore"):
        T.states = {"dim": T.VH < -19.0,
                    "bright_noveg": (T.VH >= -19.0) & (T.NDVI <= 0.3),
                    "bright_veg": (T.VH >= -19.0) & (T.NDVI > 0.3)}
    return T


def fold_rows(fold):
    """(calibration rows, test rows) in s1f11_chips.csv order.

    Test: every hand-labelled chip of the test country (no filter). Calibration:
    the other chips that pass the two training-time filters, judged with the
    trainer's own computation (output/loco/s1f11_filter.csv). For Paper the
    calibration set is fixed to the 243 chips the Paper runs trained on."""
    ch = list(csv.DictReader(open(f"{C.P}/s1f11_chips.csv")))
    idx = {r["chip"]: i for i, r in enumerate(ch)}
    filt = list(csv.DictReader(open(f"{OUT}/s1f11_filter.csv")))
    passing = {r["chip"] for r in filt if r["passes_trainer"] == "True"}
    pool = list(csv.DictReader(open(f"{ROOT}/experiments/loco_pool.csv")))
    if fold.startswith("S1F11"):
        # every Sen1Floods11 chip is the test; calibration comes from Test B (see main)
        return None, np.arange(len(ch))
    if fold == "Paper":
        test = np.array(sorted(idx[r["chip"]] for r in pool if r["fold_Paper"] == "test"))
        cal_chips = [r["chip"] for r in csv.DictReader(open(f"{OUT}/paper_calib_243.csv"))]
        cal = np.array(sorted(idx[c] for c in cal_chips))
        return cal, test
    if fold == "Official64":
        test_chips = {r["chip"] for r in pool if r["fold_Official64"] == "test"}
        test = np.array([i for i, r in enumerate(ch) if r["chip"] in test_chips])
    else:
        test = np.array([i for i, r in enumerate(ch) if r["event"] == fold])
    cal = np.array([i for i, r in enumerate(ch) if i not in set(test) and r["chip"] in passing])
    return cal, test


def t18_ckpt(run):
    return torch.load(f"{OUT}/runs/{run}/best_model.pt", map_location="cpu", weights_only=False)


def predict_ckpt(ck, X, dev):
    return C.predict(U.build_model(ck, dev), C.preprocess(X, ck), dev, bs=8)


# ------------------------------------------------------------------ scoring
def dilate(mask, px):
    out = np.empty_like(mask)
    for i in range(len(mask)):
        out[i] = distance_transform_edt(~mask[i]) <= px if mask[i].any() else mask[i]
    return out


def s61(T, m):
    m = m & T.valid
    r = C._prf(T.gt, m, T.valid)
    land = T.valid & ~T.gt
    miss = T.gt & ~T.rule
    return dict(iou=round(r["iou"], 4), precision=round(r["precision"], 4),
                recall=round(r["recall"], 4),
                land_false_alarm=round(float((land & m).sum() / max(int(land.sum()), 1)), 5),
                missed_share=round(float(miss.sum() / max(int(T.gt.sum()), 1)), 4),
                recovery=round(float((miss & m).sum() / max(int(miss.sum()), 1)), 4),
                tp=r["tp"], fp=r["fp"], fn=r["fn"])


def s62(T, m):
    """Water and land split by rule / translation correctness (Fig. 2a)."""
    m = m & T.valid
    out = {}
    for name, sel, want in (("water", T.gt, True), ("land", T.valid & ~T.gt, False)):
        r_ok = (T.rule == want) & sel
        t_ok = (m == want) & sel
        out[name] = dict(px=int(sel.sum()), both_right=int((r_ok & t_ok).sum()),
                         translation_fixed=int((sel & ~r_ok & t_ok).sum()),
                         translation_broke=int((sel & r_ok & ~t_ok).sum()),
                         both_wrong=int((sel & ~r_ok & ~t_ok).sum()))
    return out


def s63(T, m):
    """Recovery by distance from rule (a), and the share of recovery within 50 m."""
    m = m & T.valid
    miss = T.gt & ~T.rule
    rec = miss & m
    bands = {}
    for lo, hi in BANDS:
        sel = miss & (T.D > lo) & (T.D <= hi) if lo > 0 else miss & (T.D <= hi)
        bands[f"{lo}-{hi if np.isfinite(hi) else 'inf'}m"] = dict(
            missed_px=int(sel.sum()),
            recovery=round(float((rec & sel).sum() / max(int(sel.sum()), 1)), 4))
    return dict(bands=bands, recovered_px=int(rec.sum()),
                within_50m=round(float((rec & (T.D <= 50)).sum() / max(int(rec.sum()), 1)), 4),
                within_100m=round(float((rec & (T.D <= 100)).sum() / max(int(rec.sum()), 1)), 4))


def s64(T, m):
    """Median and quartiles of VV, VH, MNDWI, NDVI in the four pixel groups."""
    m = m & T.valid
    groups = {"rule_found_water": T.gt & T.rule, "recovered_water": T.gt & ~T.rule & m,
              "unrecovered_water": T.gt & ~T.rule & ~m, "land": T.valid & ~T.gt}
    out = {}
    for g, sel in groups.items():
        d = {"px": int(sel.sum())}
        for k in ("VV", "VH", "MNDWI", "NDVI"):
            a = getattr(T, k)[sel]
            a = a[np.isfinite(a)]
            d[k] = [round(float(np.percentile(a, p)), 3) for p in Q] if a.size else None
        out[g] = d
    return out


def s65(T, m):
    """False alarms on land farther than 30 m from rule (a), by surface class (Fig. 5)."""
    m = m & T.valid
    far = T.valid & ~T.gt & (T.D > 30)
    with np.errstate(invalid="ignore"):
        dim = T.VH < -19.0
        veg = (T.VH >= -19.0) & (T.NDVI > 0.3)
    return dict(land_px=int(far.sum()), all=int((far & m).sum()),
                vh_below_m19=int((far & m & dim).sum()),
                vh_ge_m19_ndvi_gt03=int((far & m & veg).sum()))


def score_all(T, m):
    return {"6-1": s61(T, m), "6-2": s62(T, m), "6-3": s63(T, m), "6-4": s64(T, m),
            "6-5": s65(T, m)}


def slice_set(T, sel):
    """Chip subset carrying only the fields the scoring uses."""
    S = C.TestSet()
    S.name = T.name
    for k in ("VV", "VH", "MNDWI", "NDVI", "valid", "gt", "rule", "M", "D"):
        setattr(S, k, getattr(T, k)[sel])
    S.states = {k: v[sel] for k, v in T.states.items()}
    S.groups = T.groups[sel]
    return S


def by_event(T, m):
    out = {}
    for g in sorted(set(T.groups)):
        sel = np.flatnonzero(T.groups == g)
        S = slice_set(T, sel)
        out[g] = dict(n_chips=int(len(sel)), **{"6-1": s61(S, m[sel]), "6-3": s63(S, m[sel])})
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    fold = args.fold
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [f"t18-{fold}-s{s}" for s in seeds]
    need = 1 if len(runs) == 1 else (2 if len(runs) <= 3 else 3)
    os.makedirs(f"{OUT}/eval", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)
    dev = C.device()

    cal_rows, test_rows = fold_rows(fold)
    if cal_rows is None:
        # S1F11excl / S1F11incl: the 54 hand-labelled FloodPlanet chips of Test B,
        # excluded from training in both variants, calibrate tau and rule (b)
        cal = C.load("B")
        cal_rows = cal.rows
    else:
        cal = load_s1f11(cal_rows)
    T = load_s1f11(test_rows)
    print(f"[{fold}] calibration {len(cal_rows)} chips ({cal.name}), test {len(test_rows)} chips "
          f"({sorted(set(T.groups))}); valid {int(T.valid.sum()):,} water {int(T.gt.sum()):,} "
          f"rule-missed {int((T.gt & ~T.rule).sum()):,}", flush=True)

    res = {"fold": fold, "runs": runs, "n_calib": int(len(cal_rows)), "n_test": int(len(test_rows)),
           "test_events": sorted(set(T.groups)), "test_chips": T.names, "calibration": {},
           "detectors": {}, "by_event": {}}

    # ---- section 5: thresholds ----
    t1, t2, f1 = U.sweep_lee(cal.VV, cal.VH, cal.gt_lab, cal.lab_ok)
    res["calibration"]["rule_b"] = dict(lee_vv_db=t1, lee_vh_db=t2, f1=round(f1, 4))
    res["calibration"]["rule_a"] = dict(lee_vv_db=C.RULE_VV, lee_vh_db=C.RULE_VH)
    print(f"  rule (b): VV < {t1} VH < {t2} (F1 {f1:.4f})", flush=True)
    taus, preds = {}, {}
    for run in runs:
        ck = t18_ckpt(run)
        pc = predict_ckpt(ck, cal.X, dev)
        tau, f1u = U.sweep_unet(pc, cal.gt_lab, cal.lab_ok)
        del pc
        taus[run] = dict(tau=tau, f1=round(f1u, 4), epoch=int(ck.get("epoch", -1)))
        preds[run] = predict_ckpt(ck, T.X, dev)
        print(f"  {run}: tau {tau:.2f} (F1 {f1u:.4f})", flush=True)
    res["calibration"]["t18"] = taus
    res["calibration"]["cur"] = {r: C.tau_of(r, "A") for r in C.RUNS}

    # ---- detectors ----
    masks = {"rule_a": T.rule, "rule_b": (T.VV < t1) & (T.VH < t2),
             "rule_a_dilate_30m": T.rule | (T.D <= 30)}
    votes = np.zeros(T.gt.shape, np.int8)
    for run in runs:
        m = preds[run] > taus[run]["tau"]
        masks[run] = m
        votes += m
    masks["t18_majority"] = votes >= need
    del preds
    cv = np.zeros(T.gt.shape, np.int8)
    for run in C.RUNS:
        ck = C.ckpt(run)
        cv += (C.predict(C.build_model(ck, dev), C.preprocess(T.X, ck), dev, bs=8)
               > C.tau_of(run, "A"))
    masks["cur_majority"] = cv >= 3
    np.save(f"{OUT}/masks/{fold}_t18.npy", np.packbits(masks["t18_majority"] & T.valid, axis=None))
    np.save(f"{OUT}/masks/{fold}_cur.npy", np.packbits(masks["cur_majority"] & T.valid, axis=None))
    res["mask_shape"] = list(T.gt.shape)

    for name, m in masks.items():
        res["detectors"][name] = score_all(T, m)
        r = res["detectors"][name]["6-1"]
        print(f"  {name:18} IoU {r['iou']:.4f} P {r['precision']:.4f} R {r['recall']:.4f} "
              f"recovery {r['recovery']:.4f} within50m {res['detectors'][name]['6-3']['within_50m']:.3f}",
              flush=True)
    # ---- 6-6: seed spread ----
    per = [res["detectors"][r] for r in runs]
    res["seed_spread"] = {k: dict(mean=round(float(np.mean([p["6-1"][k] for p in per])), 4),
                                  sd=round(float(np.std([p["6-1"][k] for p in per])), 4))
                          for k in ("iou", "precision", "recall", "recovery", "land_false_alarm")}
    res["seed_spread"]["within_50m"] = dict(
        mean=round(float(np.mean([p["6-3"]["within_50m"] for p in per])), 4),
        sd=round(float(np.std([p["6-3"]["within_50m"] for p in per])), 4))
    # ---- per event (matters for Official64) ----
    for name in ("rule_a", "t18_majority", "cur_majority"):
        res["by_event"][name] = by_event(T, masks[name])
    C.save_json(res, f"{OUT}/eval/{fold}.json")
    print("saved", f"{OUT}/eval/{fold}.json", flush=True)


if __name__ == "__main__":
    main()
