#!/usr/bin/env python3
"""Tier 0 outputs in the cur-s format (request 4.0).

For every t18-Paper-s<k> run writes runs/<run>/optical_axis_s1f11/summary.json
with the same keys as cur-s (tau calibrated on the 243 Sen1Floods11 chips the
Paper runs trained on, scored on the 139 paper test chips), and re-calibrates
the five cur-s runs on the same 243 chips for comparison. The rule is scored at
the fixed -5/-22 dB (a) and at the values re-fitted on the 243 chips (b).

Output: output/loco/runs/t18-Paper-s<k>/optical_axis_s1f11/summary.json
        output/loco/eval/Paper_summaries.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_loco as EL  # noqa: E402
from eval_loco import C, U, OUT  # noqa: E402

LABEL_NOTE = "Sentinel-2 dominant hand label; S1 VH was also consulted"


def pooled(gt, pred, valid):
    r = C._prf(gt, pred, valid)
    return {k: float(r[k]) for k in ("tp", "fp", "fn", "iou", "precision", "recall", "f1")}


def decomposition(T, m):
    vis = T.gt & T.rule
    inv = T.gt & ~T.rule
    return dict(gt_water_px=int(T.gt.sum()), sar_visible_px=int(vis.sum()),
                sar_invisible_px=int(inv.sum()),
                sar_invisible_share=round(float(inv.sum() / max(int(T.gt.sum()), 1)), 4),
                unet_recall_on_sar_visible=round(float((vis & m).sum() / max(int(vis.sum()), 1)), 4),
                unet_recall_on_sar_invisible=round(float((inv & m).sum() / max(int(inv.sum()), 1)), 4),
                note="Lee has recall 1.0 on the visible subset and 0.0 on the invisible subset by construction")


def summary(run, T, m, tau, vv, vh, n_cal):
    rule = (T.VV < vv) & (T.VH < vh) & T.valid
    return {"axis": "s1f11", "run": run, "label_note": LABEL_NOTE, "calib": "train_events_filtered",
            "n_chips_scored": int(len(T.names)), "n_chips_calibrated": int(n_cal),
            "thresholds": {"unet": float(tau), "lee_vv_db": float(vv), "lee_vh_db": float(vh)},
            "pooled": {"classic_SAR_thresholding": pooled(T.gt, rule, T.valid),
                       "unet": pooled(T.gt, m & T.valid, T.valid)},
            "decomposition": decomposition(T, m & T.valid)}


def main():
    dev = C.device()
    cal_rows, test_rows = EL.fold_rows("Paper")
    cal = EL.load_s1f11(cal_rows)
    T = EL.load_s1f11(test_rows)
    assert len(cal_rows) == 243 and len(test_rows) == 139, (len(cal_rows), len(test_rows))
    print(f"calibration 243 chips {sorted(set(cal.groups))}; test 139 chips {sorted(set(T.groups))}",
          flush=True)
    t1, t2, f1 = U.sweep_lee(cal.VV, cal.VH, cal.gt_lab, cal.lab_ok)
    res = {"n_calib": 243, "n_test": 139,
           "rule_a": dict(lee_vv_db=C.RULE_VV, lee_vh_db=C.RULE_VH),
           "rule_b_243": dict(lee_vv_db=t1, lee_vh_db=t2, f1=round(f1, 4)),
           "t18": {}, "cur_recalib_243": {}}
    print(f"rule (b) on 243: VV < {t1} VH < {t2} (F1 {f1:.4f})", flush=True)

    runs = [f"t18-Paper-s{s}" for s in range(1, 6)]
    for run in runs:
        ck = EL.t18_ckpt(run)
        pc = EL.predict_ckpt(ck, cal.X, dev)
        tau, f1u = U.sweep_unet(pc, cal.gt_lab, cal.lab_ok)
        del pc
        m = EL.predict_ckpt(ck, T.X, dev) > tau
        s = summary(run, T, m, tau, C.RULE_VV, C.RULE_VH, 243)
        s["thresholds_rule_b_243"] = {"lee_vv_db": t1, "lee_vh_db": t2}
        s["pooled"]["classic_SAR_thresholding_rule_b"] = pooled(T.gt, (T.VV < t1) & (T.VH < t2) & T.valid, T.valid)
        s["calib_f1"] = round(f1u, 4)
        s["epoch"] = int(ck.get("epoch", -1))
        d = f"{OUT}/runs/{run}/optical_axis_s1f11"
        os.makedirs(d, exist_ok=True)
        C.save_json(s, f"{d}/summary.json")
        res["t18"][run] = s
        print(f"  {run}: tau {tau:.2f} (cal F1 {f1u:.4f})  test IoU {s['pooled']['unet']['iou']:.4f} "
              f"P {s['pooled']['unet']['precision']:.4f} R {s['pooled']['unet']['recall']:.4f}  "
              f"invisible recall {s['decomposition']['unet_recall_on_sar_invisible']:.4f}", flush=True)
        del m

    for run in C.RUNS:
        ck = C.ckpt(run)
        pc = C.predict(C.build_model(ck, dev), C.preprocess(cal.X, ck), dev, bs=8)
        tau, f1u = U.sweep_unet(pc, cal.gt_lab, cal.lab_ok)
        del pc
        m = C.predict(C.build_model(ck, dev), C.preprocess(T.X, ck), dev, bs=8) > tau
        s = summary(run, T, m, tau, C.RULE_VV, C.RULE_VH, 243)
        s["tau_original_307"] = C.tau_of(run, "A")
        res["cur_recalib_243"][run] = s
        print(f"  {run}: tau_243 {tau:.2f} (was {s['tau_original_307']:.2f})  test IoU "
              f"{s['pooled']['unet']['iou']:.4f}", flush=True)
        del m
    C.save_json(res, f"{OUT}/eval/Paper_summaries.json")
    print("saved", f"{OUT}/eval/Paper_summaries.json", flush=True)


if __name__ == "__main__":
    main()
