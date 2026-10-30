#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the optical-reference evaluation axes with the score split in two.

Why split
  "Pooled IoU is higher than Lee" alone does not show why. Split the GT water pixels into
    SAR-visible   : pixels that Lee (SAR thresholding) detected
    SAR-invisible : pixels that Lee missed
  and report recall on each separately; the claim then takes the form
    "recovers XX% of the optical water that Lee missed, and matches Lee on the SAR-visible side"
  which is a direct measurement of what the SAR->MSI translation adds.

  Since "answer water everywhere" raises recall, precision and the FP rate are always reported too.

Axes:
  --axis s1f11 : Sen1Floods11 held-out (S2-dominant label; SAR was also consulted, conservative axis)
  --axis fp    : FloodPlanet held-out (3.3 m PlanetScope hand label; SAR not consulted, independent axis)

Threshold rule (EXPERIMENT_DESIGN.md §3, fixed 2026-09-06)
  The thresholds (U-Net MNDWI / Lee VV,VH) are method hyperparameters: chosen once as the
  F1-maximising value on labelled training-side data and never touched again. Scoring chips
  are never used.
  - axis A: calibrate on the 7 training-side Sen1Floods11 events (307 chips) -> score on the 139 held-out chips
  - axis B: reuse the same values (--thresholds_from). The FloodPlanet training-side labels
    do not match the water in the 10 m imagery (in 8 of 11 regions neither VH nor MNDWI
    separates water/land; calibrating Lee degenerates to VH<-10 dB "all water";
    output/axisB_calibration_check/result.md), so they cannot serve as a calibration source. Axis B is
    therefore a transfer test scored with thresholds chosen without ever looking at FloodPlanet.
  - --calib oracle is only an upper-bound reference (robustness row). Not reported as a score.

Run: torch210 container (torch + numpy)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RMSD_flood_eval import UNet, linear_to_db_safe, metrics_from_masks  # noqa: E402
from otsu_baseline import otsu_mask  # noqa: E402

AXES = {
    "s1f11": dict(X="s1f11_X_VV_VH_RVI_raw.npy", L="s1f11_label.npy",
                  J="s1f11_jrc.npy", C="s1f11_chips.csv",
                  label_note="Sentinel-2 dominant hand label; S1 VH was also consulted"),
    "fp": dict(X="fp_eval_X_VV_VH_RVI_raw.npy", L="fp_eval_label.npy",
               J="fp_eval_jrc.npy", C="fp_eval_chips.csv",
               label_note="hand label drawn on 3.3 m PlanetScope; SAR was NOT consulted"),
}

# Events held out on axis A (must match A_HOLDOUT in make_split_assignment.py).
# Bolivia is held out as a whole region from the start.
HOLDOUT_EVENTS = {"India", "Somalia", "Spain", "Bolivia"}


def lee(img, enl=4.4, size=7):
    from scipy.ndimage import uniform_filter
    nan = ~np.isfinite(img)
    x = np.where(nan, 0.0, img)
    w = (~nan).astype(np.float64)
    s = uniform_filter(x, size)
    n = uniform_filter(w, size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = uniform_filter(x * x, size) / n - mean * mean
        ci2 = var / (mean * mean)
        wgt = np.clip(1.0 - (1.0 / enl) / np.maximum(ci2, 1e-12), 0.0, 1.0)
    out = mean + wgt * (x - mean)
    out[nan] = np.nan
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(AXES), required=True)
    ap.add_argument("--run", required=True, help="output/<run>/best_model.pt")
    ap.add_argument("--root", default="/home/kk/jp_claude/RMSD")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--calib", choices=["train_events", "holdout_split", "oracle"],
                    default="train_events",
                    help="Threshold calibration source. train_events (default) = chosen once "
                         "by max F1 on the Sen1Floods11 training-side events (307 chips); "
                         "used as is on axis A and carried to axis B via --thresholds_from "
                         "(scoring chips are never touched). "
                         "oracle = upper bound chosen on the scoring chips themselves; not a score, "
                         "used only for the robustness row (ranking does not change with the "
                         "threshold choice). Output goes to a separate optical_axis_<axis>_oracle/. "
                         "holdout_split = calibrate on part of the scoring chips (old scheme; not used "
                         "because chips of the same event end up on both sides)")
    ap.add_argument("--calib_frac", type=float, default=0.3,
                    help="calibration fraction for --calib holdout_split")
    ap.add_argument("--thresholds_from", default="",
                    help="Required on axis B with --calib train_events. Path to the axis A "
                         "optical_axis_s1f11/summary.json; the thresholds chosen there are "
                         "reused as is (prevents per-axis tuning)")
    ap.add_argument("--lee_cache", default="auto",
                    help="Cache of the Lee filter result (VV/VH dB). It does not depend on the run "
                         "but takes several minutes per run, so it is essential for 86 runs. "
                         "auto = created automatically at processed/<axis>_lee_db.npy / off = disabled")
    ap.add_argument("--otsu", action="store_true",
                    help="Also score per-chip Otsu without label calibration: "
                         "Otsu on Lee-filtered VV/VH (the standard operational automatic threshold) "
                         "and Otsu on the predicted MNDWI. Default off (so the output of the "
                         "86 runs is unchanged)")
    ap.add_argument("--otsu_ashman", type=float, default=2.0,
                    help="Lower bound of Ashman D used as the Otsu bimodality test. A variant "
                         "that treats chips below it as water-free is also reported")
    a = ap.parse_args()
    cfg = AXES[a.axis]
    P = f"{a.root}/processed"
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The axis A file holds all 446 chips, so restrict to the held-out events.
    # Otherwise chips of events seen in training are scored too and generalisation cannot be measured.
    sel_rows = None
    if a.axis == "s1f11":
        ch = list(csv.DictReader(open(f"{P}/{cfg['C']}", newline="")))
        sel_rows = np.array([i for i, r in enumerate(ch)
                             if r["event"] in HOLDOUT_EVENTS], dtype=np.int64)
        cal_rows = np.array([i for i, r in enumerate(ch)
                             if r["event"] not in HOLDOUT_EVENTS], dtype=np.int64)
        evs = sorted({ch[i]["event"] for i in sel_rows})
        print(f"axis A: held-out events {evs} -> {len(sel_rows)} chips "
              f"(calibration candidates {len(cal_rows)} chips)", flush=True)

    X = np.load(f"{P}/{cfg['X']}").astype(np.float32)
    LB = np.load(f"{P}/{cfg['L']}")
    JR = np.load(f"{P}/{cfg['J']}") if os.path.exists(f"{P}/{cfg['J']}") else np.zeros_like(LB)
    N = X.shape[0]
    print(f"axis={a.axis}  chips={N}  label: {cfg['label_note']}", flush=True)

    valid_sar = np.isfinite(X[:, 0]) & (X[:, 0] > 0)
    eval_px = valid_sar & (LB >= 0) & (JR == 0)
    gt = (LB == 1) & eval_px
    print(f"eval pixels {eval_px.sum():,}   GT water {gt.sum():,} "
          f"({100*gt.sum()/max(eval_px.sum(),1):.1f}%)", flush=True)

    # ---------------- Lee (SAR thresholding) ----------------
    # Lee depends only on X, not on the run. Recomputing it for 86 runs takes hours
    # on axis A alone, so read the cache if one with matching shape exists.
    cache = None if a.lee_cache == "off" else (
        f"{P}/{a.axis}_lee_db.npz" if a.lee_cache == "auto" else a.lee_cache)
    VV = VH = None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z["VV"].shape == X[:, 0].shape and z["VH"].shape == X[:, 1].shape:
            VV, VH = z["VV"], z["VH"]
            print(f"lee cache hit: {cache}", flush=True)
        else:
            print(f"lee cache shape mismatch -> rebuilding ({cache})", flush=True)
    if VV is None:
        VV = np.empty_like(X[:, 0])
        VH = np.empty_like(X[:, 1])
        for i in range(N):
            vv = X[i, 0].astype(np.float64); vv[vv <= 0] = np.nan
            vh = X[i, 1].astype(np.float64); vh[vh <= 0] = np.nan
            VV[i] = linear_to_db_safe(lee(vv))
            VH[i] = linear_to_db_safe(lee(vh))
        print("lee done", flush=True)
        if cache:
            # write to a temporary name then rename, so concurrent runs do not collide
            tmp = f"{cache}.{os.getpid()}.tmp.npz"
            np.savez(tmp, VV=VV, VH=VH)
            os.replace(tmp, cache)
            print(f"lee cache saved: {cache}", flush=True)

    # ---------------- model ----------------
    ck = torch.load(f"{a.root}/output/{a.run}/best_model.pt", map_location="cpu",
                    weights_only=False)
    Z = X.copy()
    for c in (0, 1):
        if np.nanmedian(Z[:, c]) > 0:
            Z[:, c] = linear_to_db_safe(Z[:, c])
    lo = (np.asarray(ck["X_mean"]) - 5 * np.asarray(ck["X_std"])).reshape(1, -1, 1, 1)
    hi = (np.asarray(ck["X_mean"]) + 5 * np.asarray(ck["X_std"])).reshape(1, -1, 1, 1)
    Z[(Z < lo) | (Z > hi)] = np.nan
    for c in range(3):
        Z[:, c] = np.clip((Z[:, c] - ck["X_min"][c]) / (ck["X_max"][c] - ck["X_min"][c]), 0, 1)
    Z = np.nan_to_num(Z, nan=0.0)
    model = UNet(3, 3, bilinear=True).to(dev)
    model.load_state_dict(ck["model_xy_state_dict"]); model.eval()
    pred = np.empty((N, X.shape[2], X.shape[3]), np.float32)
    with torch.no_grad():
        for s in range(0, N, a.bs):
            xb = torch.from_numpy(Z[s:s + a.bs]).float().to(dev)
            with torch.cuda.amp.autocast(enabled=dev.type == "cuda"):
                pred[s:s + a.bs] = model(xb).float().cpu().numpy()[:, 0]
    del Z
    print("inference done", flush=True)

    # ---------------- calibration / scoring split ----------------
    scoring = sel_rows if a.axis == "s1f11" else np.arange(N)
    if a.calib == "train_events" and a.axis == "s1f11":
        # calibrate on the training-side events, score on all held-out events
        cal, tst = cal_rows, sel_rows
    elif a.calib == "train_events" and a.axis == "fp":
        # Axis B reuses the thresholds chosen on the axis A training side (--thresholds_from).
        # Calibrating on the FloodPlanet training-side labels (98 chips) was also tried, but in
        # 8 of 11 regions label water and land do not separate in VH or MNDWI, and Lee
        # degenerated to "all water" (output/axisB_calibration_check/result.md). Not a usable source.
        # Scoring chips are never used, so axis B is a transfer test with thresholds
        # chosen without looking at FloodPlanet.
        cal, tst = None, np.arange(N)
    elif a.calib == "oracle":
        # Upper-bound reference. Picks the F1-maximising threshold on the scoring chips themselves.
        # Used only for the robustness row (U-Net vs Lee ranking does not change with the threshold choice).
        cal, tst = scoring, scoring
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(scoring)
        ncal = max(1, int(round(a.calib_frac * len(perm))))
        cal, tst = perm[:ncal], perm[ncal:]
    print(f"calibration chips {'(from external file)' if cal is None else len(cal)} / "
          f"scoring chips {len(tst)}", flush=True)

    def sweep_unet(sel):
        best = (0.5, -1.0)
        for t in np.linspace(0.02, 0.98, 49):
            r = metrics_from_masks(gt[sel], (pred[sel] > t) & eval_px[sel], eval_px[sel])
            if np.isfinite(r["f1"]) and r["f1"] > best[1]:
                best = (float(t), r["f1"])
        return best[0]

    def sweep_lee(sel):
        best = (None, None, -1.0)
        for t1 in np.linspace(-25, -5, 21):
            m1 = VV[sel] < t1
            for t2 in np.linspace(-32, -10, 23):
                r = metrics_from_masks(gt[sel], (m1 & (VH[sel] < t2)) & eval_px[sel],
                                       eval_px[sel])
                if np.isfinite(r["f1"]) and r["f1"] > best[2]:
                    best = (float(t1), float(t2), r["f1"])
        return best[0], best[1]

    if cal is None:
        if not a.thresholds_from or not os.path.exists(a.thresholds_from):
            sys.exit("Evaluating axis B with --calib train_events requires --thresholds_from "
                     "pointing to the axis A summary.json "
                     "(calibrating per axis would be tuning to the benchmark)")
        th = json.load(open(a.thresholds_from))["thresholds"]
        tU, c1, c2 = th["unet"], th["lee_vv_db"], th["lee_vh_db"]
        print(f"thresholds reused from {a.thresholds_from}", flush=True)
    else:
        tU = sweep_unet(cal)                       # model dependent; must be redone every time
        # The Lee threshold sweep is 483 combinations x all calibration-chip pixels and also run independent.
        # Over 86 runs it takes hours, so cache it like the Lee filter itself.
        tc = (None if not cache else
              cache.replace(".npz", f"_th_{a.calib}_{len(cal)}.json"))
        c1 = c2 = None
        if tc and os.path.exists(tc):
            _t = json.load(open(tc))
            c1, c2 = _t["lee_vv_db"], _t["lee_vh_db"]
            print(f"lee threshold cache hit: {tc}", flush=True)
        if c1 is None:
            c1, c2 = sweep_lee(cal)
            if tc:
                json.dump({"lee_vv_db": c1, "lee_vh_db": c2, "n_calib": int(len(cal))},
                          open(tc, "w"))
    print(f"thresholds: unet>{tU:.3f}   lee VV<{c1:.1f}dB VH<{c2:.1f}dB", flush=True)

    pU = (pred[tst] > tU) & eval_px[tst]
    pL = (VV[tst] < c1) & (VH[tst] < c2) & eval_px[tst]
    g, v = gt[tst], eval_px[tst]

    mU = metrics_from_masks(g, pU, v)
    mL = metrics_from_masks(g, pL, v)

    # ---------------- * SAR-visible / SAR-invisible decomposition ----------------
    vis = g & pL             # GT water pixels that Lee detected
    inv = g & ~pL            # GT water pixels that Lee missed
    rec_u_vis = float((pU & vis).sum()) / max(int(vis.sum()), 1)
    rec_u_inv = float((pU & inv).sum()) / max(int(inv.sum()), 1)

    # ---------------- Otsu (automatic threshold without label calibration) ----------------
    # Lee / Unet above both pick one F1-maximising point using the GT labels,
    # i.e. a setting that favours the baseline. Otsu uses no labels and adapts
    # per chip, so measuring that gap checks how robust the claim is.
    otsu = {}
    if a.otsu:
        vt = eval_px[tst]
        specs = [
            ("otsu_sar", None), ("otsu_sar_bimodal", a.otsu_ashman),
        ]
        for name, ash in specs:
            m1, u1, r1 = otsu_mask(VV[tst], vt, True, ash)
            m2, u2, r2 = otsu_mask(VH[tst], vt, True, ash)
            pm = m1 & m2
            mm = metrics_from_masks(g, pm, v)
            otsu[name] = {k: float(x) for k, x in mm.items()}
            otsu[name].update(chips_used_vv=u1, chips_rejected_vv=r1,
                              chips_used_vh=u2, chips_rejected_vh=r2)
            # recall on the visible/invisible subsets defined by the supervised Lee
            otsu[name]["recall_on_sar_visible"] = round(
                float((pm & vis).sum()) / max(int(vis.sum()), 1), 4)
            otsu[name]["recall_on_sar_invisible"] = round(
                float((pm & inv).sum()) / max(int(inv.sum()), 1), 4)
            if name == "otsu_sar_bimodal":
                # invisible share and Unet recovery rate when this baseline is the reference instead
                iv2 = g & ~pm
                otsu["decomposition_vs_otsu_bimodal"] = {
                    "sar_invisible_share": round(float(iv2.sum()) / max(int(g.sum()), 1), 4),
                    "unet_recall_on_sar_invisible": round(
                        float((pU & iv2).sum()) / max(int(iv2.sum()), 1), 4),
                }
        pmu, uu, ru = otsu_mask(pred[tst], vt, False, None)
        mm = metrics_from_masks(g, pmu, v)
        otsu["otsu_unet_mndwi"] = {k: float(x) for k, x in mm.items()}
        otsu["otsu_unet_mndwi"].update(chips_used=uu, chips_rejected=ru)
        otsu["otsu_unet_mndwi"]["recall_on_sar_visible"] = round(
            float((pmu & vis).sum()) / max(int(vis.sum()), 1), 4)
        otsu["otsu_unet_mndwi"]["recall_on_sar_invisible"] = round(
            float((pmu & inv).sum()) / max(int(inv.sum()), 1), 4)
        otsu["note"] = ("Per-chip Otsu uses no labels at all. The supervised rows above "
                        "pick one globally F1-optimal threshold on calibration data, "
                        "which favours the baseline, so these rows test how much of the "
                        "conclusion survives when neither method is label calibrated.")

    out = {
        "axis": a.axis, "run": a.run, "label_note": cfg["label_note"], "calib": a.calib,
        "n_chips_scored": int(len(tst)), "n_chips_calibrated": (None if cal is None else int(len(cal))),
        "thresholds": {"unet": tU, "lee_vv_db": c1, "lee_vh_db": c2},
        "pooled": {"classic_SAR_thresholding": {k: float(x) for k, x in mL.items()},
                   "unet": {k: float(x) for k, x in mU.items()}},
        "decomposition": {
            "gt_water_px": int(g.sum()),
            "sar_visible_px": int(vis.sum()),
            "sar_invisible_px": int(inv.sum()),
            "sar_invisible_share": round(float(inv.sum()) / max(int(g.sum()), 1), 4),
            "unet_recall_on_sar_visible": round(rec_u_vis, 4),
            "unet_recall_on_sar_invisible": round(rec_u_inv, 4),
            "note": "Lee has recall 1.0 on the visible subset and 0.0 on the invisible "
                    "subset by construction. The invisible-subset recall is what the "
                    "SAR-to-optical translation adds; read it together with precision.",
        },
    }
    if otsu:
        out["otsu_uncalibrated"] = otsu
    # anything other than the default (train_events) goes to a separate directory so the official score is not overwritten
    od = f"{a.root}/output/{a.run}/optical_axis_{a.axis}" + (
        "" if a.calib == "train_events" else f"_{a.calib}")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/summary.json", "w"), indent=2)

    print(f"\n{'':22} {'IoU':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"{'Classic SAR (Lee)':22} {mL['iou']:7.3f} {mL['precision']:7.3f} "
          f"{mL['recall']:7.3f} {mL['f1']:7.3f}")
    print(f"{'Unet-MNDWI':22} {mU['iou']:7.3f} {mU['precision']:7.3f} "
          f"{mU['recall']:7.3f} {mU['f1']:7.3f}")
    for nm, lab in (("otsu_sar", "Otsu SAR"),
                    ("otsu_sar_bimodal", f"Otsu SAR (D>{a.otsu_ashman:g})"),
                    ("otsu_unet_mndwi", "Otsu Unet-MNDWI")):
        if nm in otsu:
            r = otsu[nm]
            print(f"{lab:22} {r['iou']:7.3f} {r['precision']:7.3f} "
                  f"{r['recall']:7.3f} {r['f1']:7.3f}   "
                  f"(uncalibrated, rejected chips "
                  f"{r.get('chips_rejected_vv', r.get('chips_rejected', 0))})")
    d = out["decomposition"]
    print(f"\nGT water px {d['gt_water_px']:,} = SAR-visible {d['sar_visible_px']:,} + "
          f"SAR-invisible {d['sar_invisible_px']:,} ({100*d['sar_invisible_share']:.1f}%)")
    print(f"  Unet recall  SAR-visible   {100*d['unet_recall_on_sar_visible']:5.1f}%  (Lee=100%)")
    print(f"  Unet recall  SAR-invisible {100*d['unet_recall_on_sar_invisible']:5.1f}%  (Lee=0%)")
    print(f"\nsaved {od}/summary.json", flush=True)


if __name__ == "__main__":
    main()
