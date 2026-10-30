"""REQUEST_terra_20261026_l1only_mndwi0.md §3.1 (Tier 0): Test A, paper pixel set (TestSet.lab_ok / gt_lab).

Decision: translated MNDWI > 0 (pred > -Y_min[0] / (Y_max[0] - Y_min[0]) of each checkpoint), no tau calibration.
Groups (all on Test A, 139 chips):
  P0        output/l1mndwi0/runs/P0-s1..5, one direction (SAR->MSI, 09_onedir/train_onedir.py; requester decision
            2026-10-26), each seed + 5-seed vote (>= 3); also P0 s1..3 vote (>= 2) for the mndwi-weight comparison
  P0bi      the same configs trained bidirectionally before the switch (output/l1mndwi0/runs_bidir/P0-s1..5)
  L1w2xy_m0 L1w2xy-s1..3 = la-L1only configs (mndwi_weight 2.0) trained one direction, cut at MNDWI 0, 3-seed vote:
            P0_s123 vs L1w2xy_m0 = effect of the MNDWI weight alone
  REF_cal   t18-Paper-s1..5, tau cal243 (taken from output/loss_ablation/metrics.json, same sweep), 5-seed vote = paper
  REF_m0    t18-Paper-s1..5 cut at translated MNDWI 0, 5-seed vote
  L1w2_m0   la-L1only-s1..3 (mndwi_weight 2.0, bidirectional) cut at translated MNDWI 0, 3-seed vote
  rule      backscatter rule (VV < -5 dB & VH < -22 dB), unchanged
Per mask: eval_sameday.table2 (Table II, distance bands, far-land FP by VH/NDVI, per event, WorldCover) on A_all,
A_sameday (Bolivia, India) and A_gap (Somalia, Spain), plus extras(): unrecovered rate, WorldCover x distance band,
VV/VH quantiles + 1-dB histograms of the four groups (overall and per WorldCover class), NDVI of the four groups.
Outputs (output/l1mndwi0/): metrics_tier0.json, masks_P0_A.npy, masks_P0-s<k>_A.npy, masks_P0_s123_A.npy,
masks_REF_cal_A.npy, masks_REF_m0_A.npy, masks_L1w2_m0_A.npy (np.packbits, axis=None), mndwi_P0_mean_A.npy
(float16, 5-seed mean translated MNDWI, NaN outside valid SAR).
Nothing outside scripts/12_l1mndwi0 is modified (imports only).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "06_ablation"))
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "10_sameday"))
sys.path.insert(2, os.path.join(os.path.dirname(HERE), "11_loss_ablation"))
import common as C  # noqa: E402
import eval_variant as EV  # noqa: E402
import eval_sameday as ES  # noqa: E402
import eval_loss_ablation as LA  # noqa: E402

S = f"{C.ROOT}/output/l1mndwi0"
WC_FILE = LA.WC_FILE
SUBS = {"A_all": None, "A_sameday": {"Bolivia", "India"}, "A_gap": {"Somalia", "Spain"}}
BANDS = ES.BANDS
HIST_EDGES = np.arange(-40.0, 10.0001, 1.0)
QS = (10, 25, 50, 75, 90)


def ckpt_path(run):
    if run.startswith("t18-Paper-"):
        return f"{C.ROOT}/output/loco/runs/{run}/best_model.pt"
    if run.startswith("P0bi-"):
        return f"{S}/runs_bidir/{run.replace('P0bi-', 'P0-')}/best_model.pt"
    if run.startswith("la-"):
        return f"{C.ROOT}/output/loss_ablation/runs/{run}/best_model.pt"
    return f"{S}/runs/{run}/best_model.pt"


def tau0(ck):
    return LA.tau_mndwi0(ck)


def dist(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    h, _ = np.histogram(np.clip(x, HIST_EDGES[0], HIST_EDGES[-1]), HIST_EDGES)
    return dict(n_px=int(x.size), quantiles={f"p{q}": C._r(float(v)) for q, v in zip(QS, np.percentile(x, QS))},
                hist=h.tolist())


def extras(A, m):
    m = m & A.valid
    miss = A.gt & ~A.rule
    rec, unrec = miss & m, miss & ~m
    groups = {"rule_water": A.rule & A.gt, "recovered": rec, "unrecovered": unrec, "land": A.valid & ~A.gt}
    out = dict(unrecovered_rate=C._r(unrec.sum() / max(int(miss.sum()), 1)),
               unrecovered_share_of_label_water=C._r(unrec.sum() / max(int(A.gt.sum()), 1)))
    out["backscatter_by_group"] = {g: dict(vv_db=dist(A.VV[s]), vh_db=dist(A.VH[s]), ndvi=dist(A.NDVI[s]))
                                   for g, s in groups.items()}
    out["ndvi_median_unrecovered"] = C._r(float(np.nanmedian(A.NDVI[unrec])))
    wc = {}
    for c, nm in ES.WC_NAMES.items():
        cs = A.WC == c
        if not (miss & cs).any():
            continue
        bands = {}
        for lo, hi in BANDS:
            sel = miss & cs & ((A.D > lo) if lo > 0 else True) & (A.D <= hi)
            bands[f"{lo}-{hi if np.isfinite(hi) else 'inf'}m"] = dict(
                missed_px=int(sel.sum()), recovered_px=int((rec & sel).sum()),
                recovery=C._r((rec & sel).sum() / max(int(sel.sum()), 1)))
        wc[f"{c}_{nm}"] = dict(recovery_by_distance=bands,
                               backscatter_by_group={g: dict(vv_db=dist(A.VV[s & cs]), vh_db=dist(A.VH[s & cs]))
                                                     for g, s in groups.items()})
    out["worldcover_detail"] = wc
    return out


def evaluate(A, subs, m):
    r = {k: ES.table2(ES.sub_T(A, v), m[v]) for k, v in subs.items()}
    r["A_all"].update(extras(A, m))
    return r


def short(r):
    a = r["A_all"]
    return (f"IoU {a['iou']} P {a['precision']} R {a['recall']} FP% {a['fp_pct']} recov {a['recovery']} "
            f"w50 {a['recovery_within_50m']} lost {a['rule_water_lost']} | same IoU {r['A_sameday']['iou']} "
            f"P {r['A_sameday']['precision']} rec {r['A_sameday']['recovery']} | gap IoU {r['A_gap']['iou']} "
            f"P {r['A_gap']['precision']} rec {r['A_gap']['recovery']}")


def main():
    EV.ckpt_path = ckpt_path
    dev = C.device()
    t0 = time.time()
    A = C.load("A")
    A.valid, A.gt = A.lab_ok, A.gt_lab  # paper pixel set (gate: rule 0.4017, t18-Paper 5-seed 0.5012)
    assert hashlib.md5(open(WC_FILE, "rb").read()).hexdigest() == LA.WC_MD5
    A.WC = np.load(WC_FILE)
    n = len(A.names)
    subs = {k: (np.ones(n, bool) if ev is None else np.isin(A.groups, list(ev))) for k, ev in SUBS.items()}
    la = json.load(open(f"{C.ROOT}/output/loss_ablation/metrics.json"))
    tau_cal = {r: v["cal243"] for r, v in la["arms"]["REF"]["tau"].items()}
    print(f"loaded A {n} chips in {time.time() - t0:.0f}s; t18-Paper cal243 tau {tau_cal}", flush=True)

    res = dict(note=("Tier 0. mask = (translated MNDWI > 0) & valid, valid/gt = paper pixel set (lab_ok/gt_lab). "
                     "Votes: 5 seeds >= 3, 3 seeds >= 2. REF_cal = t18-Paper tau cal243 (paper value). "
                     "Subsets: A_sameday = Bolivia+India, A_gap = Somalia+Spain. Histograms: 1-dB bins "
                     f"{HIST_EDGES[0]:.0f}..{HIST_EDGES[-1]:.0f} (values clipped into the end bins)."),
               subsets={k: dict(n_chips=int(v.sum()), events=sorted(set(A.groups[v]))) for k, v in subs.items()},
               hist_edges_db=HIST_EDGES.tolist(), chips=A.names)
    res["rule"] = evaluate(A, subs, A.rule.copy())
    print("rule:", short(res["rule"]), flush=True)
    sum_m = np.zeros(A.valid.shape, np.float32)
    groups = {"P0": [f"P0-s{k}" for k in range(1, 6)], "REF": [f"t18-Paper-s{k}" for k in range(1, 6)],
              "P0bi": [f"P0bi-s{k}" for k in range(1, 6)], "L1w2xy": [f"L1w2xy-s{k}" for k in range(1, 4)],
              "L1w2": [f"la-L1only-s{k}" for k in range(1, 4)]}
    res["tau"], res["seeds"], res["groups"] = {}, {}, {}
    for g, runs in groups.items():
        masks, cal_masks = [], []
        for run in runs:
            t1 = time.time()
            ck, var, model = EV.load_run(run, dev)
            p = EV.score(A, ck, var, model, dev)
            del model
            t = tau0(ck)
            res["tau"][run] = dict(mndwi0=round(t, 4), epoch=int(ck.get("epoch", -1)),
                                   y_min_mndwi=float(ck["Y_min"][0]), y_max_mndwi=float(ck["Y_max"][0]))
            m = (p > t) & A.valid
            masks.append(m)
            if g == "REF":
                cal_masks.append((p > tau_cal[run]) & A.valid)
            if g == "P0":
                tm = p * (ck["Y_max"][0] - ck["Y_min"][0]) + ck["Y_min"][0]
                sum_m += np.nan_to_num(tm, nan=0.0).astype(np.float32)
                np.save(f"{S}/masks_{run}_A.npy", np.packbits(m, axis=None))
            if g in ("P0", "P0bi", "L1w2xy", "L1w2"):
                res["seeds"][run] = evaluate(A, subs, m)
                res["seeds"][run]["translated_mndwi"] = LA.translated_mndwi(A, p, ck)
                print(f"  {run}: tau0 {t:.4f} | {short(res['seeds'][run])} | {time.time() - t1:.0f}s", flush=True)
            else:
                print(f"  {run}: scored ({time.time() - t1:.0f}s)", flush=True)
            del p
        need = 3 if len(masks) == 5 else 2
        V = np.sum(masks, axis=0) >= need
        name = {"P0": "P0", "P0bi": "P0bi", "REF": "REF_m0", "L1w2xy": "L1w2xy_m0", "L1w2": "L1w2_m0"}[g]
        res["groups"][name] = dict(runs=runs, vote=f">= {need} of {len(masks)}", **evaluate(A, subs, V))
        np.save(f"{S}/masks_{name}_A.npy", np.packbits(V, axis=None))
        print(f"VOTE {name}: {short(res['groups'][name])}", flush=True)
        if g == "REF":
            Vc = np.sum(cal_masks, axis=0) >= 3
            res["groups"]["REF_cal"] = dict(runs=runs, vote=">= 3 of 5", tau=tau_cal, **evaluate(A, subs, Vc))
            np.save(f"{S}/masks_REF_cal_A.npy", np.packbits(Vc, axis=None))
            print(f"VOTE REF_cal: {short(res['groups']['REF_cal'])}", flush=True)
        if g in ("P0", "P0bi"):
            V3 = np.sum(masks[:3], axis=0) >= 2
            res["groups"][f"{g}_s123"] = dict(runs=runs[:3], vote=">= 2 of 3", **evaluate(A, subs, V3))
            np.save(f"{S}/masks_{g}_s123_A.npy", np.packbits(V3, axis=None))
            print(f"VOTE {g}_s123: {short(res['groups'][f'{g}_s123'])}", flush=True)
        if g == "P0":
            mean = np.where(np.isfinite(A.VV) & np.isfinite(A.VH), sum_m / 5, np.nan).astype(np.float16)
            np.save(f"{S}/mndwi_P0_mean_A.npy", mean)
        del masks, cal_masks
        C.save_json(res, f"{S}/metrics_tier0.json")
    res["masks"] = {"format": f"np.packbits(mask, axis=None); np.unpackbits(x)[:{n}*512*512].reshape({n},512,512).astype(bool)",
                    "chips": "see 'chips' (= common.load('A').names)"}
    C.save_json(res, f"{S}/metrics_tier0.json")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
