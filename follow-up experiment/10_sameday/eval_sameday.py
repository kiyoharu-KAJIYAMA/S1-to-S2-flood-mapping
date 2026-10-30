"""Evaluation for REQUEST_terra_20261022_sameday_l1c_tau.md (§3-4).

For every arm (SD, MX, NC, L1, R0; plus CUR = the published cur-s1..3_run01 checkpoints as an
untrained-here reference) and every seed:
  tau is chosen three ways with eval_variant.sweep_tau (same F1 sweep, lab_ok pixels):
    cal243   the 243 Sen1Floods11 training chips the Paper runs trained on (output/loco/paper_calib_243.csv)
    cal_sd   the same-day events inside those 243 chips (Pakistan, Paraguay, USA; 134 chips)
    fixed05  tau = 0.5
  Test A (139 chips) is scored per seed and as the 3-seed majority (>= 2 votes) on three subsets:
    A_all, A_sameday (Bolivia + India, 83), A_gap (Spain + Somalia, 56),
  with the Table II metrics of scripts/08_loco/eval_loco.py (s61 / s62), recovery by distance band
  (0-10, 10-30, 30-50, 50-200, >200 m), land false alarms beyond 30 m split by VH and NDVI, and
  per-event land false alarms next to the rule's. The rule itself is scored on the same subsets.
  tau sweep 0.30..0.90 for the majority of each arm (cal243 seeds' masks are not reused: the vote
  is recomputed at each tau) -> tau_sweep.json.
WorldCover (Table III): processed/s1f11_testA_worldcover_2020.npy does not exist on terra, so the
class breakdown is skipped and recorded as "not_available"; if the file appears it is used.

Nothing under scripts/06_ablation is modified (eval_variant.ckpt_path is overridden in-process).
Outputs: output/sameday/metrics.json (merged over --arms), tau_sweep.json, masks_<arm>_A.npy
(majority, cal243, np.packbits), masks_<arm>_<method>_A.npy for the other two methods.
usage: eval_sameday.py --arms SD,MX [--seeds 1,2,3]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(os.path.dirname(HERE), "06_ablation")
sys.path.insert(0, ABL)
import common as C  # noqa: E402
import eval_variant as EV  # noqa: E402

S = f"{C.ROOT}/output/sameday"
SAMEDAY_CAL = {"Pakistan", "Paraguay", "USA"}
SUBSETS = {"A_all": None, "A_sameday": {"Bolivia", "India"}, "A_gap": {"Spain", "Somalia"}}
BANDS = [(0, 10), (10, 30), (30, 50), (50, 200), (200, np.inf)]
METHODS = ("cal243", "cal_sd", "fixed05")
SWEEP = [round(x, 2) for x in np.arange(0.30, 0.9001, 0.05)]
WC_FILE = f"{C.P}/s1f11_testA_worldcover_2020.npy"
WC_NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow",
            80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def ckpt_path(run):
    if run.startswith("cur-"):
        return f"{C.ROOT}/output/{run}/best_model.pt"
    return f"{S}/runs/{run}/best_model.pt"


def sub_T(T, sel):
    """A TestSet restricted to chips sel (bool over chips) with the fields the metrics need."""
    U = C.TestSet()
    U.name = T.name
    for k in ("valid", "gt", "rule", "D", "VH", "NDVI", "lab_ok", "gt_lab", "groups"):
        setattr(U, k, getattr(T, k)[sel])
    U.states = {k: v[sel] for k, v in T.states.items()}
    if hasattr(T, "WC"):
        U.WC = T.WC[sel]
    return U


def table2(T, m):
    """Table II row + Fig. 2a split (eval_loco.s61 / s62)."""
    m = m & T.valid
    r = C._prf(T.gt, m, T.valid)
    land = T.valid & ~T.gt
    miss = T.gt & ~T.rule
    r_ok_w = T.rule & T.gt
    out = dict(iou=C._r(r["iou"]), precision=C._r(r["precision"]), recall=C._r(r["recall"]),
               fn_pct=C._r(100 * r["fn"] / max(int(T.gt.sum()), 1)),
               fp_pct=C._r(100 * (land & m).sum() / max(int(land.sum()), 1)),
               missed_share=C._r(miss.sum() / max(int(T.gt.sum()), 1)),
               recovery=C._r((miss & m).sum() / max(int(miss.sum()), 1)),
               rule_water_lost=C._r((r_ok_w & ~m).sum() / max(int(r_ok_w.sum()), 1)),
               tp=r["tp"], fp=r["fp"], fn=r["fn"], gt_px=int(T.gt.sum()), land_px=int(land.sum()),
               missed_px=int(miss.sum()), recovered_px=int((miss & m).sum()),
               rule_tp_px=int(r_ok_w.sum()), rule_tp_lost_px=int((r_ok_w & ~m).sum()))
    rec = miss & m
    bands = {}
    for lo, hi in BANDS:
        sel = miss & (T.D > lo) & (T.D <= hi) if lo > 0 else miss & (T.D <= hi)
        bands[f"{lo}-{hi if np.isfinite(hi) else 'inf'}m"] = dict(
            missed_px=int(sel.sum()), recovered_px=int((rec & sel).sum()),
            recovery=C._r((rec & sel).sum() / max(int(sel.sum()), 1)))
    out["recovery_by_distance"] = bands
    out["recovery_within_50m"] = C._r((rec & (T.D <= 50)).sum() / max(int(rec.sum()), 1))
    far = land & ~T.rule & (T.D > 30)
    with np.errstate(invalid="ignore"):
        splits = {"all": far, "vh_dark(<-19)": far & (T.VH < -19), "vh_bright(>=-19)": far & (T.VH >= -19),
                  "ndvi_veg(>0.3)": far & (T.NDVI > 0.3), "ndvi_noveg(<=0.3)": far & (T.NDVI <= 0.3)}
    out["far_land_fp"] = {k: dict(land_px=int(v.sum()), fp_px=int((m & v).sum()),
                                  fp_pct=C._r(100 * (m & v).sum() / max(int(v.sum()), 1))) for k, v in splits.items()}
    ev = {}
    for name in sorted(set(T.groups)):
        s = T.groups == name
        l_all, l_far = land & s[:, None, None], far & s[:, None, None]
        ev[name] = dict(n_chips=int(s.sum()),
                        fp_pct_all_land=C._r(100 * (m & l_all).sum() / max(int(l_all.sum()), 1)),
                        fp_pct_far_land=C._r(100 * (m & l_far).sum() / max(int(l_far.sum()), 1)),
                        rule_fp_pct_all_land=C._r(100 * (T.rule & l_all).sum() / max(int(l_all.sum()), 1)),
                        iou=C._r(C._prf(T.gt[s], m[s], T.valid[s])["iou"]),
                        recovery=C._r((miss & m & s[:, None, None]).sum() / max(int((miss & s[:, None, None]).sum()), 1)))
    out["by_event"] = ev
    if hasattr(T, "WC"):
        wc = {}
        unrec = miss & ~m
        for c, nm in WC_NAMES.items():
            cs = T.WC == c
            n_miss = int((miss & cs).sum())
            if n_miss == 0 and int(cs.sum()) == 0:
                continue
            wc[f"{c}_{nm}"] = dict(missed_px=n_miss, recovery=C._r((rec & cs).sum() / max(n_miss, 1)),
                                   share_of_unrecovered=C._r((unrec & cs).sum() / max(int(unrec.sum()), 1)))
        out["worldcover"] = wc
    else:
        out["worldcover"] = "not_available (processed/s1f11_testA_worldcover_2020.npy missing on terra)"
    return out


def rule_row(T):
    """The same Table II numbers for the fixed rule itself (recovery is 0 by definition)."""
    return table2(T, T.rule.copy())


def training_summary(run):
    p = f"{S}/runs/{run}/loss_history.csv"
    if not os.path.exists(p):
        return None
    h = pd.read_csv(p)
    vc = [c for c in h.columns if "val" in c.lower() and "loss" in c.lower()]
    if not vc:
        return {"columns": list(h.columns)}
    v = h[vc[0]].to_numpy(float)
    i = int(np.nanargmin(v))
    return dict(epochs=int(len(h)), best_val=C._r(float(v[i])), best_epoch=i + 1,
                last_val=C._r(float(v[-1])), val_rise_after_best=C._r(float(v[-1] - v[i])),
                val_min_first_quarter=C._r(float(np.nanmin(v[: max(1, len(v) // 4)]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="SD,MX")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--need", type=int, default=2)
    a = ap.parse_args()
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    EV.ckpt_path = ckpt_path
    dev = C.device()
    t0 = time.time()

    CAL, A = C.load("CAL"), C.load("A")
    if os.path.exists(WC_FILE):
        A.WC = np.load(WC_FILE)
        print("WorldCover loaded", A.WC.shape, flush=True)
    else:
        print("WorldCover array missing -> Table III skipped", flush=True)
    cal243 = {r["chip"] for r in csv.DictReader(open(f"{C.ROOT}/output/loco/paper_calib_243.csv"))}
    sel243 = np.array([c in cal243 for c in CAL.names])
    selsd = sel243 & np.isin(CAL.groups, sorted(SAMEDAY_CAL))
    CAL243, CALSD = sub_T(CAL, sel243), sub_T(CAL, selsd)
    subs = {k: (np.ones(len(A.names), bool) if v is None else np.isin(A.groups, sorted(v))) for k, v in SUBSETS.items()}
    print(f"loaded: CAL {len(CAL.names)} (243 set {int(sel243.sum())}, same-day {int(selsd.sum())}), "
          f"A {len(A.names)} {{{', '.join(f'{k}:{int(v.sum())}' for k, v in subs.items())}}} in {time.time() - t0:.0f}s", flush=True)

    out_path = f"{S}/metrics.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.setdefault("note", "tau methods: cal243 = F1-optimal on paper_calib_243 chips; cal_sd = same on the "
                           "same-day events (Pakistan, Paraguay, USA) within them; fixed05 = 0.5. masks = (pred > tau) & valid; "
                           "vote = >= need of 3 seeds. Metrics as eval_loco.s61/s62: fp_pct over labelled land, fn_pct over "
                           "labelled water, rule_water_lost = rule-correct water the model drops / rule-correct water.")
    res["subsets"] = {k: dict(n_chips=int(v.sum()), events=sorted(set(A.groups[v]))) for k, v in subs.items()}
    res["calibration_sets"] = dict(cal243=int(sel243.sum()), cal_sd=int(selsd.sum()),
                                   cal_sd_events=sorted(SAMEDAY_CAL))
    res["rule"] = {k: rule_row(sub_T(A, v)) for k, v in subs.items()}
    res.setdefault("arms", {})
    sweep = json.load(open(f"{S}/tau_sweep.json")) if os.path.exists(f"{S}/tau_sweep.json") else {}

    for arm in arms:
        runs = [f"cur-s{s}_run01" for s in seeds] if arm == "CUR" else [f"{arm}-s{s}" for s in seeds]
        missing = [r for r in runs if not os.path.exists(ckpt_path(r))]
        if missing:
            print(f"SKIP {arm}: missing {missing}", flush=True)
            continue
        R = {"runs": runs, "tau": {}, "training": {}, "seeds": {}, "vote": {}}
        preds = []
        for run in runs:
            t1 = time.time()
            ck, var, model = EV.load_run(run, dev)
            pc = EV.score(CAL, ck, var, model, dev)
            taus = {}
            taus["cal243"] = EV.sweep_tau(pc[sel243], CAL243)
            taus["cal_sd"] = EV.sweep_tau(pc[selsd], CALSD)
            taus["fixed05"] = (0.5, float("nan"))
            R["tau"][run] = {m: dict(tau=round(t, 4), cal_f1=C._r(f)) for m, (t, f) in taus.items()}
            R["tau"][run]["epoch"] = int(ck.get("epoch", -1))
            R["training"][run] = training_summary(run)
            pa = EV.score(A, ck, var, model, dev)
            preds.append(pa)
            R["seeds"][run] = {}
            for meth in METHODS:
                m = (pa > taus[meth][0]) & A.valid
                R["seeds"][run][meth] = {k: table2(sub_T(A, v), m[v]) for k, v in subs.items()}
            print(f"  {run}: tau {' '.join(f'{m}={taus[m][0]:.2f}' for m in METHODS)} | "
                  f"A_all cal243 IoU {R['seeds'][run]['cal243']['A_all']['iou']} recov {R['seeds'][run]['cal243']['A_all']['recovery']} "
                  f"fp% {R['seeds'][run]['cal243']['A_all']['fp_pct']} | {time.time() - t1:.0f}s", flush=True)
            del model
        for meth in METHODS:
            V = np.sum([(preds[i] > R["tau"][runs[i]][meth]["tau"]) & A.valid for i in range(len(runs))], axis=0) >= a.need
            R["vote"][meth] = {k: table2(sub_T(A, v), V[v]) for k, v in subs.items()}
            suffix = "" if meth == "cal243" else f"_{meth}"
            np.save(f"{S}/masks_{arm}{suffix}_A.npy", np.packbits(V, axis=None))
            r = R["vote"][meth]["A_all"]
            print(f"  VOTE {arm} {meth}: A_all IoU {r['iou']} P {r['precision']} R {r['recall']} recov {r['recovery']} "
                  f"fp% {r['fp_pct']} lost {r['rule_water_lost']} | A_sameday IoU {R['vote'][meth]['A_sameday']['iou']} "
                  f"recov {R['vote'][meth]['A_sameday']['recovery']} fp% {R['vote'][meth]['A_sameday']['fp_pct']}", flush=True)
        sw = {}
        for t in SWEEP:
            V = np.sum([(p > t) & A.valid for p in preds], axis=0) >= a.need
            sw[f"{t:.2f}"] = {}
            for k in ("A_all", "A_sameday", "A_gap"):
                r = table2(sub_T(A, subs[k]), V[subs[k]])
                sw[f"{t:.2f}"][k] = dict(iou=r["iou"], recovery=r["recovery"], fp_pct=r["fp_pct"],
                                         far_fp_pct=r["far_land_fp"]["all"]["fp_pct"], rule_water_lost=r["rule_water_lost"])
        sweep[arm] = sw
        res["arms"][arm] = R
        C.save_json(res, out_path)
        C.save_json(sweep, f"{S}/tau_sweep.json")
        print(f"  {arm} saved ({time.time() - t0:.0f}s elapsed)", flush=True)
    res["masks"] = {"format": "np.packbits(mask, axis=None); np.unpackbits(x)[:139*512*512].reshape(139,512,512).astype(bool)",
                    "chips": A.names}
    C.save_json(res, out_path)
    print(f"done in {time.time() - t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
