"""Tier 0 supplement (requester 2026-10-26): P0 / P0bi / L1w2xy with tau calibrated on the 243 calibration chips,
exactly as t18-Paper (eval_variant.sweep_tau, 49 steps, F1-optimal on paper_calib_243.csv, paper pixel set lab_ok),
to separate the effect of the decision rule (MNDWI = 0 vs label-calibrated tau) from the effect of the loss.
Votes: 5 seeds >= 3 (P0, P0bi), 3 seeds >= 2 (L1w2xy, P0 s1..3). Metrics as eval_tier0.evaluate.
Outputs: output/l1mndwi0/metrics_tier0_cal.json, masks_P0_cal_A.npy, masks_P0bi_cal_A.npy, masks_L1w2xy_cal_A.npy.
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_tier0 as E  # noqa: E402

C, EV, ES, LA = E.C, E.EV, E.ES, E.LA
S = E.S


def main():
    EV.ckpt_path = E.ckpt_path
    dev = C.device()
    t0 = time.time()
    CAL, A = C.load("CAL"), C.load("A")
    A.valid, A.gt = A.lab_ok, A.gt_lab
    A.WC = np.load(E.WC_FILE)
    s243 = {r["chip"] for r in csv.DictReader(open(f"{C.ROOT}/output/loco/paper_calib_243.csv"))}
    sel = np.array([c in s243 for c in CAL.names])
    CAL243 = ES.sub_T(CAL, sel)
    n = len(A.names)
    subs = {k: (np.ones(n, bool) if ev is None else np.isin(A.groups, list(ev))) for k, ev in E.SUBS.items()}
    res = dict(note=__doc__, tau={}, seeds={}, groups={})
    groups = {"P0": [f"P0-s{k}" for k in range(1, 6)], "P0bi": [f"P0bi-s{k}" for k in range(1, 6)],
              "L1w2xy": [f"L1w2xy-s{k}" for k in range(1, 4)]}
    for g, runs in groups.items():
        masks = []
        for run in runs:
            t1 = time.time()
            ck, var, model = EV.load_run(run, dev)
            tc, f1 = EV.sweep_tau(EV.score(CAL, ck, var, model, dev)[sel], CAL243)
            p = EV.score(A, ck, var, model, dev)
            del model
            m = (p > tc) & A.valid
            del p
            masks.append(m)
            res["tau"][run] = dict(cal243=round(tc, 4), cal_f1=C._r(f1), mndwi0=round(LA.tau_mndwi0(ck), 4))
            res["seeds"][run] = {k: ES.table2(ES.sub_T(A, v), m[v]) for k, v in subs.items()}
            print(f"  {run}: tau {tc:.2f} (mndwi0 {res['tau'][run]['mndwi0']}) | {E.short(res['seeds'][run])} | "
                  f"{time.time() - t1:.0f}s", flush=True)
        need = 3 if len(runs) == 5 else 2
        V = np.sum(masks, axis=0) >= need
        res["groups"][f"{g}_cal"] = dict(runs=runs, vote=f">= {need} of {len(runs)}", **E.evaluate(A, subs, V))
        np.save(f"{S}/masks_{g}_cal_A.npy", np.packbits(V, axis=None))
        print(f"VOTE {g}_cal: {E.short(res['groups'][f'{g}_cal'])}", flush=True)
        if len(runs) == 5:
            V3 = np.sum(masks[:3], axis=0) >= 2
            res["groups"][f"{g}_cal_s123"] = dict(runs=runs[:3], vote=">= 2 of 3", **E.evaluate(A, subs, V3))
            print(f"VOTE {g}_cal_s123: {E.short(res['groups'][f'{g}_cal_s123'])}", flush=True)
        C.save_json(res, f"{S}/metrics_tier0_cal.json")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
