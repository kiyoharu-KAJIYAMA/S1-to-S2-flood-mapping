"""REQUEST_terra_20261026_l1only_mndwi0.md §3.2 (Tier 1, Fig. 5): P0-v7c (VV+VH), P0-v7b (VV only), P0-v7a (VH only),
seeds 1-3, trained on the t18-Paper pool (trn18 fold_Paper, requester's decision 2026-10-26).

Decision: translated MNDWI > 0 per checkpoint (no tau); 3-seed vote >= 2. Pixel set = paper (lab_ok / gt_lab);
note the 2026-10-16 fp_cells.py used common.valid, so its far-land pixel counts differ slightly.
far = valid & ~gt & ~rule & (D > 30 m); all_land = valid & ~gt.
Cells (output/l1mndwi0/fig5_cell_definition.md, thresholds from cell_thresholds.json):
  primary  VH split = Otsu (1979) on the CAL243 VH histogram (-19.45 dB; label-free, requester's decision 2026-10-26,
           replacing the ML crossing -18.65 dB); NDVI Sobrino 2004: bare < 0.2, mixed 0.2-0.5, veg > 0.5
           cells6 (dark|bright x bare|mixed|veg), cells2x2 (dark|bright x bare|veg, mixed left out),
           cells4 (dark, bright_bare, bright_mixed, bright_veg); NaN VH or NaN NDVI in no cell (counted separately)
  legacy   2026-10-16 definition (VH -19, NDVI 0.3; cells3 dim/bright_noveg/bright_veg, cells4) for continuity
  sens_lit VH in {primary (Otsu), primary +-1, ML crossing, KI, -19, -18, -20} x veg boundary b in {0.2, 0.3, 0.4, 0.5},
           2x2 with bare = NDVI < 0.2 and veg = NDVI > b (b = 0.2 gives bare | non-bare)
  sens_req request grid VH {-18, -19, -20} x NDVI {0.2, 0.3, 0.4}, legacy style (veg = NDVI > b, noveg = NDVI <= b)
Ratios (vote, far): dark cell VH-only / VV+VH and veg cell VV-only / VV+VH for every definition.
Also eval_sameday.table2 per seed and vote (recovery overall, by distance band, by event; Table II).
Outputs: output/l1mndwi0/fp_cells.json, metrics_tier1.json, masks_P0-v7?_A.npy (vote), masks_P0-v7?-s<k>_A.npy.
"""
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
VARIANTS = {"v7c": "VV+VH", "v7b": "VV only", "v7a": "VH only"}
SEEDS = (1, 2, 3)


def cells_of(A, th):
    with np.errstate(invalid="ignore"):
        vh_ok, nd_ok = np.isfinite(A.VH), np.isfinite(A.NDVI)

        def split(t):
            return A.VH < t, (A.VH >= t)

        def ndvi3(lo=0.2, hi=0.5):
            return A.NDVI < lo, (A.NDVI >= lo) & (A.NDVI <= hi), A.NDVI > hi

        defs = {}
        d, b = split(th["primary"])
        bare, mixed, veg = ndvi3()
        defs["primary"] = {
            "cells6": {f"{x}_{y}": dx & ny for x, dx in (("dark", d), ("bright", b))
                       for y, ny in (("bare", bare), ("mixed", mixed), ("veg", veg))},
            "cells2x2": {"dark_bare": d & bare, "dark_veg": d & veg, "bright_bare": b & bare, "bright_veg": b & veg},
            "cells4": {"dark": d & nd_ok, "bright_bare": b & bare, "bright_mixed": b & mixed, "bright_veg": b & veg}}
        d, b = split(-19.0)
        v, nv = A.NDVI > 0.3, A.NDVI <= 0.3
        defs["legacy"] = {"cells3": {"dim": d, "bright_noveg": b & nv, "bright_veg": b & v},
                          "cells4": {"dark_bare": d & ~v, "dark_veg": d & v, "bright_bare": b & ~v, "bright_veg": b & v}}
        for name, t in th["sens_vh"].items():
            d, b = split(t)
            for bv in (0.2, 0.3, 0.4, 0.5):
                v = A.NDVI > bv
                defs[f"sens_lit/vh={name}({t})/veg>{bv}"] = {"cells2x2": {
                    "dark_bare": d & bare, "dark_veg": d & v, "bright_bare": b & bare, "bright_veg": b & v}}
        for t in (-18.0, -19.0, -20.0):
            d, b = split(t)
            for bv in (0.2, 0.3, 0.4):
                v, nv = A.NDVI > bv, A.NDVI <= bv
                defs[f"sens_req/vh={t}/ndvi={bv}"] = {"cells2x2": {
                    "dark_bare": d & nv, "dark_veg": d & v, "bright_bare": b & nv, "bright_veg": b & v},
                    "cells3": {"dim": d, "bright_noveg": b & nv, "bright_veg": b & v}}
    return defs, ~vh_ok, ~nd_ok


def count(m, region, defs):
    def one(sel):
        n, f = int(sel.sum()), int((m & sel).sum())
        return dict(land_px=n, fp_px=f, fp_rate=C._r(f / max(n, 1)))
    out = {"all": one(region)}
    for dn, groups in defs.items():
        out[dn] = {g: {c: one(region & s) for c, s in cells.items()} for g, cells in groups.items()}
    return out


def ratios(vote_far):
    """dark-cell VH only / VV+VH and veg-cell VV only / VV+VH, per definition and cell group."""
    out = {}
    base = vote_far["v7c"]
    for dn in base:
        if dn == "all":
            continue
        for g, cells in base[dn].items():
            r = {}
            for c in cells:
                b = cells[c]["fp_rate"]
                r[c] = dict(vh_only_over_vvvh=C._r(vote_far["v7a"][dn][g][c]["fp_rate"] / b) if b else None,
                            vv_only_over_vvvh=C._r(vote_far["v7b"][dn][g][c]["fp_rate"] / b) if b else None)
            out[f"{dn}|{g}"] = r
    return out


def main():
    EV.ckpt_path = lambda run: f"{S}/runs/{run}/best_model.pt"
    for v in VARIANTS:
        for s in SEEDS:
            assert os.path.exists(EV.ckpt_path(f"P0-{v}-s{s}")), f"P0-{v}-s{s}"
    dev = C.device()
    t0 = time.time()
    A = C.load("A")
    A.valid, A.gt = A.lab_ok, A.gt_lab
    A.WC = np.load(LA.WC_FILE)
    ct = json.load(open(f"{S}/cell_thresholds.json"))["vh_cal243"]
    th = {"primary": ct["otsu"],
          "sens_vh": {"primary_otsu": ct["otsu"], "primary-1": round(ct["otsu"] - 1, 2),
                      "primary+1": round(ct["otsu"] + 1, 2), "ml_crossing": ct["ml_crossing"], "ki": ct["ki"],
                      "legacy": -19.0, "req-18": -18.0, "req-20": -20.0}}
    defs, nan_vh, nan_nd = cells_of(A, th)
    land = A.valid & ~A.gt
    far = land & ~A.rule & (A.D > 30)
    n = len(A.names)
    subs = {"A_all": np.ones(n, bool), "A_sameday": np.isin(A.groups, ["Bolivia", "India"]),
            "A_gap": np.isin(A.groups, ["Somalia", "Spain"])}
    zero = np.zeros_like(far)
    fp = dict(note=__doc__, thresholds=th, n_chips=n, chips=A.names,
              population={"far": count(zero, far, defs), "all_land": count(zero, land, defs),
                          "nan_vh_px": {"far": int((far & nan_vh).sum()), "all_land": int((land & nan_vh).sum())},
                          "nan_ndvi_px": {"far": int((far & nan_nd).sum()), "all_land": int((land & nan_nd).sum())}},
              tau={}, variants={})
    met = dict(note="Tier 1 Table II rows (eval_sameday.table2), translated MNDWI > 0, paper pixel set", variants={})
    print(f"loaded A in {time.time() - t0:.0f}s; far {int(far.sum()):,} px; thresholds {th}", flush=True)
    vote_far = {}
    for v, lab in VARIANTS.items():
        runs = [f"P0-{v}-s{s}" for s in SEEDS]
        fp["variants"][v] = {"label": lab, "runs": runs}
        met["variants"][v] = {"label": lab, "seeds": {}}
        masks = []
        for run in runs:
            t1 = time.time()
            ck, var, model = EV.load_run(run, dev)
            p = EV.score(A, ck, var, model, dev)
            del model
            t = LA.tau_mndwi0(ck)
            m = (p > t) & A.valid
            del p
            masks.append(m)
            fp["tau"][run] = dict(mndwi0=round(t, 4), epoch=int(ck.get("epoch", -1)), in_mask=var["in_mask"])
            fp["variants"][v][run] = {"far": count(m, far, defs), "all_land": count(m, land, defs)}
            met["variants"][v]["seeds"][run] = {k: ES.table2(ES.sub_T(A, s), m[s]) for k, s in subs.items()}
            np.save(f"{S}/masks_{run}_A.npy", np.packbits(m, axis=None))
            r = met["variants"][v]["seeds"][run]["A_all"]
            print(f"  {run}: tau0 {t:.4f} IoU {r['iou']} P {r['precision']} FP% {r['fp_pct']} recov {r['recovery']} "
                  f"far FP {fp['variants'][v][run]['far']['all']['fp_rate']} {time.time() - t1:.0f}s", flush=True)
        V = np.sum(masks, axis=0) >= 2
        np.save(f"{S}/masks_P0-{v}_A.npy", np.packbits(V, axis=None))
        fp["variants"][v]["vote"] = {"far": count(V, far, defs), "all_land": count(V, land, defs)}
        vote_far[v] = fp["variants"][v]["vote"]["far"]
        met["variants"][v]["vote"] = {k: ES.table2(ES.sub_T(A, s), V[s]) for k, s in subs.items()}
        pr = vote_far[v]["primary"]
        print(f"VOTE {v}: IoU {met['variants'][v]['vote']['A_all']['iou']} recov {met['variants'][v]['vote']['A_all']['recovery']} "
              f"far {vote_far[v]['all']['fp_rate']} | primary 2x2 "
              f"{ {c: x['fp_rate'] for c, x in pr['cells2x2'].items()} } | legacy cells3 "
              f"{ {c: x['fp_rate'] for c, x in vote_far[v]['legacy']['cells3'].items()} }", flush=True)
        C.save_json(fp, f"{S}/fp_cells.json")
        C.save_json(met, f"{S}/metrics_tier1.json")
    fp["ratios_vote_far"] = ratios(vote_far)
    for k in ("primary|cells2x2", "primary|cells4", "legacy|cells3", "legacy|cells4"):
        print(k, fp["ratios_vote_far"][k], flush=True)
    fp["masks"] = {"format": f"np.packbits(mask, axis=None); np.unpackbits(x)[:{n}*512*512].reshape({n},512,512).astype(bool)"}
    C.save_json(fp, f"{S}/fp_cells.json")
    C.save_json(met, f"{S}/metrics_tier1.json")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
