"""REQUEST_terra_20261026_l1only_mndwi0.md §3.3 (Tier 2): REQUEST_terra_20261023_sameday_v2_paperpool.md §3 tables
with the P0 decision (translated MNDWI > 0, no tau), paper pixel set (lab_ok / gt_lab), WorldCover 2020.

Arms P0-SD / MS / L1 / MX (output/sameday_v2/pools) and P0-NC (output/sameday/pools/NC), seeds 1-3, vote >= 2;
reference P0 = P0-s1..3 vote (>= 2) and P0-s1..5 vote (>= 3) (Tier 0 runs).
Subsets A_all (139), A_sameday (Bolivia + India, 83), A_gap (Somalia + Spain, 56); eval_sameday.table2 per subset.
Outputs: output/l1mndwi0/metrics_tier2.json, masks_P0-<arm>_A.npy (vote, np.packbits axis=None).
"""
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
ARMS = {"P0_3seed": [f"P0-s{k}" for k in (1, 2, 3)], "P0_5seed": [f"P0-s{k}" for k in range(1, 6)],
        **{a: [f"P0-{a}-s{k}" for k in (1, 2, 3)] for a in ("SD", "MS", "L1", "MX", "NC")}}


def main():
    EV.ckpt_path = lambda run: f"{S}/runs/{run}/best_model.pt"
    for runs in ARMS.values():
        for r in runs:
            assert os.path.exists(EV.ckpt_path(r)), r
    dev = C.device()
    t0 = time.time()
    A = C.load("A")
    A.valid, A.gt = A.lab_ok, A.gt_lab
    A.WC = np.load(LA.WC_FILE)
    n = len(A.names)
    subs = {"A_all": np.ones(n, bool), "A_sameday": np.isin(A.groups, ["Bolivia", "India"]),
            "A_gap": np.isin(A.groups, ["Somalia", "Spain"])}
    res = dict(note=__doc__, subsets={k: dict(n_chips=int(v.sum()), events=sorted(set(A.groups[v]))) for k, v in subs.items()},
               rule={k: ES.table2(ES.sub_T(A, v), A.rule[v]) for k, v in subs.items()}, tau={}, arms={})
    cache = {}
    for arm, runs in ARMS.items():
        R = {"runs": runs, "seeds": {}}
        masks = []
        for run in runs:
            if run not in cache:
                ck, var, model = EV.load_run(run, dev)
                p = EV.score(A, ck, var, model, dev)
                del model
                t = LA.tau_mndwi0(ck)
                res["tau"][run] = dict(mndwi0=round(t, 4), epoch=int(ck.get("epoch", -1)))
                cache[run] = (p > t) & A.valid
                del p
            m = cache[run]
            masks.append(m)
            R["seeds"][run] = {k: ES.table2(ES.sub_T(A, v), m[v]) for k, v in subs.items()}
        need = 3 if len(runs) == 5 else 2
        V = np.sum(masks, axis=0) >= need
        R["vote_rule"] = f">= {need} of {len(runs)}"
        R["vote"] = {k: ES.table2(ES.sub_T(A, v), V[v]) for k, v in subs.items()}
        if not arm.startswith("P0_"):
            np.save(f"{S}/masks_P0-{arm}_A.npy", np.packbits(V, axis=None))
        res["arms"][arm] = R
        a, sd, gp = R["vote"]["A_all"], R["vote"]["A_sameday"], R["vote"]["A_gap"]
        print(f"VOTE {arm}: IoU {a['iou']} P {a['precision']} FP% {a['fp_pct']} recov {a['recovery']} lost {a['rule_water_lost']} "
              f"| same IoU {sd['iou']} P {sd['precision']} rec {sd['recovery']} | gap IoU {gp['iou']} P {gp['precision']} "
              f"rec {gp['recovery']} | seeds {[R['seeds'][r]['A_all']['iou'] for r in runs]} ({time.time() - t0:.0f}s)", flush=True)
        C.save_json(res, f"{S}/metrics_tier2.json")
    res["masks"] = {"format": f"np.packbits(mask, axis=None); np.unpackbits(x)[:{n}*512*512].reshape({n},512,512).astype(bool)",
                    "chips": A.names}
    C.save_json(res, f"{S}/metrics_tier2.json")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
