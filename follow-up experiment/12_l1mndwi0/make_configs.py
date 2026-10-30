#!/usr/bin/env python3
"""REQUEST_terra_20261026_l1only_mndwi0.md §1-2: P0 configs from the template
output/loss_ablation/runs/la-L1only-s1 (kl 0, dice 0, mndwi_weight 2.0) with mndwi_weight 1.0.
Only mndwi_weight, seed, date_out, outpath change (Tier 0). Prints the diff to the template and checks that
t18-Paper-s1..5 differ from each other only in seed/date_out (so seed is the only per-run change).
Tier 1 (Fig. 5): P0 config with in_mask 1,1,0 (v7c VV+VH) / 1,0,0 (v7b VV only) / 0,1,0 (v7a VH only), seeds 1-3;
  pool = t18-Paper (trn18 fold_Paper), as confirmed by the requester on 2026-10-26 (the 2026-10-16 v7 runs used trn16).
Tier 2 (sameday): P0 config with rootpath/datapath (and data_prefix for NC) of output/sameday_v2/runs/<arm>-s1
  (SD, MS, L1, MX) and output/sameday/runs/NC-s1 (previous NC pool), seeds 1-3.
usage: make_configs.py --tier 0|1|2
"""
import argparse
import json
import os

R = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
D = f"{R}/scripts/12_l1mndwi0"
TPL = f"{R}/output/loss_ablation/runs/la-L1only-s1/config_la-L1only-s1.json"
OUT = f"{R}/output/l1mndwi0/runs"


def tier0():
    tpl = json.load(open(TPL))
    ref = [json.load(open(f"{R}/output/loco/runs/t18-Paper-s{k}/config_t18-Paper-s{k}.json")) for k in range(1, 6)]
    for k, c in enumerate(ref, 1):
        d = {x for x in c if c[x] != ref[0].get(x)}
        assert d <= {"seed", "date_out"} and c["seed"] == k, (k, d)
    d = {x: (ref[0].get(x), tpl[x]) for x in tpl if tpl[x] != ref[0].get(x)}
    print("template la-L1only-s1 vs t18-Paper-s1:", d)
    out = []
    for k in range(1, 6):
        cfg = dict(tpl, mndwi_weight=1.0, seed=k, date_out=f"P0-s{k}", outpath=OUT)
        diff = {x: (tpl.get(x), cfg[x]) for x in cfg if tpl.get(x) != cfg[x]}
        assert set(diff) <= {"mndwi_weight", "seed", "date_out", "outpath"}, diff
        p = f"{D}/configs/{cfg['date_out']}.json"
        json.dump(cfg, open(p, "w"), indent=4)
        out.append(p)
        print(cfg["date_out"], "<- la-L1only-s1", diff)
    return out


def tier1():
    out = []
    for arm, im in (("v7c", "1,1,0"), ("v7b", "1,0,0"), ("v7a", "0,1,0")):
        for k in range(1, 4):
            base = json.load(open(f"{D}/configs/P0-s{k}.json"))
            cfg = dict(base, in_mask=im, date_out=f"P0-{arm}-s{k}")
            diff = {x: (base.get(x), cfg[x]) for x in cfg if base.get(x) != cfg[x]}
            assert set(diff) <= {"in_mask", "date_out"}, diff
            p = f"{D}/configs/{cfg['date_out']}.json"
            json.dump(cfg, open(p, "w"), indent=4)
            out.append(p)
            print(cfg["date_out"], "<- P0-s%d" % k, diff)
    return out


def tier2():
    src = {a: f"{R}/output/sameday_v2/runs/{a}-s1/config_{a}-s1.json" for a in ("SD", "MS", "L1", "MX")}
    src["NC"] = f"{R}/output/sameday/runs/NC-s1/config_NC-s1.json"
    out = []
    for k in range(1, 4):
        for arm in ("SD", "MS", "L1", "MX", "NC"):
            s = json.load(open(src[arm]))
            base = json.load(open(f"{D}/configs/P0-s{k}.json"))
            cfg = dict(base, rootpath=s["rootpath"], datapath=s["datapath"], data_prefix=s["data_prefix"],
                       date_out=f"P0-{arm}-s{k}")
            diff = {x: (base.get(x), cfg[x]) for x in cfg if base.get(x) != cfg[x]}
            assert set(diff) <= {"rootpath", "datapath", "data_prefix", "date_out"}, diff
            p = f"{D}/configs/{cfg['date_out']}.json"
            json.dump(cfg, open(p, "w"), indent=4)
            out.append(p)
            print(cfg["date_out"], "<- P0-s%d" % k, diff)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0)
    a = ap.parse_args()
    paths = {0: tier0, 1: tier1, 2: tier2}[a.tier]()
    q = f"{D}/queue_tier{a.tier}.txt"
    if not os.path.exists(q):
        open(q, "w").write("\n".join(paths) + "\n")
