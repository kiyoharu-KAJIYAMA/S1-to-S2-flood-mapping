"""Evaluate retrained variants: per-seed detection threshold tau is F1-optimal on
the Sen1Floods11 training events (same sweep as eval_optical_axis.py --calib
train_events), then Test A/B are scored per seed and as a majority vote.

usage: eval_variant.py --name <label> --runs run1[,run2,run3] [--need N]
Runs live in output/ablation_recovery/runs/<run>/best_model.pt; a run named
cur-sN_run01 is read from output/<run>/ instead (original checkpoints)."""
import argparse
import json
import os

import numpy as np
import torch

import common as C
from variant_models import make_xy_model


def ckpt_path(run):
    p = f"{C.OUT}/runs/{run}/best_model.pt"
    return p if os.path.exists(p) else f"{C.ROOT}/output/{run}/best_model.pt"


def load_run(run, dev):
    ck = torch.load(ckpt_path(run), map_location="cpu", weights_only=False)
    v = ck.get("variant", dict(arch="unet", in_mask=[1, 1, 1], target="msi", n_out=3))
    m = make_xy_model(v["arch"], 3, v["n_out"]).to(dev)
    m.load_state_dict(ck["model_xy_state_dict"])
    return ck, v, m.eval()


def score(T, ck, v, model, dev):
    Z = C.preprocess(T.X, ck)
    for c, keep in enumerate(v["in_mask"]):
        if not keep:
            Z[:, c] = 0.5
    pred = C.predict(model, Z, dev, bs=8)
    if v["target"] == "label":
        pred = 1.0 / (1.0 + np.exp(-pred))
    return pred


def sweep_tau(pred, T):
    best = (0.5, -1.0)
    ok = T.lab_ok
    for t in np.linspace(0.02, 0.98, 49):
        r = C._prf(T.gt_lab, (pred > t) & ok, ok)
        if np.isfinite(r["f1"]) and r["f1"] > best[1]:
            best = (float(t), r["f1"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--need", type=int, default=0, help="votes needed (default: majority)")
    a = ap.parse_args()
    runs = a.runs.split(",")
    dev = C.device()
    sets = {n: C.load(n) for n in ("CAL", "A", "B")}
    out = {"name": a.name, "runs": runs, "per_seed": {}, "tau": {}}
    masks = {"A": [], "B": []}
    for run in runs:
        ck, v, model = load_run(run, dev)
        out["variant"] = v
        tau, f1 = sweep_tau(score(sets["CAL"], ck, v, model, dev), sets["CAL"])
        out["tau"][run] = dict(tau=tau, cal_f1=round(f1, 4), epoch=int(ck["epoch"]))
        out["per_seed"][run] = {}
        for n in ("B", "A"):
            m = (score(sets[n], ck, v, model, dev) > tau) & sets[n].valid
            masks[n].append(m)
            out["per_seed"][run][n] = C.evaluate(m, sets[n], per_group=False)
            print(f"  {run:14} tau {tau:.2f} {n}: {C.brief(out['per_seed'][run][n])}", flush=True)
    need = a.need or (len(runs) // 2 + 1)
    out["vote_need"] = need
    out["seed_mean"] = {}
    out["vote"] = {}
    for n in ("B", "A"):
        out["seed_mean"][n] = C.average([out["per_seed"][r][n] for r in runs])
        V = np.sum(masks[n], axis=0) >= need
        np.save(f"{C.OUT}/variants/votes_{a.name}_{n}.npy", V)
        out["vote"][n] = C.evaluate(V, sets[n])
        print(f"  VOTE({need}/{len(runs)}) {n}: {C.brief(out['vote'][n])}", flush=True)
    C.save_json(out, f"{C.OUT}/variants/{a.name}.json")


if __name__ == "__main__":
    main()
