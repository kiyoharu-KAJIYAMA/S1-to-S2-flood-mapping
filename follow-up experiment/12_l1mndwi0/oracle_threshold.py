"""Oracle backscatter thresholds per event (requester question 2026-10-26): how good can the threshold rule get if
its thresholds are chosen per event with the Test A labels known (an upper bound, not a usable method)?

Rules searched on a 0.25-dB grid (VV, VH in dB, Lee-filtered as common.load), paper pixel set (lab_ok / gt_lab):
  vv_and_vh  water = VV < a & VH < b   (the paper's rule form; fixed a = -5, b = -22)
  vh_only    water = VH < b
  vv_only    water = VV < a
Best IoU per event and for all 139 chips pooled (one global pair), and per-event oracle pairs pooled back together
(per-event thresholds, IoU over all 139 chips). Computed from 2-D histograms of label water / land counts.
Output: output/l1mndwi0/oracle_threshold.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "06_ablation"))
import common as C  # noqa: E402

STEP = 0.25
EDGES = np.arange(-45.0, 15.0 + STEP / 2, STEP)   # bin i = [EDGES[i], EDGES[i+1])


def hist2(vv, vh, w):
    h, _, _ = np.histogram2d(np.clip(vv, EDGES[0], EDGES[-1] - 1e-6), np.clip(vh, EDGES[0], EDGES[-1] - 1e-6),
                             bins=[EDGES, EDGES], weights=w)
    return h


def cum(h):
    """c[i, j] = count with VV < EDGES[i] and VH < EDGES[j]."""
    c = np.zeros((len(EDGES), len(EDGES)))
    c[1:, 1:] = h.cumsum(0).cumsum(1)
    return c


def best(cw, cl, nw, kind):
    tp, fp = cw, cl
    iou = tp / np.maximum(nw + fp, 1)
    if kind == "vh_only":
        iou = iou[-1:, :]
    elif kind == "vv_only":
        iou = iou[:, -1:]
    i, j = np.unravel_index(np.argmax(iou), iou.shape)
    if kind == "vh_only":
        i = len(EDGES) - 1
    if kind == "vv_only":
        j = len(EDGES) - 1
    t, f = cw[i, j], cl[i, j]
    return dict(vv=None if kind == "vh_only" else float(EDGES[i]), vh=None if kind == "vv_only" else float(EDGES[j]),
                iou=C._r(t / max(nw + f, 1)), precision=C._r(t / max(t + f, 1)), recall=C._r(t / max(nw, 1)),
                tp=int(t), fp=int(f)), (i, j)


def fixed(cw, cl, nw):
    i, j = int(np.argmin(abs(EDGES + 5))), int(np.argmin(abs(EDGES + 22)))
    t, f = cw[i, j], cl[i, j]
    return dict(vv=-5.0, vh=-22.0, iou=C._r(t / max(nw + f, 1)), precision=C._r(t / max(t + f, 1)),
                recall=C._r(t / max(nw, 1)))


def main():
    A = C.load("A")
    ok, gt = A.lab_ok, A.gt_lab
    out = dict(note=__doc__, grid_db=STEP, events={})
    H = {}
    for name in sorted(set(A.groups)) + ["ALL"]:
        s = np.ones(len(A.names), bool) if name == "ALL" else (A.groups == name)
        v = ok[s]
        vv, vh, g = A.VV[s][v], A.VH[s][v], gt[s][v]
        cw, cl = cum(hist2(vv[g], vh[g], None)), cum(hist2(vv[~g], vh[~g], None))
        nw = int(g.sum())
        H[name] = (cw, cl, nw)
        r = dict(n_chips=int(s.sum()), water_px=nw, land_px=int((~g).sum()), fixed_rule=fixed(cw, cl, nw))
        for k in ("vv_and_vh", "vh_only", "vv_only"):
            r[k], _ = best(cw, cl, nw, k)
        out["events"][name] = r
        print(name, "fixed", r["fixed_rule"]["iou"], {k: (r[k]["vv"], r[k]["vh"], r[k]["iou"], r[k]["precision"])
                                                     for k in ("vv_and_vh", "vh_only", "vv_only")}, flush=True)
    # per-event oracle thresholds pooled over all 139 chips
    pooled = {}
    for k in ("vv_and_vh", "vh_only"):
        t = f = 0
        for name in sorted(set(A.groups)):
            _, (i, j) = best(*H[name], k)
            t += H[name][0][i, j]
            f += H[name][1][i, j]
        nw = H["ALL"][2]
        pooled[k] = dict(iou=C._r(t / max(nw + f, 1)), precision=C._r(t / max(t + f, 1)), recall=C._r(t / max(nw, 1)))
    out["per_event_oracle_pooled_all139"] = pooled
    print("pooled per-event oracle", pooled, flush=True)
    C.save_json(out, f"{C.ROOT}/output/l1mndwi0/oracle_threshold.json")


if __name__ == "__main__":
    main()
