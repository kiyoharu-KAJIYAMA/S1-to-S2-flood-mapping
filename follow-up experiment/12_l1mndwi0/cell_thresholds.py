"""Fig. 5 cell boundaries for REQUEST_terra_20261026_l1only_mndwi0.md §3.2 (re-defined on the requester's
instruction of 2026-10-26: derive the VH / NDVI splits from the literature and the training side, not by hand).

VH (dB, Lee-filtered as in common.load), computed on the 243 calibration chips (paper_calib_243.csv, training side,
paper pixel set lab_ok) so Test A is not used to set the boundary:
  ml_crossing  maximum-likelihood (equal-prior Bayes) boundary between the label-water and label-land VH densities:
               first 0.1-dB bin between the two modes where p(VH | water) <= p(VH | land). "Dark" land = land whose
               VH is more likely under the water class, i.e. a water look-alike for a backscatter classifier.
  otsu         Otsu (1979) on all valid pixels (unsupervised, between-class variance)
  ki           Kittler & Illingworth (1986) minimum-error threshold on all valid pixels (as in Martinis et al. 2009)
  The same three are also reported for Test A for reference only.
NDVI: Sobrino et al. (2004, RSE 90:434-440, p. 436) NDVI-threshold classes: < 0.2 bare soil, 0.2-0.5 mixed,
  > 0.5 fully vegetated. Fixed values, not fitted.
Output: output/l1mndwi0/cell_thresholds.json
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "06_ablation"))
import common as C  # noqa: E402

LO, HI, STEP = -40.0, 5.0, 0.1


def _hist(x):
    e = np.arange(LO, HI + STEP / 2, STEP)
    h, _ = np.histogram(np.clip(x, LO, HI), e)
    return h / max(h.sum(), 1), (e[:-1] + e[1:]) / 2


def ki(x):
    h, c = _hist(x)
    best, bt = np.inf, None
    for t in range(5, len(c) - 5):
        p1, p2 = h[:t].sum(), h[t:].sum()
        if p1 < 1e-4 or p2 < 1e-4:
            continue
        m1, m2 = (h[:t] * c[:t]).sum() / p1, (h[t:] * c[t:]).sum() / p2
        s1 = np.sqrt((h[:t] * (c[:t] - m1) ** 2).sum() / p1)
        s2 = np.sqrt((h[t:] * (c[t:] - m2) ** 2).sum() / p2)
        if s1 <= 0 or s2 <= 0:
            continue
        J = 1 + 2 * (p1 * np.log(s1) + p2 * np.log(s2)) - 2 * (p1 * np.log(p1) + p2 * np.log(p2))
        if J < best:
            best, bt = J, c[t]
    return float(bt)


def otsu(x):
    h, c = _hist(x)
    w, mu = np.cumsum(h), np.cumsum(h * c)
    sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return float(c[np.argmax(sb)])


def ml_crossing(xw, xl):
    dw, c = _hist(xw)
    dl, _ = _hist(xl)
    mw, ml = c[np.argmax(dw)], c[np.argmax(dl)]
    i = np.where((c > mw) & (c < ml) & (dw <= dl))[0]
    return float(c[i[0]])


def stats(T, sel=None):
    g = (lambda a: a) if sel is None else (lambda a: a[sel])
    ok, gt, vh = g(T.lab_ok), g(T.gt_lab), g(T.VH)
    v = ok & np.isfinite(vh)
    return dict(ml_crossing=round(ml_crossing(vh[v & gt], vh[v & ~gt]), 2), otsu=round(otsu(vh[v]), 2),
                ki=round(ki(vh[v]), 2), water_vh_median=round(float(np.median(vh[v & gt])), 2),
                land_vh_median=round(float(np.median(vh[v & ~gt])), 2), n_px=int(v.sum()))


def main():
    cal = C.load("CAL")
    s243 = {r["chip"] for r in csv.DictReader(open(f"{C.ROOT}/output/loco/paper_calib_243.csv"))}
    sel = np.array([n in s243 for n in cal.names])
    out = dict(note=__doc__, bin_db=STEP, vh_cal243=stats(cal, sel), vh_testA_reference_only=stats(C.load("A")),
               ndvi=dict(bare_lt=0.2, full_veg_gt=0.5, source="Sobrino, Jimenez-Munoz & Paolini 2004, RSE 90:434-440, p.436"))
    C.save_json(out, f"{C.ROOT}/output/l1mndwi0/cell_thresholds.json")
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main()
