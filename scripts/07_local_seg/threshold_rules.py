"""Why the calibrated rule is VV < -5 & VH < -22 dB, and threshold rules that use both polarizations.
Each rule is calibrated by F1 on the 307 Sen1Floods11 training-event chips (same pixel set as eval_optical_axis.py),
from 2-D histograms of Lee VV/VH (0.25 dB bins), then scored on Test A exactly like Table II."""
import sys
import numpy as np
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/06_ablation")
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
sys.path.insert(0, "/home/kk/jp_claude/RMSD/output/from_b200/ckpt/_code")
import common as C  # noqa: E402
from testA_common import TestA  # noqa: E402

E = np.arange(-40, 10.01, 0.25)
cen = (E[:-1] + E[1:]) / 2
CV, CH = np.meshgrid(cen, cen, indexing="ij")

cal = C.load("CAL")
ok = cal.lab_ok & np.isfinite(cal.VV) & np.isfinite(cal.VH)
w, l = ok & cal.gt_lab, ok & ~cal.gt_lab
Hw = np.histogram2d(np.clip(cal.VV[w], -40, 10), np.clip(cal.VH[w], -40, 10), [E, E])[0]
Hl = np.histogram2d(np.clip(cal.VV[l], -40, 10), np.clip(cal.VH[l], -40, 10), [E, E])[0]
NW = Hw.sum()


def f1(sel):
    tp = Hw[sel].sum(); fp = Hl[sel].sum()
    return 2 * tp / (tp + fp + NW)


print("CAL water median VV %.1f VH %.1f | land median VV %.1f VH %.1f dB" % (
    np.median(cal.VV[w]), np.median(cal.VH[w]), np.median(cal.VV[l]), np.median(cal.VH[l])))
print("CAL water VV 95th pct %.1f dB, land VV 5th/50th pct %.1f / %.1f dB" % (
    np.percentile(cal.VV[w], 95), np.percentile(cal.VV[l], 5), np.percentile(cal.VV[l], 50)))

rules = {}
best = max(((a, b) for a in np.arange(-25, 5.01, 0.5) for b in np.arange(-32, -10.01, 0.5)),
           key=lambda t: f1((CV < t[0]) & (CH < t[1])))
rules["AND, VV grid to +5"] = ("and", best)
for vv in (-5.0, -10.0, -12.0, -15.0, -18.0):
    b = max(np.arange(-32, -10.01, 0.5), key=lambda b: f1((CV < vv) & (CH < b)))
    rules["AND, VV < %.0f fixed" % vv] = ("and", (vv, b))
rules["VH only"] = ("and", (99.0, max(np.arange(-32, -10.01, 0.25), key=lambda b: f1(CH < b))))
rules["VV only"] = ("and", (max(np.arange(-30, 0.01, 0.25), key=lambda a: f1(CV < a)), 99.0))
rules["OR"] = ("or", max(((a, b) for a in np.arange(-30, -5.01, 0.5) for b in np.arange(-32, -15.01, 0.5)),
                         key=lambda t: f1((CV < t[0]) | (CH < t[1]))))
lin = max(((k, c) for k in np.arange(-1.0, 1.501, 0.05) for c in np.arange(-60, -10.01, 0.25)),
          key=lambda t: f1(CH + t[0] * CV < t[1]))
rules["linear VH + k*VV < c"] = ("lin", lin)
sdwi = lambda vv, vh: np.log(10 * 10 ** (vv / 10) * 10 ** (vh / 10)) - 8  # noqa: E731  Jia et al. 2019
S = sdwi(CV, CH)  # water has low SDWI here; equivalent to VV + VH < c
rules["SDWI < t (VV+VH)"] = ("sdwi", (max(np.arange(-30, 0.01, 0.05), key=lambda t: f1(S < t)),))


def mask(kind, p, VV, VH):
    if kind == "and":
        return (VV < p[0]) & (VH < p[1])
    if kind == "or":
        return (VV < p[0]) | (VH < p[1])
    if kind == "lin":
        return VH + p[0] * VV < p[1]
    return sdwi(VV, VH) < p[0]


T = TestA()
fin = T.fin


def score(M):
    M = M & fin & T.valid
    tp = (M & T.water).sum(); fp = (M & T.land).sum(); fn = (~M & T.water).sum()
    return tp / (tp + fp + fn), tp / max(tp + fp, 1), tp / max(tp + fn, 1), 100 * fp / T.land.sum()


print("\n%-24s %-22s %7s | %6s %6s %6s %6s" % ("rule", "params", "CAL F1", "IoU", "P", "R", "FP%"))
for name, (kind, p) in rules.items():
    sel = mask(kind, p, CV, CH)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        Mt = mask(kind, p, T.VV, T.VH)
    print("%-24s %-22s %7.4f | %6.3f %6.3f %6.3f %6.1f" % (name, ",".join("%.2f" % x for x in p), f1(sel), *score(Mt)))

OTSU = {"Bolivia": -20.44, "India": -21.56, "Somalia": -21.06, "Spain": -25.13}  # Sen1Floods11 metadata VH_thresh
ev = np.array(T.ev)
Mo = np.zeros_like(T.rule)
for e, t in OTSU.items():
    Mo[ev == e] = T.VH[ev == e] < t
print("%-24s %-22s %7s | %6.3f %6.3f %6.3f %6.1f" % ("per-event Otsu VH", "metadata", "-", *score(Mo)))
