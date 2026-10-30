"""Every Results number of the GRSL text for a given majority vote (cur-s or t18-Paper), on the Test A pixel set of the
figures (testA_common.TestA). Run with the vote file as argument."""
import sys
import numpy as np
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import TestA, O  # noqa: E402

t = TestA()
if len(sys.argv) > 1:
    t.U = np.load(O + sys.argv[1]).astype(bool)
    t.rec = t.miss & t.U
U, w, l, rule = t.U, t.water, t.land, t.rule
YO_M, YO_N = t.MNDWI if hasattr(t, "MNDWI") else None, None
tr_tp, ru_tp = (U & w).sum(), (rule & w).sum()
print("missed share of water       %.1f%%" % (100 * t.miss.sum() / w.sum()))
print("recovery                    %.1f%%" % (100 * t.rec.sum() / t.miss.sum()))
print("recovered, share of water   %.1f%%" % (100 * t.rec.sum() / w.sum()))
print("lost, share of water        %.1f%%" % (100 * (w & rule & ~U).sum() / w.sum()))
print("land rule->land             %.1f%%" % (100 * (l & rule & ~U).sum() / l.sum()))
print("land land->water            %.1f%%" % (100 * (l & ~rule & U).sum() / l.sum()))
print("FA rule / translation       %.1f%% / %.1f%%" % (100 * (l & rule).sum() / l.sum(), 100 * (l & U).sum() / l.sum()))
print("extra water px              %d" % (tr_tp - ru_tp))
for n, p in (("rule", rule), ("translation", U)):
    tp, fp, fn = (p & w).sum(), (p & l).sum(), (w & ~p).sum()
    print("%-12s IoU %.3f prec %.3f rec %.3f" % (n, tp / (tp + fp + fn), tp / (tp + fp), tp / (tp + fn)))
D = t.D
bins = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 100), (100, 200), (200, np.inf)]
print("recovery by bin " + " ".join("%s-%s:%.1f" % (a, b, 100 * t.rec[t.miss & (D > a) & (D <= b) if a else t.miss & (D <= b)].mean()) for a, b in bins))
print("recovered within 50 m       %.1f%%" % (100 * (t.rec & (D <= 50)).sum() / t.rec.sum()))
print("missed beyond 50 / 200 m    %.1f%% / %.1f%%" % (100 * (t.miss & (D > 50)).sum() / t.miss.sum(), 100 * (t.miss & (D > 200)).sum() / t.miss.sum()))
import numpy as _np
P = "/home/kk/jp_claude/RMSD/processed/"
import csv
ch = list(csv.DictReader(open(P + "s1f11_chips.csv")))
rows = [i for i, r in enumerate(ch) if r["event"] in {"India", "Somalia", "Spain", "Bolivia"}]
YO = _np.asarray(_np.load(P + "s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])
M, N = YO[:, 0], YO[:, 1]
cls = (("thresholded", w & rule), ("recovered", t.rec), ("unrecovered", t.miss & ~U), ("land", l))
for n, m in cls:
    print("%-12s VV %.1f VH %.1f NDVI %.2f MNDWI %.2f" % (n, _np.median(t.VV[m]), _np.median(t.VH[m]), _np.nanmedian(N[m]), _np.nanmedian(M[m])))
un = t.miss & ~U; nv = N[un]
print("unrecovered NDVI 0.1-0.2 %.0f%%  >0.3 %.0f%%" % (100 * ((nv >= 0.1) & (nv < 0.2)).mean(), 100 * (nv > 0.3).mean()))
