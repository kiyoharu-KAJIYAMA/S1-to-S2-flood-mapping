"""Table III, second version: land-cover shares of thresholded, recovered, unrecovered, all labeled water and land, the
recovery rate within 50 m by land cover, and the land-cover shares of the recovered and the unrecovered water within
50 m (each column summing to 100)."""
import sys
import numpy as np
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import TestA, P

t = TestA()
WC = np.load(P + "s1f11_testA_worldcover_2020.npy")
NAMES = [(40, "Cropland"), (30, "Grassland"), (90, "Herbaceous wetland"), (20, "Shrubland"), (60, "Bare / sparse"),
         (10, "Tree cover"), (50, "Built-up"), (80, "Water 2020")]
near = t.D <= 50
allwater = t.water & (t.rule | t.miss)
cols = [("Thr.", t.water & t.rule), ("Rec.", t.rec), ("Unrec.", t.miss & ~t.U), ("All water", allwater), ("Land", t.land),
        ("Rec. <=50", t.rec & near), ("Unrec. <=50", t.miss & ~t.U & near)]
tot = {c: m.sum() for c, m in cols}
print("%-20s" % "class" + "".join("%12s" % c for c, _ in cols) + "%12s" % "rate <=50")
sums = {c: 0.0 for c, _ in cols}
for k, name in NAMES:
    row = []
    for c, m in cols:
        v = 100 * (m & (WC == k)).sum() / tot[c]; sums[c] += v; row.append(v)
    m_near = t.miss & (WC == k) & near
    rate = 100 * t.rec[m_near].mean()
    print("%-20s" % name + "".join("%11.1f%%" % v for v in row) + "%11.1f%%" % rate)
print("%-20s" % "sum of rows" + "".join("%11.1f%%" % sums[c] for c, _ in cols))
print("%-20s" % "n pixels" + "".join("%12d" % tot[c] for c, _ in cols))
print("all labeled water n=%d, of which within 50 m of thresholded water (missed part) n=%d" % (allwater.sum(), (t.miss & near).sum()))
