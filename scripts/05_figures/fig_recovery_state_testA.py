"""Candidate Fig. 5: recovery of the water missed by VH-22 against (a) its Lee VH and (b) its observed Sentinel-2 NDVI,
one line per distance bin of Table III / Fig. 3c, with the probability density of labeled land beneath each
(Gaussian kernel density on 200,000 sampled land pixels).
Run with /usr/bin/python3; uses the current vote file (testA_common VOTES)."""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import gaussian_kde
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import TestA, O, P, EVENTS

try:
    font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/arial.ttf")
    font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/arialbd.ttf")
    plt.rcParams["font.family"] = "Arial"
except Exception:
    pass
FS = 30
plt.rcParams.update({"font.size": FS, "axes.linewidth": 1.6, "pdf.fonttype": 42})
INK, C_LAND = "#0b0b0b", "#dcc9ad"
BINS = [(0, 10), (10, 30), (30, 50), (50, 200), (200, np.inf)]          # same bins as Table III and Fig. 3c
BIN_LAB = ["0-10 m", "10-30 m", "30-50 m", "50-200 m", "> 200 m"]
BIN_COL = ["#00A087", "#4DBBD5", "#8491B4", "#3C5488", "#1a1a1a"]
MIN_PX = 2000                                                            # sparser bins are not drawn

t = TestA()
ch = list(csv.DictReader(open(P + "s1f11_chips.csv")))
rows = [i for i, r in enumerate(ch) if r["event"] in EVENTS]
NDVI = np.asarray(np.load(P + "s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])[:, 1]
miss, rec, D = t.miss, t.rec, t.D


def panel(ax, axl, A, edges, xlabel, xticks, legend):
    xs = (edges[:-1] + edges[1:]) / 2
    for (lo, hi), lab, col in zip(BINS, BIN_LAB, BIN_COL):
        g = miss & (D > lo) & (D <= hi) & np.isfinite(A)
        kb = np.digitize(A[g], edges) - 1
        ok = (kb >= 0) & (kb < len(xs))
        n = np.bincount(kb[ok], minlength=len(xs)); r = np.bincount(kb[ok], weights=rec[g][ok], minlength=len(xs))
        y = np.where(n >= MIN_PX, 100 * r / np.maximum(n, 1), np.nan)
        ax.plot(xs, y, "-o", color=col, lw=3.2, ms=9, label=lab)
        print(xlabel, lab, " ".join("%.0f" % v if np.isfinite(v) else "-" for v in y))
    ax.set_ylabel("Recovery (%)", color=INK); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    if legend:
        # one column at the right edge of the recovery panel, so the entries stack in the order of the lines
        ax.legend(frameon=False, fontsize=FS - 10, loc="upper right", ncol=1, handlelength=1.4, labelspacing=0.25,
                  borderaxespad=0.2, title="Distance from\nwater detected\nby VH-22", title_fontsize=FS - 10)
    lv = A[t.land]; lv = lv[np.isfinite(lv)]
    sub = np.random.default_rng(0).choice(lv, size=min(200000, lv.size), replace=False)
    gx = np.linspace(edges[0] - 0.5 * (edges[1] - edges[0]), edges[-1] + 0.5 * (edges[1] - edges[0]), 400)
    pdf = gaussian_kde(sub)(gx)
    axl.fill_between(gx, pdf, color=C_LAND, lw=0); axl.plot(gx, pdf, color="#B09C85", lw=2.4)
    axl.set_ylim(0, 1.15 * pdf.max())
    axl.set_ylabel("Land\ndensity", color=INK, fontsize=FS - 8); axl.set_xlabel(xlabel, color=INK)
    h = np.histogram(lv, bins=edges)[0]
    axl.set_xlim(edges[0] - 0.6 * (edges[1] - edges[0]), edges[-1] + 0.6 * (edges[1] - edges[0])); axl.set_xticks(xticks)
    for a in (ax, axl):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(direction="out", length=8, width=1.6, colors=INK)
    plt.setp(ax.get_xticklabels(), visible=False)
    print(xlabel, "land shown %.1f%% of land pixels" % (h.sum() / lv.size * 100))


# single column; the paper uses the VH panel only, --ndvi adds the NDVI panel below as (b)
# --scale k shrinks the canvas (include the PDF at k * columnwidth), so the printed font size stays the same
NDVI_PANEL = "--ndvi" in sys.argv
SCALE = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.0
n = 2 if NDVI_PANEL else 1
fig = plt.figure(figsize=(11 * SCALE, (16 if NDVI_PANEL else 8.6) * SCALE))
gs = fig.add_gridspec(n, 1, hspace=0.32)
axs = []
for j in range(n):
    sg = gs[j, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    a = fig.add_subplot(sg[0]); b = fig.add_subplot(sg[1], sharex=a)
    axs.append((a, b))
panel(*axs[0], t.VH, np.arange(-22, -9.99, 1.0), "VH (dB)", [-22, -20, -18, -16, -14, -12, -10], True)
if NDVI_PANEL:
    panel(*axs[1], NDVI, np.round(np.arange(-0.3, 0.801, 0.05), 2), "NDVI", [-0.2, 0, 0.2, 0.4, 0.6, 0.8], False)
if NDVI_PANEL:
    for (a, _), letter in zip(axs, "ab"):
        a.text(-0.2, 1.02, letter, transform=a.transAxes, fontsize=FS + 6, fontweight="bold", va="bottom")
tag = "_ndvi" if NDVI_PANEL else ""
fig.savefig(O + "fig_recovery_state_testA%s.png" % tag, dpi=250, bbox_inches="tight", facecolor="white")
fig.savefig(O + "fig_recovery_state_testA%s.pdf" % tag, bbox_inches="tight")
