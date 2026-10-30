"""Test A, two-column figure: (a) a 1.5 km window of India_747992 where Sentinel-2 shows water that the threshold
misses and the translation does not recover (true colour, VV, VH, MNDWI, NDVI, outcome), (b) the four classes in
SAR space (VH, VV) and (c) in optical space (NDVI, MNDWI), with the density of each class along each axis on the top
and the right. Contours enclose the densest 25% and 50% of the pixels.
Pixels in b and c: all label water (threshold / recovered / unrecovered) and label land with a valid Sentinel-2
observation. Usage: python3 fig_unrecovered_space_testA.py [CHIP ROW COL]"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
import rasterio as rio
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import TestA, P, R, O, true_colour

OUT = O + "fig_unrecovered_space_testA"
S2 = R + "data/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
for f in ("STIXGeneral", "STIXGeneralBol", "STIXGeneralItalic"):
    try:
        font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/%s.ttf" % f)
    except Exception:
        pass
FS = 20
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS, "axes.linewidth": 1.2, "pdf.fonttype": 42})
INK = "#0b0b0b"
C_RULE, C_TR, C_MISS, C_LAND = "#2a78d6", "#1baf7a", "#8c8c8c", "#dcc9ad"
C_LAND_LINE = "#b8894a"
CHIP, ORIGIN, WIN = "India_773682", (325, 175), 150
if len(sys.argv) > 3:
    CHIP, ORIGIN = sys.argv[1], (int(sys.argv[2]), int(sys.argv[3]))
N_SAMPLE, SEED, LEVELS = 8000, 0, (0.50, 0.25)

t = TestA()
ch = list(csv.DictReader(open(P + "s1f11_chips.csv")))
rows = [i for i, r in enumerate(ch) if r["event"] in {"India", "Somalia", "Spain", "Bolivia"}]
YO = np.asarray(np.load(P + "s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])
M, NDVI = YO[:, 0], YO[:, 1]
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- (a) example window
k = t.names.index(CHIP); y0, x0 = ORIGIN
sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
rgb = true_colour(S2 + CHIP + "_S2Hand.tif", sl)
out = np.zeros((WIN, WIN), np.uint8)
out[(t.water & t.rule)[k][sl]] = 1; out[t.rec[k][sl]] = 2; out[(t.miss & ~t.U)[k][sl]] = 3

fig = plt.figure(figsize=(22, 9.4))
gs = fig.add_gridspec(2, 3, width_ratios=[3.45, 0.85, 1.9], wspace=0.0, hspace=0.20, left=0.03, right=0.99, top=0.94, bottom=0.08)
# left block: image, colorbar, gap, image, colorbar, gap, image, colorbar
gsl = gs[:, 0].subgridspec(2, 8, width_ratios=[1, 0.05, 0.20, 1, 0.05, 0.20, 1, 0.05], wspace=0.06, hspace=0.20)
axs = [fig.add_subplot(gsl[r, c]) for r in range(2) for c in (0, 3, 6)]
caxs = [fig.add_subplot(gsl[r, c]) for r in range(2) for c in (1, 4, 7)]
panels = [("Sentinel-2 (R, G, B)", None), ("VV (dB)", (t.VV[k][sl], "gray", -22, -4)), ("VH (dB)", (t.VH[k][sl], "gray", -28, -10)),
          ("MNDWI", (M[k][sl], "RdBu", -0.6, 0.6)), ("NDVI", (NDVI[k][sl], "YlGn", -0.2, 0.8)), ("Detection vs. label", None)]
for ax, cax, (title, spec) in zip(axs, caxs, panels):
    if title.startswith("Sentinel"):
        ax.imshow(rgb, interpolation="nearest")
    elif title == "Detection vs. label":
        ax.imshow(out, cmap=ListedColormap([C_LAND, C_RULE, C_TR, C_MISS]), vmin=0, vmax=3, interpolation="nearest")
    else:
        a, cm, v0, v1 = spec
        im = ax.imshow(a, cmap=cm, vmin=v0, vmax=v1, interpolation="nearest")
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS - 6, length=3)
    ax.set_title(title, fontsize=FS - 1, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    if spec is None:
        cax.axis("off")
for ax, letter in zip(axs, "abcdef"):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=FS + 6, fontweight="bold", ha="left", va="top",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
h = [plt.Line2D([], [], marker="s", ls="", ms=20, mfc=c, mec="none") for c in (C_RULE, C_TR, C_MISS, C_LAND)]
# legend of the outcome classes, attached to panel (f) only
axs[5].legend(h, ["VH-22", "Recovered", "Unrecovered", "Land"], loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=FS - 1, handletextpad=0.3, columnspacing=1.0, labelspacing=0.3)

# ---------------------------------------------------------------- (b), (c) class positions with marginal densities
groups = [("VH-22", t.water & t.rule, C_RULE), ("Recovered", t.rec, C_TR),
          ("Unrecovered", t.miss & ~t.U, C_MISS), ("Land", t.land, C_LAND_LINE)]


def hdr_contours(ax, x, y, col, xlim, ylim):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size > N_SAMPLE:
        i = rng.choice(x.size, N_SAMPLE, replace=False); x, y = x[i], y[i]
    kde = gaussian_kde(np.vstack([x, y]))
    dens = kde(np.vstack([x, y]))
    thr = [np.quantile(dens, 1 - q) for q in LEVELS]
    xx, yy = np.meshgrid(np.linspace(*xlim, 120), np.linspace(*ylim, 120))
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, z, levels=sorted(thr), colors=[col], linewidths=[1.3, 2.8])
    ax.plot(np.median(x), np.median(y), "o", color=col, ms=10, mec="white", mew=1.6, zorder=5)


def panel(cell, xs, ys, xlim, ylim, xlab, ylab, letter):
    g = cell.subgridspec(2, 2, width_ratios=[1.0, 0.26], height_ratios=[0.26, 1.0], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(g[1, 0]); ax_t = fig.add_subplot(g[0, 0], sharex=ax); ax_r = fig.add_subplot(g[1, 1], sharey=ax)
    xg = np.linspace(*xlim, 300); yg = np.linspace(*ylim, 300)
    for lab, m, col in groups:
        x, y = xs[m], ys[m]
        ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        hdr_contours(ax, x, y, col, xlim, ylim)
        k = rng.choice(x.size, min(N_SAMPLE, x.size), replace=False)
        dx = gaussian_kde(x[k])(xg); dy = gaussian_kde(y[k])(yg)
        ax_t.plot(xg, dx, color=col, lw=2.4); ax_t.fill_between(xg, 0, dx, color=col, alpha=0.12, lw=0)
        ax_r.plot(dy, yg, color=col, lw=2.4); ax_r.fill_betweenx(yg, 0, dy, color=col, alpha=0.12, lw=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel(xlab, labelpad=2); ax.set_ylabel(ylab)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.2, colors=INK, labelsize=FS - 2)
    ax_t.set_ylim(0, None); ax_t.set_yticks([]); ax_t.tick_params(axis="x", labelbottom=False, length=0)
    ax_r.set_xlim(0, None); ax_r.set_xticks([]); ax_r.tick_params(axis="y", labelleft=False, length=0)
    for a_, sp in ((ax_t, ("top", "right", "left")), (ax_r, ("top", "right", "bottom"))):
        for s_ in sp:
            a_.spines[s_].set_visible(False)
    ax_t.text(-0.16, 1.02, letter, transform=ax_t.transAxes, fontsize=FS + 8, fontweight="bold", va="bottom")
    return ax


ab = panel(gs[0, 2], t.VH, t.VV, (-32, -9), (-27, -3), "VH (dB)", "VV (dB)", "g")
ac = panel(gs[1, 2], NDVI, M, (-0.4, 0.9), (-0.45, 1.0), "NDVI", "MNDWI", "h")
hl = [plt.Line2D([], [], color=c, lw=2.8) for _, _, c in groups]
ac.legend(hl, [g[0] for g in groups], loc="upper right", frameon=False, fontsize=FS - 4, handlelength=1.5, borderaxespad=0.2)
fig.savefig(OUT + ".png", dpi=250, bbox_inches="tight", facecolor="white")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
for lab, m, _ in groups:
    v = NDVI[m]; v = v[np.isfinite(v)]
    print("%-22s n=%9d  NDVI median %.3f  <0.1 %.1f%%  0.1-0.2 %.1f%%  0.2-0.3 %.1f%%  >0.3 %.1f%%" % (
        lab, v.size, np.median(v), 100 * (v < 0.1).mean(), 100 * ((v >= 0.1) & (v < 0.2)).mean(),
        100 * ((v >= 0.2) & (v < 0.3)).mean(), 100 * (v > 0.3).mean()))
mv = (t.miss & ~t.U)[k][sl]
print("window %s (%d,%d): unrecovered px %d, MNDWI %.2f NDVI %.2f VV %.1f VH %.1f" % (CHIP, y0, x0, mv.sum(), np.nanmedian(M[k][sl][mv]),
      np.nanmedian(NDVI[k][sl][mv]), np.nanmedian(t.VV[k][sl][mv]), np.nanmedian(t.VH[k][sl][mv])))
