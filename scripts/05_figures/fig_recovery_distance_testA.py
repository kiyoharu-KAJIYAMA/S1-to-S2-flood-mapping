"""Test A distance figure: recovery of threshold-missed water against distance from thresholded water
on the four held-out Sen1Floods11 events.
(a) Sentinel-2 true colour of a 3 km crop of one chip, (b) the same crop with thresholded / recovered / missed water
and the 50 m line, (c) threshold-missed water by distance bin as a share of all missed water, split into recovered and not.
Usage: python3 fig_recovery_distance_testA.py [CHIP [ROW COL]]  -> figure (default MAP_CHIP / ORIGIN)
       python3 fig_recovery_distance_testA.py search              -> rank crops whose recovered water all lies
                                                                     within RMAX of thresholded water, draw a sheet"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as cc_label, distance_transform_edt
import rasterio as rio
sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import TestA, O, PX, true_colour

OUT = O + "fig_recovery_distance_testA"
S2 = "/home/kk/jp_claude/RMSD/data/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
try:
    font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/arial.ttf")
    font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/arialbd.ttf")
    plt.rcParams["font.family"] = "Arial"
except Exception:
    pass
FS = 32
plt.rcParams.update({"font.size": FS, "axes.linewidth": 1.6, "pdf.fonttype": 42})
INK = "#0b0b0b"
# same palette as the outcome figure: blue threshold, green recovered, grey missed, beige land
C_RULE, C_TR, C_MISS, C_LAND = "#2a78d6", "#1baf7a", "#b9b8b2", "#dcc9ad"
C_MISS_LINE = "#6f6f6f"                                  # darker grey for the cumulative line of all missed water
MAP_CHIP, WIN = "India_399883", 300
ORIGIN = (209, 185)                                      # (row, col) crop origin; None = best window of the search
ISO = 50                                                 # m, isoline drawn around the larger thresholded bodies
MIN_BODY = 20                                            # px; smaller specks are ignored by the isoline
RMAX = 200                                               # m; a candidate crop keeps all its recovered water within RMAX
LABEL_POS = (1500, 1500)                                      # manual position (m) of the contour label, None = automatic
BINS = [(0, 10), (10, 30), (30, 50), (50, 200), (200, np.inf)]          # same bins as Table III
BIN_LAB = ["0-10", "10-30", "30-50", "50-200", "> 200"]

t = TestA()
D, miss, rec, rule, U = t.D, t.miss, t.rec, t.rule, t.U
n_miss = miss.sum()
share = np.array([100 * (miss & (D > lo) & (D <= hi)).sum() / n_miss for lo, hi in BINS])
rate = np.array([rec[miss & (D > lo) & (D <= hi)].mean() for lo, hi in BINS])


def box_sum(a):
    ii = np.pad(a.astype(float).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return ii[WIN:, WIN:] - ii[:-WIN, WIN:] - ii[WIN:, :-WIN] + ii[:-WIN, :-WIN]


def iso_distance(k):
    """Distance (m) to the thresholded bodies of at least MIN_BODY px; specks below that are ignored by the isoline."""
    lab, n = cc_label(rule[k])
    big = np.isin(lab, np.flatnonzero(np.bincount(lab.ravel())[1:] >= MIN_BODY) + 1)
    d = distance_transform_edt(~big) * PX if big.any() else np.full(rule[k].shape, np.inf)
    return d, big


def search(top=8):
    """Rank WIN x WIN crops: recovered water and thresholded specks outside the ISO isoline are penalised,
    all three classes must be visible."""
    cands = []
    for k in range(len(t.names)):
        if rec[k].sum() < 1500:
            continue
        d_iso, big = iso_distance(k)
        n_rec, n_rule = box_sum(rec[k]), box_sum(rule[k])
        out_rec = box_sum(rec[k] & (d_iso > ISO))                # recovered water outside the isoline
        out_rule = box_sum(rule[k] & ~big)                       # thresholded specks outside the isoline
        n_far_miss = box_sum(miss[k] & ~U[k] & (D[k] > 2 * ISO))
        ok = (n_rec >= 1000) & (n_rule >= 3000) & (n_far_miss >= 500) & (out_rec <= 0.06 * n_rec) & (out_rule <= 0.06 * n_rule)
        if not ok.any():
            continue
        score = np.where(ok, np.minimum(n_rec, 6000) - 10 * out_rec - 10 * out_rule + 0.3 * np.minimum(n_far_miss, 6000), -1e9)
        y0, x0 = np.unravel_index(int(np.argmax(score)), score.shape)
        cands.append((score[y0, x0], k, int(y0), int(x0), int(n_rec[y0, x0]), int(out_rec[y0, x0]), int(out_rule[y0, x0]), int(n_far_miss[y0, x0])))
    cands.sort(reverse=True)
    return cands[:top]


def crop(k, origin):
    y0, x0 = origin
    sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
    m = np.zeros((WIN, WIN), np.uint8)
    m[rule[k][sl]] = 1; m[rec[k][sl]] = 2; m[(miss[k] & ~U[k])[sl]] = 3
    d_iso, _ = iso_distance(k)
    return m, d_iso[sl]


def draw_rgb(ax, k, origin):
    """Sentinel-2 true colour of the same crop, with the 50 m line in white."""
    y0, x0 = origin
    sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
    rgb = true_colour(S2 + t.names[k] + "_S2Hand.tif", sl)
    ext = (0, WIN * PX, WIN * PX, 0)
    ax.imshow(rgb, interpolation="nearest", extent=ext)
    _, d = crop(k, origin); xs = (np.arange(WIN) + 0.5) * PX
    ax.contour(xs, xs, d, levels=[ISO], colors=["white"], linewidths=[2.6])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_map(am, k, origin, label_pos=None):
    m, d = crop(k, origin)
    ext = (0, WIN * PX, WIN * PX, 0)
    am.imshow(m, cmap=ListedColormap([C_LAND, C_RULE, C_TR, C_MISS]), vmin=0, vmax=3, interpolation="nearest", extent=ext)
    xs = (np.arange(WIN) + 0.5) * PX
    cs = am.contour(xs, xs, d, levels=[ISO], colors=[INK], linewidths=[2.6])
    am.clabel(cs, fmt=lambda v: "%d m" % ISO, fontsize=FS, inline=True, inline_spacing=14,
              manual=[label_pos] if label_pos else False)
    am.set_xticks([]); am.set_yticks([])
    for s in am.spines.values():
        s.set_visible(False)


def draw_bars(ax):
    """Stacked bars of the threshold-missed water by distance bin (left axis) and, on the right axis, the cumulative
    share of the recovered and of the unrecovered water up to the upper edge of each bin, with the 50 m and 100 m
    values written out."""
    x = np.arange(len(BINS)); r = share * rate
    ax.bar(x, r, color=C_TR, width=0.8)
    ax.bar(x, share - r, bottom=r, color=C_MISS, width=0.8)
    for i in x:
        ax.text(i, share[i] + 0.5, "%.0f%%" % (100 * rate[i]), ha="center", va="bottom", fontsize=FS - 2, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(BIN_LAB, fontsize=FS - 9)
    ax.set_ylim(0, 2.2 * share.max()); ax.set_yticks(np.arange(0, share.max() + 1, 10))
    ax.set_ylabel("Share of water missed by VH-22 (%)", color=INK)
    ax.set_xlabel("Distance from water detected by VH-22 (m)", color=INK, labelpad=12)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", length=8, width=1.6, colors=INK)
    # cumulative shares at the upper edge of each bin (the last bin is open, so its edge is drawn at the bar's right side)
    xe = x + 0.4
    cum_rec = 100 * np.cumsum(r) / r.sum()
    cum_all = 100 * np.cumsum(share - r) / (share - r).sum()      # unrecovered water only, as the grey bars
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)   # bars and their labels in front of the curve
    ax2.plot(xe, cum_rec, color=C_TR, lw=3.5, marker="o", ms=11, mec="white", mew=1.8, zorder=5)
    ax2.text(xe[2] + 0.12, cum_rec[2] - 3, "%.0f%%" % cum_rec[2], color=C_TR, fontsize=FS, ha="left", va="top", fontweight="bold")   # 50 m edge
    ax2.set_ylim(0, 105); ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel("Cumulative share of recovery (%)", color=INK)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(direction="out", length=8, width=1.6, colors=INK)
    h = [plt.Line2D([], [], color=C_TR, lw=3.5, marker="o", ms=9, mec="white")]
    ax2.legend(h, ["Recovered, cumulative"], loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=FS - 4,
               handlelength=1.8, borderaxespad=0.0)
    print("cumulative recovered at 50 / 100 m: %.1f / %.1f%%; unrecovered: %.1f / %.1f%%" % (cum_rec[2], cum_rec[4], cum_all[2], cum_all[4]))


VH_E = np.arange(-22, -9.99, 1.0); VH_X = (VH_E[:-1] + VH_E[1:]) / 2
VH_MIN_PX = 2000                                         # bins with fewer missed-water pixels are not drawn
BIN_COL = ["#00A087", "#4DBBD5", "#8491B4", "#3C5488", "#1a1a1a"]


def draw_vh(ax, axl):
    """(d) Recovery of the missed water against its VH, one line per distance bin of (c), and the VH of labeled land
    beneath with its median dashed."""
    for (lo, hi), lab, col in zip(BINS, BIN_LAB, BIN_COL):
        g = miss & (D > lo) & (D <= hi)
        kb = np.digitize(t.VH[g], VH_E) - 1
        ok = (kb >= 0) & (kb < len(VH_X))
        n = np.bincount(kb[ok], minlength=len(VH_X)); r = np.bincount(kb[ok], weights=rec[g][ok], minlength=len(VH_X))
        y = np.where(n >= VH_MIN_PX, 100 * r / np.maximum(n, 1), np.nan)
        ax.plot(VH_X, y, "-o", color=col, lw=3.2, ms=9, label=lab + " m")
    ax.set_ylabel("Recovery (%)", color=INK); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=FS - 8, loc="upper right", handlelength=1.4, labelspacing=0.25, borderaxespad=0.1)
    lv = t.VH[t.land]
    h, _ = np.histogram(lv, bins=VH_E)
    axl.bar(VH_X, 100 * h / lv.size, width=0.9, color=C_LAND)
    axl.axvline(np.median(lv), color=INK, lw=2, ls="--")
    axl.set_ylabel("Land (%)", color=INK, fontsize=FS - 6); axl.set_xlabel("VH (dB)", color=INK)
    axl.set_xlim(-22.6, -9.4); axl.set_xticks([-22, -20, -18, -16, -14, -12, -10])
    for a in (ax, axl):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(direction="out", length=8, width=1.6, colors=INK)
    plt.setp(ax.get_xticklabels(), visible=False)


if len(sys.argv) > 1 and sys.argv[1] == "search":
    cands = search()
    fig, axs = plt.subplots(2, 4, figsize=(24, 13))
    for ax, c in zip(axs.ravel(), cands):
        _, k, y0, x0, nr, orc, orl, nf = c
        draw_map(ax, k, (y0, x0))
        ax.set_title("%s (%d,%d)\nrec %d out %d specks %d farmiss %d" % (t.names[k], y0, x0, nr, orc, orl, nf), fontsize=16)
        print(t.names[k], t.ev[k], y0, x0, "rec", nr, "rec outside iso", orc, "specks", orl, "far miss", nf)
    for ax in axs.ravel()[len(cands):]:
        ax.axis("off")
    fig.savefig(OUT + "_search.png", dpi=50, bbox_inches="tight", facecolor="white")
    sys.exit()

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MAP_CHIP = ARGS[0] if ARGS else MAP_CHIP
if len(ARGS) > 2:
    ORIGIN = (int(ARGS[1]), int(ARGS[2]))
k = t.names.index(MAP_CHIP)
far = D[k][rec[k]].max() if rec[k].any() else 0
sl = (slice(ORIGIN[0], ORIGIN[0] + WIN), slice(ORIGIN[1], ORIGIN[1] + WIN))
far_crop = D[k][sl][rec[k][sl]].max() if rec[k][sl].any() else 0
d_iso_k, big_k = iso_distance(k)
out_rec = int((rec[k][sl] & (d_iso_k[sl] > ISO)).sum()); out_rule = int((rule[k][sl] & ~big_k[sl]).sum())

ROW = "--row" in sys.argv                               # one row a | b | c | d
GRID = "--2x2" in sys.argv                              # 2 x 2 with d; default is the paper's a | b | c
if not (ROW or GRID):
    fig = plt.figure(figsize=(34, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.06, 1.25], wspace=0.22)
    gsl = gs[0, 0].subgridspec(1, 2, wspace=0.06)
    ar = fig.add_subplot(gsl[0, 0]); am = fig.add_subplot(gsl[0, 1]); ab = fig.add_subplot(gs[0, 1])
    gd = None
elif ROW:
    OUT += "_row"
    fig = plt.figure(figsize=(52, 10.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.2, 1.2], wspace=0.52)
    ar = fig.add_subplot(gs[0, 0]); am = fig.add_subplot(gs[0, 1]); ab = fig.add_subplot(gs[0, 2])
    gd = gs[0, 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
else:
    OUT += "_2x2"
    fig = plt.figure(figsize=(26, 24))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.34, wspace=0.38)
    ar = fig.add_subplot(gs[0, 0]); am = fig.add_subplot(gs[0, 1]); ab = fig.add_subplot(gs[1, 0])
    gd = gs[1, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
draw_rgb(ar, k, ORIGIN)
draw_map(am, k, ORIGIN, LABEL_POS)
# panel titles as in Fig. 2
ar.set_title("Sentinel-2 (R, G, B)", fontsize=FS, pad=12)
am.set_title("Detection vs. label", fontsize=FS, pad=12)
hand = [plt.Line2D([], [], marker="s", ls="", ms=22, mfc=col, mec="none") for col in (C_RULE, C_TR, C_MISS, C_LAND)]
xc = (ar.get_position().x0 + am.get_position().x1) / 2          # centred under the two maps
fig.legend(hand, ["VH-22", "Recovered", "Unrecovered", "Land"], loc="upper center",
           bbox_to_anchor=(xc, am.get_position().y0 - 0.01), ncol=4, frameon=False, fontsize=FS - 6,
           handletextpad=0.2, columnspacing=0.8)
draw_bars(ab)
if gd is not None:
    ad = fig.add_subplot(gd[0]); adl = fig.add_subplot(gd[1], sharex=ad)
    draw_vh(ad, adl)
    ad.text(-0.16, 1.02, "d", transform=ad.transAxes, fontsize=FS + 6, fontweight="bold", va="bottom")
ar.text(-0.02, 1.02, "a", transform=ar.transAxes, fontsize=FS + 6, fontweight="bold", va="bottom", ha="right")
am.text(-0.02, 1.02, "b", transform=am.transAxes, fontsize=FS + 6, fontweight="bold", va="bottom", ha="right")
ab.text(-0.16, 1.02, "c", transform=ab.transAxes, fontsize=FS + 6, fontweight="bold", va="bottom")
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("map chip %s (%s), crop origin row %d col %d, %.1f km across; farthest recovered pixel in crop %.0f m (chip %.0f m)"
      % (MAP_CHIP, t.ev[k], ORIGIN[0], ORIGIN[1], WIN * PX / 1000, far_crop, far))
print("in crop: recovered px %d, outside the %d m isoline %d; thresholded specks outside %d" % (int(rec[k][sl].sum()), ISO, out_rec, out_rule))
print("share by bin:", " ".join("%.1f" % v for v in share))
print("recovery rate by bin:", " ".join("%.1f" % (100 * v) for v in rate))
print("recovered within 100 m: %.1f%%; missed within 100 m: %.1f%%" % (100 * (D[rec] <= 100).mean(), 100 * (D[miss] <= 100).mean()))
