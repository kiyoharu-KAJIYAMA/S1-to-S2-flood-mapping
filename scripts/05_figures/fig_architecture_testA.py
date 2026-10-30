"""Test A version of fig_architecture.py: same strip, thumbnails from a Sen1Floods11 held-out chip (India_747992) and
the hand label in panel (b) is the Sen1Floods11 label. Run with the rmsdcpu env (torch)."""
import csv, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = "/home/kk/jp_claude/RMSD/"; P = ROOT + "processed/"; CK = ROOT + "output/from_b200/ckpt/"; O = ROOT + "output/water_states/"
sys.path.insert(0, CK + "_code"); sys.path.insert(0, ROOT + "scripts/02_preprocess")
import os
# RMSD_MODELS=cur reproduces the cur-s figures; the default is the paper model t18-Paper
MODELS = os.environ.get("RMSD_MODELS", "t18")
CKM = ROOT + {"cur": "output/from_b200/ckpt/", "P0": "output/from_terra/ckpt_P0/"}.get(MODELS, "output/from_terra/ckpt/")
RUNF = {"cur": "cur-s%d_run01", "P0": "P0-s%d"}.get(MODELS, "t18-Paper-s%d_run01")
VOTES = os.environ.get("RMSD_VOTES", O + "testA_translation_votes.npy")
O = os.environ.get("RMSD_O", O)
from RMSD_flood_eval import UNet, linear_to_db_safe  # noqa: E402
from arch_builder import build_from_ckpt  # noqa: E402

CHIP = sys.argv[1] if len(sys.argv) > 1 else "India_747992"
ch = list(csv.DictReader(open(P + "s1f11_chips.csv")))
rows = [i for i, r in enumerate(ch) if r["event"] in {"India", "Somalia", "Spain", "Bolivia"}]
I = [ch[i]["chip"] for i in rows].index(CHIP); G = rows[I]                 # Test A row / global index
X = np.array(np.load(P + "s1f11_X_VV_VH_RVI_raw.npy", mmap_mode="r")[G:G + 1], np.float32)
Y = np.array(np.load(P + "s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[G:G + 1], np.float32)
lab = np.asarray(np.load(P + "s1f11_label.npy", mmap_mode="r")[G])
votes = np.asarray(np.load(VOTES, mmap_mode="r")[I]).astype(bool)
Xd = X.copy()
for b in (0, 1):
    Xd[:, b] = linear_to_db_safe(Xd[:, b])
mndwi, runs, Yh, taus = [], [], None, []
for s in range(1, 6):
    ck = torch.load(CKM + RUNF % s + "/best_model.pt", map_location="cpu", weights_only=False)
    if MODELS == "P0":                                    # translated MNDWI > 0, no tau
        tau = -ck["Y_min"][0] / (ck["Y_max"][0] - ck["Y_min"][0])
    else:
        tau = json.load(open(CKM + RUNF % s + ("/summary_fp.json" if MODELS == "cur" else "/summary_s1f11.json")))["thresholds"]["unet"]
    Z = Xd.copy()
    lo = (np.asarray(ck["X_mean"]) - 5 * np.asarray(ck["X_std"])).reshape(1, -1, 1, 1)
    hi = (np.asarray(ck["X_mean"]) + 5 * np.asarray(ck["X_std"])).reshape(1, -1, 1, 1)
    Z[(Z < lo) | (Z > hi)] = np.nan
    for b in range(3):
        Z[:, b] = (Z[:, b] - ck["X_min"][b]) / (ck["X_max"][b] - ck["X_min"][b])
    Z = np.nan_to_num(np.clip(Z, 0, 1))
    m = build_from_ckpt(ck, UNet); m.load_state_dict(ck["model_xy_state_dict"]); m.eval()
    with torch.no_grad():
        y = m(torch.from_numpy(Z).float()).numpy()
    ymin, ymax = np.asarray(ck["Y_min"]).reshape(1, 3, 1, 1), np.asarray(ck["Y_max"]).reshape(1, 3, 1, 1)
    yd = y * (ymax - ymin) + ymin
    runs.append(y[0, 0] > tau); mndwi.append(yd[0, 0]); taus.append(tau * (ymax[0, 0, 0, 0] - ymin[0, 0, 0, 0]) + ymin[0, 0, 0, 0])
    if s == 1:
        Yh = yd
S = slice(0, X.shape[-1])

FS = 38
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": FS, "mathtext.fontset": "dejavusans"})
C_SAR, C_MSI, C_LOSS, C_INF, INK, MUTED = "#6b62d6", "#1a9e6e", "#d0473a", "#2a78d6", "#1a1a1a", "#8a8a8a"
WATER = (0.16, 0.47, 0.84)
FW, FH = 42.5, 10.0
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
YC = 5.8


def arrow(p, q, col, lw=3.5, ms=34, both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms, color=col, lw=lw))


def img(x, y, a, cmap=None, rng=(None, None), col=INK, size=2.3):
    h = fig.add_axes([x / FW, y / FH, size / FW, size / FH])
    h.imshow(a, cmap=cmap, vmin=rng[0], vmax=rng[1], interpolation="nearest"); h.set_xticks([]); h.set_yticks([])
    for s in h.spines.values():
        s.set_edgecolor(col); s.set_linewidth(3)


def stack(x, imgs, cmap, rngs, col, size=2.3, dx=0.42, dy=0.32):
    y0 = YC - size / 2 + dy
    for k in range(len(imgs) - 1, -1, -1):
        img(x + k * dx, y0 - k * dy, imgs[k], cmap, rngs[k], col, size)
    return x + (len(imgs) - 1) * dx + size


def text(x, y, t, col=INK, size=FS, **kw):
    ax.text(x, y, t, ha="center", va="center", fontsize=size, color=col, **kw)


def panel_title(x, letter, word):
    ax.text(x, 9.5, letter, ha="left", va="center", fontsize=46, color=INK, weight="bold")
    ax.text(x + 0.9, 9.5, word, ha="left", va="center", fontsize=46, color=INK, weight="bold")


def mask(a):
    o = np.ones(a.shape + (3,)); o[a] = WATER; return o


def unet_icon(x0, width):
    hs = [1.5, 1.15, 0.85, 0.6, 0.4]; ws = [0.1, 0.13, 0.17, 0.21, 0.26]
    top = YC + 0.95
    ys = [top - k * 0.32 - hs[k] for k in range(5)]
    xe = [x0 + k * 0.3 for k in range(5)]; xd = [x0 + width - k * 0.3 - ws[k] for k in range(5)]
    xe[4] = xd[4] = x0 + width / 2 - ws[4] / 2
    ys[4] = ys[3] - hs[4] * 0.55
    for k in range(5):
        for xx in ((xe[k], xd[k]) if k < 4 else (xe[k],)):
            ax.add_patch(Rectangle((xx, ys[k]), ws[k], hs[k], fc=C_SAR, ec="none", alpha=0.4 + 0.14 * k))
    for k in range(4):
        cy = ys[k] + hs[k] * 0.6
        ax.plot([xe[k] + ws[k] + 0.04, xd[k] - 0.04], [cy, cy], color=MUTED, lw=1.5, ls=(0, (3, 3)))
    return ys[4]


# (a) training
YL1, YL2 = YC - 2.5, YC - 3.35
panel_title(0.1, "a", "Training")
r = stack(0.5, [Xd[0, 0, S, S], Xd[0, 1, S, S], X[0, 2, S, S]], "gray", [(-25, -5), (-30, -12), (0, 1.5)], C_SAR)
text((0.5 + r) / 2, YL1, "Sentinel-1", C_SAR, weight="bold"); text((0.5 + r) / 2, YL2, "VV, VH, RVI")
arrow((r + 0.2, YC), (r + 0.85, YC), C_SAR)
ub = unet_icon(r + 1.0, 2.7)
text(r + 2.35, ub - 0.5, "U-Net", C_SAR, weight="bold")
x1 = r + 3.95
arrow((x1, YC), (x1 + 0.65, YC), C_SAR)
r2 = stack(x1 + 0.8, [Yh[0, k, S, S] for k in range(3)], "RdBu", [(-0.6, 0.6)] * 3, C_MSI)
text((x1 + 0.8 + r2) / 2, YL1, "Translated", C_MSI, weight="bold"); text((x1 + 0.8 + r2) / 2, YL2, "MNDWI, NDVI, NDWI")
x3 = r2 + 8.0
xc = (r2 + x3) / 2
arrow((r2 + 0.25, YC), (x3 - 0.25, YC), C_LOSS, both=True)
if MODELS == "P0":
    text(xc, YC + 0.6, "pixel difference", C_LOSS, size=FS)
else:
    text(xc, YC + 0.6, r"$\mathcal{L} = \mathcal{L}_{1} + 0.01\,\mathcal{L}_{\mathrm{KL}} + 0.2\,\mathcal{L}_{\mathrm{Dice}}$", C_LOSS, size=FS)
    for k, (a_, b_) in enumerate(((r"$\mathcal{L}_1$", "pixel difference"), (r"$\mathcal{L}_{\mathrm{KL}}$", "histogram difference"),
                                  (r"$\mathcal{L}_{\mathrm{Dice}}$", "water-mask overlap"))):
        ax.text(xc - 1.3, YC - 0.75 - k * 0.72, a_, ha="right", va="center", fontsize=FS - 4, color=C_LOSS)
        ax.text(xc - 1.1, YC - 0.75 - k * 0.72, b_, ha="left", va="center", fontsize=FS - 4, color=INK)
r3 = stack(x3, [Y[0, k, S, S] for k in range(3)], "RdBu", [(-0.6, 0.6)] * 3, C_MSI)
text((x3 + r3) / 2, YL1, "Sentinel-2", C_MSI, weight="bold"); text((x3 + r3) / 2, YL2, "MNDWI, NDVI, NDWI")

# (b) test: translation and backscatter threshold, both scored against the Sen1Floods11 hand label
z = np.load(P + "s1f11_lee_db.npz"); VVl, VHl = z["VV"][G], z["VH"][G]
thr = (VVl < -5) & (VHl < -22)
C_LAB = "#b07d1e"
xb = r3 + 5.4
panel_title(xb - 0.25, "b", "Test")
s5, d5, YT, YB = 1.6, 0.22, 7.2, 2.6
for k in range(4, -1, -1):
    img(xb + k * d5, YT - s5 / 2 + 2 * d5 - k * d5, mndwi[k][S, S], "RdBu", (-0.6, 0.6), C_MSI, s5)
e1 = xb + 4 * d5 + s5
xv = e1 + 1.3 + 4 * d5 + s5 + 1.3                                 # same right edge as before
arrow((e1 + 0.1, YT), (xv - 0.1, YT), C_INF)
if MODELS == "P0":
    text((e1 + 0.1 + xv) / 2, YT + 1.05, "MNDWI > 0", C_INF, size=FS - 2)
    text((e1 + 0.1 + xv) / 2, YT + 0.45, "3 of 5 runs", C_INF, size=FS - 2)
else:
    text((e1 + 0.1 + xv) / 2, YT + 1.05, "MNDWI > τ", C_INF, size=FS - 2)
    text((e1 + 0.1 + xv) / 2, YT + 0.45, "%.2f ≤ τ ≤ %.2f" % (min(taus), max(taus)), C_INF, size=FS - 2)
img(xv, YT - 1.1, mask(votes[S, S]), col=C_INF, size=2.2)
text((xb + xv + 2.2) / 2, YT - 1.75, "S1toS2  (5 runs)", C_INF, weight="bold")
img(xb + 0.3, YB - s5 / 2 + 0.3, VVl[S, S], "gray", (-25, -5), C_SAR, s5)
img(xb, YB - s5 / 2, VHl[S, S], "gray", (-30, -12), C_SAR, s5)
arrow((xb + 2.1, YB), (xv - 0.1, YB), INK)
text((xb + 2.1 + xv) / 2, YB + 1.05, "VV < −5 dB", INK, size=FS - 2)
text((xb + 2.1 + xv) / 2, YB + 0.45, "VH < −22 dB", INK, size=FS - 2)
img(xv, YB - 1.1, mask(thr[S, S]), col=INK, size=2.2)
text((xb + xv + 2.2) / 2, YB - 1.75, "VH-22", INK, weight="bold")
xf = xv + 3.8; ym = (YT + YB) / 2
img(xf, ym - 1.3, mask(lab[S, S] == 1), col=C_LAB, size=2.6)
text(xf + 1.3, ym + 1.8, "Sen1Floods11", C_LAB, weight="bold"); text(xf + 1.3, ym - 1.8, "hand label", C_LAB)
arrow((xv + 2.3, YT - 0.4), (xf - 0.1, ym + 0.5), MUTED, lw=3, both=True)
arrow((xv + 2.3, YB + 0.4), (xf - 0.1, ym - 0.5), MUTED, lw=3, both=True)
text(xv + 3.0, ym, "vs", MUTED)
for ext in ("png", "pdf"):
    fig.savefig(O + "fig_architecture_testA_%s.%s" % (CHIP, ext), dpi=130, bbox_inches="tight")
print("saved", CHIP, "row", I, "global", G)
