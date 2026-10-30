"""Shared Test A (Sen1Floods11 held-out events) loading for the distance / surroundings figures and statistics.
Same conventions as scripts/04_eval/ndvi_effect_AB.py and fig_surroundings_map.py (Test B):
threshold = (VV < -5) & (VH < -22); distance = EDT from thresholded pixels x 10 m per chip;
window mean = 41 x 41 pixel (410 m) mean of Lee VH over finite pixels."""
import csv
import os
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

R = "/home/kk/jp_claude/RMSD/"; P = R + "processed/"; O = R + "output/water_states/"
# RMSD_VOTES picks another vote file and RMSD_O another output folder, so a candidate model never overwrites the paper figures
VOTES = os.environ.get("RMSD_VOTES", O + "testA_translation_votes.npy")
O = os.environ.get("RMSD_O", O)
EVENTS = {"India", "Somalia", "Spain", "Bolivia"}
PX = 10.0
W = 41


class TestA:
    def __init__(self):
        ch = list(csv.DictReader(open(P + "s1f11_chips.csv")))
        rows = [i for i, r in enumerate(ch) if r["event"] in EVENTS]
        self.names = [ch[i]["chip"] for i in rows]; self.ev = [ch[i]["event"] for i in rows]
        lab = np.asarray(np.load(P + "s1f11_label.npy", mmap_mode="r")[rows])
        jrc = np.asarray(np.load(P + "s1f11_jrc.npy", mmap_mode="r")[rows]) > 0
        YO = np.asarray(np.load(P + "s1f11_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])
        OV = np.asarray(np.load(P + "s1f11_Yobs_valid.npy", mmap_mode="r")[rows], bool)
        z = np.load(P + "s1f11_lee_db.npz"); self.VV, self.VH = z["VV"][rows], z["VH"][rows]
        self.U = np.load(VOTES).astype(bool)
        self.fin = np.isfinite(self.VV) & np.isfinite(self.VH)
        self.valid = (lab >= 0) & OV & np.isfinite(YO[:, 0]) & ~jrc & self.fin
        self.lab = lab
        self.rule = (self.VV < -5) & (self.VH < -22) & self.fin
        self.water = (lab == 1) & self.valid
        self.land = (lab == 0) & self.valid
        self.miss = self.water & ~self.rule
        self.rec = self.miss & self.U
        self.D = np.stack([distance_transform_edt(~r) if r.any() else np.full(r.shape, np.inf) for r in self.rule]) * PX
        self.S = np.stack([uniform_filter(np.where(self.fin[i], self.VH[i], 0.0), W, mode="constant") /
                           np.maximum(uniform_filter(self.fin[i].astype(float), W, mode="constant"), 1e-6)
                           for i in range(len(self.VH))])


def true_colour(path, sl=None):
    """Sentinel-2 true colour with one display rule for every figure: a per-channel haze offset (1st percentile
    of the cloud-free part of the whole chip, L1C is top of atmosphere) and a single gain (98th percentile of the
    cloud-free chip). `sl` selects a window after the offset and gain are fixed on the whole chip."""
    import rasterio as rio
    with rio.open(path) as src:
        full = np.dstack([src.read(j).astype(np.float32) for j in (4, 3, 2)])
    ok = np.all(full > 0, axis=2)
    bright = full.mean(axis=2)
    clear = ok & (bright < np.percentile(bright[ok], 90))
    off = np.array([np.percentile(full[..., c][clear], 1) for c in range(3)])
    hi = np.percentile((full - off)[clear], 98)
    rgb = np.clip((full - off) / hi, 0, 1)
    rgb[~ok] = 1.0
    return rgb if sl is None else rgb[sl]
