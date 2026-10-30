"""Shared data loading, preprocessing, inference and evaluation for the
recovery-ablation experiments (Test A = Sen1Floods11 held-out events,
Test B = FloodPlanet, CAL = Sen1Floods11 training events).

Definitions follow the request document: everything is measured against the
fixed baseline rule (Lee VV < -5 dB & VH < -22 dB), and the distance D is
always taken from that rule, whatever detector is being scored.
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, uniform_filter

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
P = f"{ROOT}/processed"
OUT = f"{ROOT}/output/ablation_recovery"
sys.path.insert(0, f"{ROOT}/scripts/04_eval")
from RMSD_flood_eval import UNet, linear_to_db_safe  # noqa: E402
from arch_builder import build_from_ckpt  # noqa: E402

RUNS = [f"cur-s{s}_run01" for s in range(1, 6)]
HOLD = {"India", "Somalia", "Spain", "Bolivia"}
RULE_VV, RULE_VH = -5.0, -22.0
BANDS = [(0, 30), (30, 50), (50, 100), (100, 200), (200, 500), (500, np.inf)]
BEYOND = (10, 30, 100)
PX_M = 10.0


class TestSet:
    pass


def _rule_distance(rule):
    D = np.empty(rule.shape, np.float32)
    for i in range(len(rule)):
        D[i] = distance_transform_edt(~rule[i]) * PX_M if rule[i].any() else np.inf
    return D


def load(test):
    """test: 'A' (139 held-out chips), 'B' (54 FloodPlanet chips), 'CAL' (307 train-event chips)."""
    T = TestSet()
    T.name = test
    if test == "B":
        ch = list(csv.DictReader(open(f"{P}/fp_eval_chips.csv")))
        rows = np.arange(len(ch))
        T.names = [r["chip"] for r in ch]
        T.groups = np.array([r["region"] for r in ch])
        pre, lee = "fp_eval", "fp_lee_db.npz"
    else:
        ch = list(csv.DictReader(open(f"{P}/s1f11_chips.csv")))
        keep = (lambda e: e in HOLD) if test == "A" else (lambda e: e not in HOLD)
        rows = np.array([i for i, r in enumerate(ch) if keep(r["event"])])
        T.names = [ch[i]["chip"] for i in rows]
        T.groups = np.array([ch[i]["event"] for i in rows])
        pre, lee = "s1f11", "s1f11_lee_db.npz"
    T.rows = rows
    T.X = np.asarray(np.load(f"{P}/{pre}_X_VV_VH_RVI_raw.npy", mmap_mode="r")[rows], np.float32)
    lab = np.asarray(np.load(f"{P}/{pre}_label.npy", mmap_mode="r")[rows])
    jrc = np.asarray(np.load(f"{P}/{pre}_jrc.npy", mmap_mode="r")[rows]) > 0
    YO = np.asarray(np.load(f"{P}/{pre}_Yobs_MNDWI_NDVI_NDWI.npy", mmap_mode="r")[rows])
    OV = np.asarray(np.load(f"{P}/{pre}_Yobs_valid.npy", mmap_mode="r")[rows], bool)
    z = np.load(f"{P}/{lee}")
    T.VV = np.asarray(z["VV"][rows], np.float32)
    T.VH = np.asarray(z["VH"][rows], np.float32)
    T.MNDWI = YO[:, 0].astype(np.float32)
    T.NDVI = YO[:, 1].astype(np.float32)
    T.valid = (lab >= 0) & OV & np.isfinite(T.MNDWI) & ~jrc
    T.gt = (lab == 1) & T.valid
    # pixel set used by eval_optical_axis.py for threshold calibration (no MSI condition)
    T.lab_ok = np.isfinite(T.X[:, 0]) & (T.X[:, 0] > 0) & (lab >= 0) & ~jrc
    T.gt_lab = (lab == 1) & T.lab_ok
    T.rule = (T.VV < RULE_VV) & (T.VH < RULE_VH)
    T.M = (T.MNDWI > 0.14) & T.valid
    T.D = _rule_distance(T.rule)
    with np.errstate(invalid="ignore"):
        s1 = T.VH < -19.0
        s2 = (T.VH >= -19.0) & (T.NDVI <= 0.3)
        s3 = (T.VH >= -19.0) & (T.NDVI > 0.3)
    T.states = {"dim": s1, "bright_noveg": s2, "bright_veg": s3}
    return T


# ---------------------------------------------------------------- inference
_CK = {}


def ckpt(run):
    if run not in _CK:
        _CK[run] = torch.load(f"{ROOT}/output/{run}/best_model.pt", map_location="cpu",
                              weights_only=False)
    return _CK[run]


def preprocess(X, ck):
    """Same as venn_cell_attributes.infer_votes: dB, +-5 sigma to NaN, min-max, clip, NaN->0."""
    Z = X.astype(np.float32).copy()
    for c in (0, 1):
        if np.nanmedian(Z[:, c]) > 0:
            Z[:, c] = linear_to_db_safe(Z[:, c])
    lo = (np.asarray(ck["X_mean"]) - 5 * np.asarray(ck["X_std"])).reshape(1, -1, 1, 1)
    hi = (np.asarray(ck["X_mean"]) + 5 * np.asarray(ck["X_std"])).reshape(1, -1, 1, 1)
    Z[(Z < lo) | (Z > hi)] = np.nan
    for c in range(3):
        Z[:, c] = (Z[:, c] - ck["X_min"][c]) / (ck["X_max"][c] - ck["X_min"][c])
    np.clip(Z, 0, 1, out=Z)
    return np.nan_to_num(Z, nan=0.0)


def build_model(ck, dev):
    m = build_from_ckpt(ck, UNet).to(dev)
    m.load_state_dict(ck["model_xy_state_dict"])
    return m.eval()


def predict(model, Z, dev, bs=16, tile=0, overlap=False):
    """Channel-0 prediction (normalized MNDWI). tile>0 runs the model on tile x tile
    pieces: overlap=False stitches whole tiles, overlap=True uses stride tile/2 on a
    reflect-padded image and keeps only the central half of every tile."""
    N, C, H, W = Z.shape
    if tile <= 0:
        out = np.empty((N, H, W), np.float32)
        with torch.no_grad():
            for s in range(0, N, bs):
                xb = torch.from_numpy(Z[s:s + bs]).to(dev)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    out[s:s + bs] = model(xb).float().cpu().numpy()[:, 0]
        return out
    t = tile
    if not overlap:
        k = H // t
        tiles = Z.reshape(N, C, k, t, k, t).transpose(0, 2, 4, 1, 3, 5).reshape(N * k * k, C, t, t)
        pt = predict(model, np.ascontiguousarray(tiles), dev, bs=max(bs, 4096 // (t * t // 256)))
        return pt.reshape(N, k, k, t, t).transpose(0, 1, 3, 2, 4).reshape(N, H, W)
    q = t // 4
    Zp = np.pad(Z, ((0, 0), (0, 0), (q, q), (q, q)), mode="reflect")
    k = H // (t // 2)
    pos = [(r * (t // 2)) for r in range(k)]
    tiles = np.stack([Zp[:, :, r:r + t, c:c + t] for r in pos for c in pos], axis=1)
    tiles = tiles.reshape(N * k * k, C, t, t)
    pt = predict(model, np.ascontiguousarray(tiles), dev, bs=max(bs, 4096 // (t * t // 256)))
    pt = pt.reshape(N, k, k, t, t)[:, :, :, q:q + t // 2, q:q + t // 2]
    return pt.transpose(0, 1, 3, 2, 4).reshape(N, H, W)


def tau_of(run, test):
    axis = "fp" if test == "B" else "s1f11"
    return json.load(open(f"{ROOT}/output/{run}/optical_axis_{axis}/summary.json"))["thresholds"]["unet"]


def votes(X, test, dev, runs=RUNS, taus=None, modify=None, seed=0, tile=0, overlap=False, bs=16):
    """Majority vote (>= 3 of 5, or > half for other ensemble sizes). modify(Z, rng, run)
    may alter the normalized input before inference."""
    V = np.zeros(X.shape[0:1] + X.shape[2:], np.int8)
    for r, run in enumerate(runs):
        ck = ckpt(run)
        Z = preprocess(X, ck)
        if modify is not None:
            Z = modify(Z, np.random.default_rng(seed * 100 + r), run)
        tau = taus[run] if taus is not None else tau_of(run, test)
        pred = predict(build_model(ck, dev), Z, dev, bs=bs, tile=tile, overlap=overlap)
        V += (pred > tau)
    need = 3 if len(runs) == 5 else len(runs) // 2 + 1
    return V >= need


# ---------------------------------------------------------------- features
def window_mean(A, size):
    """NaN-aware moving average (per chip), size in pixels (odd)."""
    out = np.empty_like(A, np.float32)
    for i in range(len(A)):
        a = A[i]
        ok = np.isfinite(a)
        x = np.where(ok, a, 0.0).astype(np.float64)
        n = uniform_filter(ok.astype(np.float64), size, mode="reflect")
        s = uniform_filter(x, size, mode="reflect")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(n > 0, s / n, np.nan)
    return out


def window_std(A, size):
    m = window_mean(A, size)
    m2 = window_mean(A * A, size)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum(m2 - m * m, 0)).astype(np.float32)


# ---------------------------------------------------------------- evaluation
def _prf(gt, pred, valid):
    gt = gt & valid
    pred = pred & valid
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * pr * rc / (pr + rc) if (tp + fp and tp + fn and pr + rc) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, iou=iou, precision=pr, recall=rc, f1=f1)


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _beyond(gt, m, sel):
    n = int(sel.sum())
    ng = int((gt & sel).sum())
    npred = int((m & sel).sum())
    tp = int((gt & m & sel).sum())
    return dict(n_px=n, n_gt=ng, n_pred=npred,
                chance=_r(ng / n if n else np.nan),
                precision=_r(tp / npred if npred else np.nan),
                recall=_r(tp / ng if ng else np.nan))


def evaluate(mask, T, per_group=True):
    v, gt, rule, D = T.valid, T.gt, T.rule, T.D
    m = mask & v
    miss = gt & ~rule
    rec = miss & m
    out = {"test": T.name, "overall": {k: (_r(x) if isinstance(x, float) else x)
                                       for k, x in _prf(gt, m, v).items()}}
    out["missed_share"] = _r(miss.sum() / max(gt.sum(), 1))
    out["recovery"] = dict(missed_px=int(miss.sum()), recovered_px=int(rec.sum()),
                           rate=_r(rec.sum() / max(miss.sum(), 1)),
                           within_100m=_r((rec & (D <= 100)).sum() / max(rec.sum(), 1)))
    bands = {}
    for lo, hi in BANDS:
        sel = miss & (D > lo) & (D <= hi) if lo > 0 else miss & (D <= hi)
        bands[f"{lo}-{hi if np.isfinite(hi) else 'inf'}"] = dict(
            missed_px=int(sel.sum()), rate=_r((rec & sel).sum() / max(sel.sum(), 1)))
    out["recovery_by_distance"] = bands
    bey = {}
    for thr in BEYOND:
        reg = v & ~rule & (D > thr)
        bey[f">{thr}m"] = {"all": _beyond(gt, m, reg)}
        for s, sm in T.states.items():
            bey[f">{thr}m"][s] = _beyond(gt, m, reg & sm)
    out["beyond"] = bey
    if per_group:
        g = {}
        for name in sorted(set(T.groups)):
            sel = T.groups == name
            sub = TestSet()
            sub.name, sub.valid, sub.gt, sub.rule, sub.D = name, v[sel], gt[sel], rule[sel], D[sel]
            sub.states = {k: x[sel] for k, x in T.states.items()}
            r = evaluate(m[sel], sub, per_group=False)
            g[name] = dict(overall=r["overall"], recovery=r["recovery"],
                           beyond30=r["beyond"][">30m"])
        out["groups"] = g
    return out


def average(items):
    """Element-wise mean (and sd) of numeric leaves over a list of nested dicts."""
    def rec(vals):
        v0 = vals[0]
        if isinstance(v0, dict):
            return {k: rec([v[k] for v in vals]) for k in v0}
        if isinstance(v0, (int, float)) and not isinstance(v0, bool):
            arr = np.array([np.nan if v is None else v for v in vals], float)
            m = np.nanmean(arr) if np.isfinite(arr).any() else np.nan
            return _r(m) if len(vals) == 1 else dict(mean=_r(m), sd=_r(np.nanstd(arr)))
        return v0
    return rec(items)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def _m(x):
    """Numeric leaf, also for averaged results where leaves are {mean, sd}."""
    if isinstance(x, dict):
        x = x.get("mean")
    return float("nan") if x is None else float(x)


def brief(r):
    o, rc, b = r["overall"], r["recovery"], r["beyond"][">30m"]["all"]
    return (f"IoU {_m(o['iou']):.3f} P {_m(o['precision']):.3f} R {_m(o['recall']):.3f} | "
            f"recov {100 * _m(rc['rate']):.1f}% (<=100m {100 * _m(rc['within_100m']):.0f}%) | "
            f">30m chance {_m(b['chance']):.3f} P {_m(b['precision']):.3f} R {_m(b['recall']):.3f}")


def device(pref=None):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
