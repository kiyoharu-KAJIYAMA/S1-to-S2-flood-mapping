#!/usr/bin/env python3
"""Training script for the recovery ablations (experiments 6-8).

Same data pipeline, losses, optimizer, LR schedule, AMP, step budget and
checkpoint format as scripts/03_train/RMSD_model_bidirectional_fast2.py (the
cur-s* runs), plus four hooks:
  --arch     unet | pixel | shallow | unet2         (experiment 6, X->Y model only)
  --in_mask  "1,1,1" keep flags for VV,VH,RVI; dropped channels are the constant 0.5
             after normalization, in training and inference   (experiment 7)
  --target   msi | label | mndwi   label = water mask, masked BCE + Dice, no Y->X branch
             mndwi = same as label, but the mask is observed MNDWI > 0 of the pool target
             (direct S1 -> water segmentation on exactly the translation pool)
                                                    (experiment 8a)
  --subset   "" | subset8  use the labelled subset built by build_subset8.py
                                                    (experiment 8)
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
sys.path.insert(0, f"{ROOT}/scripts/04_eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from RMSD_flood_eval import UNet, linear_to_db_safe  # noqa: E402
from variant_models import make_xy_model  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t")


ap = argparse.ArgumentParser()
ap.add_argument("--do", type=float, default=0.0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--optimizer", default="AdamW")
ap.add_argument("--loss", default="L1")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--num_epochs", type=int, default=100)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--shuffle", type=str2bool, default=True)
ap.add_argument("--date_out", required=True)
ap.add_argument("--rootpath", required=True)
ap.add_argument("--datapath", required=True)
ap.add_argument("--outpath", required=True)
ap.add_argument("--vis_interval", type=int, default=0)
ap.add_argument("--kl_weight", type=float, default=0.01)
ap.add_argument("--kl_bins", type=int, default=32)
ap.add_argument("--kl_sigma", type=float, default=0.03)
ap.add_argument("--amp", type=str2bool, default=True)
ap.add_argument("--tf32", type=str2bool, default=True)
ap.add_argument("--kl_max_px", type=int, default=50000)
ap.add_argument("--data_prefix", default="trn16")
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--augment", type=str2bool, default=True)
ap.add_argument("--val_frac", type=float, default=0.2)
ap.add_argument("--lr_patience", type=int, default=5)
ap.add_argument("--lr_factor", type=float, default=0.5)
ap.add_argument("--mndwi_weight", type=float, default=2.0)
ap.add_argument("--dice_weight", type=float, default=0.2)
ap.add_argument("--dice_temp", type=float, default=0.05)
ap.add_argument("--components", default="sen12,s1f11,c2s,fp,kuros2,kurr,sturm")
ap.add_argument("--split_keep", default="train")
ap.add_argument("--drop_low_water", type=float, default=0.005)
ap.add_argument("--total_steps", type=int, default=14000)
ap.add_argument("--ckpt_every", type=int, default=0)
ap.add_argument("--tst_prefix", default="tst4")
ap.add_argument("--arch", default="unet", choices=["unet", "pixel", "shallow", "unet2"])
ap.add_argument("--in_mask", default="1,1,1")
ap.add_argument("--target", default="msi", choices=["msi", "label", "mndwi"])
ap.add_argument("--subset", default="")
ap.add_argument("--label_dice_w", type=float, default=1.0)
config = ap.parse_args()

data_dir = config.datapath
outpath_model = os.path.join(config.outpath, config.date_out)
os.makedirs(outpath_model, exist_ok=True)
json.dump(vars(config), open(os.path.join(outpath_model, f"config_{config.date_out}.json"), "w"),
          indent=4)
IN_MASK = [int(v) for v in config.in_mask.split(",")]
assert len(IN_MASK) == 3

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Selected Device: {device}", flush=True)
random.seed(config.seed)
np.random.seed(config.seed)
np.random.randint(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)
USE_AMP = bool(config.amp) and device.type == "cuda"
if config.tf32 and device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"[variant] arch={config.arch} in_mask={IN_MASK} target={config.target} "
      f"subset='{config.subset}' AMP={USE_AMP}", flush=True)

# ------------------------------------------------------------------ data
L_trn = None
if config.subset:
    D = f"{ROOT}/output/ablation_recovery/data"
    X_trn = np.load(f"{D}/{config.subset}_X_VV_VH_RVI_raw.npy")
    Y_trn = np.load(f"{D}/{config.subset}_Y_MNDWI_NDVI_NDWI_raw.npy")
    L_trn = np.load(f"{D}/{config.subset}_label.npy")
    scene_ids = pd.read_csv(f"{D}/{config.subset}_samples.csv")["scene"].to_numpy()
    print(f"[variant] subset {config.subset}: {len(X_trn)} samples, "
          f"{len(set(scene_ids))} scenes", flush=True)
else:
    _p = config.data_prefix
    _comp = [c.strip() for c in config.components.split(",") if c.strip()]
    _ci = pd.read_csv(os.path.join(data_dir, f"{_p}_component.csv"))
    _Xm = np.load(os.path.join(data_dir, f"{_p}_X_VV_VH_RVI_raw.npy"), mmap_mode="r")
    _Ym = np.load(os.path.join(data_dir, f"{_p}_Y_MNDWI_NDVI_NDWI_raw.npy"), mmap_mode="r")
    _keep_mask = _ci["component"].isin(_comp).to_numpy().copy()
    if config.split_keep:
        _sp = pd.read_csv(os.path.join(data_dir, f"{_p}_split.csv"))
        _ok = (_sp["split"] == config.split_keep).to_numpy()
        print(f"[fast2/v18] split_keep='{config.split_keep}': "
              f"{int((_keep_mask & ~_ok).sum())} chips excluded", flush=True)
        _keep_mask &= _ok
    if config.drop_low_water > 0:
        _cmeta = pd.read_csv(os.path.join(data_dir, f"{_p}_chip_meta.csv"), low_memory=False)
        _lowm = _cmeta["water_frac_measured"].to_numpy() < config.drop_low_water
        print(f"[fast2/v18] drop_low_water<{config.drop_low_water}: "
              f"{int((_keep_mask & _lowm).sum())} chips excluded", flush=True)
        _keep_mask &= ~_lowm
    _sel = np.flatnonzero(_keep_mask)
    X_trn = np.empty((len(_sel),) + _Xm.shape[1:], np.float32)
    Y_trn = np.empty((len(_sel),) + _Ym.shape[1:], np.float32)
    for _s in range(0, len(_sel), 64):
        _e = min(_s + 64, len(_sel))
        X_trn[_s:_e] = _Xm[_sel[_s:_e]]
        Y_trn[_s:_e] = _Ym[_sel[_s:_e]]
    del _Xm, _Ym
    scene_ids = _ci["scene"].to_numpy()[_sel]
    if config.target == "mndwi":
        # weak label from the pool target itself: water where observed MNDWI > 0, -1 where not observed
        L_trn = np.where(np.isfinite(Y_trn[:, 0]), (Y_trn[:, 0] > 0).astype(np.int8), np.int8(-1))
        LABEL_SOURCE = "observed MNDWI > 0"
        config.target = "label"
    print(f"[fast2/v17] components={','.join(_comp)} -> {len(_sel)} chips "
          f"{_ci.iloc[_sel]['component'].value_counts().to_dict()}", flush=True)

_x_valid = np.array([float(np.mean((X_trn[i, 0] > 0) & np.isfinite(X_trn[i, 0])))
                     for i in range(len(X_trn))])
if config.target == "label":
    _y_valid = np.array([float(np.mean(L_trn[i] >= 0)) for i in range(len(L_trn))])
else:
    _y_valid = np.array([float(np.mean(np.isfinite(Y_trn[i, 0]))) for i in range(len(Y_trn))])
_keep = (_x_valid >= 0.2) & (_y_valid >= 0.2)
print(f"[fast2] degenerate filter: keep {int(_keep.sum())}/{len(_keep)}", flush=True)
X_trn, Y_trn, scene_ids = X_trn[_keep], Y_trn[_keep], scene_ids[_keep]
if L_trn is not None:
    L_trn = L_trn[_keep]

for ch in (0, 1):
    if np.isfinite(np.nanmedian(X_trn[:, ch])) and np.nanmedian(X_trn[:, ch]) > 0:
        X_trn[:, ch] = linear_to_db_safe(X_trn[:, ch])

_rng = np.random.default_rng(config.seed)
_scenes = np.array(sorted(set(scene_ids)))
_rng.shuffle(_scenes)
_n_val = max(1, int(len(_scenes) * config.val_frac))
val_scenes, trn_scenes = set(_scenes[:_n_val]), set(_scenes[_n_val:])
idx_trn = np.flatnonzero(np.array([s in trn_scenes for s in scene_ids]))
idx_val = np.flatnonzero(np.array([s in val_scenes for s in scene_ids]))
print(f"[fast2] scene split: {len(trn_scenes)} train scenes ({len(idx_trn)} imgs) / "
      f"{len(val_scenes)} val scenes ({len(idx_val)} imgs)", flush=True)
json.dump({"train_scenes": sorted(trn_scenes), "val_scenes": sorted(val_scenes),
           "n_train_imgs": int(len(idx_trn)), "n_val_imgs": int(len(idx_val))},
          open(os.path.join(outpath_model, "scene_split.json"), "w"), indent=2)


def mask_outside_nsigma(arr, mean, std, nsigma=5.0):
    mean = np.asarray(mean).reshape(1, -1, 1, 1)
    std = np.asarray(std).reshape(1, -1, 1, 1)
    arr[(arr < mean - nsigma * std) | (arr > mean + nsigma * std)] = np.nan
    return arr


def normalize_inplace(arr, vmin, vmax):
    for c in range(arr.shape[1]):
        r = vmax[c] - vmin[c] if vmax[c] != vmin[c] else 1.0
        a = arr[:, c]
        a -= np.float32(vmin[c])
        a /= np.float32(r)
        np.clip(a, 0.0, 1.0, out=a)
    return arr


X_trn_mean = np.nanmean(X_trn[idx_trn], axis=(0, 2, 3))
X_trn_std = np.nanstd(X_trn[idx_trn], axis=(0, 2, 3))
X_trn = mask_outside_nsigma(X_trn, X_trn_mean, X_trn_std)
X_trn_min = np.nanpercentile(X_trn[idx_trn], 1, axis=(0, 2, 3))
X_trn_max = np.nanpercentile(X_trn[idx_trn], 99, axis=(0, 2, 3))
X_trn = normalize_inplace(X_trn, X_trn_min, X_trn_max)
if config.target == "msi":
    Y_trn_mean = np.nanmean(Y_trn[idx_trn], axis=(0, 2, 3))
    Y_trn_std = np.nanstd(Y_trn[idx_trn], axis=(0, 2, 3))
    Y_trn = mask_outside_nsigma(Y_trn, Y_trn_mean, Y_trn_std)
    Y_trn_min = np.nanpercentile(Y_trn[idx_trn], 1, axis=(0, 2, 3))
    Y_trn_max = np.nanpercentile(Y_trn[idx_trn], 99, axis=(0, 2, 3))
    Y_trn = normalize_inplace(Y_trn, Y_trn_min, Y_trn_max)
    T0_WATER = float((0.0 - Y_trn_min[0]) / (Y_trn_max[0] - Y_trn_min[0]))
    print(f"[fast2] water threshold (raw MNDWI>0): normalized t0={T0_WATER:.4f}", flush=True)
else:
    del Y_trn
    Y_trn = np.where(L_trn < 0, np.nan, L_trn.astype(np.float32))[:, None]
    Y_trn_mean = Y_trn_std = np.array([np.nan])
    Y_trn_min, Y_trn_max = np.array([0.0]), np.array([1.0])
    T0_WATER = 0.5

MX = np.isfinite(X_trn)
MY = np.isfinite(Y_trn)
X_all = torch.from_numpy(np.nan_to_num(X_trn, nan=0.0, copy=False)).float()
Y_all = torch.from_numpy(np.nan_to_num(Y_trn, nan=0.0, copy=False)).float()
MX_all, MY_all = torch.from_numpy(MX), torch.from_numpy(MY)
for c, keep in enumerate(IN_MASK):
    if not keep:
        X_all[:, c] = 0.5
        MX_all[:, c] = False
del X_trn, Y_trn, MX, MY
N_OUT = 3 if config.target == "msi" else 1


class PairPatchDataset(torch.utils.data.Dataset):
    def __init__(self, indices, crop=0, augment=False):
        self.indices = np.asarray(indices)
        self.crop, self.augment = int(crop), bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        x, y, mx, my = X_all[j], Y_all[j], MX_all[j], MY_all[j]
        H = x.shape[-1]
        if self.crop and self.crop < H:
            c = self.crop
            top = torch.randint(0, H - c + 1, (1,)).item()
            left = torch.randint(0, H - c + 1, (1,)).item()
            sl = (slice(None), slice(top, top + c), slice(left, left + c))
            x, y, mx, my = x[sl], y[sl], mx[sl], my[sl]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x, y, mx, my = (torch.flip(t, [-1]) for t in (x, y, mx, my))
            if torch.rand(1).item() < 0.5:
                x, y, mx, my = (torch.flip(t, [-2]) for t in (x, y, mx, my))
            k = torch.randint(0, 4, (1,)).item()
            if k:
                x, y, mx, my = (torch.rot90(t, k, [-2, -1]) for t in (x, y, mx, my))
        return x.contiguous(), y.contiguous(), mx.contiguous(), my.contiguous()


loader_trn = DataLoader(PairPatchDataset(idx_trn, config.crop, config.augment),
                        batch_size=config.bs, shuffle=config.shuffle, num_workers=0,
                        pin_memory=device.type == "cuda")
loader_val = DataLoader(PairPatchDataset(idx_val, 0, False), batch_size=config.bs,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

# ------------------------------------------------------------------ model / loss
model_xy = make_xy_model(config.arch, 3, N_OUT).to(device)
params = list(model_xy.parameters())
model_yx = None
if config.target == "msi":
    model_yx = UNet(3, 3, bilinear=True).to(device)
    params += list(model_yx.parameters())
optimizer = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": optim.SGD,
             "RMSprop": optim.RMSprop}[config.optimizer](params, lr=config.lr)
base_loss = {"L1": nn.L1Loss, "MAE": nn.L1Loss, "MSE": nn.MSELoss,
             "SmoothL1": nn.SmoothL1Loss}[config.loss](reduction="none")
Y_CH_W = (torch.tensor([config.mndwi_weight, 1.0, 1.0], device=device)
          if config.mndwi_weight != 1.0 else None)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                 factor=config.lr_factor,
                                                 patience=config.lr_patience)
bce = nn.BCEWithLogitsLoss(reduction="none")


def masked_loss(pred, target, mask, channel_weights=None):
    loss = base_loss(pred, target)
    if channel_weights is not None:
        loss = loss * channel_weights.view(1, -1, 1, 1)
    mask = mask.to(torch.bool)
    return loss[mask].mean() if torch.any(mask) else pred.sum() * 0.0


def soft_water_dice(pred, target, mask, t0, temp=0.05, eps=1e-6):
    m = mask[:, 0].to(torch.bool)
    if not torch.any(m):
        return pred.sum() * 0.0
    p = torch.sigmoid((pred[:, 0] - t0) / temp)[m]
    g = (target[:, 0] > t0).float()[m]
    return 1.0 - (2.0 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def soft_histogram_kl_loss(pred, target, mask, bins=32, sigma=0.03, eps=1e-6, max_px=50000):
    mask = mask.to(torch.bool)
    centers = torch.linspace(0.0, 1.0, bins, device=pred.device, dtype=pred.dtype)
    kls = []
    for ch in range(pred.shape[1]):
        m = mask[:, ch]
        if not torch.any(m):
            continue
        pv, tv = pred[:, ch][m], target[:, ch][m]
        if max_px and pv.numel() > max_px:
            sel = torch.randperm(pv.numel(), device=pv.device)[:max_px]
            pv, tv = pv[sel], tv[sel]
        pw = torch.exp(-0.5 * ((pv.reshape(-1, 1).clamp(0, 1) - centers) / sigma) ** 2).sum(0)
        tw = torch.exp(-0.5 * ((tv.reshape(-1, 1).clamp(0, 1) - centers) / sigma) ** 2).sum(0)
        kls.append(F.kl_div(torch.log(pw / (pw.sum() + eps) + eps), tw / (tw.sum() + eps),
                            reduction="sum"))
    return torch.stack(kls).mean() if kls else pred.sum() * 0.0


def label_loss(logit, y, my):
    m = my[:, 0].to(torch.bool)
    if not torch.any(m):
        return logit.sum() * 0.0
    l = bce(logit[:, 0], y[:, 0])[m].mean()
    p = torch.sigmoid(logit[:, 0])[m]
    g = y[:, 0][m]
    dice = 1.0 - (2 * (p * g).sum() + 1e-6) / (p.sum() + g.sum() + 1e-6)
    return l + config.label_dice_w * dice


def step_losses(Xb, Yb, MXb, MYb):
    with torch.cuda.amp.autocast(enabled=USE_AMP):
        pred_y = model_xy(Xb)
        pred_x = model_yx(Yb) if model_yx is not None else None
    pred_y = pred_y.float()
    if config.target == "label":
        l = label_loss(pred_y, Yb, MYb)
        return l, l, torch.zeros((), device=device), torch.zeros((), device=device)
    pred_x = pred_x.float()
    loss_xy = masked_loss(pred_y, Yb, MYb, Y_CH_W)
    loss_yx = masked_loss(pred_x, Xb, MXb)
    kl = (soft_histogram_kl_loss(pred_y, Yb, MYb, config.kl_bins, config.kl_sigma,
                                 max_px=config.kl_max_px)
          + soft_histogram_kl_loss(pred_x, Xb, MXb, config.kl_bins, config.kl_sigma,
                                   max_px=config.kl_max_px))
    loss = loss_xy + loss_yx + config.kl_weight * kl
    if config.dice_weight > 0:
        loss = loss + config.dice_weight * soft_water_dice(pred_y, Yb, MYb, T0_WATER,
                                                           config.dice_temp)
    return loss, loss_xy, loss_yx, kl


if config.total_steps > 0:
    _spe = max(1, len(loader_trn))
    _ne = max(1, int(np.ceil(config.total_steps / _spe)))
    print(f"[fast2/v18] total_steps={config.total_steps}: {_spe} steps/epoch -> "
          f"num_epochs {config.num_epochs} -> {_ne}", flush=True)
    config.num_epochs = _ne

FIELDS = ["epoch", "train_loss", "train_xy_loss", "train_yx_loss", "train_kl_loss",
          "val_loss", "val_xy_loss", "val_yx_loss", "val_kl_loss", "best_val_loss"]
history, best_val = [], np.inf
t_start = time.time()
for epoch in range(config.num_epochs):
    model_xy.train()
    if model_yx is not None:
        model_yx.train()
    acc = np.zeros(4)
    n = 0
    for Xb, Yb, MXb, MYb in tqdm(loader_trn, desc=f"Epoch {epoch + 1}/{config.num_epochs} train",
                                 mininterval=30):
        Xb, Yb, MXb, MYb = (t.to(device) for t in (Xb, Yb, MXb, MYb))
        optimizer.zero_grad()
        loss, lxy, lyx, kl = step_losses(Xb, Yb, MXb, MYb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc += [loss.item(), lxy.item(), lyx.item(), kl.item()]
        n += 1
    trn = acc / max(n, 1)
    model_xy.eval()
    if model_yx is not None:
        model_yx.eval()
    acc = np.zeros(4)
    n = 0
    with torch.no_grad():
        for Xb, Yb, MXb, MYb in loader_val:
            Xb, Yb, MXb, MYb = (t.to(device) for t in (Xb, Yb, MXb, MYb))
            loss, lxy, lyx, kl = step_losses(Xb, Yb, MXb, MYb)
            acc += [loss.item(), lxy.item(), lyx.item(), kl.item()]
            n += 1
    val = acc / max(n, 1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    scheduler.step(val[0])
    print(f"  lr = {optimizer.param_groups[0]['lr']:.2e}", flush=True)
    print(f"Epoch {epoch + 1:03d}/{config.num_epochs} | train loss = {trn[0]:.5f} "
          f"(xy = {trn[1]:.5f}, yx = {trn[2]:.5f}, kl = {trn[3]:.5f}) | val loss = {val[0]:.5f} "
          f"(xy = {val[1]:.5f}, yx = {val[2]:.5f}, kl = {val[3]:.5f}) | "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    is_best = val[0] < best_val
    best_val = min(best_val, val[0])
    history.append(dict(zip(FIELDS, [epoch + 1, *map(float, trn), *map(float, val),
                                     float(best_val)])))
    with open(os.path.join(outpath_model, "loss_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(history)
    json.dump(history, open(os.path.join(outpath_model, "loss_history.json"), "w"), indent=4)
    if is_best:
        torch.save({
            "epoch": epoch,
            "model_xy_state_dict": model_xy.state_dict(),
            "model_yx_state_dict": model_yx.state_dict() if model_yx is not None else None,
            "train_loss": trn[0], "val_loss": val[0],
            "X_min": X_trn_min, "X_max": X_trn_max, "Y_min": Y_trn_min, "Y_max": Y_trn_max,
            "X_mean": X_trn_mean, "X_std": X_trn_std, "Y_mean": Y_trn_mean, "Y_std": Y_trn_std,
            "variant": dict(arch=config.arch, in_mask=IN_MASK, target=config.target,
                            subset=config.subset, n_out=N_OUT,
                            label_source=globals().get("LABEL_SOURCE", "hand label")),
        }, os.path.join(outpath_model, "best_model.pt"))
print(f"done {config.date_out}: best val {best_val:.5f}, {(time.time() - t_start) / 60:.1f} min",
      flush=True)
