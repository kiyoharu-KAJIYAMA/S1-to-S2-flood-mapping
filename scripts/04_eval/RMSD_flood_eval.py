#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flood-detection evaluation: downstream-task performance of the SAR->MS translation model (model_xy).

Steps:
  1. load best_model.pt (v2_run02) and tst2 (338 pairs)
  2. same preprocessing as training (dB conversion -> 5-sigma mask -> min-max normalisation, stats from ckpt)
  3. infer SAR -> [MNDWI, NDVI, NDWI] with model_xy
  4. binarise GT/prediction with water mask = raw MNDWI > 0 (standard criterion of Xu 2006)
  5. compute IoU / F1 / Precision / Recall
       - all pixels pooled (global)
       - per image (only images with GT water >= 0.5%)
       - stratified by flood / non-flood date (flood flag in the tst2 CSV)
  6. baseline: VV_dB threshold (best threshold searched on the test set = favourable to the baseline)
  7. also MNDWI L1 (raw-value scale)

Output: summary.json, per_image.csv, example_*.png under output/<run>/flood_eval/
Run: inside the torch container.
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- UNet (same definition as the training script) -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear); self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear); self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ----------------------------- preprocessing (same as training) -----------------------------
def linear_to_db_safe(sar):
    out = np.full_like(sar, np.nan, dtype=np.float32)
    valid = np.isfinite(sar) & (sar > 0)
    out[valid] = 10.0 * np.log10(sar[valid])
    return out

def metrics_from_masks(gt, pred, valid):
    """gt/pred: bool water masks, valid: pixels to evaluate"""
    gt = gt & valid; pred = pred & valid
    tp = int(np.sum(gt & pred)); fp = int(np.sum(~gt & pred & valid)); fn = int(np.sum(gt & ~pred))
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and (prec + rec) > 0) else (0.0 if (tp + fp + fn) > 0 else np.nan)
    return dict(tp=tp, fp=fp, fn=fn, iou=iou, precision=prec, recall=rec, f1=f1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_run02")
    ap.add_argument("--root", default="/home/kk/jp_claude/RMSD/data/SEN12FLOOD")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()

    outdir = os.path.join(a.root, "output", a.run, "flood_eval")
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    ck = torch.load(os.path.join(a.root, "output", a.run, "best_model.pt"), map_location="cpu")
    X_mean, X_std = ck["X_mean"], ck["X_std"]
    X_min, X_max = ck["X_min"], ck["X_max"]
    Y_min, Y_max = ck["Y_min"], ck["Y_max"]
    model = UNet(3, 3, bilinear=True).to(device)
    model.load_state_dict(ck["model_xy_state_dict"])
    model.eval()
    print(f"loaded model_xy (epoch {ck['epoch']})", flush=True)

    # ---- load tst2 ----
    P = os.path.join(a.root, "processed")
    X = np.load(P + "/tst2_X_VV_VH_RVI_raw.npy")
    Y = np.load(P + "/tst2_Y_MNDWI_NDVI_NDWI_raw.npy")
    df = pd.read_csv(P + "/tst2_matched_samples_good.csv")
    assert len(df) == X.shape[0]

    # ---- drop degenerate pairs (same criterion as training) ----
    xv = np.array([float(np.mean((X[i, 0] > 0) & np.isfinite(X[i, 0]))) for i in range(len(X))])
    yv = np.array([float(np.mean(np.isfinite(Y[i, 0]))) for i in range(len(Y))])
    keep = (xv >= 0.2) & (yv >= 0.2)
    print(f"degenerate filter: keep {keep.sum()}/{len(keep)}", flush=True)
    X, Y, df = X[keep], Y[keep], df[keep].reset_index(drop=True)

    # ---- same preprocessing as training ----
    for chn in (0, 1):  # VV, VH -> dB (training: convert if the global median is positive)
        if np.nanmedian(X[:, chn]) > 0:
            X[:, chn] = linear_to_db_safe(X[:, chn])
    for arr, mean, std in [(X, X_mean, X_std), (Y, ck["Y_mean"], ck["Y_std"])]:
        lo = (np.asarray(mean) - 5 * np.asarray(std)).reshape(1, -1, 1, 1)
        hi = (np.asarray(mean) + 5 * np.asarray(std)).reshape(1, -1, 1, 1)
        bad = (arr < lo) | (arr > hi)
        arr[bad] = np.nan
    def norm(arr, vmin, vmax):
        for c in range(arr.shape[1]):
            rng = vmax[c] - vmin[c] if vmax[c] != vmin[c] else 1.0
            arr[:, c] -= np.float32(vmin[c]); arr[:, c] /= np.float32(rng)
        np.clip(arr, 0.0, 1.0, out=arr)
        return arr
    X = norm(X, X_min, X_max); Y = norm(Y, Y_min, Y_max)
    MX = np.isfinite(X); MY = np.isfinite(Y)
    Xz = np.nan_to_num(X, nan=0.0)

    # ---- inference ----
    preds = np.empty_like(Y, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(Xz), a.bs):
            xb = torch.from_numpy(Xz[s:s + a.bs]).float().to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pb = model(xb)
            preds[s:s + a.bs] = pb.float().cpu().numpy()
    print(f"inference done: {len(preds)} images", flush=True)

    # ---- water mask (raw MNDWI > 0 expressed as a normalised threshold) ----
    t0 = float((0.0 - Y_min[0]) / (Y_max[0] - Y_min[0]))
    print(f"water threshold: raw MNDWI>0 -> normalized {t0:.4f}", flush=True)
    gt_w = (Y[:, 0] > t0)
    pr_w = (preds[:, 0] > t0)
    valid = MY[:, 0] & MX[:, 0]

    # ---- metrics: global / flood-stratified / per image ----
    flood = df["flood"].to_numpy(bool)
    summary = {"run": a.run, "epoch": int(ck["epoch"]), "n_images": int(len(preds)),
               "n_flood_images": int(flood.sum()), "water_threshold_norm": t0}
    summary["global"] = metrics_from_masks(gt_w, pr_w, valid)
    summary["flood_images"] = metrics_from_masks(gt_w[flood], pr_w[flood], valid[flood])
    summary["nonflood_images"] = metrics_from_masks(gt_w[~flood], pr_w[~flood], valid[~flood])

    # MNDWI L1 (raw-value scale)
    d = np.abs(preds[:, 0] - Y[:, 0])[valid]
    summary["mndwi_L1_norm"] = float(np.mean(d))
    summary["mndwi_L1_raw"] = float(np.mean(d) * (Y_max[0] - Y_min[0]))

    # headroom from calibrating the prediction threshold (search the F1-maximising threshold)
    best = (t0, -1)
    gtv = gt_w & valid
    for t in np.linspace(t0 - 0.15, t0 + 0.15, 31):
        m = metrics_from_masks(gtv, (preds[:, 0] > t) & valid, valid)
        if np.isfinite(m["f1"]) and m["f1"] > best[1]:
            best = (float(t), m["f1"])
    summary["pred_threshold_best"] = {"threshold_norm": best[0], "f1": best[1]}
    # stratified metrics at the calibrated threshold (performance after prediction-bias correction)
    pr_w_cal = (preds[:, 0] > best[0])
    summary["calibrated_global"] = metrics_from_masks(gt_w, pr_w_cal, valid)
    summary["calibrated_flood_images"] = metrics_from_masks(gt_w[flood], pr_w_cal[flood], valid[flood])
    summary["calibrated_nonflood_images"] = metrics_from_masks(gt_w[~flood], pr_w_cal[~flood], valid[~flood])

    # ---- baseline: VV_dB threshold (best on the test set = lenient condition) ----
    vvn = Xz[:, 0]
    bl = (-1, None)
    for t in np.linspace(0.05, 0.95, 46):
        m = metrics_from_masks(gtv, (vvn < t) & valid, valid)   # water = low backscatter
        if np.isfinite(m["f1"]) and m["f1"] > bl[0]:
            bl = (m["f1"], {"threshold_norm": float(t), **m})
    summary["baseline_VV_threshold_oracle"] = bl[1]

    # ---- per image (only GT water >= 0.5%) ----
    rows = []
    for i in range(len(preds)):
        v = valid[i]
        wfrac = float(np.mean(gt_w[i] & v))
        m = metrics_from_masks(gt_w[i], pr_w[i], v)
        rows.append({"idx": i, "folder": os.path.basename(str(df.loc[i, "folder"])),
                     "date": df.loc[i, "date"], "flood": bool(flood[i]),
                     "gt_water_frac": wfrac, **m})
    dfr = pd.DataFrame(rows)
    dfr.to_csv(os.path.join(outdir, "per_image.csv"), index=False)
    dfw = dfr[dfr.gt_water_frac >= 0.005]
    summary["per_image_mean_iou_waterimgs"] = float(dfw["iou"].mean())
    summary["per_image_median_iou_waterimgs"] = float(dfw["iou"].median())
    summary["n_water_images"] = int(len(dfw))

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- example figures: best / median / worst IoU among flood images ----
    cand = dfw[dfw.flood].sort_values("iou") if dfw.flood.any() else dfw.sort_values("iou")
    if len(cand):
        picks = [cand.iloc[-1]["idx"], cand.iloc[len(cand)//2]["idx"], cand.iloc[0]["idx"]]
        names = ["best", "median", "worst"]
        for nm, i in zip(names, picks):
            i = int(i)
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(np.where(MY[i,0], Y[i,0], np.nan), vmin=0, vmax=1); axs[0].set_title("GT MNDWI (norm)")
            axs[1].imshow(preds[i,0], vmin=0, vmax=1); axs[1].set_title("Pred MNDWI (SAR->MS)")
            axs[2].imshow(np.where(valid[i], gt_w[i], np.nan), vmin=0, vmax=1, cmap="Blues"); axs[2].set_title("GT water")
            axs[3].imshow(np.where(valid[i], pr_w[i], np.nan), vmin=0, vmax=1, cmap="Blues")
            r = dfr[dfr.idx == i].iloc[0]
            axs[3].set_title(f"Pred water IoU={r.iou:.2f} F1={r.f1:.2f}")
            for ax in axs: ax.axis("off")
            fig.suptitle(f"{nm}: scene {r.folder} {r.date} flood={r.flood}")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"example_{nm}.png"), dpi=110)
            plt.close(fig)

    print(json.dumps(summary, indent=2, default=float))
    print("saved to", outdir, flush=True)

if __name__ == "__main__":
    main()
