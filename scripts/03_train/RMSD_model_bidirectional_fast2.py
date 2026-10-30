import os
import time
import math
import json
import argparse
import random
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def add_metrics(x, y, ax=None, s=0, show_corr=True, show_ubRMSD=True, show_bias=True, show_line_eq=True, 
                corner='top-right', color='black', alpha=0.7, fontsize = 8, vminmax=None, fontcolor = None):
    """
    Add metrics (correlation, ubRMSD, bias, regression line equation) to a matplotlib axis.
    
    Parameters:
        x (array-like): x-axis data.
        y (array-like): y-axis data.
        ax (matplotlib.axes.Axes): Axis to plot the scatter plot and metrics text.
        s (float): Marker size for the scatter plot.
        show_corr (bool): Whether to show the correlation coefficient (R).
        show_ubRMSD (bool): Whether to show the unbiased RMSD (ubRMSD).
        show_bias (bool): Whether to show the bias.
        show_line_eq (bool): Whether to show the regression line equation.
        corner (str): Corner of the axis to place the metrics ('top-right', 'top-left', 'bottom-right', 'bottom-left').
        color (str): Color of the scatter points and regression line.
        alpha (float): Transparency of the scatter points.
    """
    x = np.array(x) # reference
    y = np.array(y) # prediction
    
    # Remove NaN values
    nan_mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[nan_mask], y[nan_mask]

    # Calculate metrics
    corr = np.corrcoef(x, y)[0, 1]
    ubRMSD = ubrmsd(x, y)
    bias_ = np.nanmean(y - x)
    
    # Linear regression
    line = LinearRegression()
    line.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    
    # Prepare the metrics text based on user preference
    metrics = []
    if show_corr:
        metrics.append(r'$R=%.3f$' % corr)
    if show_ubRMSD:
        metrics.append(r'$ubRMSD=%.3f$' % ubRMSD)
    if show_bias:
        metrics.append(r'$bias=%.3f$' % bias_)
    if show_line_eq:
        metrics.append(r'$y = {:.4f}x {:+.3f}$'.format(line.coef_[0, 0], line.intercept_[0]))
        
    # Combine metrics text
    textstr = '\n'.join(metrics)
    
    # Determine text position based on corner argument
    corner_positions = {
        'top-right': (0.95, 0.95),
        'top-left': (0.05, 0.95),
        'bottom-right': (0.95, 0.05),
        'bottom-left': (0.05, 0.05),
    }
    position = corner_positions.get(corner, (0.95, 0.95))  # Default to top-right if invalid corner is given
    if ax==None:
        fig,ax = plt.subplots()
        ax.scatter(x, y, s=s, alpha=alpha, color=color)
    else:
        # Plot scatter points
        ax.scatter(x, y, s=s, alpha=alpha, color=color)

    # Get x-axis limits and extend regression line to the full range
    x_limits = ax.get_xlim()
    if vminmax==None:
        x_pred = np.linspace(x_limits[0], x_limits[1], 100)
    else:
        x_pred = np.linspace(vminmax[0], vminmax[1], 100)
    y_pred = line.predict(x_pred.reshape(-1, 1))
    
    # Plot regression line
    ax.plot(x_pred, y_pred, color=color, alpha=0.8, linewidth=2)
    
    # Add text to the specified corner with the chosen text color
    if fontcolor != None:
        color = fontcolor
    else:
        None
    
    props = dict(boxstyle='round', facecolor='white', alpha=.9)
    ax.text(position[0], position[1], textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top' if 'top' in corner else 'bottom',
            horizontalalignment='right' if 'right' in corner else 'left',
            bbox=props, color=color)
    
    # Set face color
    ax.set_facecolor('w')
    
def corr_ubrmsd_bias(V1, V2):
    mask = np.isfinite(V1) & np.isfinite(V2)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan
    # mask = np.isnan(V1) & np.isnan(V2)
    # if np.sum(mask) < 2:
    #     return np.nan, np.nan, np.nan
    
    V1_masked = V1[mask]
    V2_masked = V2[mask]

    # regression impossible when all values are identical
    if np.all(V1_masked == V1_masked[0]):
        return np.nan, np.nan, np.nan

    try:
        r = corr(V1_masked, V2_masked)
        ubrmsd_ = ubrmsd(V1_masked, V2_masked)
        bias_ = bias(V1_masked, V2_masked)
        
        return r, ubrmsd_, bias_
    except Exception:
        return np.nan, np.nan, np.nan
    
def corr(predictions, targets):
    x = predictions
    y = targets
    x1 = np.array(x); y1 = np.array(y);
    x1_nanloc = np.isnan(x1)
    y1_nanloc = np.isnan(y1)
    nanloc = x1_nanloc + y1_nanloc    
    x1 = x1[nanloc==0]
    y1 = y1[nanloc==0]
    rho = np.nanmean((x1 - x1.mean()) * (y1 - y1.mean())) / (x1.std() * y1.std())    
    return rho

def ubrmsd(predictions, targets):
    return np.sqrt(np.nanmean((((predictions - np.nanmean(predictions)) - (targets - np.nanmean(targets))) ** 2)))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def bias(predictions, targets): 
    return (predictions - targets).mean()

def bias2(targets, predictions): 
    return (targets - predictions ).mean()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            # nn.Dropout2d(.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            # nn.Dropout2d(.1),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = nn.Linear(logits)
        return logits
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def save_config(args, save_dir='.'):
    """
    Save the argparse settings to a JSON file.
    """
    # 1. create the save directory if missing
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. convert the args object to a dict (vars())
    config_dict = vars(args)

    # 3. build the file name (e.g. config_20250101_12_00_00.json)
    # include date_out in the name when present, to tell runs apart
    file_name = f"config_{config_dict.get('date_out', 'experiment')}.json"
    file_path = os.path.join(save_dir, file_name)

    # 4. write as JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4) # indent=4 for readability
        
    print(f"Configuration saved to: {file_path}")

def cdn(tensor):
    return tensor.cpu().detach().numpy()

# =============================================================================
# experiment configuration
# =============================================================================
parser = argparse.ArgumentParser(description="Configuration for the model development.")

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument('--do', type=float, default=0.0, help='dropout rate')
parser.add_argument('--bs', type=int, default=4, help='batch size')
parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer., string, see https://pytorch.org/docs/stable/optim.html')
parser.add_argument('--loss', type=str, default='L1', help='Loss function')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rates')
parser.add_argument('--num_epochs', type=int, default=100, help='number of iteration')
parser.add_argument('--gpu', type=int, default=0, help='gpu number i want to use')
parser.add_argument('--note', type=str, default="Contact to Jaese Lee (jaeselee@gist.ac.kr)", help='Note')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle in dataloader')
parser.add_argument('--date_out', type=str, default=time.strftime('%y%m%d %H_%M_%S', time.localtime(time.time())) , help='save exp date')
parser.add_argument('--rootpath', type=str, default='/home/kk/jp_claude/RMSD/data/SEN12FLOOD/', help='root path that contains processed/ and output/')
parser.add_argument('--vis_interval', type=int, default=1, help='save validation visualization every N epochs; set 0 to disable interval saves')
parser.add_argument('--vis_sample_idx', type=int, default=0, help='sample index in the first validation batch to visualize')
parser.add_argument('--kl_weight', type=float, default=0.01, help='weight for auxiliary channel-wise histogram KL loss')
parser.add_argument('--kl_bins', type=int, default=32, help='number of bins for auxiliary histogram KL loss')
parser.add_argument('--kl_sigma', type=float, default=0.03, help='soft histogram bin width for auxiliary KL loss')
parser.add_argument('--amp', type=str2bool, default=True, help='[fast] use CUDA AMP (fp16) to speed up train/val')
parser.add_argument('--tf32', type=str2bool, default=True, help='[fast] allow TF32 matmul/conv + cudnn.benchmark on CUDA')
parser.add_argument('--kl_max_px', type=int, default=50000, help='[fast] max valid px/channel sampled for KL histogram (0=all)')
parser.add_argument('--data_prefix', type=str, default='trn2', help='[v2] npy prefix: trn2 (v2 data) or trn (v1 data)')
parser.add_argument('--crop', type=int, default=256, help='[v2] random-crop size for training patches (0=full image)')
parser.add_argument('--augment', type=str2bool, default=True, help='[v2] random flip/rot90 augmentation on training patches')
parser.add_argument('--val_frac', type=float, default=0.2, help='[v2] fraction of SCENES held out for validation')
parser.add_argument('--lr_patience', type=int, default=5, help='[v2] ReduceLROnPlateau patience (epochs)')
parser.add_argument('--lr_factor', type=float, default=0.5, help='[v2] ReduceLROnPlateau factor')
parser.add_argument('--mndwi_weight', type=float, default=1.0, help='[v3] X->Y L1 loss weight for MNDWI channel (1.0=off)')
parser.add_argument('--dice_weight', type=float, default=0.0, help='[v3] weight of soft water-mask Dice loss on predicted MNDWI (0=off)')
parser.add_argument('--dice_temp', type=float, default=0.05, help='[v3] sigmoid temperature for soft water mask (normalized units)')
parser.add_argument('--components', type=str, default='',
                    help='[v17] comma-separated dataset components to KEEP from the --data_prefix array '
                         '(e.g. "sen12,s1f11,c2s,fp,kuros2"). Empty = use the whole array (legacy behaviour). '
                         'Requires processed/{prefix}_component.csv with columns idx,component,scene.')
parser.add_argument('--split_keep', type=str, default='',
                    help='[v18] keep only rows whose split column in processed/{prefix}_split.csv '
                         'equals this value (e.g. train). Empty = no filtering. Prevents the evaluation held-out from leaking in.')
parser.add_argument('--drop_low_water', type=float, default=0.0,
                    help='[v18] drop chips whose water_frac_measured in {prefix}_chip_meta.csv '
                         'is below this value (0=off). For the contrast experiment on whether low-water chips '
                         'act as negatives or merely dilute. E.g. 0.005')
parser.add_argument('--subsample', type=int, default=0,
                    help='[v18] use only N randomly chosen chips after selection (0=all). '
                         'For random-removal / subset size curves.')
parser.add_argument('--subsample_seed', type=int, default=0,
                    help='[v18] sampling seed for --subsample (independent of the training seed)')
parser.add_argument('--total_steps', type=int, default=0,
                    help='[v18] fix the number of gradient updates (0=use --num_epochs as is). '
                         'Equalizes compute across runs with different N. num_epochs is derived automatically.')
parser.add_argument('--ckpt_every', type=int, default=10,
                    help='[v17] save model_epoch_NNN.pt every N epochs (0=best_model.pt only, for ablations)')
parser.add_argument('--tst_prefix', type=str, default='',
                    help='[v17] test-set prefix override; empty = data_prefix with trn->tst')

if __name__ == "__main__":
    config_run, _ = parser.parse_known_args()
else:
    config_run = parser.parse_args([])
config = config_run

root_dir = os.path.abspath(os.path.expanduser(config.rootpath))
data_dir = os.path.join(root_dir, 'processed')
out_dir = os.path.join(root_dir, 'output')
outpath_model = os.path.join(out_dir, config.date_out)
outpath_vis = os.path.join(outpath_model, 'validation_figures')
config.datapath = data_dir
config.outpath = out_dir
os.makedirs(outpath_model, exist_ok=True)
os.makedirs(outpath_vis, exist_ok=True)
save_config(config, outpath_model)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # uncomment to pin a specific GPU

# pick the device automatically
if torch.cuda.is_available():
    # NVIDIA GPU on Windows/Linux
    device = torch.device(f"cuda:{config.gpu}")
    print('Device: CUDA')
    print('Count of using GPUs:', torch.cuda.device_count())
    # print('Current cuda device:', torch.cuda.current_device())

elif torch.backends.mps.is_available():
    # MPS on Mac (Apple Silicon)
    device = torch.device("mps")
    print('Device: MPS (Apple Silicon)')

else:
    # fall back to CPU when no GPU is available
    device = torch.device("cpu")
    print('Device: CPU')

print(f'Selected Device: {device}')

random.seed(config.seed)
np.random.seed(config.seed)
np.random.randint(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

# ----- speed knobs (numeric precision / kernel selection only; training semantics unchanged) -----
USE_AMP = bool(config.amp) and (device.type == "cuda")
if config.tf32 and device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"[fast2] AMP={USE_AMP} TF32/cudnn={bool(config.tf32) and device.type=='cuda'} "
      f"data={config.data_prefix} crop={config.crop} augment={config.augment} "
      f"kl_max_px={config.kl_max_px}", flush=True)

import pandas as _pd
_p = config.data_prefix                       # 'trn2' (v2) or 'trn' (v1)
_pt = config.tst_prefix or _p.replace('trn', 'tst')
_comp = [c.strip() for c in str(config.components).split(',') if c.strip()]

if _comp:
    # [v17] ablation: pull only the rows of the requested components from the concatenated array into RAM.
    # No per-component npy copies are made, so disk/transfer cost is zero.
    _ci = _pd.read_csv(os.path.join(data_dir, f'{_p}_component.csv'))
    _known = sorted(_ci['component'].unique().tolist())
    _bad = [c for c in _comp if c not in _known]
    assert not _bad, f"unknown component(s) {_bad}; available: {_known}"
    _Xm = np.load(os.path.join(data_dir, f'{_p}_X_VV_VH_RVI_raw.npy'), mmap_mode='r')
    _Ym = np.load(os.path.join(data_dir, f'{_p}_Y_MNDWI_NDVI_NDWI_raw.npy'), mmap_mode='r')
    assert len(_ci) == _Xm.shape[0], f"component CSV {len(_ci)} != X N {_Xm.shape[0]}"
    _keep_mask = _ci['component'].isin(_comp).to_numpy()

    # [v18] drop the evaluation held-out / leak sources. The split file is assumed row-aligned with the component CSV.
    if config.split_keep:
        _sp = _pd.read_csv(os.path.join(data_dir, f'{_p}_split.csv'))
        assert len(_sp) == len(_ci), f"split CSV {len(_sp)} != component CSV {len(_ci)}"
        _ok = (_sp['split'] == config.split_keep).to_numpy()
        _drop = int((_keep_mask & ~_ok).sum())
        _keep_mask &= _ok
        print(f"[fast2/v18] split_keep='{config.split_keep}': {_drop} chips dropped", flush=True)

    # [v18] drop low-water chips (contrast: do they act as negatives or merely dilute?)
    if config.drop_low_water > 0:
        _cmeta = _pd.read_csv(os.path.join(data_dir, f'{_p}_chip_meta.csv'))
        assert len(_cmeta) == len(_ci), \
            f"chip_meta {len(_cmeta)} != component CSV {len(_ci)}"
        _wf = _cmeta['water_frac_measured'].to_numpy()
        _lowm = _wf < config.drop_low_water
        _n = int((_keep_mask & _lowm).sum())
        _keep_mask &= ~_lowm
        print(f"[fast2/v18] drop_low_water<{config.drop_low_water}: {_n} chips dropped",
              flush=True)

    _sel = np.flatnonzero(_keep_mask)

    # [v18] random subsample for the size curve
    if config.subsample and config.subsample < len(_sel):
        _rng = np.random.default_rng(config.subsample_seed)
        _sel = np.sort(_rng.choice(_sel, size=config.subsample, replace=False))
        print(f"[fast2/v18] subsample: {config.subsample} chips "
              f"(seed={config.subsample_seed})", flush=True)
    X_trn = np.empty((len(_sel),) + _Xm.shape[1:], dtype=np.float32)
    Y_trn = np.empty((len(_sel),) + _Ym.shape[1:], dtype=np.float32)
    for _s in range(0, len(_sel), 64):                       # chunked copy, page-cache friendly
        _e = min(_s + 64, len(_sel))
        _ix = _sel[_s:_e]
        X_trn[_s:_e] = _Xm[_ix]
        Y_trn[_s:_e] = _Ym[_ix]
    del _Xm, _Ym
    scene_ids = _ci['scene'].to_numpy()[_sel]                # scene is "<component>/<folder>"
    _n_by_c = _ci.iloc[_sel]['component'].value_counts().to_dict()
    print(f"[fast2/v17] components={','.join(_comp)} -> {len(_sel)} chips {_n_by_c}", flush=True)
else:
    X_trn = np.load(os.path.join(data_dir, f'{_p}_X_VV_VH_RVI_raw.npy'))
    Y_trn = np.load(os.path.join(data_dir, f'{_p}_Y_MNDWI_NDVI_NDWI_raw.npy'))
    # per-sample scene ID (folder column of the good CSV) — used for the scene-level split
    _df_good = _pd.read_csv(os.path.join(data_dir, f'{_p}_matched_samples_good.csv'))
    assert len(_df_good) == X_trn.shape[0], \
        f"good CSV rows ({len(_df_good)}) != X_trn N ({X_trn.shape[0]})"
    scene_ids = _df_good['folder'].map(os.path.basename).to_numpy()

X_tst = np.load(os.path.join(data_dir, f'{_pt}_X_VV_VH_RVI_raw.npy'))
Y_tst = np.load(os.path.join(data_dir, f'{_pt}_Y_MNDWI_NDVI_NDWI_raw.npy'))
print(f"[fast2] loaded X_trn={X_trn.shape} from prefix '{_p}', "
      f"{len(set(scene_ids))} unique scenes", flush=True)

# [v2.1] degenerate-pair removal: pixels outside the S1 footprint are 0-filled (not NaN),
# so pairs with an essentially empty SAR (almost no VV>0) or a fully clouded optical (Y almost all NaN) can slip in.
# They contaminate the Y->X direction in particular, so drop them at load time.
_x_valid = np.array([float(np.mean((X_trn[i, 0] > 0) & np.isfinite(X_trn[i, 0])))
                     for i in range(X_trn.shape[0])])
_y_valid = np.array([float(np.mean(np.isfinite(Y_trn[i, 0])))
                     for i in range(Y_trn.shape[0])])
_keep = (_x_valid >= 0.2) & (_y_valid >= 0.2)
print(f"[fast2] degenerate filter: keep {int(_keep.sum())}/{len(_keep)} "
      f"(drop: SAR-empty={int((_x_valid < 0.2).sum())}, optical-empty={int((_y_valid < 0.2).sum())})",
      flush=True)
X_trn = X_trn[_keep]
Y_trn = Y_trn[_keep]
scene_ids = scene_ids[_keep]

def mask_outside_nsigma(arr, mean, std, nsigma=3.0, copy=True):
    """
    arr shape: (N, C, H, W)
    mean, std shape: (C,)
    
    Set values outside mean ± nsigma*std to NaN, per channel.
    """
    if copy:
        arr = arr.copy()

    mean = np.asarray(mean).reshape(1, -1, 1, 1)
    std = np.asarray(std).reshape(1, -1, 1, 1)

    lower = mean - nsigma * std
    upper = mean + nsigma * std

    bad = (arr < lower) | (arr > upper)
    arr[bad] = np.nan

    return arr

def get_optimizer(name):
    optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
    }
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}. Choose from {list(optimizers)}")
    return optimizers[name]

def get_masked_loss(name):
    losses = {
        "L1": nn.L1Loss,
        "MAE": nn.L1Loss,
        "MSE": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }
    if name not in losses:
        raise ValueError(f"Unsupported loss: {name}. Choose from {list(losses)}")

    base_loss = losses[name](reduction="none")

    def masked_loss(pred, target, mask, channel_weights=None):
        loss = base_loss(pred, target)
        if channel_weights is not None:
            # [v3] per-channel weights (e.g. up-weight MNDWI to allocate more capacity to water information)
            loss = loss * channel_weights.to(loss.device).view(1, -1, 1, 1)
        mask = mask.to(dtype=torch.bool, device=loss.device)
        if not torch.any(mask):
            return pred.sum() * 0.0
        return loss[mask].mean()

    return masked_loss

def soft_water_dice(pred, target, mask, t0, temp=0.05, eps=1e-6):
    """
    [v3] Soft Dice loss aligning the water mask of the predicted MNDWI (ch0) with the GT water mask.

    L1 averages uniformly over all pixels, so on ambiguous pixels the prediction is
    biased toward the majority class (land = low MNDWI) (observed as Precision 0.97 / Recall 0.55 in evaluation).
    Dice is a differentiable version of "overlap of the water masks thresholded at t0"
    and is invariant to class imbalance. Adding it directly to the loss aims to
    recover Recall without threshold calibration.
    """
    m = mask[:, 0].to(dtype=torch.bool, device=pred.device)
    if not torch.any(m):
        return pred.sum() * 0.0
    p = torch.sigmoid((pred[:, 0] - t0) / temp)[m]      # water-likeness (0-1)
    g = (target[:, 0] > t0).float()[m]                  # GT water mask
    inter = (p * g).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + g.sum() + eps)

def soft_histogram_kl_loss(pred, target, mask, bins=32, sigma=0.03, value_range=(0.0, 1.0), eps=1e-6, max_px=50000):
    """
    Auxiliary KL loss matching the per-channel soft histogram distributions.
    pred/target are assumed to be normalized 0-1 values.
    """
    mask = mask.to(dtype=torch.bool, device=pred.device)
    bin_centers = torch.linspace(
        value_range[0],
        value_range[1],
        bins,
        device=pred.device,
        dtype=pred.dtype,
    )

    kl_losses = []
    for ch in range(pred.shape[1]):
        ch_mask = mask[:, ch]
        if not torch.any(ch_mask):
            continue

        pv = pred[:, ch][ch_mask]
        tv = target[:, ch][ch_mask]
        if max_px and max_px > 0 and pv.numel() > max_px:
            sel = torch.randperm(pv.numel(), device=pv.device)[:max_px]
            pv = pv[sel]; tv = tv[sel]
        pred_vals = pv.reshape(-1, 1).clamp(value_range[0], value_range[1])
        target_vals = tv.reshape(-1, 1).clamp(value_range[0], value_range[1])

        pred_weights = torch.exp(-0.5 * ((pred_vals - bin_centers) / sigma) ** 2)
        target_weights = torch.exp(-0.5 * ((target_vals - bin_centers) / sigma) ** 2)

        pred_hist = pred_weights.sum(dim=0)
        target_hist = target_weights.sum(dim=0)

        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)

        kl_losses.append(
            F.kl_div(
                torch.log(pred_prob + eps),
                target_prob,
                reduction="sum",
            )
        )

    if len(kl_losses) == 0:
        return pred.sum() * 0.0
    return torch.stack(kl_losses).mean()

def plot_validation_grid(X, Y_true, Y_pred, X_pred, X_mask, Y_mask, save_path, epoch, sample_idx=0):
    X_true = X.detach().cpu().numpy()
    Y_true = Y_true.detach().cpu().numpy()
    Y_pred = Y_pred.detach().cpu().numpy()
    X_pred = X_pred.detach().cpu().numpy()
    X_mask = X_mask.detach().cpu().numpy().astype(bool)
    Y_mask = Y_mask.detach().cpu().numpy().astype(bool)

    sample_idx = min(sample_idx, X_true.shape[0] - 1)
    X_true_sample = X_true[sample_idx].copy()
    X_pred_sample = X_pred[sample_idx].copy()
    Y_true_sample = Y_true[sample_idx].copy()
    Y_pred_sample = Y_pred[sample_idx].copy()
    X_mask_sample = X_mask[sample_idx]
    Y_mask_sample = Y_mask[sample_idx]

    X_true_sample[~X_mask_sample] = np.nan
    X_pred_sample[~X_mask_sample] = np.nan
    Y_true_sample[~Y_mask_sample] = np.nan
    Y_pred_sample[~Y_mask_sample] = np.nan

    rows = [
        ("X_true", X_true_sample, ["VV", "VH", "RVI"]),
        ("X_pred (Y->X)", X_pred_sample, ["VV", "VH", "RVI"]),
        ("Y_true", Y_true_sample, ["MNDWI", "NDVI", "NDWI"]),
        ("Y_pred (X->Y)", Y_pred_sample, ["MNDWI", "NDVI", "NDWI"]),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(11, 12), dpi=180, constrained_layout=True)
    for row_idx, (row_name, arr, channel_names) in enumerate(rows):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr[col_idx], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{row_name}: {channel_names[col_idx]}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"Bidirectional validation sample - epoch {epoch + 1:03d}", fontsize=12)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_loss_history(history, save_dir):
    csv_path = os.path.join(save_dir, "loss_history.csv")
    json_path = os.path.join(save_dir, "loss_history.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_xy_loss",
                "train_yx_loss",
                "train_kl_loss",
                "val_loss",
                "val_xy_loss",
                "val_yx_loss",
                "val_kl_loss",
                "best_val_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def linear_to_db_safe(sar):
    sar = sar.astype(np.float32, copy=True)
    out = np.full_like(sar, np.nan, dtype=np.float32)
    valid = np.isfinite(sar) & (sar > 0)
    out[valid] = 10.0 * np.log10(sar[valid])
    return out

def convert_sar_channels_to_db(arr, channels=(0, 1)):
    arr = arr.copy()
    for ch in channels:
        ch_data = arr[:, ch]
        ch_median = np.nanmedian(ch_data)

        # if already in dB the median is usually negative; convert to dB only for linear backscatter.
        if np.isfinite(ch_median) and ch_median > 0:
            arr[:, ch] = linear_to_db_safe(ch_data)
        else:
            arr[:, ch] = ch_data
    return arr

X_trn = convert_sar_channels_to_db(X_trn)
X_tst = convert_sar_channels_to_db(X_tst)

# ==============================================================================
# [v2] scene-level split (done before computing statistics, to avoid leaking into val)
# ==============================================================================
_rng = np.random.default_rng(config.seed)
_scenes = np.array(sorted(set(scene_ids)))
_rng.shuffle(_scenes)
_n_val_scenes = max(1, int(len(_scenes) * config.val_frac))
val_scenes = set(_scenes[:_n_val_scenes])
trn_scenes = set(_scenes[_n_val_scenes:])
idx_trn = np.flatnonzero(np.array([s in trn_scenes for s in scene_ids]))
idx_val = np.flatnonzero(np.array([s in val_scenes for s in scene_ids]))
print(f"[fast2] scene split: {len(trn_scenes)} train scenes ({len(idx_trn)} imgs) / "
      f"{len(val_scenes)} val scenes ({len(idx_val)} imgs)", flush=True)
with open(os.path.join(outpath_model, "scene_split.json"), "w") as _f:
    json.dump({"train_scenes": sorted(trn_scenes), "val_scenes": sorted(val_scenes),
               "n_train_imgs": int(len(idx_trn)), "n_val_imgs": int(len(idx_val))}, _f, indent=2)

# train statistics — [v2] computed from training scenes only
X_trn_mean = np.nanmean(X_trn[idx_trn], axis=(0, 2, 3))
X_trn_std  = np.nanstd(X_trn[idx_trn], axis=(0, 2, 3))
Y_trn_mean = np.nanmean(Y_trn[idx_trn], axis=(0, 2, 3))
Y_trn_std  = np.nanstd(Y_trn[idx_trn], axis=(0, 2, 3))

# 5-sigma masking (in-place to save RAM)
X_trn = mask_outside_nsigma(X_trn, X_trn_mean, X_trn_std, nsigma=5, copy=False)
X_tst = mask_outside_nsigma(X_tst, X_trn_mean, X_trn_std, nsigma=5, copy=False)
Y_trn = mask_outside_nsigma(Y_trn, Y_trn_mean, Y_trn_std, nsigma=5, copy=False)
Y_tst = mask_outside_nsigma(Y_tst, Y_trn_mean, Y_trn_std, nsigma=5, copy=False)

X_trn_min = np.nanpercentile(X_trn[idx_trn], 1, axis=(0, 2, 3))
X_trn_max = np.nanpercentile(X_trn[idx_trn], 99, axis=(0, 2, 3))
Y_trn_min = np.nanpercentile(Y_trn[idx_trn], 1, axis=(0, 2, 3))
Y_trn_max = np.nanpercentile(Y_trn[idx_trn], 99, axis=(0, 2, 3))

def _normalize_inplace(arr, vmin, vmax):
    """Apply (arr-min)/range -> clip[0,1] per channel, staying in float32."""
    for c in range(arr.shape[1]):
        rng_c = vmax[c] - vmin[c] if vmax[c] != vmin[c] else 1.0
        a = arr[:, c]
        a -= np.float32(vmin[c])
        a /= np.float32(rng_c)
        np.clip(a, 0.0, 1.0, out=a)
    return arr

X_trn = _normalize_inplace(X_trn, X_trn_min, X_trn_max)
X_tst = _normalize_inplace(X_tst, X_trn_min, X_trn_max)
Y_trn = _normalize_inplace(Y_trn, Y_trn_min, Y_trn_max)
Y_tst = _normalize_inplace(Y_tst, Y_trn_min, Y_trn_max)

# [v3] water threshold (normalized version of raw MNDWI>0) — alignment target of the Dice loss
T0_WATER = float((0.0 - Y_trn_min[0]) / (Y_trn_max[0] - Y_trn_min[0]))
print(f"[fast2] water threshold (raw MNDWI>0): normalized t0={T0_WATER:.4f} | "
      f"mndwi_weight={config.mndwi_weight} dice_weight={config.dice_weight}", flush=True)

def make_bidirectional_tensors(X, Y):
    X_mask = np.isfinite(X)
    Y_mask = np.isfinite(Y)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(X_mask),
        torch.from_numpy(Y_mask),
    )

X_all_tensor, Y_all_tensor, MX_all_tensor, MY_all_tensor = make_bidirectional_tensors(X_trn, Y_trn)

# ==============================================================================
# [v2] Dataset: train = random crop + flip/rot90 augmentation, val = full image, no augmentation
# ==============================================================================
class PairPatchDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, MX, MY, indices, crop=0, augment=False):
        self.X, self.Y, self.MX, self.MY = X, Y, MX, MY
        self.indices = np.asarray(indices)
        self.crop = int(crop)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        x, y = self.X[j], self.Y[j]
        mx, my = self.MX[j], self.MY[j]
        H = x.shape[-1]
        if self.crop and self.crop < H:
            c = self.crop
            top = torch.randint(0, H - c + 1, (1,)).item()
            left = torch.randint(0, H - c + 1, (1,)).item()
            sl = (slice(None), slice(top, top + c), slice(left, left + c))
            x, y, mx, my = x[sl], y[sl], mx[sl], my[sl]
        if self.augment:
            if torch.rand(1).item() < 0.5:   # horizontal flip
                x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
                mx = torch.flip(mx, [-1]); my = torch.flip(my, [-1])
            if torch.rand(1).item() < 0.5:   # vertical flip
                x = torch.flip(x, [-2]); y = torch.flip(y, [-2])
                mx = torch.flip(mx, [-2]); my = torch.flip(my, [-2])
            k = torch.randint(0, 4, (1,)).item()   # rot90
            if k:
                x = torch.rot90(x, k, [-2, -1]); y = torch.rot90(y, k, [-2, -1])
                mx = torch.rot90(mx, k, [-2, -1]); my = torch.rot90(my, k, [-2, -1])
        return x.contiguous(), y.contiguous(), mx.contiguous(), my.contiguous()

dataset_trn = PairPatchDataset(
    X_all_tensor, Y_all_tensor, MX_all_tensor, MY_all_tensor,
    idx_trn, crop=config.crop, augment=config.augment,
)
dataset_val = PairPatchDataset(
    X_all_tensor, Y_all_tensor, MX_all_tensor, MY_all_tensor,
    idx_val, crop=0, augment=False,
)

loader_trn = DataLoader(
    dataset_trn,
    batch_size=config.bs,
    shuffle=config.shuffle,
    num_workers=0,
    pin_memory=device.type == "cuda"
)

loader_val = DataLoader(
    dataset_val,
    batch_size=config.bs,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda"
)
# ==============================================================================
# Model
# ==============================================================================
model_xy = UNet(n_channels=3, n_classes=3, bilinear=True).to(device)
model_yx = UNet(n_channels=3, n_classes=3, bilinear=True).to(device)

optimizer_ = get_optimizer(config.optimizer)
optimizer = optimizer_(
    list(model_xy.parameters()) + list(model_yx.parameters()),
    lr=config.lr,
)
criterion = get_masked_loss(config.loss)
# [v3] channel weights for Y=[MNDWI,NDVI,NDWI] (applied to the X->Y direction only)
Y_CH_W = None
if config.mndwi_weight != 1.0:
    Y_CH_W = torch.tensor([config.mndwi_weight, 1.0, 1.0], dtype=torch.float32, device=device)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience,
)

# [v18] equalize the number of gradient updates. With a fixed epoch count, runs with larger N get
# more updates, confounding "better data" with "more compute". Mandatory for ablations.
if config.total_steps > 0:
    _spe = max(1, len(loader_trn))
    _ne = max(1, int(np.ceil(config.total_steps / _spe)))
    print(f"[fast2/v18] total_steps={config.total_steps}: "
          f"{_spe} steps/epoch -> num_epochs {config.num_epochs} -> {_ne} "
          f"(actual steps {_spe * _ne})", flush=True)
    config.num_epochs = _ne

loss_arr = []
loss_arr_val = []
loss_history = []
best_val_loss = np.inf

# plt.imshow(Y_val_tensor[0, 1]);plt.colorbar()

for epoch in range(config.num_epochs):
    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    model_xy.train()
    model_yx.train()
    trn_losses = []
    trn_xy_losses = []
    trn_yx_losses = []
    trn_kl_losses = []
    trn_dice_losses = []

    for Xb, Yb, MXb, MYb in tqdm(loader_trn, desc=f"Epoch {epoch+1}/{config.num_epochs} train"):
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        MXb = MXb.to(device)
        MYb = MYb.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            pred_y = model_xy(Xb)
            pred_x = model_yx(Yb)
        pred_y = pred_y.float()
        pred_x = pred_x.float()
        loss_xy = criterion(pred_y, Yb, MYb, channel_weights=Y_CH_W)
        loss_yx = criterion(pred_x, Xb, MXb)
        kl_xy = soft_histogram_kl_loss(pred_y, Yb, MYb, bins=config.kl_bins, sigma=config.kl_sigma, max_px=config.kl_max_px)
        kl_yx = soft_histogram_kl_loss(pred_x, Xb, MXb, bins=config.kl_bins, sigma=config.kl_sigma, max_px=config.kl_max_px)
        kl_loss = kl_xy + kl_yx
        loss = loss_xy + loss_yx + config.kl_weight * kl_loss
        if config.dice_weight > 0:
            dice = soft_water_dice(pred_y, Yb, MYb, T0_WATER, temp=config.dice_temp)
            loss = loss + config.dice_weight * dice
            trn_dice_losses.append(dice.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        trn_losses.append(loss.item())
        trn_xy_losses.append(loss_xy.item())
        trn_yx_losses.append(loss_yx.item())
        trn_kl_losses.append(kl_loss.item())

    trn_loss = np.mean(trn_losses)
    trn_xy_loss = np.mean(trn_xy_losses)
    trn_yx_loss = np.mean(trn_yx_losses)
    trn_kl_loss = np.mean(trn_kl_losses)
    loss_arr.append(trn_loss)

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    model_xy.eval()
    model_yx.eval()
    val_losses = []
    val_xy_losses = []
    val_yx_losses = []
    val_kl_losses = []
    val_dice_losses = []
    val_vis_batch = None

    with torch.no_grad():
        for Xb, Yb, MXb, MYb in tqdm(loader_val, desc=f"Epoch {epoch+1}/{config.num_epochs} val"):
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            MXb = MXb.to(device)
            MYb = MYb.to(device)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred_y = model_xy(Xb)
                pred_x = model_yx(Yb)
            pred_y = pred_y.float()
            pred_x = pred_x.float()
            loss_xy = criterion(pred_y, Yb, MYb, channel_weights=Y_CH_W)
            loss_yx = criterion(pred_x, Xb, MXb)
            kl_xy = soft_histogram_kl_loss(pred_y, Yb, MYb, bins=config.kl_bins, sigma=config.kl_sigma, max_px=config.kl_max_px)
            kl_yx = soft_histogram_kl_loss(pred_x, Xb, MXb, bins=config.kl_bins, sigma=config.kl_sigma, max_px=config.kl_max_px)
            kl_loss = kl_xy + kl_yx
            loss = loss_xy + loss_yx + config.kl_weight * kl_loss
            if config.dice_weight > 0:
                dice = soft_water_dice(pred_y, Yb, MYb, T0_WATER, temp=config.dice_temp)
                loss = loss + config.dice_weight * dice
                val_dice_losses.append(dice.item())

            val_losses.append(loss.item())
            val_xy_losses.append(loss_xy.item())
            val_yx_losses.append(loss_yx.item())
            val_kl_losses.append(kl_loss.item())
            if val_vis_batch is None:
                # [v2.1] visualize the sample with the most valid Y pixels in the batch
                # (a fully clouded sample would give a blank figure)
                val_vis_sample_idx = int(torch.argmax(
                    MYb.float().mean(dim=(1, 2, 3))).item())
                val_vis_batch = (
                    Xb.detach().cpu(),
                    Yb.detach().cpu(),
                    pred_y.detach().cpu(),
                    pred_x.detach().cpu(),
                    MXb.detach().cpu(),
                    MYb.detach().cpu(),
                )

    val_loss = np.mean(val_losses)
    val_xy_loss = np.mean(val_xy_losses)
    val_yx_loss = np.mean(val_yx_losses)
    val_kl_loss = np.mean(val_kl_losses)
    loss_arr_val.append(val_loss)

    scheduler.step(val_loss)
    _cur_lr = optimizer.param_groups[0]['lr']
    if config.dice_weight > 0 and trn_dice_losses:
        print(f"  lr = {_cur_lr:.2e} | dice: train {np.mean(trn_dice_losses):.4f} "
              f"val {np.mean(val_dice_losses):.4f}", flush=True)
    else:
        print(f"  lr = {_cur_lr:.2e}", flush=True)

    print(
        f"Epoch {epoch+1:03d}/{config.num_epochs} | "
        f"train loss = {trn_loss:.5f} "
        f"(xy = {trn_xy_loss:.5f}, yx = {trn_yx_loss:.5f}, kl = {trn_kl_loss:.5f}) | "
        f"val loss = {val_loss:.5f} "
        f"(xy = {val_xy_loss:.5f}, yx = {val_yx_loss:.5f}, kl = {val_kl_loss:.5f})"
    )

    is_best = val_loss < best_val_loss
    current_best_val_loss = min(best_val_loss, val_loss)
    loss_history.append(
        {
            "epoch": epoch + 1,
            "train_loss": float(trn_loss),
            "train_xy_loss": float(trn_xy_loss),
            "train_yx_loss": float(trn_yx_loss),
            "train_kl_loss": float(trn_kl_loss),
            "val_loss": float(val_loss),
            "val_xy_loss": float(val_xy_loss),
            "val_yx_loss": float(val_yx_loss),
            "val_kl_loss": float(val_kl_loss),
            "best_val_loss": float(current_best_val_loss),
        }
    )
    save_loss_history(loss_history, outpath_model)

    # -------------------------------------------------
    # Save best model
    # -------------------------------------------------
    should_save_vis = (
        val_vis_batch is not None
        and config.vis_interval > 0
        and (epoch + 1) % config.vis_interval == 0
    )
    if should_save_vis:
        plot_validation_grid(
            *val_vis_batch,
            save_path=os.path.join(outpath_vis, f"val_epoch_{epoch+1:03d}.png"),
            epoch=epoch,
            sample_idx=val_vis_sample_idx,
        )

    if is_best:
        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_xy_state_dict": model_xy.state_dict(),
                "model_yx_state_dict": model_yx.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": trn_loss,
                "train_xy_loss": trn_xy_loss,
                "train_yx_loss": trn_yx_loss,
                "train_kl_loss": trn_kl_loss,
                "val_loss": val_loss,
                "val_xy_loss": val_xy_loss,
                "val_yx_loss": val_yx_loss,
                "val_kl_loss": val_kl_loss,
                "X_min": X_trn_min,
                "X_max": X_trn_max,
                "Y_min": Y_trn_min,
                "Y_max": Y_trn_max,
                "X_mean": X_trn_mean,
                "X_std": X_trn_std,
                "Y_mean": Y_trn_mean,
                "Y_std": Y_trn_std,
            },
            os.path.join(outpath_model, "best_model.pt")
        )
        if val_vis_batch is not None:
            plot_validation_grid(
                *val_vis_batch,
                save_path=os.path.join(outpath_vis, "val_best.png"),
                epoch=epoch,
                sample_idx=val_vis_sample_idx,
            )

    # optional: per-epoch checkpoint
    if config.ckpt_every > 0 and (epoch + 1) % config.ckpt_every == 0:
        torch.save(
            {
                "model_xy_state_dict": model_xy.state_dict(),
                "model_yx_state_dict": model_yx.state_dict(),
            },
            os.path.join(outpath_model, f"model_epoch_{epoch+1:03d}.pt")
        )
