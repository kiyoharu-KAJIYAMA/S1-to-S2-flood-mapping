"""Forward (SAR -> target) models for experiment 6. The default U-Net is the one
used by the cur-s* runs (RMSD_flood_eval.UNet, bilinear=True)."""
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd"),
                                "scripts/04_eval"))
from RMSD_flood_eval import UNet, DoubleConv, Down, Up, OutConv  # noqa: E402


class PixelNet(nn.Module):
    """1x1 convolutions only: no spatial context (receptive field 1 px)."""

    def __init__(self, n_in=3, n_out=3, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_in, width, 1), nn.ReLU6(inplace=True),
            nn.Conv2d(width, width, 1), nn.ReLU6(inplace=True),
            nn.Conv2d(width, n_out, 1))

    def forward(self, x):
        return self.net(x)


class ShallowNet(nn.Module):
    """Three 3x3 convolutions (receptive field 7x7 px = 70 m), no pooling."""

    def __init__(self, n_in=3, n_out=3, width=64):
        super().__init__()
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                    nn.ReLU6(inplace=True)]
        self.net = nn.Sequential(*blk(n_in, width), *blk(width, width), *blk(width, width),
                                 nn.Conv2d(width, n_out, 1))

    def forward(self, x):
        return self.net(x)


class UNet2(nn.Module):
    """U-Net with two down-sampling stages instead of four (same blocks, bilinear)."""

    def __init__(self, n_in=3, n_out=3):
        super().__init__()
        self.inc = DoubleConv(n_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up1 = Up(256, 64, True)
        self.up2 = Up(128, 64, True)
        self.outc = OutConv(64, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)


def make_xy_model(arch, n_in=3, n_out=3):
    if arch == "unet":
        return UNet(n_in, n_out, bilinear=True)
    if arch == "pixel":
        return PixelNet(n_in, n_out)
    if arch == "shallow":
        return ShallowNet(n_in, n_out)
    if arch == "unet2":
        return UNet2(n_in, n_out)
    raise ValueError(arch)


def receptive_field_px(arch):
    return {"unet": 140, "pixel": 1, "shallow": 7, "unet2": 44}.get(arch)
