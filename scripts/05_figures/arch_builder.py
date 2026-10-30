"""Shared helper that rebuilds the forward model (X->Y) from a checkpoint's "arch" config.

Old checkpoints (no arch key) remain UNet(3,3,bilinear=True).
Light runs from fast2 v19 onward assemble a LightAttentionUNet with the same config.
The reverse model g has the identical architecture, so use build_from_ckpt and
load model_yx_state_dict.
"""


def build_from_ckpt(ck, unet_cls):
    arch = ck.get("arch")
    if not arch or arch.get("model", "unet") == "unet":
        return unet_cls(3, 3, bilinear=True)
    if arch["model"] == "light-ens":
        return LightEnsemble(arch)
    from rmsd_light_model import LightAttentionUNet
    return LightAttentionUNet(
        n_channels=3, n_classes=3,
        base_channels=arch["base_channels"],
        use_depthwise=arch["use_depthwise"],
        use_residual=arch["use_residual"],
        use_eca=arch["use_eca"],
        use_skip_attention=arch["use_skip_attention"],
        bottleneck_dilation=arch["bottleneck_dilation"],
        normalization=arch["normalization"],
        activation=arch["activation"],
    )


class LightEnsemble(__import__("torch").nn.Module):
    """seed ensemble: absorb each member's normalization difference with an affine map, then average.

    Input z is normalized with the statistics of the reference run (member 1). Member i
    receives z_i = z * in_scale[i] + in_shift[i]; its output is mapped back into the
    reference normalized space with out_scale/out_shift before averaging. scale/shift are
    persistent buffers stored in the state_dict (values differ between xy and yx).
    """

    def __init__(self, arch):
        import torch
        from rmsd_light_model import LightAttentionUNet
        super().__init__()
        n = arch["n_members"]
        self.members = torch.nn.ModuleList([
            LightAttentionUNet(
                n_channels=3, n_classes=3,
                base_channels=arch["base_channels"],
                use_depthwise=arch["use_depthwise"],
                use_residual=arch["use_residual"],
                use_eca=arch["use_eca"],
                use_skip_attention=arch["use_skip_attention"],
                bottleneck_dilation=arch["bottleneck_dilation"],
                normalization=arch["normalization"],
                activation=arch["activation"],
            ) for _ in range(n)])
        self.register_buffer("in_scale", torch.ones(n, 3))
        self.register_buffer("in_shift", torch.zeros(n, 3))
        self.register_buffer("out_scale", torch.ones(n, 3))
        self.register_buffer("out_shift", torch.zeros(n, 3))

    def forward(self, z):
        import torch
        outs = []
        for i, m in enumerate(self.members):
            zi = (z * self.in_scale[i].view(1, 3, 1, 1)
                  + self.in_shift[i].view(1, 3, 1, 1)).clamp(0, 1)
            yi = m(zi)
            outs.append(yi * self.out_scale[i].view(1, 3, 1, 1)
                        + self.out_shift[i].view(1, 3, 1, 1))
        return torch.stack(outs).mean(0)
