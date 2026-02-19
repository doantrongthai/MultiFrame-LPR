"""Standalone RRDBNet architecture (Real-ESRGAN generator).

Ported from basicsr/xinntao without any basicsr dependency.
num_block=23, num_feat=64, num_grow_ch=32 matches RealESRGAN_x4plus.pth pretrain.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import urllib.request

PRETRAIN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Pixel unshuffle: reduces spatial size, enlarges channel size.
    Used for scale=1 and scale=2 inputs in RRDBNet.
    """
    b, c, h, w = x.size()
    out_channel = c * (scale ** 2)
    assert h % scale == 0 and w % scale == 0
    h_out, w_out = h // scale, w // scale
    x = x.view(b, c, h_out, scale, w_out, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, out_channel, h_out, w_out)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (5 conv layers with dense connections).
    No BatchNorm — empirically found to reduce artifacts in SR.
    """
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,               num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization: default kaiming, but scale down for stable training
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        # Scale residual by 0.2 for stable training (empirical from paper)
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    3 chained ResidualDenseBlock with a global residual connection.
    """
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Scale by 0.2 for stable training
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet generator used in Real-ESRGAN.

    Default config matches RealESRGAN_x4plus.pth pretrained weights:
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4

    Args:
        num_in_ch: Input channels (3 for RGB).
        num_out_ch: Output channels (3 for RGB).
        scale: Upscale factor. 4 for x4plus pretrain.
        num_feat: Intermediate feature channels. Default 64.
        num_block: Number of RRDB blocks. 23 for full Real-ESRGAN.
        num_grow_ch: Growth channels in RDB. Default 32.
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale

        # Pixel unshuffle for scale < 4 to match pretrain input channels
        in_ch = num_in_ch
        if scale == 2:
            in_ch = num_in_ch * 4   # pixel_unshuffle scale=2 → *4 channels
        elif scale == 1:
            in_ch = num_in_ch * 16  # pixel_unshuffle scale=4 → *16 channels

        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)

        # Trunk: 23 RRDB blocks
        self.body = nn.Sequential(*[
            RRDB(num_feat, num_grow_ch) for _ in range(num_block)
        ])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsample (x4 = two 2x nearest + conv)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pixel unshuffle for scale < 4
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # Global residual

        # x4 upsample: 2x → 2x
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return feat


def load_pretrained_rrdbnet(
    model: RRDBNet,
    pretrain_path: str,
    device: torch.device,
    strict: bool = True,
) -> RRDBNet:
    # Tự tải nếu chưa có
    if not os.path.exists(pretrain_path):
        os.makedirs(os.path.dirname(os.path.abspath(pretrain_path)), exist_ok=True)
        print(f"⬇️  Downloading Real-ESRGAN pretrain → {pretrain_path}")
        urllib.request.urlretrieve(PRETRAIN_URL, pretrain_path)
        print("✅ Download xong.")

    print(f"📦 Loading Real-ESRGAN pretrain: {pretrain_path}")
    ckpt = torch.load(pretrain_path, map_location=device)

    if 'params_ema' in ckpt:
        state_dict = ckpt['params_ema']
    elif 'params' in ckpt:
        state_dict = ckpt['params']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=strict)
    print(f"✅ Real-ESRGAN pretrain loaded ({len(state_dict)} keys)")
    return model