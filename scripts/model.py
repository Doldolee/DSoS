"""Compact U-Net from channel IQ to joint SoS and attenuation maps."""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(1, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1),
                         nn.GroupNorm(1, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, base_channels=16, output_shape=(256, 128)):
        super().__init__()
        b = base_channels
        self.output_shape = tuple(output_shape)
        # Learned time compression keeps the full IQ input; it is a network
        # feature transform, not a claim that RF pixels are tissue coordinates.
        self.e0 = block(2, b, stride=(4, 1))
        self.e1, self.e2, self.e3 = block(b, b*2), block(b*2, b*4), block(b*4, b*8)
        self.d2, self.d1, self.d0 = block(b*12, b*4), block(b*6, b*2), block(b*3, b)
        # Output channel order is fixed: SoS, then ATT.
        self.head = nn.Conv2d(b, 2, 1)

    def forward(self, x):
        a = self.e0(x)
        b = self.e1(F.avg_pool2d(a, 2))
        c = self.e2(F.avg_pool2d(b, 2))
        d = self.e3(F.avg_pool2d(c, 2))
        def join(v, skip):
            return torch.cat([F.interpolate(v, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)
        v = self.d2(join(d, c))
        v = self.d1(join(v, b))
        v = self.d0(join(v, a))
        return self.head(F.interpolate(v, size=self.output_shape, mode="bilinear", align_corners=False))
