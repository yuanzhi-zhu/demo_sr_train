import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return x + out


class MSRResNet(nn.Module):
    """modified SRResNet"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_body = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(nf, nf * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hres = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv_first(x))
        x1 = self.conv_body(x1)

        if self.upscale == 4:
            x1 = self.lrelu(self.pixel_shuffle(self.upconv1(x1)))
            x1 = self.lrelu(self.pixel_shuffle(self.upconv2(x1)))
        elif self.upscale in [2, 3]:
            x1 = self.lrelu(self.pixel_shuffle(self.upconv1(x1)))

        x1 = self.conv_last(self.lrelu(self.conv_hres(x1)))
        x = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        return x1 + x
