import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """([Conv2D] => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block with maxpool"""

    def __init__(self, in_channels, out_channels, dropout=False, first=False):
        super().__init__()
        if first:
            dropout_rate = 0.25
        else:
            dropout_rate = 0.5
        if dropout:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Dropout(dropout_rate),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with bilinear upsample or ConvTranspose"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, dropout):
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        if dropout:
            x = self.dropout(x)
        return self.conv(x)


class OutConv(nn.Module):
    """ Network head """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """ Unet Network """
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, cut_last_convblock=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        self.cut_last_convblock = cut_last_convblock

        # Use bilinear upsampling or transposed convolutions
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, self.dropout, first=True)
        self.down2 = Down(128, 256, self.dropout)
        if not cut_last_convblock:
            self.down3 = Down(256, 512, self.dropout)
            self.down4 = Down(512, 1024 // factor, self.dropout)
            self.up1 = Up(1024, 512 // factor, bilinear)

        else:
            self.down3 = Down(256, 512 // factor, self.dropout)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if not self.cut_last_convblock:
            x5 = self.down4(x4)
            x = self.up1(x5, x4, self.dropout)
        else:
            x = x4
        x = self.up2(x, x3, self.dropout)
        x = self.up3(x, x2, self.dropout)
        x = self.up4(x, x1, self.dropout)
        logits = self.outc(x)
        return logits
