from typing import Sequence
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """ (Conv 3x3 -> BatchNorm -> ReLU) * 2 + Optional Dropout2d"""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 1, 
        features: Sequence[int] = (64, 128, 256, 512),
        dropout_p: float = 0.2,
        name: str = "unet"
    ):
        super().__init__()
        self.name = name
        self.inc = DoubleConv(in_channels, features[0], dropout_p=0.0)
        self.down1 = DownBlock(features[0], features[1], dropout_p=0.0)
        self.down2 = DownBlock(features[1], features[2], dropout_p=0.0)
        self.down3 = DownBlock(features[2], features[3], dropout_p=dropout_p * 0.5)
        self.bottleneck = DownBlock(features[3], features[3] * 2, dropout_p=dropout_p)
        
        self.up1 = UpBlock(features[3] * 2, features[3])
        self.up2 = UpBlock(features[3], features[2])
        self.up3 = UpBlock(features[2], features[1])
        self.up4 = UpBlock(features[1], features[0])
        
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bottleneck(s4)
        
        d1 = self.up1(b, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)
        
        return self.outc(d4)
