import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import interpolate


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 128)
        self.conv5 = DoubleConv(128, 64)
        self.conv6 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True))
        x3 = self.conv3(interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=True))
        x4 = self.conv4(interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True))
        x5 = self.conv5(interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True))
        out = self.conv6(x5)
        return out

class Stem(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Stem, self).__init__()
        self.stem_block = nn.Sequential(
            nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chs // 2),
            nn.ReLU(),
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # x is of shape (b, c, h, w, t)
        batch_size, channels, height, width, time_steps = x.shape
        outputs = []
        
        # Process each time step separately
        for t in range(time_steps):
            x_t = x[:, :, :, :, t]  # Extract the t-th time slice (shape: (b, c, h, w))
            out_t = self.stem_block(x_t)  # Apply the stem block to this time slice
            outputs.append(out_t.unsqueeze(-1))  # Add the time dimension back
        
        # Stack the outputs along the time dimension
        output = torch.cat(outputs, dim=-1)  # The output shape will be (b, out_chs, h, w, t)
        
        return output

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        outputs = []
        for t in range(time_steps):
            x_t = x[:, :, :, :, t]  # Extract the t-th time slice (shape: (b, c, h, w))
            out_t = self.network(x)
            outputs.append(out_t.unsqueeze(-1))

        output = torch.cat(outputs, dim=-1)
        return output
    