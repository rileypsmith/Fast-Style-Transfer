"""
Implementation of transformer net described in the paper found below:
https://arxiv.org/pdf/1603.08155.pdf%7C.

Per the license statement on Justin Johnson's GitHub, this network is free
to use for personal use, but you must contact Justin Johnson for commercial
use.

Implemented in PyTorch by: Riley Smith
Date: 12-12-2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Vanilla convolutional residual block from seminal paper by He et al.

    Use of instance normalization suggested by Ulyanov et al. in
    https://arxiv.org/pdf/1607.08022.pdf%C2%A0%C2%A0%C2%A0%C2%A0.
    """
    def __init__(self, filters=128):
        super(ResidualBlock, self).__init__()

        # Create convolutions and use instance normalization
        self.conv1 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.in_norm1 = nn.InstanceNorm2d(filters, affine=True)
        self.conv2 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.in_norm2 = nn.InstanceNorm2d(filters, affine=True)

    def forward(self, x):
        a = self.conv1(x)
        b = self.in_norm1(a)
        c = F.relu(b)
        d = self.conv2(c)
        e = self.in_norm2(d)
        return F.relu(e + x)

class ImageTransformationNet(nn.Module):
    """
    The image transformation network described in the paper by Johnson et al.,
    with instance normalization as suggested by Ulyanov et al.
    """
    def __init__(self, vangoh=False):
        super(ImageTransformationNet, self).__init__()

        # First layer has 9x9 filters
        self.conv1 = nn.Conv2d(3, 32, (9, 9), padding=(4, 4), padding_mode='reflect')
        self.in_norm1 = nn.InstanceNorm2d(32, affine=True)

        # Use two convolutions with stride 2 to downsample image
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1), padding_mode='reflect', stride=2)
        self.in_norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), padding_mode='reflect', stride=2)
        self.in_norm3 = nn.InstanceNorm2d(128, affine=True)

        # Use 5 residual blocks with 128 filters each
        self.block1 = ResidualBlock()
        self.block2 = ResidualBlock()
        self.block3 = ResidualBlock()
        self.block4 = ResidualBlock()
        self.block5 = ResidualBlock()

        # To reduce artifacts, upsample with nearest neighbor, then use convolution
        self.conv4 = nn.Conv2d(128, 64, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.in_norm4 = nn.InstanceNorm2d(64, affine=True)
        if vangoh:  # VanGoh net was trained with convtranspose instead of regular convolution
            self.conv5 = nn.ConvTranspose2d(64, 32, (3, 3), padding=(1, 1))
        else:
            self.conv5 = nn.Conv2d(64, 32, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.in_norm5 = nn.InstanceNorm2d(32, affine=True)

        # Upsampling operation is standard nearest neighbor upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Last layer uses 9x9 filters and reduces to 3 channels
        self.conv6 = nn.Conv2d(32, 3, (9, 9), padding=(4, 4), padding_mode='reflect')

    def forward(self, x):
        x = self.conv1(x)
        x = self.in_norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.in_norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.in_norm3(x)
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.upsample(x)
        x = self.conv4(x)
        x = self.in_norm4(x)
        x = F.relu(x)

        x = self.upsample(x)
        x = self.conv5(x)
        x = self.in_norm5(x)
        x = F.relu(x)

        x = self.conv6(x)

        # Skip tanh because it is unclear how to scale it (noted by gordicaleksa)
        return x
