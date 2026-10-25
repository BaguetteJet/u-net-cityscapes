import torch
import torch.nn.functional as F

NUM_CLASSES = 19    # number of output channels (19 Cityscapes classes)
BASE = 32           # initial number of feature channels, doubles after each downsampling step

# U-Net implementation based on kerrgarr's model
# Source: https://github.com/kerrgarr/SemanticSegmentationCityscapes/blob/main/seg_code/models.py

# input = [B, 3, 512, 1024]      B = Batch Size, 3 = RGB channels, 512x1024 = image size
# output = [B, 19, 512, 1024]    B = Batch Size, 19 = number of classes, 512x1024 = image size


class DoubleConv(torch.nn.Module):
    """
    Two convolution layers, optionally with a parallel shortcut path
    Benefits of residual=True: network can preserve info, and can optimise gradient flow during backpropagation
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()

        # main convolution path to learn two features, almost identical to original implementation
        # exclude bias because it is redundant with GroupNorm after
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = torch.nn.GroupNorm(8, out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(8, out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        # optional shortcut path to carry information forward
        # applies a shortcut to convert the input shape to match output shape so that it can be added to the main conv path
        self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    torch.nn.GroupNorm(8, out_channels)
                )
            else:
                # if channels already match, return the input unchanged, no new parameters
                self.shortcut = torch.nn.Identity()

    # the forward method actually executes the layers, it's called automatically when you apply DoubleConv(x)
    def forward(self, x):
        # execute the layers defined above
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        # add the shortcut representation to the main conv path before the final relu, if residual is enabled
        if self.residual:
            out = out + self.shortcut(x)

        return self.relu(out)


class Down(torch.nn.Module):
    """Encoder downsampling with maxpool to reduce spatial dimensions then double conv to extract features"""

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),  # 2x2 max pooling to reduce spatial dimensions by half
            DoubleConv(in_channels, out_channels, residual) # two convolution layers, optionally residual
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Decoder upsampling with bilinear interpolation then double conv"""

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()

        # bilinear interpolation upsamples the spatial dimensions without learnable parameters
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # after concatenating the skip features, the conv input has in_channels + out_channels
        self.conv = DoubleConv(in_channels + out_channels, out_channels, residual)

    def forward(self, x1, x2):
        """x1 = low-res decoder features, x2 = high-res encoder features"""
        x1 = self.up(x1)    # upsampling

        # image tensor input has shape [Batch, Channels, Height, Width]
        # compute differences because pooling and upsampling don't always perfectly reverse each other
        # e.g. 255/2 = 127.5 -> 128, then 128*2 = 256
        diffY = x2.size()[2] - x1.size()[2]     # difference in height
        diffX = x2.size()[3] - x1.size()[3]     # difference in width

        # pad the upsampled decoder features to match the encoder features
        # order = [left, right, top, bottom]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # create skip connection along channel dimension (dim=1)
        # combines high-res encoder features with low-res decoder features
        x = torch.cat([x2, x1], dim=1)

        # feature fusion to learn combined encoder and decoder features
        return self.conv(x)


class OutConv(torch.nn.Module):
    """Final classification layer. NOTE: no softmax because the model outputs raw logits"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1x1 convolution to map the feature channels to the 19 output classes
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, base=BASE):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base = base
        # example input image size                           # [3, 512, 1024]

        # encoder - reducing spatial dimensions and increasing feature channels
        self.inc = DoubleConv(in_channels, base)             # [32, 512, 1024] RGB image to feature channels
        self.down1 = Down(base,     base * 2)                # [64, 256, 512]
        self.down2 = Down(base * 2, base * 4)                # [128, 128, 256]
        self.down3 = Down(base * 4, base * 8, residual=True) # [256, 64, 128]

        # bottleneck - deepest layer with the smallest spatial dimensions and most feature channels
        self.down4 = Down(base * 8, base * 16, residual=True) # [512, 32, 64]

        # decoder - increasing spatial dimensions and reducing feature channels
        self.up1 = Up(base * 16, base * 8, residual=True)    # [256, 64, 128]
        self.up2 = Up(base * 8,  base * 4)                   # [128, 128, 256]
        self.up3 = Up(base * 4,  base * 2)                   # [64, 256, 512]
        self.up4 = Up(base * 2,  base)                       # [32, 512, 1024]

        # output layer map feature channels to 19 classes
        self.out = OutConv(base, num_classes)                # [19, 512, 1024]

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # bottleneck
        b = self.down4(x4)

        # decoder with skip connections
        x = self.up1(b, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.out(x)