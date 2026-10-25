import torch
import torch.nn.functional as F

NUM_CLASSES = 19    # number of output channels (19 Cityscapes classes)
BASE = 32           # initial number of feature channels, doubles after each downsampling step (32, 64, 128, 256, 512)

# U-Net implementation based on kerrgarr's model
# Source: https://github.com/kerrgarr/SemanticSegmentationCityscapes/blob/main/seg_code/models.py

# input = [B, 3, 256, 256]      B = Batch Size, 3 = RGB channels, 256x256 = image size
# output = [B, 19, 256, 256]     B = Batch Size, 19 = number of classes, 256x256 = image size


class ResidualDoubleConv(torch.nn.Module):
    """
    Creates two parallel paths: main convolution path, shortcut path
    Benefits: network can preserve info, and can optimise gradient flow during backpropagation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # main convolution path to learn two features, almost identical to original implementation
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        # new shortcut path to carry information forward
        # applies a shortcut to convert the input shape to match output shape so that it can be added to the main conv path
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            # if channels already match, return the input unchanged, no new parameters
            self.shortcut = torch.nn.Identity()

    # the forward method actually executes the layers, it's called automatically when you apply ResidualDoubleConv(x)
    def forward(self, x):
        # create the shortcut representation: either x if channels match, or bn(conv(x)) if not
        identity = self.shortcut(x)

        # execute the layers defined at the top of the __init__ method
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # add together the output of the main conv path and the shortcut path, then apply relu
        out = self.relu(out + identity)

        return out


class Down(torch.nn.Module):
    """Encoder downsampling with maxpool to reduce spatial dimensions then double conv to extract features"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),  # 2x2 max pooling to reduce spatial dimensions by half
            ResidualDoubleConv(in_channels, out_channels) # two convolution layers with residual connections
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Decoder upsampling with transposed convolution or bilinear interpolation then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # transposed convolution learns the upsampling weights, has learnable parameters
        self.up = torch.nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualDoubleConv(in_channels, out_channels)

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

        # transforms input RGB image to 32 feature channels
        self.inc = ResidualDoubleConv(in_channels, base)    # [32, 256, 256]

        # encoder - reducing spatial dimensions and increasing feature channels
        self.down1 = Down(base,     base * 2)               # [64, 128, 128]
        self.down2 = Down(base * 2, base * 4)               # [128, 64, 64]
        self.down3 = Down(base * 4, base * 8)               # [256, 32, 32]
        self.down4 = Down(base * 8, (base * 16))            # [512, 16, 16]

        # decoder - increasing spatial dimensions and reducing feature channels
        # each concatenated to the corresponding Down encoding step via skip connections
        self.up1 = Up(base * 16, (base * 8))                # [256, 32, 32]
        self.up2 = Up(base * 8,  (base * 4))                # [128, 64, 64]
        self.up3 = Up(base * 4,  (base * 2))                # [64, 128, 128]
        self.up4 = Up(base * 2,  base)                      # [32, 256, 256]

        # final output layer (32 feature channels -> 19 classes)
        self.outc = OutConv(base, num_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)