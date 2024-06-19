import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaUNet2D(nn.Module):
    def __init__(self, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH):
        super(VanillaUNet2D, self).__init__()
        self.IMG_CHANNELS = IMG_CHANNELS
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH

        # Contraction path (encoder)
        self.down_conv_1 = self.conv_block(IMG_CHANNELS, 16, dropout=0.1)
        self.down_conv_2 = self.conv_block(16, 32, dropout=0.1)
        self.down_conv_3 = self.conv_block(32, 64, dropout=0.2)
        self.down_conv_4 = self.conv_block(64, 128, dropout=0.2)
        self.bottom_conv = self.conv_block(128, 256, dropout=0.3)

        # Expansive path (decoder)
        self.up_trans_1 = self.up_trans_block(256, 128)
        self.up_conv_1 = self.conv_block(256, 128, dropout=0.2)
        self.up_trans_2 = self.up_trans_block(128, 64)
        self.up_conv_2 = self.conv_block(128, 64, dropout=0.2)
        self.up_trans_3 = self.up_trans_block(64, 32)
        self.up_conv_3 = self.conv_block(64, 32, dropout=0.1)
        self.up_trans_4 = self.up_trans_block(32, 16)
        self.up_conv_4 = self.conv_block(32, 16, dropout=0.1)

        self.out = nn.Conv2d(16, 1, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.xavier_uniform_()
            if module.bias is not None:
                module.bias.data.zero_()

    def conv_block(self, in_channels, out_channels, dropout=0.1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )
        return block

    def up_trans_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Encoder
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_2(F.max_pool2d(x1, 2))
        x3 = self.down_conv_3(F.max_pool2d(x2, 2))
        x4 = self.down_conv_4(F.max_pool2d(x3, 2))
        x5 = self.bottom_conv(F.max_pool2d(x4, 2))

        # Decoder
        x = self.up_trans_1(x5)
        x = self.up_conv_1(torch.cat([x, x4], 1))
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x3], 1))
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x2], 1))
        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.out(x)

        return x
