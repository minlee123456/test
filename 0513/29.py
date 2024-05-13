import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __inti__(selfself, in_channels, out_channels, kernel_size=3):
        super(baicBlock, self).__init__()

        self.c1 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, in_channels,
                            kernel_size=kernel_size, padding=1)
        self.downsample = nn.Conv2d(in_channels, out_channels,
                            kernel_size=1)
