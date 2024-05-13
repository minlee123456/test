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

        self.fn1 = nn.BatchNorm2d(num_features=out_channels)
        self.fn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU

    def forward(self, x):

        x_ = x

        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        x_ = self.downsample(x_)

        x +=x_
        x = self.relu(x)

        return x

