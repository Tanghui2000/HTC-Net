import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TMFblock(nn.Module):
    def __init__(self, in_channels):
        super(TMFblock, self).__init__()
        self.dilate1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, dilation=1, padding=0)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)
        self.se = SELayer(channel=in_channels)
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=3, dilation=1, kernel_size=3, padding=1)
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        branches_1= self.dilate1(x)
        branches_1= self.conv1x1(self.se(branches_1))
        branches_2 = self.dilate2(x)
        branches_2 = self.conv1x1(self.se(branches_2))
        branches_3 = self.dilate3(x)
        branches_3 = self.conv1x1(self.se(branches_3))
        out = branches_3 + branches_2 + branches_1

        out = self.relu(self.conv(out))

        att = self.conv3x3(out)
        att = F.softmax(att, dim=1)
        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        att_3 = att[:, 2, :, :].unsqueeze(1)
        fusion = att_1 * branches_1 + att_2 * branches_2 + att_3 * branches_3

        ax = self.relu(self.gamma * fusion + (1 - self.gamma) * x)
        ax = self.conv_last(ax)
        return ax
