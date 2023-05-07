import torch
import torch.nn as nn

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
# # SE block add to U-My_Net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)


class SpatialAttentionModule(nn.Module):
    def __init__(self,in_dim, out_dim, drop_out=False):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2dReLU(
            in_dim,
            out_dim,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.dropout = drop_out

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        original_out = x
        out = out * original_out
        out = self.conv(out)

        return out


# a = torch.randn(16, 192, 56, 56)
# model = SpatialAttentionModule(in_dim = 192, out_dim=64)
# b = model(a)
# print(b.shape)




