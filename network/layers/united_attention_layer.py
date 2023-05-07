import torch.nn as nn
from network.layers.spatial_attention_layer import SpatialAttentionModule


# # SE block add to U-My_Net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)


class UAblock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(UAblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes <= 16:
            self.globalAvgPool = nn.AvgPool2d((224, 224), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((224, 224), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 112), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((112, 112), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((56, 56), stride=1)    # (56, 75) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((56, 56), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((28, 28), stride=1)    # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((28, 28), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((14, 14), stride=1)    # (14, 18) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((14, 14), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttentionModule(in_dim = inplanes, out_dim= planes)

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)
        # 并联
        y = self.sa(x)
        out = y + out
        # 串联
        # out = self.SA(out)
        return out


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



class UAblock_Swin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(UAblock_Swin, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes <= 16:
            self.globalAvgPool = nn.AvgPool2d((224, 224), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((224, 224), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 112), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((112, 112), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((56, 56), stride=1)    # (56, 75) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((56, 56), stride=1)
        elif planes == 192:
            self.globalAvgPool = nn.AvgPool2d((28, 28), stride=1)    # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((28, 28), stride=1)
        elif planes == 384:
            self.globalAvgPool = nn.AvgPool2d((14, 14), stride=1)    # (14, 18) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((14, 14), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttentionModule(in_dim = inplanes, out_dim= planes)
        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)
        # 并联
        y = self.sa(x)
        out = y + out
        # 串联
        # out = self.SA(out)
        return out
