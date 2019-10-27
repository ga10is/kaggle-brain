import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .. import config
from .resnet_cbam import resnet50_cbam


class ResNet(nn.Module):
    def __init__(self, dropout_rate):
        super(ResNet, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torchvision.models.resnet34(pretrained=True)
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        n_out_channels = 512  # resnet18, 34: 512, resnet50: 512*4

        # FC
        self.fc = nn.Linear(n_out_channels, 6)

    def forward(self, x, label=None):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # GAP
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def gem(x, p, eps):
    x = F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), (1, 1)).pow(1. / p)
    return x


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class HighResNet(nn.Module):
    def __init__(self, dropout_rate):
        super(HighResNet, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        resnet = torchvision.models.resnet34(pretrained=True)
        # self.resnet = torchvision.models.resnet18(pretrained=True)

        self.feature = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        n_out_channels = 256

        # GeM
        self.gem = GeM(p=3)

        # FC
        self.fc = nn.Linear(n_out_channels, 6)

    def forward(self, x):
        x = self.feature(x)

        # GeM
        x = F.relu(x)
        x = self.gem(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class HighSEResNeXt(nn.Module):
    def __init__(self, dropout_rate):
        super(HighSEResNeXt, self).__init__()

        senet = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            num_classes=1000, pretrained='imagenet')
        # remove layer4
        self.feature = nn.Sequential(
            senet.layer0,
            senet.layer1,
            senet.layer2,
            senet.layer3,
        )
        n_out_channels = 1024

        # GeM
        self.gem = GeM(p=3.0)

        # FC
        self.last_layer = nn.Sequential(
            nn.Linear(n_out_channels, n_out_channels),
            # nn.BatchNorm1d(n_out_channels),
            # nn.ReLU(),
            nn.Linear(n_out_channels, 6)
        )

    def forward(self, x):
        x = self.feature(x)

        # GeM
        x = F.relu(x)
        x = self.gem(x)
        x = x.view(x.size(0), -1)
        # FC
        x = self.last_layer(x)

        return x


class HighSEResNeXt2(nn.Module):
    """The model has 2 blocks of 4 resual blocks"""

    def __init__(self, dropout_rate):
        super(HighSEResNeXt2, self).__init__()

        senet = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            num_classes=1000, pretrained='imagenet')
        # remove layer4
        self.feature = nn.Sequential(
            senet.layer0,
            senet.layer1,
            senet.layer2,
            senet.layer3,
            senet.layer4,
        )
        n_out_channels = 2048

        # GeM
        self.gem = GeM(p=3.0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # FC
        self.last_layer = nn.Sequential(
            nn.Linear(n_out_channels * 3, n_out_channels),
            nn.Dropout(p=0.5),
            nn.Linear(n_out_channels, 6)
        )

    def forward(self, x):
        x = self.feature(x)

        # GeM
        x = F.relu(x)
        x = torch.cat([self.avg_pool(x), self.gem(x), self.max_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        # FC
        x = self.last_layer(x)

        return x


class HighCbamResNet(nn.Module):
    def __init__(self, dropout_rate):
        super(HighCbamResNet, self).__init__()

        resnet = resnet50_cbam(pretrained=True)
        # remove layer4
        self.feature = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        n_out_channels = 1024

        # GeM
        self.gem = GeM(p=3.0)

        # FC
        self.last_layer = nn.Sequential(
            nn.Linear(n_out_channels, n_out_channels),
            # nn.BatchNorm1d(n_out_channels),
            # nn.ReLU(),
            nn.Linear(n_out_channels, 6)
        )

    def forward(self, x):
        x = self.feature(x)

        # GeM
        x = F.relu(x)
        x = self.gem(x)
        x = x.view(x.size(0), -1)
        # FC
        x = self.last_layer(x)

        return x
