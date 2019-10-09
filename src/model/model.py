import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .. import config
from .loss import ArcMarginProduct


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


class SEResNet(nn.Module):
    def __init__(self, dropout_rate, latent_dim):
        # def __init__(self, dropout_rate, latent_dim, temperature, m):
        super(SEResNet, self).__init__()

        senet = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            num_classes=1000, pretrained='imagenet')
        self.layer0 = senet.layer0
        self.layer1 = senet.layer1
        self.layer2 = senet.layer2
        self.layer3 = senet.layer3
        self.layer4 = senet.layer4

        n_out_channels = 2048
        # FC
        self.fc = nn.Linear(n_out_channels, 6)

    def forward(self, x, label=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # GAP
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # FC
        x = self.fc(x)

        return x


class CompareNet(nn.Module):
    def __init__(self, dropout_rate, latent_dim):
        super(CompareNet, self).__init__()

        self.net = SEResNet(dropout_rate, latent_dim)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )

    def forward(self, x1, x2):
        x1 = self.net(x1)
        x2 = self.net(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.head(x)

        return x
