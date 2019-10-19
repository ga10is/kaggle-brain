from __future__ import division
from __future__ import print_function
import math
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch


class ArcMarginProduct(nn.Module):
    # source from https://github.com/ronghuaiyang/arcface-pytorch
    # Thank you, ronghuaiyang
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            # output = cosine * self.s
            output = cosine
            return output

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output[0])

        return output


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, x):
        return x


class FocalBinaryLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalBinaryLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        p = torch.sigmoid(input)
        loss = torch.mean(-1 * target * torch.pow(1 - p, self.gamma) * torch.log(p + 1e-10) +
                          -1 * (1 - target) * torch.pow(p, self.gamma) * torch.log(1 - p + 1e-10)) * 4
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return focal_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class WeightedBCE(nn.Module):
    def __init__(self, label_weight):
        """
        Initialize instance

        Parameters
        ----------
        label_weight: torch.Tensor, size of [n_labels]
            weight of labels
        """
        super(WeightedBCE, self).__init__()
        # self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.label_weight = label_weight

    def forward(self, input, target):
        # loss size: [n_batches, n_labels]
        # loss = self.bce(input, target)
        # w_loss size: [n_batches]
        # w_loss = torch.mv(loss, self.label_weight)
        eps = 1e-15
        prob = torch.sigmoid(input)
        prob = prob.clamp(min=eps, max=1 - eps)
        loss = F.binary_cross_entropy(prob, target, reduction='none')

        # loss * label_weight size: [n_batches, n_labels] -> [n_batches]
        # label_weight.sum() = 7
        w_loss2 = (loss * self.label_weight).sum(dim=1) / \
            self.label_weight.sum()

        return w_loss2.mean()


class FocalBCELoss(nn.Module):
    def __init__(self, bce_weight, label_weight, gamma=2):
        super(FocalBCELoss, self).__init__()
        self.focal = BinaryFocalLoss(gamma=gamma)
        self.bce = WeightedBCE(label_weight)
        self.bce_w = bce_weight

    def forward(self, input, target):
        alpha = self.bce_w
        loss = alpha * self.bce(input, target) + \
            (1 - alpha) * self.focal(input, target)
        return loss
