import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from task.loss import DummyLoss


class DummyLayer(nn.Module):
    def forward(self, x):
        return x


class CGNet(nn.Module):
    """
    dummy network for task 2.
    """
    def __init__(self, **params):
        super(CGNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        #self.resnet.avgpool = nn.AvgPool2d(kernel_size=32, stride=1, padding=0)
        self.resnet.fc = DummyLayer()
        #self.resnet.fc = nn.Linear(2048, 3)
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False
        #self.dummyloss = DummyLoss()
        self.add_hidden = nn.Linear(2048, 512)
        self.add_output = nn.Linear(512, 3)
        self.loss = nn.MSELoss()

    def forward(self, x):
        features = self.resnet(x)
        hidden = self.add_hidden(features)
        output = self.add_output(hidden)
        return output

    def calc_loss(self, preds, gts):
        l = self.loss(preds, gts)
        return [l]
