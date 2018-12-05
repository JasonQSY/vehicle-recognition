import torch
from torch import nn
import torchvision.models as models
from task.loss import DummyLoss


class JccNet(nn.Module):
    """
    Fine-tune resnet to ROB535 dataset.
    """
    def __init__(self, **params):
        super(JccNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(512 * 4, 3)
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False
        self.dummyloss = DummyLoss()

    def forward(self, x):
        return self.resnet(x)

    def calc_loss(self, preds, gts):
        l = self.dummyloss(preds, gts)
        return [l]

