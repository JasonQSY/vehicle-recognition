import torch
from torch import nn
import torchvision.models as models
from task.loss import DummyLoss


class DummyLayer(nn.Module):
    def forward(self, x):
        return x


class JccNet(nn.Module):
    """
    Fine-tune resnet to ROB535 dataset.
    """
    def __init__(self, **params):
        super(JccNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        #self.resnet.fc = nn.Linear(512 * 4, 3)
        #self.resnet.avgpool = DummyLayer()
        self.resnet.avgpool = nn.AvgPool2d(kernel_size=32, stride=1, padding=0)
        #self.resnet.fc = DummyLayer()
        self.resnet.fc = nn.Linear(2048, 3)
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False
        self.dummyloss = DummyLoss()

    def forward(self, x):
        return self.resnet(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #out = self.resnet(x)
        import pdb; pdb.set_trace()
        return self.resnet(x)

    def calc_loss(self, preds, gts):
        l = self.dummyloss(preds, gts)
        return [l]

