import torch
import time
import numpy as np
from utils.misc import make_input

class MAELoss(torch.nn.Module):
    """
    loss for MAE
    """
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred, masks, gt):
        assert pred.size() == gt.size()
        epsilon = 0.0001
        #l = (1 - (pred * gt).sum(dim = 3) / ( torch.norm(pred, 2, 3) * torch.norm(gt, 2, 3) ) ) * masks
        l = (pred * gt).sum(dim = 3) / ( torch.norm(pred, 2, 3) * torch.norm(gt, 2, 3) ) * masks
        l = torch.clamp(l, -1 + epsilon, 1 - epsilon)
        l = torch.acos(l)
        l = l.mean(dim=2).mean(dim=1)
        eucli_l = (((gt - pred) ** 2).mean(dim=3) * masks).mean(dim=2).mean(dim=1)
        return (l*2 + eucli_l*np.pi)/(np.pi+2)
#         return (l/np.pi+eucli_l/2)


def test_mae_loss():
    pred = torch.FloatTensor([[[[0.5, 0.2, 0.3], [0.1, -0.2, -0.7]]]])
    print(pred.size())
    mask = torch.FloatTensor([[[1, 1]]])
    gt = torch.FloatTensor([[[[0.5, 0.2, 0.3], [0.1, -0.2, -0.7]]]])
    loss = MAELoss.forward(MAELoss, pred, mask, gt)
    print(loss)

if __name__ == '__main__':
    test_mae_loss()
