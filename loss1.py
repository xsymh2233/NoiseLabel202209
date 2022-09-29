import torch
import torch.nn.functional as F
import numpy as np
def custom_ce(prediction, target):
    output_pos = -target * prediction
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()

class Bootstrapping(torch.nn.Module):
    def __init__(self, num_classes, t):
        super(Bootstrapping, self).__init__()
        self.num_classes = num_classes
        self.t = t   # beta

    def forward(self, pred, labels):
        # Create smoothed labels
        labels_onehot = F.one_hot(labels.to(torch.int64), self.num_classes).float()  # q
        prediction = F.softmax(pred, dim=1)
        labels_smooth = (1.0 - self.t) * labels_onehot + self.t * prediction

        pred_log = prediction.clamp(1e-7, 1.0).log()

        return custom_ce(pred_log, labels_smooth)


class Bootstrapping_Hard(torch.nn.Module):
    def __init__(self, num_classes, t):
        super(Bootstrapping_Hard, self).__init__()
        self.num_classes = num_classes
        self.t = t   # beta

    def forward(self, pred, labels):
        # Create smoothed labels
        labels_onehot = F.one_hot(labels.to(torch.int64), self.num_classes).float()  # q
        prediction = F.softmax(pred, dim=1)
        zker = pred.argmax(dim=1, keepdim=True)
        labels_smooth = (1.0 - self.t) * labels_onehot + self.t * zker

        pred_log = prediction.clamp(1e-7, 1.0).log()

        return custom_ce(pred_log, labels_smooth)
