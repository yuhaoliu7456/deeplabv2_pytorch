import torch.nn.functional as F
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
    
    def forward(self, logits, label):
        criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignore_label)
        loss = criterion(logits, label)
        return loss