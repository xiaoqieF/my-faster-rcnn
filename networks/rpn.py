import torch
import torch.nn as nn
from torch.nn import functional as F


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # every anchor predict one score to indicate whether is contains object 
        # (in paper, here are 2 scores softmax)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # every anchor predict 4 position params
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x(List[Tensor]): list of features, 
            it is possible there are many different layers of feature (eg. when use fpn)
        Return:
            Tuple[List[Tensor], List[Tensor]]
        """
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

