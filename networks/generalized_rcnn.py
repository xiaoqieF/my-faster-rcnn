import torch
from torch import nn
from collections import OrderedDict
from .transform import GeneralizedRCNNTransform


class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Args:
            images: list[Tensor] images to be processed
            targets: list[Dict[str, Tensor]]  ground truth
        """
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    assert isinstance(boxes, torch.Tensor), \
                        f"Expected target boxes to be type of Tensor, got{type(boxes)}"
                    assert len(boxes.shape) == 2 and boxes.shape[-1] == 4, \
                        f"Expected target boxes to be tensor of shape [N, 4], got {boxes.shape}"
        
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2 # last two dimensions are H and W
            original_image_sizes.append((val[0], val[1]))
        
        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_size, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return (losses, detections)