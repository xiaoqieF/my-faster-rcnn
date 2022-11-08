import random
from torchvision.transforms import functional as F

"""
Definition of Transforms which used in VOCDataSet
"""

class Compose(object):
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ToTensor(object):
    def __call__(self, img, target):
        img = F.to_tensor(img)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img, target):
        if random.random() < self.prob:
            height, width = img.shape[-2:]
            img = img.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return img, target