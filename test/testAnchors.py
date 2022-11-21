import os
import sys

# 将工作目录添加到包搜索路径, pycharm默认添加
sys.path.append(os.getcwd())

from collections import OrderedDict
from backbones.mobilenetv2 import MobileNetV2
from dataset import VOCDataSet, collate_fn
import transforms
from torch.utils.data import DataLoader
from networks.transform import GeneralizedRCNNTransform
from networks.anchor_generator import AnchorGenerator
from networks.rpn import RegionProposalNetwork
from networks.rpn import RPNHead
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    backbone = MobileNetV2().features
    backbone.to(device)
    print(backbone)
    trans = GeneralizedRCNNTransform(400, 800, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    g = AnchorGenerator()

    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)

    rpn = RegionProposalNetwork(
        g,
        RPNHead(in_channels=1280, num_anchors=9),
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=128,
        positive_fraction=0.5,
        pre_nms_top_n=rpn_pre_nms_top_n,
        post_nms_top_n=rpn_post_nms_top_n,
        nms_thresh=0.7
    )
    rpn.to(device)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.ToTensor()
    }
    train_data_set = VOCDataSet(os.path.join(os.getcwd(), "data"), data_transform["train"], isTrain=True)
    print(f"total samples:{len(train_data_set)}")

    train_dataloader = DataLoader(train_data_set, batch_size=8, shuffle=True, \
        num_workers=4, collate_fn=collate_fn)

    for imgs, target in train_dataloader:
        imgs = [img.to(device) for img in imgs]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        img_list, target = trans(imgs, target)
        feat = backbone(img_list.tensors)
        print(f"features shape:{feat.shape}")
        features = OrderedDict([("0", feat)])
        boxes, losses = rpn(img_list, features, target)