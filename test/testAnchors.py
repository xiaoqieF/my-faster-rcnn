import os
import sys

# 将工作目录添加到包搜索路径, pycharm默认添加
sys.path.append(os.getcwd())

from backbones.mobilenetv2 import MobileNetV2
from dataset import VOCDataSet, collate_fn
import transforms
from torch.utils.data import DataLoader
from networks.transform import GeneralizedRCNNTransform
from networks.anchor_generator import AnchorGenerator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    backbone = MobileNetV2().features
    backbone.to(device)
    print(backbone)
    trans = GeneralizedRCNNTransform(400, 800, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    g = AnchorGenerator()

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.ToTensor()
    }
    train_data_set = VOCDataSet(os.path.join(os.getcwd(), "data"), data_transform["train"], isTrain=True)
    print(f"total samples:{len(train_data_set)}")

    train_dataloader = DataLoader(train_data_set, batch_size=8, shuffle=True, \
        num_workers=4, collate_fn=collate_fn)

    for imgs, t in train_dataloader:
        img_list, target = trans(imgs, t)
        img_cuda = img_list.tensors.to(device)
        feat = backbone(img_cuda)
        print(f"features shape:{feat.shape}")
        anchors = g(img_list, (feat,))
        print(f"anchors len:{len(anchors)}, shape: {anchors[0].shape}")