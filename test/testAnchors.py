import os
import sys

# 将工作目录添加到包搜索路径, pycharm默认添加
sys.path.append(os.getcwd())

from dataset import VOCDataSet, collate_fn
import transforms
from torch.utils.data import DataLoader
import torch

from backbones.mobilenetv2 import MobileNetV2
from networks.generalized_rcnn import FasterRCNN
from train_utils.coco_utils import convert_voc_to_coco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    backbone = MobileNetV2().features
    backbone.out_channels = 1280
    model = FasterRCNN(backbone, 21)
    model.to(device)
    model.eval()

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.ToTensor()
    }
    train_data_set = VOCDataSet(os.path.join(os.getcwd(), "data"), data_transform["train"], isTrain=True)
    print(f"total samples:{len(train_data_set)}")

    train_dataloader = DataLoader(train_data_set, batch_size=8, shuffle=True, \
        num_workers=4, collate_fn=collate_fn)

    coco = convert_voc_to_coco(train_dataloader.dataset)
    

    for imgs, target in train_dataloader:
        imgs = [img.to(device) for img in imgs]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        loss, detections = model(imgs, target)
        print(loss)