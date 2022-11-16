import os
import sys

# 将工作目录添加到包搜索路径, pycharm默认添加
sys.path.append(os.getcwd())

import torchvision
import numpy as np
import json
import matplotlib.pyplot as plt 
from dataset import VOCDataSet, collate_fn
import transforms
from torch.utils.data import DataLoader
from networks.transform import GeneralizedRCNNTransform
from draw_boxes_utils import draw_objs

"""
    测试 GeneralizedRCNNTransform 效果，将一个 batch 数据经过该模块后显示相应的图像和 bbox 位置
    效果是先将原图进行像素归一化，然后调整大小并在右边/下边补 0 , bbox位置进行相应变化
"""

if __name__ == '__main__':
    trans = GeneralizedRCNNTransform(400, 800, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.ToTensor()
    }
    train_data_set = VOCDataSet(os.path.join(os.getcwd(), "data"), data_transform["train"], isTrain=True)
    print(f"total samples:{len(train_data_set)}")

    train_dataloader = DataLoader(train_data_set, batch_size=8, shuffle=True, \
        num_workers=4, collate_fn=collate_fn)

    cate_index = {}
    with open('./pascal_voc_classes.json', 'r') as f:
        class_dict = json.load(f)
        cate_index = {str(v): str(k) for k, v in class_dict.items()}

    for imgs, targets in train_dataloader:
        print(f"imgs.lenth:{len(imgs)}, targets[0]:{targets[0]}")
        imgs, targets = trans(imgs, targets)
        for img, target in zip(imgs.tensors, targets):
            img = torchvision.transforms.ToPILImage()(img)
            plot_img = draw_objs(img,
                                target["boxes"].numpy(),
                                target["labels"].numpy(),
                                np.ones(target["labels"].shape[0]),
                                cate_index,
                                box_thresh=0.5,
                                line_thick=2)
            plt.imshow(plot_img)
            plt.show()
        break