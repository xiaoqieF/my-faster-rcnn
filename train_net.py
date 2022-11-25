import os
import torch
from torch.utils.data import DataLoader
from train_utils.train import train_one_epoch

from backbones.mobilenetv2 import MobileNetV2
from networks.generalized_rcnn import FasterRCNN
import transforms
from dataset import VOCDataSet, collate_fn


def create_model(num_classes):
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.ToTensor()
    }

    data_root = os.path.join(os.getcwd(), "data")
    if not os.path.exists(os.path.join(data_root, "VOCdevkit")):
        raise FileNotFoundError(f"VOCdevkit dose not in path:{data_root}.")

    train_dataset = VOCDataSet(data_root, data_transform["train"], isTrain=True)
    print(f"total training samples:{len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, \
        num_workers=4, collate_fn=collate_fn)

    val_dataset = VOCDataSet(data_root, data_transform["val"], isTrain=False)
    print(f"total validation samples:{len(val_dataset)}")

    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
        num_workers=4, collate_fn=collate_fn)
    
    model = create_model(num_classes=21)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=0.0005)
    for epoch in range(20):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device, warmup=True)


if __name__ == '__main__':
    main()