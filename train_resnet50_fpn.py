import os
import torch
from torch.utils.data import DataLoader
import torchvision

from train_utils.train import train_one_epoch, evaluate
from networks.anchor_generator import AnchorGenerator
from backbones.resnet50 import resnet50_fpn
from networks.generalized_rcnn import FasterRCNN
import transforms
from dataset import VOCDataSet, collate_fn


def create_model(num_classes):
    backbone = resnet50_fpn("./backbones/resnet50.pth")

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()]),
        "val": transforms.ToTensor()
    }

    data_root = os.path.join(os.getcwd(), "data")
    if not os.path.exists(os.path.join(data_root, "VOCdevkit")):
        raise FileNotFoundError(f"VOCdevkit dose not in path:{data_root}.")

    train_dataset = VOCDataSet(data_root, data_transform["train"], isTrain=True)
    print(f"total training samples:{len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, \
        num_workers=6, collate_fn=collate_fn)

    val_dataset = VOCDataSet(data_root, data_transform["val"], isTrain=False)
    print(f"total validation samples:{len(val_dataset)}")

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
        num_workers=6, collate_fn=collate_fn)
    
    model = create_model(num_classes=21)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.333)
    num_epochs = 20
    for epoch in range(num_epochs):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device, warmup=True)
        lr_scheduler.step()

        evaluate(model, val_dataloader, device)


if __name__ == '__main__':
    main()