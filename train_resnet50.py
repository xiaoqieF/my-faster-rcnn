import os
import torch
from torch.utils.data import DataLoader

from train_utils.train import train_one_epoch, evaluate
from backbones.resnet50 import resnet50
from networks.generalized_rcnn import FasterRCNN
import transforms
from dataset import VOCDataSet, collate_fn


def create_model(num_classes):
    backbone = resnet50(weight_path='./backbones/resnet50.pth', trainable_layers=0)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
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

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    # 先冻结backbone参数，训练 init_epochs 轮
    init_epochs = 5
    for epoch in range(init_epochs):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device, warmup=True)
        evaluate(model, val_dataloader, device)

    for name, parameter in model.backbone.named_parameters():
        # 冻结底层参数，启用backbone后面参数
        if all([not name.startswith(layer) for layer in ['layer1', 'layer2']]):
            parameter.requires_grad_(True)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=2,
                                                   gamma=0.5)
    num_epochs = 15
    for epoch in range(init_epochs, num_epochs+init_epochs, 1):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device, warmup=False)
        lr_scheduler.step()

        evaluate(model, val_dataloader, device)

        if epoch > 10:
            torch.save(model.state_dict(), f"saved_params/resnet50_{str(epoch)}.pth")
        

if __name__ == '__main__':
    main()
