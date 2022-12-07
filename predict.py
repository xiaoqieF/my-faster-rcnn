import torch
import os
from PIL import Image
from torchvision.transforms import functional as F
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision

import transforms
from dataset import VOCDataSet, collate_fn
from backbones.mobilenetv2 import MobileNetV2
from backbones.resnet50 import resnet50
from networks.generalized_rcnn import FasterRCNN
from draw_boxes_utils import draw_objs
from train_utils.train import evaluate

def create_model(num_classes):
    backbone = resnet50(weight_path='', trainable_layers=0)
    backbone.out_channels = 2048

    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"device:{device}")

    cate_index = {}
    with open('./pascal_voc_classes.json', 'r') as f:
        class_dict = json.load(f)
        cate_index = {str(v): str(k) for k, v in class_dict.items()}

    model = create_model(num_classes=21)
    model.to(device)
    model.load_state_dict(torch.load('./saved_params/resnet50_15.pth'))
    model.eval()

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()]),
        "val": transforms.ToTensor()
    }
    data_root = os.path.join(os.getcwd(), "data")
    val_dataset = VOCDataSet(data_root, data_transform["val"], isTrain=False)
    print(f"total validation samples:{len(val_dataset)}")

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
        num_workers=6, collate_fn=collate_fn)

    evaluate(model, val_dataloader, device)

    # img_path = 'sample.jpg'
    # img = Image.open(img_path)
    # img = F.to_tensor(img).unsqueeze(0)
    # img = img.to(device)
    # with torch.no_grad():
    #     detection = model(img)[0]
    #     print(detection)
    #     img_1 = torchvision.transforms.ToPILImage()(img.squeeze(0))
    #     plot_img = draw_objs(img_1,
    #                         detection["boxes"].cpu().numpy(),
    #                         detection["labels"].cpu().numpy(),
    #                         detection["scores"].cpu().numpy(),
    #                         cate_index,
    #                         box_thresh=0.5,
    #                         line_thick=2)
    # plt.imshow(plot_img)
    # plt.show()