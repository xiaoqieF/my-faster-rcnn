import torch
import os
from PIL import Image
from torchvision.transforms import functional as F
import json
import matplotlib.pyplot as plt
import torchvision
import time

from backbones.mobilenetv2 import MobileNetV2
from networks.generalized_rcnn import FasterRCNN
from draw_boxes_utils import draw_objs

def create_model(num_classes):
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    model = FasterRCNN(backbone=backbone, num_classes=num_classes, box_score_thresh=0.6)
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
    model.load_state_dict(torch.load('./saved_params/saved_params_18.pth'))
    model.eval()

    img_path = 'sample.jpg'
    img = Image.open(img_path)
    img = F.to_tensor(img).unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        detection = model(img)[0]
        print(detection)
        img_1 = torchvision.transforms.ToPILImage()(img.squeeze(0))
        plot_img = draw_objs(img_1,
                            detection["boxes"].cpu().numpy(),
                            detection["labels"].cpu().numpy(),
                            detection["scores"].cpu().numpy(),
                            cate_index,
                            box_thresh=0.5,
                            line_thick=2)
        plot_img.save("result.jpg")
    plt.imshow(plot_img)
    plt.show()