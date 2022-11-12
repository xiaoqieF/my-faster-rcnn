import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


def collate_fn(batch):
    """
    Stack img and target respectively
    Args:
        batch (List[List[Tensor, Dict]])
    Return:
        Tuple[List[Tensor], List[Dict]]
    """
    return tuple(zip(*batch))

class VOCDataSet(Dataset):
    """ 
    DataSet of VOC2012
    """
    def __init__(self, voc_root, transforms=None, isTrain=True):
        super().__init__()
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.img_root = os.path.join(self.root, "JPEGImages")

        if isTrain:
            txt_path = os.path.join(self.root, "ImageSets", "Main", "train.txt")
        else:
            txt_path = os.path.join(self.root, "ImageSets", "Main", "val.txt")
        
        with open(txt_path) as txt:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml") 
                                for line in txt.readlines()]
        label_index_path = './pascal_voc_classes.json'
        assert os.path.exists(label_index_path), f"{label_index_path} file not exist"
        with open(label_index_path, "r") as f:
            self.class_dict = json.load(f)
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.xml_list)
    
    def __getitem__(self, idx):
        """
        Return:
            Tuple[Tensor, Dict[str, Tensor]]: which indicate img and target
        """
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        img = Image.open(img_path)
        if img.format != "JPEG":
            raise ValueError(f"{img_path} format not JPEG")
        
        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, f"{xml_path} lack of object info"
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            if xmax <= xmin or ymax <= ymin:
                print(f"Warning: in {xml_path}, some bbox w/h <= 0")
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        
        # to Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        img_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target= {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    @staticmethod
    def parse_xml_to_dict(xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = VOCDataSet.parse_xml_to_dict(child)
            if child.tag != "object":
                result[child.tag] = child_result[child.tag]
            else:
                if "object" not in result:
                    result["object"] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}
    
if __name__ == '__main__':
    import transforms
    import torchvision
    import random
    from draw_boxes_utils import draw_objs
    import matplotlib.pyplot as plt
    cate_index = {}
    with open('./pascal_voc_classes.json', 'r') as f:
        class_dict = json.load(f)
        cate_index = {str(v): str(k) for k, v in class_dict.items()}
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.ToTensor()
    }
    train_data_set = VOCDataSet(os.path.join(os.getcwd(), "data"), data_transform["train"], isTrain=True)
    print(len(train_data_set))
    for index in random.sample(range(0, len(train_data_set)), k=5):
        img, target = train_data_set[index]
        print(f"img shape:{img.shape}, target:{target}")
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