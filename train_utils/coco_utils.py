import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np


def convert_voc_to_coco(voc_dataset):
    """
    将 VOCDataset 转换为 pycocotools 工具中的 COCO 对象, 
    参照 COCO 的构造方法，先构建相应格式的 COCO.dataset 属性，然后调用 createIndex()
    """
    coco_ds = COCO()
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(voc_dataset)):
        hw, targets = voc_dataset.coco_index(img_idx)
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = hw[0]
        img_dict['width'] = hw[1]
        dataset['images'].append(img_dict)
        bboxes = targets['boxes']
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id

            ann_id += 1
            categories.add(labels[i])
            dataset['annotations'].append(ann)
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


class CocoEvaluator(object):
    def __init__(self, coco_gt):
        self.coco_gt = coco_gt
        self.coco_dt = COCO()
        self.coco_eval = None
        self.coco_results = []

    def _convert_boxes_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def update(self, predictions):
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction['boxes']
            boxes = self._convert_boxes_to_xywh(boxes).tolist()
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()

            self.coco_results.extend(
                [
                    {
                        'image_id': original_id,
                        'category_id': labels[i],
                        'bbox': box,
                        'score': scores[i]
                    }
                    for i, box in enumerate(boxes)
                ]
            )
    
    def evaluate(self):
        self.coco_dt = self.coco_gt.loadRes(self.coco_results) if len(self.coco_results) != 0 else COCO()
        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
        self.coco_eval.evaluate()
    
    def accumulate(self):
        return self.coco_eval.accumulate()
    
    def summarize(self):
        return self.coco_eval.summarize()