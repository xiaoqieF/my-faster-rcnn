import torch
from torch import nn, optim
from tqdm import tqdm
import time

from train_utils.coco_utils import CocoEvaluator, convert_voc_to_coco

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch(model, epoch, train_dataloader, optimizer, device, warmup=False):
    print(f"------第{epoch + 1}轮训练开始-------")
    model.to(device)
    model.train()

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for imgs, target in tqdm(train_dataloader, desc='training:'):
        imgs = [img.to(device) for img in imgs]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        optimizer.zero_grad()

        loss_dict = model(imgs, target)
        losses = sum(loss_dict.values())
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
    
    print(f"epoch: {epoch + 1}, loss: {losses.item()}")
            
@torch.no_grad()
def evaluate(model, val_dataloader, device):
    model.eval()

    coco_gt = convert_voc_to_coco(val_dataloader.dataset)
    coco_evaluator = CocoEvaluator(coco_gt)

    for image, targets in tqdm(val_dataloader, desc='validating:'):
        image = [img.to(device) for img in image]

        model_time = time.time()
        outputs = model(image)
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}

        coco_evaluator.update(res)

    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()