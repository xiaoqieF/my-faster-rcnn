import torch
from torch import nn
import math
from .image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    """
    First normalize minibatch images, Then Resize them: whether long side == max_size or 
    short side == min_size.
    Finally align all images in top-left corner, and padding to same size to be a batch
    """
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__()
        self.min_size = min_size   # short side's min val
        self.max_size = max_size   # long side's max val
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        """
        Args:
            images: List[Tensor]
            targets: List[Dict[str, Tensor]]
        Return:
            ImageList, List[Dict[str, Tensor]]]

        """
        images = [img for img in images]
        if targets is not None:
            targets = [{k: v for k,v in t.items()} for t in targets]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None

            assert image.dim() == 3  # [C, H, W]
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        
        image_sizes = [(sz[0], sz[1]) for sz in image_sizes]
        image_list = ImageList(images, image_sizes)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # mean.shape: [C] => mean[:, None, None].shape: [C, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        """
            Resize image and bbox rely on self.min_size and self.max_size
        """
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        scale_factor = self.min_size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        
        image = nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode="bilinear", 
            recompute_scale_factor=True,
            align_corners=False
        )[0]
        
        if target is not None:
            bbox = target["boxes"]
            target["boxes"] = self.resize_boxes(bbox, [h, w], image.shape[-2:])
            
        return image, target

    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratios_height, ratios_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratios_width
        xmax = xmax * ratios_width
        ymin = ymin * ratios_height
        ymax = ymax * ratios_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def batch_images(self, images, size_divisible=32):
        """
        Args:
            images(List[Tensor])
            size_divisible(int)
        """
        max_size = self.max_by_axis([list(img.shape) for img in images]) #List[int]: [maxC, maxH, maxW]
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        batch_shape = [len(images)] + max_size # [batch, C, H, W]
        
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # trans all imgs to shape of max_size, and align top-left corner
            # if img smaller than max_size, padding zero
            pad_img[: img.shape[0], : img.shape[1], :img.shape[2]].copy_(img)
        
        return batched_imgs


    @staticmethod
    def max_by_axis(the_list):
        """
            Get max val of each column
            (List[List[int]]) -> List[int]
        """
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, sz, original_sz) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = self.resize_boxes(boxes, sz, original_sz)  # cast back to original image
            result[i]["boxes"] = boxes
        return result