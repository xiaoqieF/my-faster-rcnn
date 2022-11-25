import math
import torch
import torchvision

def encode_boxes(reference_boxes, proposals, weights):
    """
        Calculate regression params use gt boxes and anchors/proposals
        t_x = (x - x_a) / w_a
        t_y = (y - y_a) / h_a
        t_w = log(w / w_a)
        t_h = log(h / h_a)
    """
    # weights of each param
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # cal params that trans box from proposals to reference_boxes
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_center_x = proposals_x1 + 0.5 * ex_widths
    ex_center_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_center_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_center_y = reference_boxes_y1 + 0.5 * gt_heights

    tx = wx * (gt_center_x - ex_center_x) / ex_widths
    ty = wy * (gt_center_y - ex_center_y) / ex_heights
    tw = ww * torch.log(gt_widths / ex_widths)
    th = wh * torch.log(gt_heights / ex_heights)

    params = torch.cat((tx, ty, tw, th), dim=1)
    return params


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)) -> None:
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Calculate regression params use gt boxes and anchors/proposals
        Args:
            reference_boxes(List[Tensor]): ground truth boxes of many images
            proposals(List[Tensor]): anchors, or proposals when second regression. 
                                    length and elem tensor shape should equals to reference_boxes
        Returns:
            regression params(List[Tensor])
        """
        boxes_per_image = [b.shape[0] for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes, boxes):
        """
        Calculate regression boxes use regression params and anchors
        Args:
            rel_codes(Tensor): regression params, shape is [N(batch_size*anchors_per_img), 4]
            boxes(List[Tensor]): anchors / proposals of many images
        Returns:
            pred_boxes(Tensor): shape is [N, num_classes, 4]
        """
        concat_boxes = torch.cat(boxes, dim=0)   # [N, 4]

        box_sum = concat_boxes.shape[0]
        
        pred_boxes = self.decode_single(rel_codes, concat_boxes)

        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        Args:
            rel_codes(Tensor: shape is [N, 4]): regression params
            boxes(Tensor: shape is [N, 4]): anchors
        """
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        center_x = boxes[:, 0] + 0.5 * widths
        center_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx         # rel_codes:[N, num_classes*4], 0::4 extract dx of each class
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_center_x = dx * widths[:, None] + center_x[:, None]
        pred_center_y = dy * widths[:, None] + center_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes_x1 = pred_center_x - torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes_y1 = pred_center_y - torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes_x2 = pred_center_x + torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes_y2 = pred_center_y + torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.cat((pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2), dim=1)
        return pred_boxes        


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Args:
        boxes1(Tensor[N, 4])
        boxes2(Tensor[M, 4])
    Returns:
        iou matrix(Tensor[N, M]): iou matrix N x M
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # calculate areas of interaction of boxes1 and boxes2
    # torch.max(Tensor[N, 1, 2], Tensor[M, 2]), follow broadcasting rules, output [N, M, 2]
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]         # [N, M]

    # broadcast
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def clip_boxes_to_image(boxes, size):
    """
    Args:
        boxes(Tensor[N, 4]): (x1, y1, x2, y2)
        size(Tuple[height, width])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # 0::2 extract x of each box, total num_classes boxes per proposal
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

def remove_small_boxes(boxes, min_size):
    """
    Remove boxes which contains at least one side smaller than min_size.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        min_size (float): minimum size

    Returns:
        keep (Tensor[K]): indices of the boxes that have both sides
            larger than min_size
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    keep = torch.where(keep)[0]  # torch.where(condition) returns a tuple, use [0] return keep box index
    return keep

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes(Tensor[N, 4])
            boxes where NMS will be performed. They
            are expected to be in (x1, y1, x2, y2) format
        scores(Tensor[N])
            scores for each one of the boxes
        idxs(Tensor[N])
            indices of the categories for each one of the boxes.
        iou_threshold(float)
            discards all overlapping boxes
            with IoU < iou_threshold
    Returns:
        keep(Tensor)
            int64 tensor with the indices
            of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()

    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    return torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)