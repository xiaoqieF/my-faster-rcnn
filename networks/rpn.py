import torch
import torch.nn as nn
from torch.nn import functional as F
from . import boxes as boxes_ops
from . import det_utils


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # every anchor predict one score to indicate whether is contains object 
        # (in paper, here are 2 scores softmax)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)  # [batch_size, ]
        # every anchor predict 4 position params
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x(List[Tensor]): list of features, 
            it is possible there are many different layers of feature (eg. when use fpn)
        Return:
            Tuple[List[Tensor], List[Tensor]]
        """
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

def permute_and_flatten(layer, N, A, C, H, W):
    """
    permute tensor and reshape, change C to last dimension
    """
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)   # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    """
    concate box_cls and box_regression of many feature layers respectively
    Args:
        box_cls(List[Tensor]): box_cls of many feature maps, shape of Tensor is [N, C, H, W], 
            C equals to num_anchors * 1, that is each pixel of feature map predict num_anchors
        box_regression(List[Tensor]): box regression params of many feature maps, shape of Tensor is 
            [N, C, H, W], C equals to num_anchors * 4, each bbox has 4 params
    Returns:
        Tuple[Tensor, Tensor]: box_cls:[N, 1], box_regression:[N, 4]
    """
    box_cls_flattened = []
    box_regression_flattened = []

    # for each feature map, permute and reshape it's box_cls and box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, num_anchors * class_num, height, width]
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)   # [N(batch_size), -1, C]
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        anchor_generator,
        head, 
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        score_thresh = 0.0
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = boxes_ops.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        self.box_similarity = boxes_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.0
    
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']
    
    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        """
        Args:
            objectness(Tensor[batch_size, -1])
            num_anchors_per_level(List[int]): num anchors of each feature map
        Returns:
            top n anchor index(Tensor[batch_size, n*batch_size])
        """
        r = []
        offset = 0
        # for each feature map layer, get top k socre anchor index
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nums_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nums_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)


    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        """
        get top k proposals and do clip, remove small boxes and nms
        Args:
            proposals(Tensor[batch_size, -1, 4]): proposals predicted by rpn_head
            objectness(Tensor[batch_size * num_anchors_per_img, 1]): cls_scores predicted by rpn_head
            image_shapes(List[Tuple(int, int)]): true image shape (no padding)
            num_anchors_per_level(List[int]): num anchors of each feature map, use this to do nms independent of feature maps
        Returns:
            Tuple[List[Tensor[N, 4]], List[Tensor[N, 1]]]: final boxes coordinates and final scores
        """
        num_images = proposals.shape[0]
        device = proposals.device

        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # [tensor[0, 0, 0, ...], tensor[1, 1, 1, ...] ...]
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device) 
                    for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, dim=0)  # shape is [n1 + n2 + ... ni, ], where n1 + n2 + ... + ni = total anchors of different levels

        levels = levels.reshape(1, -1).expand_as(objectness) # [num_images, n1 + n2 + ... + ni]，to label which level current proposal is in

        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level) # [num_images, top_n_num]

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        # got top n anchors's info
        objectness = objectness[batch_idx, top_n_idx] # [num_images, top_n_num]
        levels = levels[batch_idx, top_n_idx]         # [num_images, top_n_num]

        proposals = proposals[batch_idx, top_n_idx]  # [num_images, top_n_num, 4]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []

        # for each image, do clip, remove_small_boxes and nms
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = boxes_ops.clip_boxes_to_image(boxes, img_shape)

            keep = boxes_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            keep = torch.where(torch.ge(scores, self.score_thresh))[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = boxes_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors, targets):
        """
        assign gt box to each anchor
        Args:
            anchors(List[Tensor])
            targets(List[Dict[Str, Tensor]])

        Returns:
            Tuple[List[num_images], List[num_images]]: 
            labels:List of tensors, each tensor(shape[num_anchors_per_img]) 
            indicate current anchor's label, 1 for positive, 0 for negetive, 
            -1 for dropped sample.
            matched_gt_boxes:List of tensors, each tensor(shape[num_anchors_per_img, 4])
            indicate corresponding gt box's coordinate, only where labels is 1 is avail
        """
        labels = []
        matched_gt_boxes = []
        # for each image
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = boxes_ops.box_iou(gt_boxes, anchors_per_image)
                # gt indices corresponding to each anchor, -1 for negative sample, -2 for dropped sample
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                
                # shape:[num_anchors_per_image, 4], holds gt box coordinate to each anchor
                # use clamp(min=0) to get gt coord conveniently, only coordinates for positive samples are available
                # labels_per_image holds positive samples' indices
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)] 

                # labels_per_image holds labels of each anchor(positive sample is 1, negetive is 0, dropped sample is -2)
                labels_per_image = matched_idxs >= 0   # positive anchor label True
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                discard_indices = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLD # -2
                labels_per_image[discard_indices] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        
        return labels, matched_gt_boxes
    
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum"
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict. 
        """
        features = list(features.values())

        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)

        # calculate number of anchors in each feature map
        num_anchors_per_level_shape_tensors = [obj[0].shape for obj in objectness]  # obj:[N, C, H, W] -> obj[0]:[C, H, W]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # now shape of objectness is [N(batch_size * num_anchors_per_img), 1], shape of pred_bbox_deltas is [N, 4]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # decode regression params to bbox coordinate
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None, f"targets should not be None when training"
            # labels(List[Tensor[num_anchors_per_img]]): 1 for positive, 0 for negetive, -1 for dropped sample.
            # matched_gt_boxes(List[Tensor[num_anchors_per_img, 4]]): each anchor's corresponding gt, only positive anchor is valid
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses