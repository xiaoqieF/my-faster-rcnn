import torch
from torch import nn
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)  # [N, ] N is total proposals num of all imgs
    regression_targets = torch.cat(regression_targets, dim=0)  # [N, 4]

    classification_loss = F.cross_entropy(class_logits, labels)

    # positive samples' indices
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]
    pos_labels = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)   # [N, num_classes, 4]
    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, pos_labels],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHead(nn.Module):
    def __init__(self, 
                 box_roi_pool,    # Multi-scale RoIAlign pooling
                 box_head,        # TwoMLPHead
                 box_predictor,   # FastRCNNPredictor
                 fg_iou_thresh,
                 bg_iou_thresh,
                 batch_size_per_image,
                 positive_fraction,
                 bbox_reg_weights,
                 score_thresh,
                 nms_thresh,
                 detection_per_img):
        super().__init__()
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = box_ops.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box), dim=0) for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals
    
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        Args:
            proposals(List[Tensor]): proposals of many images
            gt_boxes(List[Tensor]): gt_boxes of many images
            gt_labels(List[Tensor[int64]]): labels of each gt_boxes
        Returns:
            matched_idxs(List[Tensor[N,]]): matched gt idxs for each proposal, only positive are avail
            labels(List[Tensor[N,]]): matched labels for each proposal, 0 for background, -1 for dropped
        """
        matched_idxs = []
        labels = []
        # for each image
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
                # matched gtbox's indice of each proposal, -1 for negative, -2 for drop, 
                # 0 and positive num is indices of gt
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                
                # negative proposal(label background)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # dropped proposal(between low and high threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLD  # -2
                labels_in_image[ignore_inds] = -1
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        """
            List[Tensor] -> List[Tensor], from labels get some samples
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # for each image
        for pos_inds_img, neg_inds_img in zip(sampled_pos_inds, sampled_neg_inds):
            img_samled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_samled_inds)
        return sampled_inds


    def select_training_samples(self, proposals, targets):
        assert targets is not None
        
        dtype = proposals[0].dtype
        device = proposals[0].device
        
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices and labels for each proposals
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        
        sampled_inds = self.subsample(labels)  # List[Tensor[batch_per_img,]]
        # sample a fixed proportion of positive-negative proposals
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        """
        Args:
            class_logits(Tensor[num_proposals, num_classes])
            box_regression(Tensor[num_proposals, 4])
            proposals(List[Tensor])
            image_shapes(List[Tensor])
        Return:
            all_boxes(List[Tensor]), all_scores(List[Tensor]), all_labels(List[Tensor])
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)  # [N, num_classes], N is sum of all boxes

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # for each image
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # boxes: [num_proposals_per_img, num_classes, 4]
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)  # [num_proposals_per_img, num_classes]

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class 
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels


    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features(Dict[Str, Tensor])
            proposals(List[Tensor[N, 4]]): N is proposals num of each img, different imgs have different N
            image_shapes(List[Tuple[int, int]])
            targets(List[Dict])
        """ 

        if self.training:
            # proposals(List[Tensor[batch_per_img, 4]])
            # labels(List[Tensor[batch_per_img,]])
            # regression_targets(same as proposals)
            proposals, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
        
        # box_features:[num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # box_features:[num_proposals, representation_size]
        box_features = self.box_head(box_features)
        
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i]
                    }
                )
        return result, losses
        