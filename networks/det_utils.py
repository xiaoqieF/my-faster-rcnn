import torch


class Matcher(object):
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False) -> None:
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLD = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        calculate matches of anchors and gtboxes. return the match gtbox's index of each anchor
        if iou < low_threshold, the index is -1, if low_threshold <= iou <= high_threshold, the index is -2
        Args:
            match_quality_matrix(Tensor[float]): an M x N tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches(Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        assert match_quality_matrix.numel() != 0, f"numel of match_quality_matrix should not be 0"

        # find best gt candidate for each anchor
        # matched_vals is the max values of each column
        # matches is the index of the max values
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
        
        below_low_threshold = matched_vals < self.low_threshold
        between_threshold = (matched_vals >= self.low_threshold) & \
            (matched_vals <= self.high_threshold)

        # set index to -1 when iou < low_threshold
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        # set index to -2 when low_threshold <= iou <= high_threshold
        matches[between_threshold] = self.BETWEEN_THRESHOLD

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        
        return matches
    
    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        For each gt, find best anchors that have max iou with it, if the anchor is unmatched, 
        match it to the gt
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # Example gt_pred_pairs_of_highest_quality(Tuple[2]):
        #   (tensor([ 0, 0, 1, 1, 1, 1, 1, 1]),
        #   tensor([ 1010, 1019, 2340, 2349, 2358, 2367, 2376, 2385]))
        gt_anchor_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # update corresponding anchor
        anchor_inds_to_update = gt_anchor_pairs_of_highest_quality[1]
        matches[anchor_inds_to_update] = all_matches[anchor_inds_to_update]


class BalancedPositiveNegativeSampler(object):
    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: (List[Tensor]): list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (List[Tensor]): positive elements that were selected, where the val is set to 1
            neg_idx (List[Tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx

def smooth_l1_loss(input, target, beta = 1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()