import torch
from torch import nn
from .image_list import ImageList


class AnchorGenerator(nn.Module):
    """
    Generate anchors for a set of feature maps and image sizes.
    Sizes and aspect_ratios have the same num of elements, which equals to num of feature maps.

    type of element in sizes and aspect_ratios is tuple, indicates how to 
    generate anchors in correspond feature map.

    Args:
        sizes(Tuple[Tuple[int]]): area of anchor = size[idx] * size[idx]
        aspect_ratios(Tuple[Tuple[int]]): ratio of height and width
    """
    def __init__(
        self, 
        sizes=((128, 256, 512),),
        aspect_ratios=(0.5, 1.0, 2.0)) -> None:
        super().__init__()

        # change element in sizes and aspect_ratios to tuple
        # for there is only one feature map, and sizes is Tuple[int]
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        # generate base_anchors for each feature map
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) 
                for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device = torch.device("cpu")):
        """
        Generate basic anchors, nums of anchors = len(scales) * len(aspect_ratios)
        Positions of anchors are based on (0, 0)
        Return:
            base_anchors(Tensor): shape:[N, 4] -> N * [top-left , right-bottom]
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) 
            for cell_anchor in self.cell_anchors]

    def grid_anchors(self, feature_map_sizes, strides):
        """
        Generate anchors in original image for all feature maps.
        calculate shifts of x and y, then plus base anchors and shifts.
        Args:
            feature_map_sizes(List[List[int]]): size of each feature map
            strides(List[int]): down sampling ratio of each feature map
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None, "cell_anchors should not be None"
        assert len(feature_map_sizes) == len(strides) == len(cell_anchors) # num of feature maps

        # for each feature map, generate anchors in origin image
        for size, stride, base_anchors in zip(feature_map_sizes, strides, cell_anchors):
            map_height, map_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, map_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, map_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    
    def forward(self, image_list, feature_maps):
        feature_map_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # stride of each feature map
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1])
            ]
            for g in feature_map_sizes 
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(feature_map_sizes, strides)
        anchors = []
        for _ in range(len(image_list)):
            # FIXME: anchors of each image of minibatch is the same, move me out of for...
            anchors_in_image = [
                anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps
            ]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors