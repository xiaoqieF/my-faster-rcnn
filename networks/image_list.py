from torch import Tensor

class ImageList(object):
    def __init__(self, tensors, image_sizes) -> None:
        """
        Args:
            tensors (Tensor): a single tensor includes minibatch of images
            image_sizes (List[tuple[int, int]]): size of each image (not include padding zeros)
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)