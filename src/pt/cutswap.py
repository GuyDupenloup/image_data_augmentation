
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_argument, check_augment_mix_args
from dataaug_utils import sample_patch_dims, sample_patch_locations, gen_patch_mask, mix_augmented_images


class RandomCutSwap(v2.Transform):
    """
   Applies the "CutSwap" data augmentation technique to a batch of images.
    
    Reference paper:
        Jianjian Qin, Chunzhi Gu, Jun Yu, and Chao Zhang (2013). "Multilevel 
        Saliency-Guided Self-Supervised Learning for Image Anomaly Detection"

    For each image in the batch, the function:
    1. Samples a patch size based on the specified area and aspect ratio ranges.
    2. Chooses random locations in the image for two patches of the sampled size.
       The two patches may overlap.
    3. Swaps the contents of the two patches.

    Patch sizes and locations are sampled independently for each image, ensuring 
    variety across the batch.

    By default, the augmented/original image ratio in the output mix is `1.0`. 
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images

        patch_area:
            A tuple of two floats specifying the range from which patch areas
            are sampled. Values must be > 0 and < 1, representing fractions 
            of the image area.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch 
            height/width aspect ratios are sampled. Values must be > 0.

        augmentation_ratio:
            A float in the interval [0, 1] specifying the augmented/original
            images ratio in the output mix. If set to 0, no images are augmented.
            If set to 1, all the images are augmented.

        bernoulli_mix:
            A boolean specifying the method to use to mix the original and augmented
            images in the output images:
              - False: the augmented/original ratio is equal to `augmentation_ratio`
                for every batch.
              - True: the augmented/original ratio varies stochastically from batch
                to batch with an expectation equal to `augmentation_ratio`.
            Augmented images are at random positions in the output mix.

    Returns:
        A tensor of the same shape and dtype as the input images, containing a mix
        of original and CutSwap-augmented images.
    """

    def __init__(
        self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        alpha: float = 1.0,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.transform_name = 'RandomCutSwap'
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.alpha = alpha
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_transform_args()
		

    def _check_transform_args(self):
        """
        Checks that the arguments passed to the transform are valid
        """
        check_argument(
            self.patch_area,
            context={'arg_name': 'patch_area', 'caller_name' : self.transform_name},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_argument(
            self.patch_aspect_ratio,
            context={'arg_name': 'patch_aspect_ratio', 'caller_name' : self.transform_name},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
        )
        check_argument(
            self.alpha,
            context={'arg_name': 'alpha', 'caller_name' : self.transform_name},
            constraints={'min_val': ('>', 0)}
        )
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        original_image_shape = images.shape
        device = images.device  # keep track of input device

        # Reshape grayscale images [B,H,W] -> [B,1,H,W]
        if images.ndim == 3:
            images = images.unsqueeze(1)

        # ---- Sample patches
        patch_dims = sample_patch_dims(images, self.patch_area, self.patch_aspect_ratio, self.alpha)

        # Sample first patches and generate mask
        corners_1 = sample_patch_locations(images, patch_dims)
        mask_1 = gen_patch_mask(images, corners_1)
        if mask_1.ndim == 3:
            mask_1 = mask_1.unsqueeze(1)  # [B,1,H,W]
        mask_1 = mask_1.to(device)

        # Sample second patches and generate mask
        corners_2 = sample_patch_locations(images, patch_dims)
        mask_2 = gen_patch_mask(images, corners_2)
        if mask_2.ndim == 3:
            mask_2 = mask_2.unsqueeze(1)  # [B,1,H,W]
        mask_2 = mask_2.to(device)

        # ---- Gather patch contents (all channels)
        indices_1 = mask_1.nonzero(as_tuple=False)  # [num_pixels, 4]
        contents_1 = images[indices_1[:, 0], :, indices_1[:, 2], indices_1[:, 3]]  # [num_pixels, C]

        indices_2 = mask_2.nonzero(as_tuple=False)
        contents_2 = images[indices_2[:, 0], :, indices_2[:, 2], indices_2[:, 3]]

        # ---- Swap patches
        images_aug = images.clone()
        images_aug[indices_1[:, 0], :, indices_1[:, 2], indices_1[:, 3]] = contents_2
        images_aug[indices_2[:, 0], :, indices_2[:, 2], indices_2[:, 3]] = contents_1

        # ---- Mix original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)
        output_images = output_images.to(device)

        # ---- Restore original shape
        output_images = output_images.reshape(original_image_shape)

        return output_images

