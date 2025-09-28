# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_patch_sampling_args, check_augment_mix_args
from dataaug_utils import gen_patch_sizes, gen_patch_mask, mix_augmented_images


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
            Patch areas are sampled from a Beta distribution. See `alpha` argument.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Minimum value must be > 0.
            Patch aspect ratios are sampled from a uniform distribution.

        alpha:
            A float specifying the alpha parameter of the Beta distribution used
            to sample patch areas. Set to 0 by default, making the distribution
            uniform.

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
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.alpha, self.transform_name)
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        original_image_shape = images.shape
        device = images.device  # keep track of input device

        # Reshape grayscale images [B,H,W] -> [B,1,H,W]
        if images.ndim == 3:
            images = images.unsqueeze(1)

        # Sample patch sizes
        patch_sizes = gen_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)

        # Sample locations of the 1st set of patches, then generate
        # a boolean mask (True inside patches, False outside)
        mask_1, _ = gen_patch_mask(images, patch_sizes)
        mask_1 = mask_1.unsqueeze(1)  # Add channel dim

        # Sample locations of the 2nd set of patches, then generate
        # a boolean mask (True inside patches, False outside)
        mask_2, _ = gen_patch_mask(images, patch_sizes)
        mask_2 = mask_2.unsqueeze(1)  # Add channel dim

        # Gather contents of the 1st set of patches
        indices_1 = mask_1.nonzero(as_tuple=False)  # [num_pixels, 4]
        contents_1 = images[indices_1[:, 0], :, indices_1[:, 2], indices_1[:, 3]]  # [num_pixels, C]

        # Gather contents of the 2nd set of patches
        indices_2 = mask_2.nonzero(as_tuple=False)
        contents_2 = images[indices_2[:, 0], :, indices_2[:, 2], indices_2[:, 3]]

        # Swap patches
        images_aug = images.clone()
        images_aug[indices_1[:, 0], :, indices_1[:, 2], indices_1[:, 3]] = contents_2
        images_aug[indices_2[:, 0], :, indices_2[:, 2], indices_2[:, 3]] = contents_1

        # Mix original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)
        output_images = output_images.to(device)

        # Restore shape of input images
        output_images = output_images.reshape(original_image_shape)

        return output_images

