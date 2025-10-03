# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_patch_sampling_args, check_argument, check_augment_mix_args
from dataaug_utils import gen_patch_sizes, gen_patch_mask, mix_augmented_images


class RandomCutBlur(v2.Transform):
    """
    Applies the "CutBlur" data augmentation technique to a batch of images.

    Reference paper:
        Jaejun Yoo, Namhyuk Ahn, and Kyung-Ah Sohn (2020). "Rethinking Data Augmentation 
        for Image Super-Resolution: : A comprehensive analysis and a new strategy".

    For each image in the batch:
    1. A low-resolution version of the image is created by downscaling and then
       upscaling it back to the original size.
    2. A random rectangular patch is cropped from the low-resolution image.
    3. The patch is pasted at the same location in the original image, producing
       an image identical to the original except for the blurred region.

    Patch sizes and locations are sampled independently for each image, ensuring 
    variety across the batch.

    By default, the augmented/original image ratio in the output mix is `1.0`. 
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.  
            Supported shapes:  
                [B, H, W, 3]  -->  Color  
                [B, H, W, 1]  -->  Grayscale  
                [B, H, W]     -->  Grayscale  

        patch_area:
            A tuple of two floats specifying the range from which patch areas 
            are sampled. Values must be > 0 and < 1, representing fractions 
            of the image area.
            Patch areas are sampled from a Beta distribution with shape parameters
            `alpha` and beta=1.0. By default, `alpha` is 1.0 making the distribution
            uniform.
            
        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Minimum value must be > 0.
            Patch aspect ratios are sampled from a uniform distribution.

        alpha:
            A float specifying the alpha parameter of the Beta distribution used
            to sample patch areas. Set to 0 by default, making the distribution
            uniform.

        blur_factor:
            A float specifying the size of the low-resolution images (they all 
            have the same size). Values must be > 0 and < 1, representing fractions 
            of the image size. Smaller values produce stronger blur.

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
        of original and CutBlur-augmented images. Pixel values are in the same
        range as the input images.
    """

    def __init__(
        self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        alpha: float = 1.0,
        blur_factor: float = 0.1,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.transform_name = 'RandomCutBlur'
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.alpha = alpha
        self.blur_factor = blur_factor
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_transform_args()
		

    def _check_transform_args(self):
        """
        Checks that the arguments passed to the transform are valid
        """
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.alpha, self.transform_name)
        check_argument(
            self.blur_factor,
            context={'arg_name': 'blur_factor', 'caller_name' : self.transform_name},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        device = images.device
        original_image_shape = images.shape

        # Reshape images with shape [B, H, W] to [B, 1, H, W]
        if images.ndim == 3:
            images = images.unsqueeze(1)
        
        # Low-res size
        original_size = images.shape[2:]
        image_size = torch.tensor(images.shape[2:], dtype=torch.float32, device=device)
        low_res_size = self.blur_factor * image_size
        low_res_size = torch.round(low_res_size).to(torch.int64)

        # Downsize then up-size images to get blurred images
        smaller_images = F.interpolate(images, size=tuple(low_res_size.tolist()), mode='bilinear', align_corners=False)
        low_res_images = F.interpolate(smaller_images, size=original_size, mode='bilinear', align_corners=False)

        # Match dtype
        low_res_images = low_res_images.to(images.dtype)

        # Get patch sizes and generate boolean mask (True inside patches)
        patch_sizes= gen_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)
        patch_mask, _ = gen_patch_mask(images, patch_sizes)

        # Fill patches with the low-res images
        images_aug = torch.where(patch_mask[:, None, :, :], low_res_images, images)

        # Mix the original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape of input images
        output_images = output_images.reshape(original_image_shape)

        return output_images
    