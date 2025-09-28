
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import (
    check_argument, check_augment_mix_args, check_fill_method_arg, check_pixels_range_args
)
from dataaug_utils import (
    sample_patch_sizes, sample_patch_locations, gen_patch_mask, 
    gen_patch_contents, mix_augmented_images, rescale_pixel_values
)


class RandomCutout(v2.Transform):
    """
    Applies the "Cutout" data augmentation technique to a batch of images.

    Reference paper:
        Terrance DeVries, Graham W. Taylor (2017). “Improved Regularization 
        of Convolutional Neural Networks with Cutout”.

    For each image in the batch, a square patch centered at a random location
    is erased and filled with solid color or noise. All the patches have 
    the same size and are entirely contained inside the images.

    Patch locations are sampled independently for each image, ensuring variety 
    across the batch.

    Args:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images
            Pixel values are expected to be in the range specified by 
            the `pixels_range` argument.

        patch_area:
            A float specifying the patch area as a fraction of the image area.
            Values must be > 0 and < 1.

        fill_method:
            A string specifying how to fill the erased patches.  
            Options:  
            - 'black': filled with black  
            - 'gray': filled with mid-gray (128)  
            - 'white': filled with white  
            - 'mean_per_channel': filled with the mean color of the image channels  
            - 'random': filled with random solid colors  
            - 'noise': filled with random pixel values

        pixels_range:
            Tuple or list of two numbers specifying the expected min and max
            pixel values in the input images, e.g. (0, 255), (0, 1), (-1, 1).
            This ensures the fill values are scaled correctly.

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
        of original and Cutout-augmented images. Pixel values are in the same range
        as the input images.
    """

    def __init__(
        self,
        patch_area: float = 0.1,
        fill_method: str = 'black',
        pixels_range: Tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
    ):
        super().__init__()

        self.transform_name = 'RandomCutout'
        self.patch_area = patch_area
        self.fill_method = fill_method
        self.pixels_range = pixels_range
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        # Check that all the arguments are valid
        self._check_transform_args()


    def _check_transform_args(self):
        """
        Checks that the arguments passed to the transform are valid
        """
        check_argument(
            self.patch_area,
            context={'arg_name': 'patch_area', 'caller_name' : self.transform_name},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_fill_method_arg(self.fill_method, self.transform_name)
        check_pixels_range_args(self.pixels_range, self.transform_name)
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    # def forward(self, images: torch.Tensor) -> torch.Tensor:
    def forward(self, images) -> torch.Tensor:

        original_image_shape = images.shape
        if images.ndim == 3:  # i.e., [B, H, W]
            images = images.unsqueeze(1)  # insert a channel dimension at index 1

        pixels_dtype = images.dtype
        images = rescale_pixel_values(images, self.pixels_range, (0, 255), dtype=torch.int32)

        # Calculate patch sizes (same in images, sample locations,  
        # and generate a boolean mask (True inside patches, False outside)
        patch_sizes = sample_patch_sizes(images, patch_area=self.patch_area, patch_aspect_ratio=1.0, alpha=1.0)
        patch_corners = sample_patch_locations(images, patch_sizes)

        # Generate boolean mask (True inside patches, False outside)
        patch_mask = gen_patch_mask(images, patch_corners)

        # Generate color contents of patches
        patch_contents = gen_patch_contents(images, self.fill_method)

        # Apply mask correctly - patch_mask is [B, H, W], need to broadcast over channel dim
        images_aug = torch.where(patch_mask[:, None, :, :], patch_contents, images)

        # Mix the original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape, data type and pixels range of input images
        output_images = output_images.reshape(original_image_shape)
        output_images = rescale_pixel_values(output_images, (0, 255), self.pixels_range, dtype=pixels_dtype)

        return output_images
