
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from argument_utils import check_patch_sampling_args, check_augment_mix_args
from dataaug_utils import gen_patch_sizes, gen_patch_mask, mix_augmented_images


class RandomCutMix(v2.Transform):
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
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        alpha: float = 1.0,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.transform_name = 'RandomCutMix'
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.alpha = alpha
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        # Check that all the arguments are valid
        self._check_transform_args()


    def _check_transform_args(self):
        """
        Checks that the arguments passed to the transform are valid
        """
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.alpha, self.transform_name)
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, data) -> torch.Tensor:

        images, labels = data

        original_shape = images.shape

        # Reshape grayscale images [B, H, W] -> [B, H, W, 1]
        if images.ndim == 3:
            images = images.unsqueeze(-1)  # [B, H, W, 1]

        batch_size, _, img_height, img_width = images.shape

        # Sample patch sizes and locations, then generate 
        # a boolean mask (True inside patches, False outside)
        patch_sizes = gen_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)
        patch_mask, _ = gen_patch_mask(images, patch_sizes)

        # Randomly select other images in the batch
        shuffle_indices = torch.randperm(batch_size, device=images.device)
        shuffled_images = images[shuffle_indices]

        # Paste the patches from the shuffled images
        images_aug = torch.where(patch_mask[:, None, :, :], shuffled_images, images)

        # Compute lambda values based on actual patch sizes
        img_area = img_height * img_width
        patch_areas = patch_sizes[0].float() * patch_sizes[1].float()
        lambda_vals = 1.0 - (patch_areas / img_area)

        # Update labels
        labels = labels.float()
        shuffled_labels = labels[shuffle_indices]
        lambda_vals = lambda_vals[:, None]    # For broadcasting
        labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

        # Mix original/augmented images and labels
        output_images, augment_mask = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)
        output_labels = torch.where(augment_mask[:, None], labels_aug, labels)

        # Restore shape of input images
        output_images = output_images.reshape(original_shape)

        return output_images, output_labels
    