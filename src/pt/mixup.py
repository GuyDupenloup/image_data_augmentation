# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from argument_utils import check_argument, check_augment_mix_args
from dataaug_utils import mix_augmented_images


class RandomMixup(v2.Transform):
    """
    Applies the "Mixup" data augmentation technique to a batch of images.

    Reference paper:
        Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz (2017). 
        "Mixup: Beyond Empirical Risk Minimization".

    For each image in the batch:
        1. Sample a blending coefficient `lambda` from a Beta distribution.
        2. Randomly select another image from the batch.
        3. Blend the two images together using the blending coefficient.
        4. Adjust the labels of the image using `lambda` to reflect the proportion
           of pixels contributed by both images.

    Blending coefficients are sampled independently for each image, ensuring variety 
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

        alpha:
            A positive float specifying the parameter `alpha` of the Beta distribution 
            from which blending coefficients `lambda` are sampled.
            If `alpha` is equal to 1.0 (default value), the distribution is uniform.

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
        alpha: float = 1.0,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.transform_name = 'RandomMixup'
        self.alpha = alpha
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        # Check that all the arguments are valid
        self._check_transform_args()


    def _check_transform_args(self):
        """
        Checks that the arguments passed to the transform are valid
        """
        check_argument(
            self.alpha,
            context={'arg_name': 'alpha', 'caller_name' : self.transform_name},
            constraints={'format': 'number'}
        )
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, data) -> torch.Tensor:

        images, labels = data
        original_shape = images.shape

        # Reshape images with shape [B, H, W] to [B, 1, H, W]
        if images.ndim == 3:
            images = images.unsqueeze(1)

        batch_size = images.shape[0]

        # Sample lambda values from Beta distribution
        gamma1 = torch.distributions.Gamma(self.alpha, 1.0).sample([batch_size])
        gamma2 = torch.distributions.Gamma(self.alpha, 1.0).sample([batch_size])
        lambda_vals = gamma1 / (gamma1 + gamma2)

        # Randomly select other images in the batch
        shuffle_indices = torch.randperm(batch_size, device=images.device)
        shuffled_images = images[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]

        # Blend the images together
        lambda_vals = lambda_vals.reshape([batch_size, 1, 1, 1])
        images_aug = lambda_vals * images + (1 - lambda_vals) * shuffled_images

        # Weigh the labels
        lambda_vals = lambda_vals.reshape([batch_size, 1])
        labels = labels.to(torch.float32)
        labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

        # Mix original/augmented images and labels
        output_images, augment_mask = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)
        output_labels = torch.where(augment_mask[:, None], labels_aug, labels)

        # Restore shape of input images
        output_images = output_images.reshape(original_shape)

        return output_images, output_labels
    