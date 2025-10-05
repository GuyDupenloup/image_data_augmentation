
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from argument_utils import check_argument, check_patch_sampling_args, check_augment_mix_args
from dataaug_utils import gen_patch_sizes, gen_patch_mask, mix_augmented_images


class RandomCutMix(v2.Transform):
    """
    Applies the "CutMix" data augmentation technique to a batch of images.

    Reference paper:
        Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe,
        Youngjoon Yoo (2019). "CutMix: Regularization Strategy to Train Strong
        Classifiers with Localizable Features."

    For each image in the batch:
        1. Sample a mixing coefficient `lambda` from a Beta distribution.
        2. Compute a patch size using `lambda` as a fraction of the image size,
           and an aspect ratio sampled from the specified range.
        3. Choose a random location for the patch.
        4. Randomly select another image from the batch and crop the patch from it.
        5. Paste the patch into the image.
        6. Update the label of the image using `lambda` to reflect the proportion
           of pixels contributed by both images.

    Lambda values, patch aspect ratios and patch locations are sampled independently 
    for each image, ensuring variety across the batch.

    By default, the augmented/original image ratio in the output mix is `1.0`. 
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images

        labels:
            Labels for the input images. Must be **one-hot encoded**.
            Data type should be tf.float32 (will be cast if not).

        alpha:
            A positive float specifying the parameter `alpha` of the Beta distribution 
            from which `lambda` values are sampled. Controls patch size variability.
            If `alpha` is equal to 1.0 (default value), the distribution is uniform.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which the height/width 
            aspect ratios of patches are sampled from. Values must be > 0.

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
        A tuple `(output_images, output_labels)` where:
            output_images:
                A tensor of the same shape and dtype as `images`, containing a
                mix of original and Cutmix-augmented images.
            output_labels:
                A tensor of the same shape as `labels`, containing the
                correspondingly mixed labels.
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
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.transform_name)
        check_argument(
            self.alpha,
            context={'arg_name': 'alpha', 'caller_name': self.transform_name},
            constraints={'min_val': ('>', 0)}
        )
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, data) -> torch.Tensor:

        images, labels = data
        original_shape = images.shape

        # Reshape images with shape [B, H, W] to [B, 1, H, W]
        if images.ndim == 3:
            images = images.unsqueeze(1)

        batch_size, _, img_height, img_width = images.shape

        # Get patch sizes and generate boolean mask (True inside patches)
        patch_sizes = gen_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)
        patch_mask, _ = gen_patch_mask(images, patch_sizes)

        # Randomly select other images in the batch
        shuffle_indices = torch.randperm(batch_size, device=images.device)
        shuffled_images = images[shuffle_indices]

        # Fill patches with contents of patches from the other images
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
    