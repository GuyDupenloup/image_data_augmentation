# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import (
    check_dataaug_function_arg, sample_patch_dims, sample_patch_locations,
    gen_patch_mask, mix_augmented_images
)


class RandomCutPaste(tf.keras.Layer):
    """
    Applies the "CutPaste" data augmentation technique to a batch of images.

    Reference paper:
        Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister (2021).
        "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization."

    For each image in the batch, the function:
    1. Samples a patch size based on the specified area and aspect ratio ranges.
    2. Chooses two random locations in the image for a source patch and a target
       patch, both with the sampled size. The two patches may overlap.
    3. Copies the contents of the source patch into the target patch, leaving 
       the source region unchanged.

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
        A tensor of the same shape and dtype as the input images, containing a mix
        of original and CutPaste-augmented images.
    """
    
    def __init__(self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)

        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_arguments()


    def _check_arguments(self):
        """
        Checks the arguments passed to the `random_cutpaste` function
        """

        check_dataaug_function_arg(
            self.patch_area,
            context={'arg_name': 'patch_area', 'function_name' : 'random_cutpaste'},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_dataaug_function_arg(
            self.patch_aspect_ratio,
            context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'random_cutpaste'},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
        )
        check_dataaug_function_arg(
            self.augmentation_ratio,
            context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_cutpaste'},
            constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
        )
        if not isinstance(self.bernoulli_mix, bool):
            raise ValueError(
                'Argument `bernoulli_mix` of function `random_cutpaste`: '
                f'expecting a boolean value\nReceived: {self.bernoulli_mix}'
            )


    def call(self, images, training=None):

        original_image_shape = tf.shape(images)

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if images.shape.rank == 3:
            images = tf.expand_dims(images, axis=-1)
        image_shape = tf.shape(images)

        # Sample patch sizes
        patch_dims = sample_patch_dims(image_shape, self.patch_area, self.patch_aspect_ratio)

        # Sample locations for the source patches and generate
        # a boolean mask (True inside the patches, False outside)
        source_corners = sample_patch_locations(image_shape, patch_dims)
        source_mask = gen_patch_mask(image_shape, source_corners)

        # Sample locations for the target patches and generate
        # a boolean mask (True inside the patches, False outside)
        target_corners = sample_patch_locations(image_shape, patch_dims)
        target_mask = gen_patch_mask(image_shape, target_corners)

        # Gather the contents of the source patches
        source_indices = tf.where(source_mask)
        source_contents = tf.gather_nd(images, source_indices)

        # Update the contents of the target patches with the contents of the source patches
        target_indices = tf.where(target_mask)
        images_aug = tf.tensor_scatter_nd_update(images, target_indices, source_contents)

        # Mix the original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore input images shape
        output_images = tf.reshape(output_images, original_image_shape)

        return output_images


    def get_config(self):
        base_config = super().get_config()
        config = {
            'patch_area': self.patch_area,
            'patch_aspect_ratio': self.patch_aspect_ratio,
            'augmentation_ratio': self.augmentation_ratio,
            'bernoulli_mix': self.bernoulli_mix
        }
        base_config.update(config)
        return base_config