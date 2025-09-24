# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import (
    check_dataaug_function_arg, sample_patch_dims, sample_patch_locations,
    gen_patch_mask, mix_augmented_images
)

def _check_random_cutswap_args(
    patch_area, patch_aspect_ratio, augmentation_ratio, bernoulli_mix):

    """
    Checks the arguments passed to the `random_cutswap` function
    """

    check_dataaug_function_arg(
        patch_area,
        context={'arg_name': 'patch_area', 'function_name' : 'random_cutswap'},
        constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
    )
    check_dataaug_function_arg(
        patch_aspect_ratio,
        context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'random_cutswap'},
        constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
    )
    check_dataaug_function_arg(
        augmentation_ratio,
        context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_cutswap'},
        constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    )
    if not isinstance(bernoulli_mix, bool):
        raise ValueError(
            'Argument `bernoulli_mix` of function `random_cutswap`: '
            f'expecting a boolean value\nReceived: {bernoulli_mix}'
        )


@tf.function
def random_cutswap(
        images: tf.Tensor,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ) -> tf.Tensor:

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

    # Check the arguments passed to the function
    _check_random_cutswap_args(
        patch_area, patch_aspect_ratio, augmentation_ratio, bernoulli_mix)

    original_image_shape = tf.shape(images)

    # Reshape grayscale images with shape [batch_size, height, width]
    # to shape [batch_size, height, width, 1]
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    image_shape = tf.shape(images)

    # Sample the dimensions of the patches
    patch_dims = sample_patch_dims(image_shape, patch_area, patch_aspect_ratio)

    # Sample locations for the first patches and generate
    # a boolean mask (True inside the patches, False outside)
    corners_1 = sample_patch_locations(image_shape, patch_dims)
    mask_1 = gen_patch_mask(image_shape, corners_1)

    # Sample locations for the second patches and generate
    # a boolean mask (True inside the patches, False outside)
    corners_2 = sample_patch_locations(image_shape, patch_dims)
    mask_2 = gen_patch_mask(image_shape, corners_2)

    # Gather the contents of the first patches
    indices_1 = tf.where(mask_1)
    contents_1 = tf.gather_nd(images, indices_1)

    # Gather the contents of the second patches
    indices_2 = tf.where(mask_2)
    contents_2 = tf.gather_nd(images, indices_2)

    # Swap the contents of the first and second patches
    images_aug = tf.tensor_scatter_nd_update(images, indices_1, contents_2)
    images_aug = tf.tensor_scatter_nd_update(images_aug, indices_2, contents_1)

    # Mix the original and augmented images
    output_images, _ = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)

    # Restore input images shape
    output_images = tf.reshape(output_images, original_image_shape)

    return output_images


class RandomCutSwap(tf.keras.Layer):
    """
    This keras layer implements the "CutSwap" data augmentation 
    technique. It is intended to be used as a preprocessing layer,
    similar to Tensorflow's built-in layers such as RandomContrast,
    RandomFlip, etc.
    
    Refer to the docstring of the random_cutswap() function for 
    an explanation of the parameters of the layer.
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

    def call(self, inputs, training=None):
        return random_cutswap(
            inputs,
            patch_area=self.patch_area,
            patch_aspect_ratio=self.patch_aspect_ratio,
            augmentation_ratio=self.augmentation_ratio,
            bernoulli_mix=self.bernoulli_mix
        )

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