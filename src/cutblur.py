# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import (
    check_dataaug_function_arg, sample_patch_dims, sample_patch_locations,
    gen_patch_mask, mix_augmented_images
)

def _check_random_cutblur_args(
    patch_area, patch_aspect_ratio, blur_factor, augmentation_ratio, bernoulli_mix):

    """
    Checks the arguments passed to the `random_cutblur` function
    """

    check_dataaug_function_arg(
        patch_area,
        context={'arg_name': 'patch_area', 'function_name' : 'random_cutblur'},
        constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
    )
    check_dataaug_function_arg(
        patch_aspect_ratio,
        context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'random_cutblur'},
        constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
    )
    check_dataaug_function_arg(
        blur_factor,
        context={'arg_name': 'blur_factor', 'function_name' : 'random_cutblur'},
        constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
    )
    check_dataaug_function_arg(
        augmentation_ratio,
        context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_cutblur'},
        constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    )
    if not isinstance(bernoulli_mix, bool):
        raise ValueError(
            'Argument `bernoulli_mix` of function `random_cutblur`: '
            f'expecting a boolean value\nReceived: {bernoulli_mix}'
        )


@tf.function
def random_cutblur(
        images: tf.Tensor,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        blur_factor: float = 0.1,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ) -> tf.Tensor:

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

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Values must be > 0.

        blur_factor:
            A float specifying the size ratio used to generate the low-resolution
            image. Smaller values produce stronger blur.

        augmentation_ratio:
            A float in the interval [0, 1] specifying the augmented/original
            images ratio in the output mix. If set to 0, no images are augmented.
            If set to 1, all the images are augmented.

        bernoulli_mix:
            A boolean specifying the method to use to mix the original and augmented
            images in the output images:
              - False: the augmented/original ratio is equal to `augmentation_ratio`
                for every batch. Augmented images are at random positions.
              - True: the augmented/original ratio varies stochastically from batch
                to batch with an expectation equal to `augmentation_ratio`.
            Augmented images are at random positions in the output mix.

    Returns:
        A tensor of the same shape and dtype as the input images, containing a mix
        of original and CutBlur-augmented images. Pixel values are in the same
        range as the input images.
    """

    # Check the arguments passed to the function
    _check_random_cutblur_args(
        patch_area, patch_aspect_ratio, blur_factor, augmentation_ratio, bernoulli_mix)

    original_image_shape = tf.shape(images)

    # Reshape images with shape [batch_size, height, width]
    # to shape [batch_size, height, width, 1]
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    image_shape = tf.shape(images)

    # Calculate the size of the low-resolution images
    image_size = image_shape[1:3]
    low_res_size = blur_factor * tf.cast(image_size, tf.float32)
    low_res_size = tf.cast(tf.round(low_res_size), tf.int32)

    # Downsize to the low resolution size, then upsize to the original size
    smaller_images = tf.image.resize(images, low_res_size)
    low_res_images = tf.image.resize(smaller_images, image_size)
    low_res_images = tf.cast(low_res_images, images.dtype)

    # Generate random patches
    patch_dims = sample_patch_dims(image_shape, patch_area, patch_aspect_ratio)
    patch_corners = sample_patch_locations(image_shape, patch_dims)
    patch_mask = gen_patch_mask(image_shape, patch_corners)

    # Erase the patches from the images and fill them with the low-res images
    images_aug = tf.where(patch_mask[..., None], low_res_images, images)

    # Mix the original and augmented images
    output_images, _ = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)

    # Restore input images shape
    output_images = tf.reshape(output_images, original_image_shape)

    return output_images


class RandomCutBlur(tf.keras.Layer):
    """
    This keras layer implements the "CutBlur" data augmentation technique.
    It is intended to be used as a preprocessing layer, similar to Tensorflow's 
    built-in layers such as RandomContrast, RandomFlip, etc.
    
    Refer to the docstring of the random_cutblur() function for an explanation
    of the parameters of the layer.
    """
    
    def __init__(self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        blur_factor: float = 0.1,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)

        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.blur_factor = blur_factor
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

    def call(self, inputs, training=None):
        return random_cutblur(
            inputs,
            patch_area=self.patch_area,
            patch_aspect_ratio=self.patch_aspect_ratio,
            blur_factor=self.blur_factor,
            augmentation_ratio=self.augmentation_ratio,
            bernoulli_mix=self.bernoulli_mix
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            'patch_area': self.patch_area,
            'patch_aspect_ratio': self.patch_aspect_ratio,
            'blur_factor': self.blur_factor,
            'augmentation_ratio': self.augmentation_ratio,
            'bernoulli_mix': self.bernoulli_mix
        }
        base_config.update(config)
        return base_config