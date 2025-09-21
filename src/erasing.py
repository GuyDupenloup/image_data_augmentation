# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import (
    check_dataaug_function_arg, rescale_pixel_values, sample_patch_dims,
    sample_patch_locations, gen_patch_mask, gen_patch_contents, mix_augmented_images
)


def _check_random_erasing_args(
    patch_area, patch_aspect_ratio, fill_method,
    pixels_range, augmentation_ratio, bernoulli_mix):

    """
    Checks the arguments passed to the `random_erasing` function
    """

    check_dataaug_function_arg(
        patch_area,
        context={'arg_name': 'patch_area', 'function_name' : 'random_erasing'},
        constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
    )

    check_dataaug_function_arg(
        patch_aspect_ratio,
        context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'random_erasing'},
        constraints={'format': 'tuple', 'min_val': ('>', 0)}
    )

    supported_fill_methods = ('black', 'gray', 'white', 'mean_per_channel', 'random', 'noise')
    if fill_method not in supported_fill_methods:
        raise ValueError(
            '\nArgument `fill_method` of function `random_erasing`: '
            f'expecting one of {supported_fill_methods}\n'
            f'Received: {fill_method}'
        )

    check_dataaug_function_arg(
        pixels_range,
        context={'arg_name': 'pixels_range', 'function_name' : 'random_erasing'},
        constraints={'format': 'tuple'}
    )

    check_dataaug_function_arg(
        augmentation_ratio,
        context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_erasing'},
        constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    )

    if not isinstance(bernoulli_mix, bool):
        raise ValueError(
            'Argument `bernoulli_mix` of function `random_erasing`: '
            f'expecting a boolean value\nReceived: {bernoulli_mix}'
        )


@tf.function
def random_erasing(
        images: tf.Tensor,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        fill_method: str = 'noise',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ) -> tf.Tensor:

    """
    Applies the "Random Erasing" data augmentation technique to a batch of images.

    Reference paper:
        Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang (2017).
        “Random Erasing Data Augmentation”.

    For each image in the batch:
    1. A patch area and aspect ratio are sampled from the specified ranges.
    2. The patch is placed at a random location, ensuring that it will be entirely
       contained inside the image.
    3. The patch is erased from the image and filled with solid color or noise.

    Patch areas, aspect ratios and locations are sampled independently for each image,
    ensuring variety across the batch.

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

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which the height/width 
            aspect ratios of patches are sampled from. Values must be > 0.

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
                for every batch. Augmented images are at random positions.
              - True: the augmented/original ratio varies stochastically from batch
                to batch with an expectation equal to `augmentation_ratio`.
            Augmented images are at random positions in the output mix.

    Returns:
        A tensor of the same shape and dtype as the input images, containing a mix
        of original and Random-Erasing-augmented images. Pixel values are in the same 
        range as the input images.
    """

    # Check the function arguments
    _check_random_erasing_args(
        patch_area,
        patch_aspect_ratio,
        fill_method,
        pixels_range,
        augmentation_ratio,
        bernoulli_mix
    )

    original_image_shape = tf.shape(images)

    # Reshape images with shape [B, H, W] to [B, H, W, 1]
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    image_shape = tf.shape(images)

    # Save images data type and rescale pixel values to (0, 255)
    pixels_dtype = images.dtype
    images = rescale_pixel_values(images, pixels_range, (0, 255), dtype=tf.int32)

    # Sample patch heights and widths
    patch_dims = sample_patch_dims(image_shape, patch_area, patch_aspect_ratio)

    # Sample patch locations, then generate a boolean mask
    # with value True inside patches, False outside
    patch_corners = sample_patch_locations(image_shape, patch_dims)
    patch_mask = gen_patch_mask(image_shape, patch_corners)

    # Generate color contents of patches
    patch_contents = gen_patch_contents(images, fill_method)

    # Erase the patches from the images and fill them
    images_aug = tf.where(patch_mask[:, :, :, None], patch_contents, images)

    # Mix the original and augmented images
    output_images, _ = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)

    # Restore shape, data type and pixels range of input images
    output_images = tf.reshape(output_images, original_image_shape)
    output_images = rescale_pixel_values(output_images, (0, 255), pixels_range, dtype=pixels_dtype)

    return output_images
 

class RandomErasing(tf.keras.Layer):
    """
    This keras layer implements the "Random Erasing" data augmentation
    technique. It is intended to be used as a preprocessing layer, similar
    to Tensorflow's built-in layers such as RandomContrast, RandomFlip, etc.
    
    Refer to the docstring of the random_erasing() function for an explanation
    of the parameters of the layer.
    """

    def __init__(self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        fill_method: str = 'noise',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)
        
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.fill_method = fill_method
        self.pixels_range = pixels_range
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix
        
    def call(self, inputs, training=None):
        return random_erasing(
            inputs,
            patch_area=self.patch_area,
            patch_aspect_ratio=self.patch_aspect_ratio,
            fill_method=self.fill_method,
            pixels_range=self.pixels_range,
            augmentation_ratio=self.augmentation_ratio,
            bernoulli_mix=self.bernoulli_mix
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "patch_area": self.patch_area,
            "patch_aspect_ratio": self.patch_aspect_ratio,
            "fill_method": self.fill_method,
            "pixels_range": self.pixels_range,
            "augmentation_ratio": self.augmentation_ratio,
            "bernoulli_mix": self.bernoulli_mix
        }       
        base_config.update(config)
        return base_config
    