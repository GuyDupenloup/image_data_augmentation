# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import check_dataaug_function_arg, rescale_pixel_values, mix_augmented_images


def _check_random_hide_and_seek_args(
        erased_patches, grid_size, fill_method,
        pixels_range, augmentation_ratio, bernoulli_mix):

    """
    Checks the arguments passed to the `random_hide_and_seek` function
    """

    check_dataaug_function_arg(
        grid_size,
        context={'arg_name': 'grid_size', 'function_name' : 'random_hide_and_seek'},
        constraints={'format': 'tuple', 'tuple_ordering': 'None', 'data_type': 'int', 'min_val': ('>', 0)}
    )

    check_dataaug_function_arg(
        erased_patches,
        context={'arg_name': 'erased_patches', 'function_name' : 'random_hide_and_seek'},
        constraints={'format': 'tuple', 'data_type': 'int', 'min_val': ('>=', 0)}
    )

    supported_fill_methods = ('black', 'gray', 'white', 'mean_per_channel', 'random', 'noise')
    if fill_method not in supported_fill_methods:
        raise ValueError(
            '\nArgument `fill_method` of function `random_hide_and_seek`: '
            f'expecting one of {supported_fill_methods}\n'
            f'Received: {fill_method}'
        )

    check_dataaug_function_arg(
        pixels_range,
        context={'arg_name': 'pixels_range', 'function_name' : 'random_hide_and_seek'},
        constraints={'format': 'tuple'}
    )

    check_dataaug_function_arg(
        augmentation_ratio,
        context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_hide_and_seek'},
        constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    )

    if not isinstance(bernoulli_mix, bool):
        raise ValueError(
            'Argument `bernoulli_mix` of function `random_hide_and_seek`: '
            f'expecting a boolean value\nReceived: {bernoulli_mix}'
        )


def _gen_hs_patch_mask(image_shape, erased_patches, grid_size):
    """
    Generates a boolean mask that will be used to erase the patches from 
    the images. Value is True inside areas to erase, False outside.
    """

    batch_size, img_height, img_width = tf.unstack(image_shape)

    # Sample the coordinates (x, y) of the grid points where
    # the corresponding patches in the image will be erased
    num_patches = tf.random.uniform([batch_size], minval=erased_patches[0], maxval=erased_patches[1] + 1, dtype=tf.int32)

    # Don't erase more patches than total available (for robustness)
    num_patches = tf.minimum(num_patches, grid_size[0] * grid_size[1])

    indices = tf.argsort(tf.random.uniform([batch_size, grid_size[0] * grid_size[1]]))

    grid_mask = tf.where(indices < num_patches[:, None], True, False)
    grid_mask = tf.reshape(grid_mask, [batch_size, grid_size[0], grid_size[1]])

    # Calculate width and height of patches
    patch_h = tf.cast(img_height / grid_size[0], tf.int32)
    patch_w = tf.cast(img_width / grid_size[1], tf.int32)
    patch_size = tf.stack([patch_h, patch_w], axis=-1)

    # Fill patches with the mask values of the corresponding grid points
    patch_mask = tf.repeat(grid_mask, repeats=patch_h + 1, axis=1)
    patch_mask = tf.repeat(patch_mask, repeats=patch_w + 1, axis=2)

    # Truncate mask to image size
    patch_mask = patch_mask[:, :img_height, :img_width]

    return patch_mask, patch_size


def _gen_hs_patch_contents(images, grid_size, patch_size, fill_method):

    """
    This function generates the color contents of the erased patches,
    accordingly to the specified fill method: random color, uniform color,
    or noise.
    It outputs a tensor with shape [batch_size, img_width, img_height, 3].
    """

    image_shape = tf.shape(images)
    batch_size, img_height, img_width, img_channels = tf.unstack(image_shape)

    if fill_method == 'black':
        contents = tf.zeros(image_shape, dtype=tf.int32)

    elif fill_method == 'gray':
        contents = tf.fill(image_shape, 128)

    elif fill_method == 'white':
        contents = tf.fill(image_shape, 255)

    elif fill_method == 'mean_per_channel':
        channel_means = tf.reduce_mean(images, axis=[1, 2])
        contents = tf.broadcast_to(channel_means[:, None, None, :], image_shape)

    elif fill_method == 'random':
        color_grid = tf.random.uniform(
            [batch_size, grid_size[0], grid_size[1], img_channels],
            minval=0, maxval=256,
            dtype=tf.int32
        )
        # Fill patches with the color of the corresponding grid points
        contents = tf.repeat(color_grid, repeats=patch_size[0], axis=1)
        contents = tf.repeat(contents, repeats=patch_size[1], axis=2)
        contents = contents[:, :img_height, :img_width]

    elif fill_method == 'noise':
        contents = tf.random.uniform(image_shape, minval=0, maxval=256, dtype=tf.int32)

    return contents


@tf.function
def random_hide_and_seek(
        images: tf.Tensor,
        grid_size: tuple[int, int] = (8, 8),
        erased_patches: tuple[int, int] = (0, 5),
        fill_method: str = 'black',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ) -> tf.Tensor:

    """
    Applies the "Hide-and-Seek" data augmentation technique to a batch of images.

    Reference paper:
        Krishna Kumar Singh, Hao Yu, Aron Sarmasi, Gautam Pradeep, Yong Jae Lee (2018).
        “Hide-and-Seek: A Data Augmentation Technique for Weakly-Supervised Localization 
        and Beyond”.

    For each image in the batch:
    - The image is divided into a regular grid of patches.
    - A random number of patches is sampled from the specified range.
    - That number of patches are erased from the image at random 
      locations in the grid.
    - The erased patches are filled with solid color or noise.
    
    The number of patches to erase and their locations in the grid are sampled
    independently for each image, ensuring variety across the batch.

    Args:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images
            Pixel values are expected to be in the range specified by 
            the `pixels_range` argument.

        grid_size:
            A tuple of two positive integers specifying the number of patches
            of the grid in each direction. Columns are first, rows second.

        erased_patches:
            A tuple of two integers specifying the range from which to sample 
            the number of patches to erase.
            The minimum value can be 0. For example, (0, 5) means that 0 to 5
            patches can be erased.

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
        of original and Hide-and-Seek-augmented images. Pixel values are in the same 
        range as the input images.
    """

    # Check the arguments passed to the function
    _check_random_hide_and_seek_args(
        erased_patches, grid_size, fill_method,
        pixels_range, augmentation_ratio, bernoulli_mix)

    original_image_shape = tf.shape(images)

    # Reshape images with shape [B, H, W] to [B, H, W, 1]
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    image_shape = tf.shape(images)

    # Save images data type and rescale pixel values to (0, 255)
    pixels_dtype = images.dtype
    images = rescale_pixel_values(images, pixels_range, (0, 255), dtype=tf.int32)

    # Generate a boolean mask with value True inside the areas to erase, False outside
    patch_mask, patch_size = _gen_hs_patch_mask(image_shape[:3], erased_patches, grid_size)

    # Generate the color contents of the erased patches
    patch_contents = _gen_hs_patch_contents(images, grid_size, patch_size, fill_method)

    # Erase the patches from the images and fill them
    images_aug = tf.where(patch_mask[:, :, :, None], patch_contents, images)

    # Mix the original and augmented images
    output_images, _ = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)

    # Restore shape, data type and pixels range of input images
    output_images = tf.reshape(output_images, original_image_shape)
    output_images = rescale_pixel_values(output_images, (0, 255), pixels_range, dtype=pixels_dtype)

    return output_images


class RandomHideAndSeek(tf.keras.Layer):
    """
    This keras layer implements the "Hide-and-Seek" data augmentation 
    technique. It is intended to be used as a preprocessing layer,
    similar to Tensorflow's built-in layers such as RandomContrast,
    RandomFlip, etc.
    
    Refer to the docstring of the random_hide_and_seek() function for 
    an explanation of the parameters of the layer.
    """

    def __init__(self,
        grid_size: tuple[int, int] = (8, 8),
        erased_patches: tuple[int, int] = (0, 5),
        fill_method: str = 'black',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)

        self.grid_size = grid_size
        self.erased_patches = erased_patches
        self.fill_method = fill_method
        self.pixels_range = pixels_range
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

    def call(self, inputs, training=None):
        return random_hide_and_seek(
            inputs,
            grid_size=self.grid_size,
            erased_patches=self.erased_patches,
            fill_method=self.fill_method,
            pixels_range=self.pixels_range,
            augmentation_ratio=self.augmentation_ratio,
            bernoulli_mix=self.bernoulli_mix
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "grid_size": self.grid_size,
            "erased_patches": self.erased_patches,
            "fill_method": self.fill_method,
            "pixels_range": self.pixels_range,
            "augmentation_ratio": self.augmentation_ratio,
            "bernoulli_mix": self.bernoulli_mix
        }
        base_config.update(config)
        return base_config
    