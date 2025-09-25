# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

from argument_utils import (
	check_dataaug_function_arg, check_fill_method_arg,
	check_pixels_range_args, check_augment_mix_args
)
from dataaug_utils import rescale_pixel_values, gen_patch_contents, mix_augmented_images


class RandomGridMask(tf.keras.Layer):
    """
   Applies the "GridMask" data augmentation technique to a batch of images.
    
    Reference paper:
        Pengguang Chen, Shu Liu, Hengshuang Zhao, and Jiaya Jia (2020).
        "GridMask Data Augmentation."

    Regular grid patterns are applied to the images. A grid is made of square "units"
    that are transparent areas with a masked (opaque) subarea in the bottom-right corner.
    When a grid is applied to an image, the transparent areas are left unchanged, while
    the masked areas are erased and filled with a solid color or noise.

    For each image in the batch:
    1. A unit side length is sampled from the specified range.
    2. A grid is generated using this unit side length.
    3. Random offsets sampled in the range [0, unit_side_length] are applied to the grid
       to shift it in both directions.
    4. The grid is applied to the image.

    Unit side lengths and grid offsets are sampled independently for each image, ensuring 
    variety across the batch.

    By default, the augmented/original images ratio in the output mix is `1.0`. 
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images
            Pixel values are expected to be in the range specified by 
            the `pixels_range` argument.

        unit_length:
            A tuple of two floats specifying the range from which the side length 
            of each grid unit is sampled. Values must be > 0 and < 1, representing 
            fractions of the image size. If the image is rectangular, the minimum
            of height and width is used.

        masked_ratio:
            A float specifying the side length of the masked region inside
            each unit as a fraction of the unit side length. Must be > 0 and < 1.

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
        of original and Grid-Mask-augmented images. Pixel values are in the same 
        range as the input images.
    """
        
    def __init__(self,
        unit_length: tuple[float, float] = (0.2, 0.4),
        masked_ratio: float = 0.5,
        fill_method: str = 'black',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)

        self.unit_length = unit_length
        self.masked_ratio = masked_ratio
        self.fill_method = fill_method
        self.pixels_range = pixels_range
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_arguments()


    def _check_arguments(self):
        """
        Checks the arguments passed to the `hide_and_seek` function
        """

        check_dataaug_function_arg(
            self.unit_length,
            context={'arg_name': 'unit_length', 'function_name' : 'random_grid_mask'},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_dataaug_function_arg(
            self.masked_ratio,
            context={'arg_name': 'masked_ratio', 'function_name' : 'random_grid_mask'},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )

        check_fill_method_arg(self.fill_method, 'RandomGridMask')
        check_pixels_range_args(self.pixels_range, 'RandomGridMask')
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, 'RandomGridMask')


    def _generate_grid_mask(self, images):
        """
        Samples unit sizes and grid offsets. Applies the offsets to the grid 
        to shift it in both directions. Generates a boolean mask with 
        value True inside masked areas and False inside transparent areas.
        If the image is rectangular, the unit size is calculated using
        the minimum of the height and width.
        """

        image_shape = tf.shape(images)
        batch_size, img_height, img_width = tf.unstack(image_shape[:3])

        # Sample unit lengths and calculate masked area lengths
        length_fract = tf.random.uniform(
            [batch_size],
            minval=self.unit_length[0],
            maxval=self.unit_length[1],
            dtype=tf.float32
        )
    
        min_img_side = tf.minimum(img_height, img_width)
        unit_sizes = length_fract * tf.cast(min_img_side, tf.float32)
        masked_area_sizes = tf.round(unit_sizes * self.masked_ratio)

        # Sample grid offsets from range [0, unit_size]
        delta_x = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32) * unit_sizes
        delta_y = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32) * unit_sizes

        # Create coordinate grids for the entire image
        x_coords, y_coords = tf.meshgrid(tf.range(img_width), tf.range(img_height))
        x_coords = tf.broadcast_to(x_coords[None, :, :], image_shape[:3])
        y_coords = tf.broadcast_to(y_coords[None, :, :], image_shape[:3])

        # Add dimensions for broadcasting
        unit_sizes = unit_sizes[:, None, None]
        masked_area_sizes = masked_area_sizes[:, None, None]
        delta_x = delta_x[:, None, None]
        delta_y = delta_y[:, None, None]

        # Apply random offsets to shift the grid's top-left corner
        shifted_x = tf.cast(x_coords, tf.float32) + delta_x
        shifted_y = tf.cast(y_coords, tf.float32) + delta_y

        # Calculate which unit each pixel belongs to after applying offsets
        unit_x = tf.floor(shifted_x / unit_sizes)
        unit_y = tf.floor(shifted_y / unit_sizes)

        # Calculate position within each unit
        pos_x_in_unit = shifted_x - unit_x * unit_sizes
        pos_y_in_unit = shifted_y - unit_y * unit_sizes

        # Determine if each pixel is in the masked area
        in_masked_area = tf.logical_and(
            pos_x_in_unit < masked_area_sizes,
            pos_y_in_unit < masked_area_sizes
        )

        # Create a boolean mask (True inside masked areas, False outside)
        mask = tf.where(in_masked_area, True, False)

        return mask


    def call(self, images, training=None):

        original_image_shape = tf.shape(images)

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if images.shape.rank == 3:
            images = tf.expand_dims(images, axis=-1)

        # Save images data type and rescale pixel values to (0, 255)
        pixels_dtype = images.dtype
        images = rescale_pixel_values(images, self.pixels_range, (0, 255), dtype=tf.int32)

        # Generate a mask to erase the masked areas of units
        mask = self._generate_grid_mask(images)

        # Generate the contents of the erased areas
        unit_contents = gen_patch_contents(images, self.fill_method)

        # Erase unit masked areas from the images and fill them
        images_aug = tf.where(mask[:, :, :, None], unit_contents, images)

        # Mix the original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape, data type and pixels range of input images
        output_images = tf.reshape(output_images, original_image_shape)
        output_images = rescale_pixel_values(output_images, (0, 255), self.pixels_range, dtype=pixels_dtype)

        return output_images


    def get_config(self):
        base_config = super().get_config()
        config = {
             'unit_length': self.unit_length,
             'masked_ratio': self.masked_ratio,
             'fill_method': self.fill_method,
             'pixels_range': self.pixels_range,
             'augmentation_ratio': self.augmentation_ratio,
             'bernoulli_mix': self.bernoulli_mix
        }
        base_config.update(config)
        return base_config
    