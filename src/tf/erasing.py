# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
    
from argument_utils import check_patch_sampling_args, check_fill_method_arg, check_pixels_range_args, check_augment_mix_args
from dataaug_utils import (
	rescale_pixel_values, sample_patch_sizes, sample_patch_locations, 
	gen_patch_mask, gen_patch_contents, mix_augmented_images
)


class RandomErasing(tf.keras.Layer):
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
        of original and Random-Erasing-augmented images. Pixel values are in the same 
        range as the input images.
    """

    def __init__(self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        alpha: float = 1.0,
        fill_method: str = 'black',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)
        
        self.layer_name = 'RandomErasing'
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.alpha = alpha
        self.fill_method = fill_method
        self.pixels_range = pixels_range
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix
        
        self._check_layer_args()
 

    def _check_layer_args(self):
        """
        Checks the arguments passed to the `random_erasing` function
        """
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.alpha, self.layer_name)
        check_fill_method_arg(self.fill_method, self.layer_name)
        check_pixels_range_args(self.pixels_range, self.layer_name)
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.layer_name)


    def call(self, images, training=None):

        original_image_shape = tf.shape(images)

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if images.shape.rank == 3:
            images = tf.expand_dims(images, axis=-1)

        # Save images data type and rescale pixel values to (0, 255)
        pixels_dtype = images.dtype
        images = rescale_pixel_values(images, self.pixels_range, (0, 255), dtype=tf.int32)

        # Sample patch sizes and locations, then generate a boolean mask
        # (True inside patches, False outside)
        patch_sizes = sample_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)
        patch_corners = sample_patch_locations(images, patch_sizes)
        patch_mask = gen_patch_mask(images, patch_corners)

        # Generate color contents of patches
        patch_contents = gen_patch_contents(images, self.fill_method)

        # Erase the patches from the images and fill them
        images_aug = tf.where(patch_mask[:, :, :, None], patch_contents, images)

        # Mix original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape, data type and pixels range of input images
        output_images = tf.reshape(output_images, original_image_shape)
        output_images = rescale_pixel_values(output_images, (0, 255), self.pixels_range, dtype=pixels_dtype)

        return output_images
 

    def get_config(self):
        base_config = super().get_config()
        config = {
            'layer_name': self.layer_name,
            'patch_area': self.patch_area,
            'patch_aspect_ratio': self.patch_aspect_ratio,
            'alpha': self.alpha,
            'fill_method': self.fill_method,
            'pixels_range': self.pixels_range,
            'augmentation_ratio': self.augmentation_ratio,
            'bernoulli_mix': self.bernoulli_mix
        }       
        base_config.update(config)
        return base_config
    