# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

from argument_utils import check_argument, check_patch_sampling_args, check_augment_mix_args
from dataaug_utils import sample_patch_sizes, sample_patch_locations, gen_patch_mask, mix_augmented_images


class RandomCutBlur(tf.keras.Layer):
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
            Patch areas are sampled from a Beta distribution. See `alpha` argument.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Minimum value must be > 0.
            Patch aspect ratios are sampled from a uniform distribution.

        alpha:
            A float specifying the alpha parameter of the Beta distribution used
            to sample patch areas. Set to 0 by default, making the distribution
            uniform.

        blur_factor:
            A float specifying the size of the low-resolution images (they all 
            have the same size). Values must be > 0 and < 1, representing fractions 
            of the image size. Smaller values produce stronger blur.

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
        of original and CutBlur-augmented images. Pixel values are in the same
        range as the input images.
    """
    
    def __init__(self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        alpha: float = 1.0,
        blur_factor: float = 0.1,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
        **kwargs):

        super().__init__(**kwargs)

        self.layer_name = 'RandomCutBlur'
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.alpha = alpha
        self.blur_factor = blur_factor
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_layer_args()


    def _check_layer_args(self):
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.alpha, self.layer_name)
        check_argument(
            self.blur_factor,
            context={'arg_name': 'blur_factor', 'caller_name': self.layer_name},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.layer_name)


    def call(self, images, training=None):

        original_image_shape = tf.shape(images)

        # Reshape images with shape [batch_size, height, width]
        # to shape [batch_size, height, width, 1]
        if images.shape.rank == 3:
            images = tf.expand_dims(images, axis=-1)

        # Calculate the size of the low-resolution images
        image_size = tf.shape(images)[1:3]
        low_res_size = self.blur_factor * tf.cast(image_size, tf.float32)
        low_res_size = tf.cast(tf.round(low_res_size), tf.int32)

        # Downsize to the low resolution size, then upsize to the original size
        smaller_images = tf.image.resize(images, low_res_size)
        low_res_images = tf.image.resize(smaller_images, image_size)
        low_res_images = tf.cast(low_res_images, images.dtype)

        # Sample patch sizes and locations, then generate a boolean mask
        # (True inside patches, False outside)
        patch_sizes = sample_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)
        patch_corners = sample_patch_locations(images, patch_sizes)
        patch_mask = gen_patch_mask(images, patch_corners)

        # Erase patches from the images and fill them with the low-res images
        images_aug = tf.where(patch_mask[..., None], low_res_images, images)

        # Mix original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape of input images
        output_images = tf.reshape(output_images, original_image_shape)

        return output_images


    def get_config(self):
        base_config = super().get_config()
        config = {
            'layer_name': self.layer_name,
            'patch_area': self.patch_area,
            'patch_aspect_ratio': self.patch_aspect_ratio,
            'alpha': self.alpha,
            'blur_factor': self.blur_factor,
            'augmentation_ratio': self.augmentation_ratio,
            'bernoulli_mix': self.bernoulli_mix
        }
        base_config.update(config)
        return base_config