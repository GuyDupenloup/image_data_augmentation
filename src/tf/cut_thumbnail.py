# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

from argument_utils import check_argument, check_augment_mix_args
from dataaug_utils import gen_patch_mask, mix_augmented_images


class RandomCutThumbnail(tf.keras.Layer):
    """
    Applies the "Cut-Thumbnail" data augmentation technique to a batch of images.

    Reference paper:
        Tianshu Xie, Xuan Cheng, Xiaomin Wang, Minghui Liu, Jiali Deng, Tao Zhou, 
        Ming Liu (2021). "Cut-thumbnail: A novel data augmentation for convolutional
        neural network".

    For each image in the batch, the function:
      1. Resizes the image to a smaller size to create a thumbnail.
      2. Pastes the thumbnail into the original image at a random location.
    All the thumbnails have the same area, specified as a fraction of the image area,
    and have the same aspect ratio as the images.

    Thumbnail locations are sampled independently for each image, ensuring variety 
    across the batch.

    By default, the augmented/original image ratio in the output mix is 1.0.
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images

        thumbnail_area:
            A float specifying the area of the thumbnail as a fraction of the 
            image area. Values must be > 0 and < 1.

        resize_method:
            A string specifying the interpolation method used by tf.image.resize() to
            create the thumbnail. Supported methods include:
            {'bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'nearest', 
            'area', 'mitchellcubic'}

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
        of original and Cut-Thumbnail-augmented images.
    """
        
    def __init__(self,
        thumbnail_area: float = 0.1,
        resize_method: str = 'bilinear',
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False,
		**kwargs):

        super().__init__(**kwargs)

        self.layer_name = 'RandomCutThumbnail'
        self.thumbnail_area = thumbnail_area
        self.resize_method = resize_method
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_layer_args()


    def _check_layer_args(self):
        """
        Checks the arguments passed to the `random_cut_thumbnail` function
        """
        check_argument(
            self.thumbnail_area,
            context={'arg_name': 'thumbnail_area', 'caller_name': self.layer_name},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )

        supported_resize_methods = (
            'bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'nearest', 'area', 'mitchellcubic')
        if self.resize_method not in supported_resize_methods:
            raise ValueError(
                f'\nArgument `resize_method` of layer {self.layer_name}: '
                f'expecting one of {supported_resize_methods}\n'
                f'Received: {self.resize_method}'
            )
        
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.layer_name)


    def _calculate_thumbnail_size(self, image_size):
        """
        Calculates height and width of the thumbnails.
        The thumbnail area is specified as a fraction of the images area.
        The aspect ratio is the same as the images.
        """
        
        img_height = tf.cast(image_size[0], tf.float32)
        img_width = tf.cast(image_size[1], tf.float32)

        area = self.thumbnail_area * img_height * img_width
        aspect_ratio = img_height / img_width

        thumb_w = tf.math.sqrt(area / aspect_ratio)
        thumb_h = thumb_w * aspect_ratio

        thumb_h = tf.cast(tf.round(thumb_h), tf.int32)
        thumb_w = tf.cast(tf.round(thumb_w), tf.int32)

        # Clip thumbnail size to image size (for robustness)
        thumb_h = tf.clip_by_value(thumb_h, 0, image_size[0])
        thumb_w = tf.clip_by_value(thumb_w, 0, image_size[1])

        return thumb_h, thumb_w


    def call(self, images, training=None):

        original_image_shape = tf.shape(images)

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if images.shape.rank == 3:
            images = tf.expand_dims(images, axis=-1)
        image_shape = tf.shape(images)

        # Calculate the size of the thumbnails and create them
        thumbnail_size = self._calculate_thumbnail_size(image_shape[1:3])
        thumbnails = tf.image.resize(images, thumbnail_size, method=self.resize_method)
        thumbnails = tf.cast(thumbnails, images.dtype)

        # Create a boolean mask to paste the thumbnails
        batch_size = image_shape[0]
        batched_thumbnail_size = (tf.repeat(thumbnail_size[0], batch_size), tf.repeat(thumbnail_size[1], batch_size))
        patch_mask, patch_corners = gen_patch_mask(images, batched_thumbnail_size)

        # Get the indices where patches should be placed
        patch_indices = tf.where(patch_mask)
        patch_indices = tf.cast(patch_indices, tf.int32)

        # Create thumbnail contents that match the patch locations
        batch_indices = patch_indices[:, 0]  # Get batch index for each pixel
        
        # For each pixel in the patch, find its corresponding pixel in the thumbnail
        y_rel = patch_indices[:, 1] - tf.gather(patch_corners[:, 0], batch_indices)
        x_rel = patch_indices[:, 2] - tf.gather(patch_corners[:, 1], batch_indices)
        
        # Create indices into the thumbnails tensor and gather pixel values
        thumbnail_indices = tf.stack([batch_indices, y_rel, x_rel], axis=1)
        thumbnail_contents = tf.gather_nd(thumbnails, thumbnail_indices)
        
        # Paste the thumbnails
        images_aug = tf.tensor_scatter_nd_update(images, patch_indices, thumbnail_contents)

        # Mix original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape of input images
        output_images = tf.reshape(output_images, original_image_shape)
        
        return output_images


    def get_config(self):
        base_config = super().get_config()
        config = {
            'layer_name': self.layer_name,
            'thumbnail_area': self.thumbnail_area,
            'resize_method': self.resize_method,
            'augmentation_ratio': self.augmentation_ratio,
            'bernoulli_mix': self.bernoulli_mix
        }
        base_config.update(config)
        return base_config
   