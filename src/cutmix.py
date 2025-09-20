# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import (
    check_dataaug_function_arg, sample_patch_locations,
    gen_patch_mask, mix_augmented_images
)


def _check_random_cutmix_args(alpha, patch_aspect_ratio, augmentation_ratio, bernoulli_mix):
    """
    Checks the arguments passed to the `random_cutmix` function
    """

    check_dataaug_function_arg(
        alpha,
        context={'arg_name': 'alpha', 'function_name' : 'random_cutmix'},
        constraints={'format': 'number'}
    )
    check_dataaug_function_arg(
        patch_aspect_ratio,
        context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'random_cutmix'},
        constraints={'format': 'tuple', 'min_val': ('>', 0)}
    )
    check_dataaug_function_arg(
        augmentation_ratio,
        context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_cutmix'},
        constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    )
    if not isinstance(bernoulli_mix, bool):
        raise ValueError(
            'Argument `bernoulli_mix` of function `random_cutmix`: '
            f'expecting a boolean value\nReceived: {bernoulli_mix}'
        )


def _get_patch_dims(image_shape, lambda_vals, patch_aspect_ratio):

    batch_size, img_height, img_width = tf.unstack(image_shape[:3])

    # Sample aspect ratios for the patches
    aspect_ratio = tf.random.uniform([batch_size], minval=patch_aspect_ratio[0], maxval=patch_aspect_ratio[1], dtype=tf.float32)

    # Calculate width and height of patches
    area = tf.sqrt(1 - lambda_vals) * tf.cast(img_height, tf.float32) * tf.cast(img_width, tf.float32)
    patch_w = tf.math.sqrt(area / aspect_ratio)
    patch_h = patch_w * aspect_ratio

    patch_h = tf.cast(tf.round(patch_h), tf.int32)
    patch_w = tf.cast(tf.round(patch_w), tf.int32)
    
    # Clip patch dimensions to image size
    patch_h = tf.clip_by_value(patch_h, 0, img_height)
    patch_w = tf.clip_by_value(patch_w, 0, img_width)

    return patch_h, patch_w


@tf.function
def random_cutmix(
        images: tf.Tensor,
        labels: tf.Tensor,
        alpha: float = 1.0,
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ) -> tf.Tensor:

    """
    Applies the "CutMix" data augmentation technique to a batch of images.

    Reference paper:
        Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe,
        Youngjoon Yoo (2019). "CutMix: Regularization Strategy to Train Strong
        Classifiers with Localizable Features."

    For each image in the batch, the function:
        1. Samples a mixing coefficient `lambda` from a Beta distribution.
        2. Computes the patch size so that its area is a function of lambda, 
           with an aspect ratio sampled from the specified range.
        3. Chooses a random location for the patch within the image.
        4. Randomly selects another image from the batch.
        5. Copies the contents of the patch from the other image into the current 
        image at the chosen location.
        6. Adjusts the labels of the image using `lambda` to reflect the proportion
        of pixels contributed by the other image.

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
            Labels for the input images. Must be *one-hot encoded".

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
                for every batch. Augmented images are at random positions.
              - True: the augmented/original ratio varies stochastically from batch
                to batch with an expectation equal to `augmentation_ratio`.

    Returns:
        A tuple `(output_images, output_labels)` where:
            output_images:
                A tensor of the same shape and dtype as `images`, containing a
                mix of original and Cutmix augmented images.
            output_labels:
                A tensor of the same shape as `labels`, containing the
                correspondingly mixed labels.
    """

    # Check the arguments passed to the function
    _check_random_cutmix_args(alpha, patch_aspect_ratio, augmentation_ratio, bernoulli_mix)

    original_image_shape = tf.shape(images)

    # Reshape images with shape [B, H, W] to [B, H, W, 1]
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    image_shape = tf.shape(images)

    # Sample lambda values
    batch_size = image_shape[0]
    gamma1 = tf.random.gamma([batch_size], alpha, dtype=tf.float32)
    gamma2 = tf.random.gamma([batch_size], alpha, dtype=tf.float32)
    lambda_vals = gamma1 / (gamma1 + gamma2)
    
    # Get patch sizes based on lambda and aspect ratio range
    patch_size = _get_patch_dims(image_shape, lambda_vals, patch_aspect_ratio)

    # Sample patch locations and generate a boolean mask
    # (True inside the patches, False outside)
    patch_corners = sample_patch_locations(image_shape, patch_size)
    patch_mask = gen_patch_mask(image_shape, patch_corners)

    # Randomly select other images in the batch
    batch_size = image_shape[0]
    shuffle_indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, shuffle_indices)

    # Paste the patches taken from the other images
    images_aug = tf.where(patch_mask[:, :, :, None], shuffled_images, images)

    # Recompute lambda to reflect the actual patch size
    # (there was rounding and clipping)
    patch_area = tf.cast(patch_size[0], tf.float32) * tf.cast(patch_size[1], tf.float32)
    lambda_vals = 1.0 - patch_area

    # Update the labels to reflect the contribution of the pasted patches
    labels = tf.cast(labels, tf.float32)
    shuffled_labels = tf.gather(labels, shuffle_indices)
    lambda_vals = lambda_vals[:, None]
    labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

    # Mix the original/augmented images and labels
    output_images, augment_mask = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)
    output_labels = tf.where(augment_mask[:, None], labels_aug, labels)

    # Restore input images shape
    output_images = tf.reshape(output_images, original_image_shape)

    return output_images, output_labels
