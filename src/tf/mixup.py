# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

from argument_utils import check_argument, check_augment_mix_args
from dataaug_utils import mix_augmented_images


def _check_random_mixup_args(alpha, augmentation_ratio, bernoulli_mix):
    """
    Checks the arguments passed to the `random_mixup` function
    """
    function_name = 'random_mixup'
    check_argument(
        alpha,
        context={'arg_name': 'alpha', 'caller_name': function_name},
        constraints={'format': 'number'}
    )
    check_augment_mix_args(augmentation_ratio, bernoulli_mix, function_name)


@tf.function
def random_mixup(
        images: tf.Tensor,
        labels: tf.Tensor,
        alpha: float = 1.0,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ) -> tf.Tensor:

    """
     Applies the "Mixup" data augmentation technique to a batch of images.

    Reference paper:
        Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz (2017). 
        "Mixup: Beyond Empirical Risk Minimization".

    For each image in the batch:
        1. Sample a blending coefficient `lambda` from a Beta distribution.
        2. Randomly select another image from the batch.
        3. Blend the two images together using the blending coefficient.
        4. Adjust the labels of the image using `lambda` to reflect the proportion
           of pixels contributed by both images.

    Blending coefficients are sampled independently for each image, ensuring variety 
    across the batch.

    Args:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images
            Pixel values are expected to be in the range specified by 
            the `pixels_range` argument.

        alpha:
            A positive float specifying the parameter `alpha` of the Beta distribution 
            from which blending coefficients `lambda` are sampled.
            If `alpha` is equal to 1.0 (default value), the distribution is uniform.

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
        of original and Cutout-augmented images. Pixel values are in the same range
        as the input images.
    """

    # Check the arguments passed to the function
    _check_random_mixup_args(alpha, augmentation_ratio, bernoulli_mix)

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

    # Randomly select other images in the batch
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Blend the images together
    lambda_vals = tf.reshape(lambda_vals, [batch_size, 1, 1, 1])
    images_aug = lambda_vals * images + (1 - lambda_vals) * shuffled_images

    # Weigh the labels
    lambda_vals =lambda_vals[:, None]   # For broadcasting
    labels = tf.cast(labels, tf.float32)
    labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

    # Mix original/augmented images and labels
    output_images, augment_mask = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)
    output_labels = tf.where(augment_mask[:, None], labels_aug, labels)

    # Restore input images shape
    output_images = tf.reshape(output_images, original_image_shape)

    return output_images, output_labels
