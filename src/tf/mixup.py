# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from dataaug_utils import check_dataaug_function_arg, check_augment_mix_args, mix_augmented_images


def _check_random_mixup_args(alpha, augmentation_ratio, bernoulli_mix):
    """
    Checks the arguments passed to the `random_mixup` function
    """

    check_dataaug_function_arg(
        alpha,
        context={'arg_name': 'alpha', 'function_name' : 'random_mixup'},
        constraints={'format': 'number'}
    )

    check_augment_mix_args(augmentation_ratio, bernoulli_mix, 'random_mixup')


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

    For each image in the batch, the function:
        1. Samples a mixing factor `lambda` from a Beta distribution.
        2. Randomly selects another image from the batch.
        3. Blends the two images in proportions determined by `lambda`.
        4. Update the labels of the image to reflect the relative contributions
           of both images.
 
    Lambda values are sampled independently for each image, ensuring variety
    across the batch.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images

        labels:
            Labels for the input images. Must be **one-hot encoded**.
            Data type should be tf.float32 (will be cast if not).

        alpha:
            A positive float specifying the parameter `alpha` of the Beta distribution 
            from which `lambda` values are sampled. Controls blending variability.
            If `alpha` is equal to 1.0 (default value), the distribution is uniform.

        augmentation_ratio:
            A float in the range [0, 1] specifying the proportion of images in
            the batch to be replaced by their Mixup-augmented counterparts.
            - 0.0 means no Mixup is applied.
            - 1.0 means Mixup is applied to every image in the batch.

        bernoulli_mix:
            A boolean controlling whether the proportion of augmented images
            should vary from batch to batch.
            - If False, the augmented/original ratio is exactly equal to
              `augmentation_ratio` in every batch.
            - If True, the number of augmented images is drawn from a Bernoulli
              distribution with expected value equal to `augmentation_ratio`.
            Augmented images are at random positions in the output mix.

    Returns:
        A tuple `(output_images, output_labels)` where:
            output_images:
                A tensor of the same shape and dtype as `images`, containing a
                mix of original and Mixup-augmented images.
            output_labels:
                A tensor of the same shape as `labels`, containing the
                correspondingly mixed labels.
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
    lambda_vals = tf.reshape(lambda_vals, [batch_size, 1])
    labels = tf.cast(labels, tf.float32)
    labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

    # Mix the original/augmented images and labels
    output_images, augment_mask = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)
    output_labels = tf.where(augment_mask[:, None], labels_aug, labels)

    # Restore input images shape
    output_images = tf.reshape(output_images, original_image_shape)

    return output_images, output_labels
