# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

from argument_utils import check_patch_sampling_args, check_augment_mix_args
from dataaug_utils import gen_patch_sizes, gen_patch_mask, mix_augmented_images


def _check_function_args(alpha, patch_area, patch_aspect_ratio, augmentation_ratio, bernoulli_mix):
    """
    Checks the arguments passed to the `random_cutmix` function
    """
    function_name = 'random_cutmix'
    check_patch_sampling_args(patch_area, patch_aspect_ratio, alpha, function_name)
    check_augment_mix_args(augmentation_ratio, bernoulli_mix, function_name)


@tf.function
def random_cutmix(
    images: tf.Tensor,
    labels: tf.Tensor,
    patch_area: tuple[float, float] = (0.05, 0.3),
    patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
    alpha: float = 1.0,
    augmentation_ratio: float = 1.0,
    bernoulli_mix: bool = False
) -> tf.Tensor:

    """
    Applies the "CutMix" data augmentation technique to a batch of images.

    Reference paper:
        Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe,
        Youngjoon Yoo (2019). "CutMix: Regularization Strategy to Train Strong
        Classifiers with Localizable Features."

    For each image in the batch:
        1. Sample a mixing coefficient `lambda` from a Beta distribution.
        2. Compute the patch size so that its area is a function of lambda, 
           with an aspect ratio sampled from the specified range.
        3. Choose a random location for the patch within the image.
        4. Randomly select another image from the batch.
        5. Copy the contents of the patch from the other image into the current 
        image at the chosen location.
        6. Adjust the labels of the image using `lambda` to reflect the proportion
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
            Labels for the input images. Must be **one-hot encoded**.
            Data type should be tf.float32 (will be cast if not).

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
                for every batch.
              - True: the augmented/original ratio varies stochastically from batch
                to batch with an expectation equal to `augmentation_ratio`.
            Augmented images are at random positions in the output mix.

    Returns:
        A tuple `(output_images, output_labels)` where:
            output_images:
                A tensor of the same shape and dtype as `images`, containing a
                mix of original and Cutmix-augmented images.
            output_labels:
                A tensor of the same shape as `labels`, containing the
                correspondingly mixed labels.
    """

    # Check the arguments passed to the function
    _check_function_args(alpha, patch_area, patch_aspect_ratio, augmentation_ratio, bernoulli_mix)

    original_image_shape = tf.shape(images)
    # Reshape images with shape [B, H, W] to [B, H, W, 1]
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)

    image_shape = tf.shape(images)
    batch_size, img_height, img_width = tf.unstack(image_shape[:3])

    # Sample patch sizes and locations, then generate a boolean mask
    # (True inside patches, False outside)
    patch_sizes = gen_patch_sizes(images, patch_area, patch_aspect_ratio, alpha)
    patch_mask, _ = gen_patch_mask(images, patch_sizes)

    # Randomly select other images in the batch
    shuffle_indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, shuffle_indices)
    shuffled_labels = tf.gather(labels, shuffle_indices)

    # Paste the patches taken from the other images
    images_aug = tf.where(patch_mask[:, :, :, None], shuffled_images, images)

    # Compute lambda values based on actual patch sizes
    img_area = tf.cast(img_height * img_width, tf.float32)
    patch_areas = tf.cast(patch_sizes[0] * patch_sizes[1], tf.float32)
    lambda_vals = 1.0 - (patch_areas / img_area)

    # Update labels
    labels = tf.cast(labels, tf.float32)
    lambda_vals = lambda_vals[:, None]    # For broadcasting
    labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

    # Mix original/augmented images and labels
    output_images, augment_mask = mix_augmented_images(images, images_aug, augmentation_ratio, bernoulli_mix)
    output_labels = tf.where(augment_mask[:, None], labels_aug, labels)

    # Restore shape of input images
    output_images = tf.reshape(output_images, original_image_shape)

    return output_images, output_labels
