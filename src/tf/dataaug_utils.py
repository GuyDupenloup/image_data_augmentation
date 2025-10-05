# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import math
import tensorflow as tf


def gen_patch_sizes(
    images: tf.Tensor,
    patch_area: tuple[float, float],
    patch_aspect_ratio: tuple[float, float],
    alpha: float = 1.0
) -> tf.Tensor:
    """
    Samples heights and widths of patches

    Arguments:
        images:
            The input images.

        patch_area:
            A tuple of two floats specifying the range from which patch areas
            are sampled. Values must be > 0 and < 1.
            A single float may be used instead of a tuple. In this case, the patch
            area is equal to `patch_area` for all the images.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Values must be > 0.
            A single float may be used instead of a tuple. In this case, the aspect ratio
            is equal to `patch_aspect_ratio` for all the images.

        alpha:
            A float greater than 0, the shape parameter of the Beta distribution
            used to sample patch area. If `alpha` is equal to 1.0 (default), 
            the distribution is uniform.

    Returns:
        A tuple of 2 tensors with shape [batch_size]:
            `(patch_height, patch_width)`. 
    """

    image_shape = tf.shape(images)
    batch_size, img_height, img_width = tf.unstack(image_shape[:3])

    if isinstance(patch_area, (tuple, list)):
        if alpha != 1.0:
            # Sample lambda values from a Beta distribution
            gamma1 = tf.random.gamma([batch_size], alpha, dtype=tf.float32)
            gamma2 = tf.random.gamma([batch_size], alpha, dtype=tf.float32)
            lambda_vals = gamma1 / (gamma1 + gamma2)
            # Linearly rescale to the specified area range (this does not change the distribution)
            area_fraction = patch_area[0] + lambda_vals * (patch_area[1] - patch_area[0])
        else:
            # Sample from a uniform distribution
            area_fraction = tf.random.uniform([batch_size], minval=patch_area[0], maxval=patch_area[1], dtype=tf.float32)
    else:
        # Constant area fraction
        area_fraction = tf.fill([batch_size], patch_area)

    if isinstance(patch_aspect_ratio, (tuple, list)):
        # Sample patch aspect ratios from a uniform distribution
        # Aspect ratios are non-linear. We use logs for the sampling range
        # to get better balance between tall and wide rectangles.
        log_min = math.log(patch_aspect_ratio[0])
        log_max = math.log(patch_aspect_ratio[1])
        log_aspect_ratio = tf.random.uniform([batch_size], minval=log_min, maxval=log_max, dtype=tf.float32)
        aspect_ratio = tf.exp(log_aspect_ratio)
    else:
        # Constant aspect ratio
        aspect_ratio = tf.fill([batch_size], patch_aspect_ratio)

    # Calculate width and height of patches
    area = area_fraction  * tf.cast(img_width, tf.float32) * tf.cast(img_height, tf.float32)
    patch_w = tf.math.sqrt(area / aspect_ratio)
    patch_h = patch_w * aspect_ratio

    patch_h = tf.cast(tf.round(patch_h), tf.int32)
    patch_w = tf.cast(tf.round(patch_w), tf.int32)

    # Clip oversized patches to image size
    patch_h = tf.clip_by_value(patch_h, 0, img_height)
    patch_w = tf.clip_by_value(patch_w, 0, img_width)

    return patch_h, patch_w


def gen_patch_mask(
    images: tf.Tensor,
    patch_size: tuple[tf.Tensor, tf.Tensor]
) -> tf.Tensor:

    """
    Samples patch locations given their heights and widths.
    Locations are constrained in such a way that the patches 
    are entirely contained inside the images.

    Arguments:
        image_shape:
            The shape of the images. Either [B, H, W, C] or
            [B, H, W] (the channel is not used).

        patch_size:
            A tuple of 2 tensors with shape [batch_size]:
                `(patch_height, patch_width)` 

    Returns:
        A tensor with shape [batch_size, 4].
        Contains the opposite corners coordinates (y1, x1, y2, x2)
        of the patches.
    """

    image_shape = tf.shape(images)
    batch_size, img_height, img_width = tf.unstack(image_shape[:3])
    patch_h, patch_w = patch_size

    # Sample uniformly between 0 and 1
    x_rand = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)
    y_rand = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Calculate valid ranges for each patch
    max_x1 = tf.cast(img_width - patch_w, tf.float32)
    max_y1 = tf.cast(img_height - patch_h, tf.float32)
    
    # Scale linearly to valid ranges (distributions remain uniform)
    x1 = tf.cast(tf.round(x_rand * max_x1), tf.int32)
    y1 = tf.cast(tf.round(y_rand * max_y1), tf.int32)
    
    # Get coordinates of opposite patch corners
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    corners = tf.stack([y1, x1, y2, x2], axis=-1)

    # Create coordinate grids
    grid_x, grid_y = tf.meshgrid(tf.range(img_width), tf.range(img_height))
	
    # Add new axis for broadcasting
    grid_x = grid_x[None, :, :]
    grid_y = grid_y[None, :, :]
    x1 = x1[:, None, None]
    x2 = x2[:, None, None]
    y1 = y1[:, None, None]
    y2 = y2[:, None, None]
    
    # Create the boolean mask
    mask = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y <  y2)
 
    return mask, corners


def gen_patch_contents(
    images: tf.Tensor,
    fill_method: str
) -> tf.Tensor:

    """
    Generates color contents for erased cells (images filled 
    with solid color or noise).

    Arguments:
        images:
            The images being augmented (4D tensor).
            Pixels must be in range [0, 255] with tf.int32 data type.

        fill_method:
            A string, the method to use to generate the contents.
            One of: {'black', 'gray', 'white', 'mean_per_channel', 'random', 'noise'}

    Returns:
        A tensor with the same shape as the images.
    """

    image_shape = tf.shape(images)

    if fill_method == 'black':
        contents = tf.zeros(image_shape, dtype=tf.int32)

    elif fill_method == 'gray':
        contents = tf.fill(image_shape, 128)

    elif fill_method == 'white':
        contents = tf.fill(image_shape, 255)

    elif fill_method == 'mean_per_channel':
        channel_means = tf.reduce_mean(images, axis=[1, 2])
        channel_means = tf.cast(channel_means, tf.int32)
        contents = tf.broadcast_to(channel_means[:, None, None, :], image_shape)

    elif fill_method == 'random':
        color = tf.random.uniform([image_shape[0], image_shape[-1]], minval=0, maxval=256, dtype=tf.int32)
        contents = tf.broadcast_to(color[:, None, None, :], image_shape)

    elif fill_method == 'noise':
        contents = tf.random.uniform(image_shape, minval=0, maxval=256, dtype=tf.int32)

    return contents


def mix_augmented_images(
    original_images: tf.Tensor,
    augmented_images: tf.Tensor,
    augmentation_ratio: int | float = 1.0,
    bernoulli_mix: bool = False
) -> tf.Tensor:

    """
    Mixes original images and augmented images according to a specified
    augmented/original images ratio and method.
    The augmented images are at random positions in the output mix.

    The original and augmented images must have the same shape, one of:
        [B, H, W, 3]  -->  RGB
        [B, H, W, 1]  -->  Grayscale
        [B, H, W]     -->  Grayscale

    Arguments:
    ---------
        original_images:
            The original images.

        augmented_images:
            The augmented images to mix with the original images.

        augmentation_ratio:
            A float in the interval [0, 1] specifying the augmented/original
            images ratio in the output mix. If set to 0, no images are
            augmented. If set to 1, all the images are augmented.

        bernoulli_mix:
            A boolean specifying the method to use to mix the images:
            - False:
                The fraction of augmented images in the mix is equal
                to `augmentation_ratio` for every batch.
            - True:
                The fraction of augmented images in the mix varies from batch
                to batch. Because Bernoulli experiments are used, the expectation
                of the fraction is equal to `augmentation_ratio`.

    Returns:
    -------
        A tensor of the same shape as the input images containing
        a mix of original and augmented images.
    """

    # Ensure original and augmented images have the same shape
    tf.debugging.assert_equal(
        tf.shape(original_images),
        tf.shape(augmented_images),
        message=('Function `mix_augmented_images`: original '
                 'and augmented images must have the same shape')
    )

    image_shape = tf.shape(original_images)
    batch_size = image_shape[0]

    if augmentation_ratio == 0.0:
        mixed_images = original_images
        mask = tf.zeros([batch_size], dtype=tf.bool)

    elif augmentation_ratio == 1.0:
        mixed_images = augmented_images
        mask = tf.ones([batch_size], dtype=tf.bool)
    else:

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if original_images.shape.rank == 3:
            original_images = tf.expand_dims(original_images, axis=-1)
            augmented_images = tf.expand_dims(augmented_images, axis=-1)

        if bernoulli_mix:
            # For each image position in the output mix, make a Bernoulli
            # experiment to decide if the augmented image takes it.
            probs = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)
            mask = tf.where(probs < augmentation_ratio, True, False)
        else:
            # Calculate the number of augmented images in the mix
            num_augmented = augmentation_ratio * tf.cast(batch_size, tf.float32)
            num_augmented = tf.cast(tf.round(num_augmented), tf.int32)

            # Generate a mask set to True for positions in the mix
            # occupied by augmented images, False by original images
            grid = tf.range(batch_size)
            mask = tf.where(grid < num_augmented, True, False)

            # Shuffle the mask so that the augmented images
            # are at random positions in the output mix
            mask = tf.random.shuffle(mask)

        # Apply the mask to images to generate the output mix
        mixed_images = tf.where(mask[:, None, None, None], augmented_images, original_images)

        # Restore the image input shape
        mixed_images = tf.reshape(mixed_images, image_shape)

    return mixed_images, mask


def rescale_pixel_values(
    images: tf.Tensor,
    input_range: tuple[int | float, int | float],
    output_range: tuple[int | float, int | float],
    dtype: tf.DType = tf.float32
) -> tf.Tensor:

    """
    Linearly rescales pixel values of images from one range to another.

    A linear transformation is applied to each pixel so that values
    originally in `input_range` are mapped to `output_range`. Output
    values are clipped to the target range and cast to the specified 
    data type `dtype`.

    Example:
        # Convert uint8 images [0, 255] to float32 [0.0, 1.0]
        images_mapped = remap_pixel_values_range(images, (0, 255), (0.0, 1.0), tf.float32)

    Args:
        images:
            Input images.
        input_range:
            (min, max) range of input pixel values.
        output_range:
            (min, max) target range for output pixel values.
        dtype:
            Desired output data type.

    Returns:
        Images with pixel values rescaled to `output_range` and cast to `dtype`.
    """

    if input_range != output_range:
        input_min, input_max = input_range
        output_min, output_max = output_range

        images = tf.cast(images, tf.float32)
        images = ((output_max - output_min) * images +
                   output_min * input_max - output_max * input_min) / (input_max - input_min)
        
        # Clip to the output range
        images = tf.clip_by_value(images, output_min, output_max)

    return tf.cast(images, dtype)
