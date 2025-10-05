# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import math
import torch


def gen_patch_sizes(
    images: torch.Tensor,
    patch_area: tuple[float, float],
    patch_aspect_ratio: tuple[float, float],
    alpha: float = 1.0
) -> torch.Tensor:
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
            used to sample patch areas. If `alpha` is equal to 1.0 (default), 
            the distribution is uniform.

    Returns:
        A tuple of 2 tensors with shape [batch_size]:
            `(patch_height, patch_width)`. 
    """

    device = images.device
    batch_size, _, img_height, img_width = images.shape

    if isinstance(patch_area, (tuple, list)):
        if alpha != 1.0:
            # Sample lambda values from a Beta distribution
            gamma1 = torch.distributions.Gamma(alpha, 1.0).sample([batch_size]).to(device)
            gamma2 = torch.distributions.Gamma(alpha, 1.0).sample([batch_size]).to(device)
            lambda_vals = gamma1 / (gamma1 + gamma2)
            # Linearly rescale to the specified area range (this does not change the distribution)
            area_fraction = patch_area[0] + lambda_vals * (patch_area[1] - patch_area[0])
        else:
            # Sample from a uniform distribution
            area_fraction = torch.rand([batch_size], device=device) * (patch_area[1] - patch_area[0]) + patch_area[0]
    else:
        # Constant area fraction
        area_fraction = torch.full([batch_size], patch_area, device=device)

    if isinstance(patch_aspect_ratio, (tuple, list)):
        # Sample patch aspect ratios from a uniform distribution
        # Aspect ratios are non-linear. We use logs for the sampling range
        # to get better balance between tall and wide rectangles.
        log_min = math.log(patch_aspect_ratio[0])
        log_max = math.log(patch_aspect_ratio[1])
        log_aspect_ratio = torch.rand([batch_size], device=device) * (log_max - log_min) + log_min
        aspect_ratio = torch.exp(log_aspect_ratio)
    else:
        # Constant aspect ratio
        aspect_ratio = torch.full([batch_size], patch_aspect_ratio, device=device)

    # Calculate width and height of patches
    area = area_fraction * img_width * img_height
    patch_w = torch.sqrt(area / aspect_ratio)
    patch_h = patch_w * aspect_ratio

    patch_h = torch.round(patch_h).long()
    patch_w = torch.round(patch_w).long()

    # Clip oversized patches to image size
    patch_h = torch.clamp(patch_h, 0, img_height)
    patch_w = torch.clamp(patch_w, 0, img_width)

    return patch_h, patch_w


def gen_patch_mask(
    images: torch.Tensor,
    patch_size: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:

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

    device = images.device
    batch_size, _, img_height, img_width = images.shape
    patch_h, patch_w = patch_size

    # Sample uniformly between 0 and 1
    x_rand = torch.rand(batch_size, device=device)
    y_rand = torch.rand(batch_size, device=device)

    # Calculate valid ranges for each patch
    max_x1 = (img_width - patch_w).float()
    max_y1 = (img_height - patch_h).float()
    
    # Scale linearly to valid ranges (distributions remain uniform)
    x1 = torch.round(x_rand * max_x1).long()
    y1 = torch.round(y_rand * max_y1).long()
    
    # Get coordinates of opposite patch corners
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    corners = torch.stack([y1, x1, y2, x2], dim=-1)

    # Create coordinate grids
    grid_y, grid_x = torch.meshgrid(torch.arange(img_height, device=device), 
                                     torch.arange(img_width, device=device), 
                                     indexing='ij')
    
    # Add new axis for broadcasting
    grid_x = grid_x[None, :, :]
    grid_y = grid_y[None, :, :]
    x1 = x1[:, None, None]
    x2 = x2[:, None, None]
    y1 = y1[:, None, None]
    y2 = y2[:, None, None]

    # Create the boolean mask
    mask = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
 
    return mask, corners


def gen_patch_contents(
    images: torch.Tensor,
    fill_method: str
) -> torch.Tensor:

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

    device = images.device
    image_shape = images.shape

    if fill_method == 'black':
        contents = torch.zeros(image_shape, dtype=torch.long, device=device)

    elif fill_method == 'gray':
        contents = torch.full(image_shape, 128, dtype=torch.long, device=device)

    elif fill_method == 'white':
        contents = torch.full(image_shape, 255, dtype=torch.long, device=device)

    elif fill_method == 'mean_per_channel':
        channel_means  = images.to(torch.float32).mean(dim=(2, 3))
        channel_means = channel_means.to(torch.int32)
        contents = torch.broadcast_to(channel_means[:, :, None, None], image_shape)

    elif fill_method == 'random':
        color = torch.randint(0, 256, (image_shape[0], image_shape[1]), 
                             dtype=torch.long, device=device)
        contents = torch.broadcast_to(color[:, :, None, None], image_shape)

    elif fill_method == 'noise':
        contents = torch.randint(0, 256, image_shape, dtype=torch.long, device=device)

    return contents


def mix_augmented_images(
    original_images: torch.Tensor,
    augmented_images: torch.Tensor,
    augmentation_ratio: int | float = 1.0,
    bernoulli_mix: bool = False
) -> torch.Tensor:

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
    assert original_images.shape == augmented_images.shape, (
        'Function `mix_augmented_images`: original '
        'and augmented images must have the same shape'
    )

    device = original_images. device
    image_shape = original_images.shape
    batch_size = image_shape[0]

    if augmentation_ratio == 0.0:
        mixed_images = original_images
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    elif augmentation_ratio == 1.0:
        mixed_images = augmented_images
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    else:

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if original_images.dim() == 3:
            original_images = original_images.unsqueeze(-1)
            augmented_images = augmented_images.unsqueeze(-1)

        if bernoulli_mix:
            # For each image position in the output mix, make a Bernoulli
            # experiment to decide if the augmented image takes it.
            probs = torch.rand(batch_size, device=device)
            mask = probs < augmentation_ratio
        else:
            # Calculate the number of augmented images in the mix
            num_augmented = round(augmentation_ratio * batch_size)

            # Generate a mask set to True for positions in the mix
            # occupied by augmented images, False by original images
            grid = torch.arange(batch_size, device=device)
            mask = grid < num_augmented

            # Shuffle the mask so that the augmented images
            # are at random positions in the output mix
            mask = mask[torch.randperm(batch_size, device=device)]

        # Apply the mask to images to generate the output mix
        mixed_images = torch.where(mask.view(-1, 1, 1, 1), augmented_images, original_images)

        # Restore the image input shape
        mixed_images = mixed_images.reshape(image_shape)

    return mixed_images, mask


def rescale_pixel_values(
    images: torch.Tensor,
    input_range: tuple[int | float, int | float],
    output_range: tuple[int | float, int | float],
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:

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

        images = images.float()
        images = ((output_max - output_min) * images +
                   output_min * input_max - output_max * input_min) / (input_max - input_min)
        
        # Clip to the output range
        images = torch.clamp(images, output_min, output_max)

    return images.to(dtype)
