# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import math
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union


def gen_patch_sizes(
    images: torch.Tensor,
    patch_area: tuple[float, float],
    patch_aspect_ratio: tuple[float, float],
    alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Samples heights and widths of patches

    Arguments:
        image_shape:
            The shape of the images (4D tensor).

        patch_area:
            A tuple of two floats specifying the range from which patch areas
            are sampled. Values must be > 0 and < 1, representing fractions 
            of the image area.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch 
            height/width aspect ratios are sampled. Values must be > 0.

    Returns:
        A tuple of 2 tensors with shape [batch_size]:
            `(patch_height, patch_width)`. 
    """

    device = images.device
    batch_size, _, img_height, img_width = images.shape

    if isinstance(patch_area, (tuple, list)):
        # Sample area fractions from Beta distribution
        gamma1 = torch.distributions.Gamma(alpha, 1.0).sample([batch_size]).to(device)
        gamma2 = torch.distributions.Gamma(alpha, 1.0).sample([batch_size]).to(device)
        lambda_vals = gamma1 / (gamma1 + gamma2)
        # Linearly rescale to the specified area range (this does not change the distribution)
        area_fraction = patch_area[0] + lambda_vals * (patch_area[1] - patch_area[0])
    else:
        # Constant area fraction
        area_fraction = torch.full([batch_size], patch_area, dtype=torch.float32, device=device)

    if isinstance(patch_aspect_ratio, (tuple, list)):
        # Sample patch aspect ratios from a uniform distribution
        # Aspect ratios are non-linear. We use logs for the sampling range
        # to get better balance between tall and wide rectangles.
        log_min = math.log(patch_aspect_ratio[0])
        log_max = math.log(patch_aspect_ratio[1])
        log_aspect = torch.empty(batch_size, device=device).uniform_(log_min, log_max)
        aspect_ratio = torch.exp(log_aspect)
    else:
        # Constant aspect ratio
        aspect_ratio = torch.full([batch_size],  patch_aspect_ratio, dtype=torch.float32, device=device)

    # Compute patch width and height
    area = area_fraction * float(img_width * img_height)
    patch_w = torch.sqrt(area / aspect_ratio)
    patch_h = patch_w * aspect_ratio

    # Round to nearest int, cast to int32
    patch_h = torch.round(patch_h).to(torch.int32)
    patch_w = torch.round(patch_w).to(torch.int32)

    # Clip to stay within image bounds
    patch_h = torch.clamp(patch_h, min=0, max=img_height)
    patch_w = torch.clamp(patch_w, min=0, max=img_width)

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

    # Sample random positions uniformly in [0, 1]
    x_rand = torch.rand(batch_size, device=device)
    y_rand = torch.rand(batch_size, device=device)

    # Compute valid start positions so patches stay inside image
    max_x1 = (torch.tensor(img_width, device=device) - patch_w).to(torch.float32)
    max_y1 = (torch.tensor(img_height, device=device) - patch_h).to(torch.float32)

    # Scale random positions to valid range & round
    x1 = torch.round(x_rand * max_x1).to(torch.int32)
    y1 = torch.round(y_rand * max_y1).to(torch.int32)

    # Get opposite corners coordinates
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    corners = torch.stack([y1, x1, y2, x2], dim=-1)

    # Create coordinate grids (on GPU)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(img_width, device=device),
        torch.arange(img_height, device=device),
        indexing='xy'
    )

    # Broadcast to [B, H, W] and apply patch logic
    mask = (
        (grid_x.unsqueeze(0) >= x1.view(-1, 1, 1)) &
        (grid_x.unsqueeze(0) <  x2.view(-1, 1, 1)) &
        (grid_y.unsqueeze(0) >= y1.view(-1, 1, 1)) &
        (grid_y.unsqueeze(0) <  y2.view(-1, 1, 1))
    )

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
    batch_size, channels = image_shape[:2]

    # Ensure integer math stays in 32-bit range
    images = images.to(torch.int32)

    if fill_method == "black":
        contents = torch.zeros(image_shape, dtype=torch.int32, device=device)

    elif fill_method == "gray":
        contents = torch.full(image_shape, 128, dtype=torch.int32, device=device)

    elif fill_method == "white":
        contents = torch.full(image_shape, 255, dtype=torch.int32, device=device)

    elif fill_method == "mean_per_channel":
        # Compute mean per channel per image, keep int precision
        channel_means = images.to(torch.float32).mean(dim=(2, 3))
        channel_means = channel_means.round().to(torch.int32)
        contents = channel_means[:, :, None, None].expand(image_shape)

    elif fill_method == "random":
        # Random color per image per channel
        color = torch.randint(0, 256, (batch_size, channels), dtype=torch.int32, device=device)
        contents = color[:, :, None, None].expand(image_shape)

    elif fill_method == "noise":
        contents = torch.randint(0, 256, image_shape, dtype=torch.int32, device=device)

    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    return contents


def rescale_pixel_values(
    images: torch.Tensor,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Linearly rescales pixel values of images from one range to another.

    A linear transformation is applied so that values in `input_range`
    are mapped to `output_range`. Output is clipped and cast to `dtype`.

    Example:
        # Convert uint8 images [0, 255] to float32 [0.0, 1.0]
        images = rescale_pixel_values(images, (0, 255), (0.0, 1.0), torch.float32)
    """
    if input_range != output_range:
        device = images.device
        input_min, input_max = input_range
        output_min, output_max = output_range

        # Ensure float math on GPU
        images = images.to(torch.float32)

        # Create tensors on the same device as images
        input_min = torch.tensor(input_min, device=device, dtype=torch.float32)
        input_max = torch.tensor(input_max, device=device, dtype=torch.float32)
        output_min = torch.tensor(output_min, device=device, dtype=torch.float32)
        output_max = torch.tensor(output_max, device=device, dtype=torch.float32)

        # Fully vectorized linear transform
        images = ((output_max - output_min) * images +
                  output_min * input_max - output_max * input_min) / (input_max - input_min)

        # Clip to output range on GPU
        images = torch.clamp(images, min=output_min, max=output_max)

    return images.to(dtype)


def mix_augmented_images(
    original_images: torch.Tensor,
    augmented_images: torch.Tensor,
    augmentation_ratio: float = 1.0,
    bernoulli_mix: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixes original and augmented images according to a specified
    augmented/original ratio and method. Fully vectorized and GPU-friendly.
    """

    # Ensure inputs have the same shape
    if original_images.shape != augmented_images.shape:
        raise ValueError("original_images and augmented_images must have the same shape")

    device = original_images.device
    batch_size = original_images.shape[0]

    if augmentation_ratio == 0.0:
        mixed_images = original_images
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    elif augmentation_ratio == 1.0:
        mixed_images = augmented_images
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    else:
        # Ensure shape is [B, H, W, C] if input was [B, H, W]
        if original_images.ndim == 3:
            original_images = original_images.unsqueeze(-1)
            augmented_images = augmented_images.unsqueeze(-1)

        if bernoulli_mix:
            # Bernoulli sampling: each position independent
            mask = torch.rand(batch_size, device=device) < augmentation_ratio
        else:
            # Deterministic number of augmented images
            batch_size_tensor = torch.tensor(batch_size, device=device, dtype=torch.float32)
            augmentation_ratio_tensor = torch.tensor(augmentation_ratio, device=device, dtype=torch.float32)
            num_augmented = int(torch.round(batch_size_tensor * augmentation_ratio_tensor).item())
            
            mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            mask[:num_augmented] = True
            # Shuffle mask on GPU
            perm = torch.randperm(batch_size, device=device)
            mask = mask[perm]

        # Apply mask: broadcasting over H, W, C
        mixed_images = torch.where(mask.view(-1, 1, 1, 1), augmented_images, original_images)

        # Restore to original input shape
        mixed_images = mixed_images.view_as(original_images)

    return mixed_images, mask
