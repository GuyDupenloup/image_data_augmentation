# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)

from cutout import RandomCutout
from erasing import RandomErasing
from hide_and_seek import RandomHideAndSeek
from grid_mask import RandomGridMask
from cutblur import RandomCutBlur
from cutpaste import RandomCutPaste
from cutswap import RandomCutSwap
from cut_thumbnail import RandomCutThumbnail
from cutmix import RandomCutMix
from mixup import RandomMixup


def _get_data_loader(image_size, batch_size, seed):
    """
    Loads the Flowers dataset from torchvision and returns a DataLoader.
    Images are resized and normalized to [0,1].
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()  # converts to float32 in [0,1]
    ])

    ds = datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)

    return data_loader


# ----------------------------------------
# Display images
# ----------------------------------------
def _display_images(image, image_aug, legend):
    """
    Displays original and augmented images side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    img_orig = image.permute(1,2,0).cpu().numpy() if image.ndim == 3 else image.cpu().numpy()
    img_aug = image_aug.permute(1,2,0).cpu().numpy() if image_aug.ndim == 3 else image_aug.cpu().numpy()

    ax1.imshow(img_orig, cmap='gray' if img_orig.ndim==2 else None)
    ax1.set_title('Original image')

    ax2.imshow(img_aug, cmap='gray' if img_aug.ndim==2 else None)
    ax2.set_title(f'Augmented: {legend}')

    plt.tight_layout()
    plt.show()
    plt.close()


def _augment_images(images, labels, function):
    """
        Calls the data augmentation functions and returns the augmented images
    """
    if function == "RandomCutout":
        cutout = RandomCutout(
            patch_area=0.1,
            fill_method='mean_per_channel',
            pixels_range=(0, 1),
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = cutout(images)

    elif function == "RandomErasing":
        erasing = RandomErasing(
            patch_area=(0.05, 0.2),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            fill_method='mean_per_channel',
            pixels_range=(0, 1),
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = erasing(images)

    elif function == "RandomHideAndSeek":
        hide_and_seek = RandomHideAndSeek(
            grid_size=(4, 4),
            erased_patches=(0, 5),
            fill_method='white',
            pixels_range=(0, 1),
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = hide_and_seek(images)

    elif function == "RandomGridMask":
        grid_mask = RandomGridMask(
            unit_length=(0.2, 0.4),
            masked_ratio=0.5,
            fill_method='gray',
            pixels_range=(0, 1),
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = grid_mask(images)

    elif function == "RandomCutBlur":
        cutblur = RandomCutBlur(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            blur_factor=0.1,
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = cutblur(images)

    elif function == "RandomCutPaste":
        cutpaste = RandomCutPaste(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = cutpaste(images)

    elif function == "RandomCutSwap":
        cutswap = RandomCutSwap(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = cutswap(images)

    elif function == "RandomCutThumbnail":
        cut_thumbnail = RandomCutThumbnail(
            thumbnail_area=0.1,
            resize_method='bilinear',
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug = cut_thumbnail(images)

    elif function == "RandomCutMix":
        cutmix = RandomCutMix(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug, _ = cutmix((images, labels))

    elif function == "RandomMixup":
        mixup = RandomMixup(
            alpha=1.0,
            augmentation_ratio=1.0,
            bernoulli_mix=False
        )
        images_aug, _ = mixup((images, labels))

    else:
        raise ValueError(f"Unknown function: {function}")

    return images_aug


def run_test(image_size, images_per_function, grayscale, test_list, shuffling_seed):

    batch_size = max(images_per_function, 64)
    data_loader = _get_data_loader(image_size, batch_size, shuffling_seed)

    for i, (images, labels) in enumerate(data_loader):
        if grayscale:
            images = images[:, 0, :, :]  # take first channel

        print(f"Running '{test_list[i]}'")
        images_aug = _augment_images(images, labels, test_list[i])

        # Plot original vs augmented images
        for j in range(images_per_function):
            _display_images(images[j], images_aug[j], test_list[i])

        if i == len(test_list) - 1:
            return
        

if __name__ == '__main__':

    image_size = (224, 224)
    images_per_function = 4
    grayscale = False
    shuffling_seed = None   # Set to an int value to always see the same sequence of images

    test_list = [
        'RandomCutout',
        'RandomErasing',
        'RandomHideAndSeek',
        'RandomGridMask',
        'RandomCutBlur',
        'RandomCutPaste',
        'RandomCutSwap',
        'RandomCutThumbnail',
        'RandomCutMix',
        'RandomMixup',
    ]

    run_test(image_size, images_per_function, grayscale, test_list, shuffling_seed)
