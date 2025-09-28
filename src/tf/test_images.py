# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from cutout import RandomCutout
from erasing import RandomErasing
from hide_and_seek import RandomHideAndSeek
from grid_mask import RandomGridMask
from cutblur import RandomCutBlur
from cutpaste import RandomCutPaste
from cutswap import RandomCutSwap
from cut_thumbnail import RandomCutThumbnail
from cutmix import random_cutmix
from mixup import random_mixup


def _get_data_loader(image_size, batch_size, seed):

    # Load dataset info to get number of classes
    ds_info = tfds.builder('tf_flowers').info
    num_classes = ds_info.features['label'].num_classes

    def preprocess_image(image, label):
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes, dtype=tf.float32)
        return image, label
    
    ds = tfds.load('tf_flowers', split='train', as_supervised=True)
    ds = ds.shuffle(buffer_size=1000, seed=seed)
    ds = ds.map(preprocess_image)
    ds = ds.batch(batch_size)

    return ds


def _display_images(image, image_aug, function_name):
    """
    Displays original and augmented images side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original image')
    ax2.imshow(image_aug, cmap='gray')
    ax2.set_title(f'Augmented: {function_name}')
    plt.tight_layout()
    plt.show()
    plt.close()


def _augment_images(images, labels, function):
    """
    Calls the data augmentation functions and returns the augmented images
    """
    
    if function == 'RandomCutout':
        cutout = RandomCutout(
            patch_area=0.1,
            fill_method='black'
        )
        images_aug = cutout(images)

    elif function == 'RandomErasing':
        erasing = RandomErasing(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            fill_method='black'
        )
        images_aug = erasing(images)

    elif function == 'RandomHideAndSeek':
        hide_and_seek = RandomHideAndSeek(
            grid_size=(4, 4),
            erased_patches=(1, 5),
            fill_method='black'
        )
        images_aug = hide_and_seek(images)

    elif function == 'RandomGridMask':
        grid_mask = RandomGridMask(
            unit_length=(0.2, 0.4),
            masked_ratio=0.5,
            fill_method='black'
        )
        images_aug = grid_mask(images)

    elif function == 'RandomCutBlur':
        cutblur = RandomCutBlur(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            blur_factor=0.2
        )
        images_aug = cutblur(images)

    elif function == 'RandomCutPaste':
        cutpaste = RandomCutPaste(
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,

        )
        images_aug = cutpaste(images)

    elif function == 'RandomCutSwap':
        cutsawp = RandomCutSwap(
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0),
            alpha=1.0,
        )
        images_aug = cutsawp(images)

    elif function == 'RandomCutThumbnail':
        cut_thumbnail = RandomCutThumbnail(
            thumbnail_area=0.1
        )
        images_aug = cut_thumbnail(images)

    elif function == 'random_cutmix':
        images_aug, _ = random_cutmix(
            images,
            labels,
            alpha=1.0,
            patch_aspect_ratio=(0.3, 3.0),
        )

    elif function == 'random_mixup':
        images_aug, _ = random_mixup(
            images,
            labels,
            alpha=1.0
        )
    else:
        raise ValueError(f"Unknown data augmentation function `{function}`")

    return images_aug


def run_test(image_size, images_per_function, grayscale, test_list, shuffling_seed):

    # Set the minimum batch size to 64 (useful for functions
    # that sample another image from the batch)
    batch_size = max(images_per_function, 64)

    data_loader = _get_data_loader(image_size, batch_size, shuffling_seed)

    for i, (images, labels) in enumerate(data_loader):
        if grayscale:
            images = images[..., 0]

        print(f"Running '{test_list[i]}'")
        images_aug = _augment_images(images, labels, test_list[i])

        # Plot the original and augmented images side-by-side
        for j in range(images_per_function):
            _display_images(images[j], images_aug[j], test_list[i])

        # Exit when all the functions have tested
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
        'random_cutmix',
        'random_mixup'
    ]

    run_test(image_size, images_per_function, grayscale, test_list, shuffling_seed)
