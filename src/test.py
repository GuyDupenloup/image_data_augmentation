# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from hide_and_seek import random_hide_and_seek
from cutout import random_cutout
from erasing import random_erasing
from grid_mask import random_grid_mask
from cutblur import random_cutblur
from cutpaste import random_cutpaste
from cutswap import random_cutswap
from cut_thumbnail import random_cut_thumbnail
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
    
    if function == 'random_cutout':
        return random_cutout(
            images,
            patch_area=0.1,
            fill_method='black'
        )
    elif function == 'random_erasing':
        return random_erasing(
            images,
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            fill_method='noise'
        )
    elif function == 'random_hide_and_seek':
        return random_hide_and_seek(
            images,
            grid_size=(4, 4),
            erased_patches=(1, 5),
            fill_method='mean_per_channel'
        )
    elif function == 'random_grid_mask':
        return random_grid_mask(
            images,
            unit_length=(0.2, 0.4),
            masked_ratio=0.5,
            fill_method='gray'
        )
    elif function == 'random_cutblur':
        return random_cutblur(
            images,
            patch_area=(0.2, 0.4),
            patch_aspect_ratio=(0.3, 0.4),
            blur_factor=0.2
        )
    elif function == 'random_cutpaste':
        return random_cutpaste(
            images,
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0)
        )
    elif function == 'random_cutswap':
        return random_cutswap(
            images,
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0)
        )
    elif function == 'random_cut_thumbnail':
        return random_cut_thumbnail(
            images,
            thumbnail_area=0.1
        )
    elif function == 'random_cutmix':
        images_aug, _ = random_cutmix(
            images,
            labels,
            alpha=1.0,
            patch_aspect_ratio=(0.3, 3.0),
        )
        return images_aug
    elif function == 'random_mixup':
        images_aug, _ = random_mixup(
            images,
            labels,
            alpha=1.0
        )
        return images_aug
    else:
        raise ValueError(f"Unknown data augmentation function `{function}`")


def run_test(image_size, images_per_function, grayscale, functions_to_test, shuffling_seed):

    # Set the minimum batch size to 64 (useful for functions
    # that sample another image from the batch)
    batch_size = max(images_per_function, 64)

    data_loader = _get_data_loader(image_size, batch_size, shuffling_seed)

    for i, (images, labels) in enumerate(data_loader):
        if grayscale:
            images = images[..., 0]

        print(f"Running function '{functions_to_test[i]}'")
        images_aug = _augment_images(images, labels, functions_to_test[i])

        # Plot the original and augmented images side-by-side
        for j in range(images_per_function):
            _display_images(images[j], images_aug[j], functions_to_test[i])

        # Exit when all the functions have tested
        if i == len(functions_to_test) - 1:
            return


if __name__ == '__main__':

    image_size = (224, 224)
    images_per_function = 4
    grayscale = False
    shuffling_seed = None   # Set to an int value to always see the same sequence of images

    functions_to_test = [
        'random_cutout', 'random_erasing', 'random_hide_and_seek',
        'random_grid_mask', 'random_cutblur', 'random_cutpaste',
        'random_cutswap', 'random_cut_thumbnail', 'random_cutmix',
        'random_mixup'
    ]

    run_test(image_size, images_per_function, grayscale, functions_to_test, shuffling_seed)
