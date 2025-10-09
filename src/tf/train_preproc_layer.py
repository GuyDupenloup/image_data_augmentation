# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Rescaling, RandomContrast, RandomBrightness, RandomFlip, RandomRotation

from cutout import RandomCutout
from erasing import RandomErasing
from hide_and_seek import RandomHideAndSeek
from grid_mask import RandomGridMask
from cutblur import RandomCutBlur
from cutpaste import RandomCutPaste
from cutswap import RandomCutSwap
from cut_thumbnail import RandomCutThumbnail


def _get_data_loaders(image_size, batch_size):
    """
    Creates training and validation data loaders for the Flowers dataset
    """
    def preprocess_image(image, label):
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label

    # Get the Flowers dataset and take 20% for validation
    dataset, info = tfds.load('tf_flowers', as_supervised=True, with_info=True)
    num_examples = info.splits['train'].num_examples
    val_size = int(num_examples * 0.2)

    print(f"Total number of examples: {num_examples}")
    print(f"Validation set size: {val_size}")

    # Create the training and validation data loaders
    full_ds = dataset['train'].map(preprocess_image)
    val_ds = full_ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = full_ds.skip(val_size).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def _get_resnet(input_shape, num_classes):
    # Get a ResNet50 model
    resnet = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the model backbone (transfer learning)
    resnet.trainable = False

    # Add classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def _get_mobilenet(input_shape, num_classes):
    mobilenet = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        alpha=0.25,
        weights='imagenet',   # use pretrained weights
        include_top=False     # drop the original 1000-class head
    )

    # Freeze the model backbone (transfer learning)
    mobilenet.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = mobilenet(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def _create_augmented_model(base_model, rescaling, pixels_range):
    """
    Adds preprocessing layers to the model to train
    """
    model_layers = []
    model_layers.append(tf.keras.Input(shape=base_model.input.shape[1:]))

    # Add a rescaling layer
    model_layers.append(Rescaling(rescaling[0], rescaling[1]))
    
    # Add the data augmentation layers
    # For demonstration purposes only
    model_layers.append(RandomContrast(0.4))
    model_layers.append(RandomBrightness(0.5))
    model_layers.append(RandomFlip(mode='horizontal'))
    model_layers.append(
        RandomCutout(
            patch_area=0.3,
            fill_method='black',
            pixels_range=pixels_range,
            augmentation_ratio=0.1,   # Augment 10% of images
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomErasing(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            fill_method='black',
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomHideAndSeek(
            grid_size=(4, 4),
            erased_patches=(1, 5),
            fill_method='black',
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomGridMask(
            unit_length=(0.2, 0.4),
            masked_ratio=0.5,
            fill_method='black',
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomCutBlur(
            patch_area=(0.2, 0.4),
            patch_aspect_ratio=(0.3, 0.4),
            blur_factor=0.2,
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomCutThumbnail(
            thumbnail_area=0.1,
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomCutPaste(
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0),
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )
    model_layers.append(
        RandomCutSwap(
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0),
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
    )

    model_layers.append(RandomRotation(factor=0.05))

    # Add the model to train and create the augmented model
    model_layers.append(base_model)
    augmented_model = tf.keras.Sequential(model_layers, name='augmented_model')

    return augmented_model


def train():

    image_shape = (128, 128, 3)
    rescaling = (1/255., 0)
    pixels_range = (0, 1)
    num_classes = 5
    batch_size = 32
    epochs = 5
    
    # Get dataloaders for the Flowers dataset
    train_ds, val_ds = _get_data_loaders(image_shape[:2], batch_size)

    # base_model = _get_resnet(image_shape, num_classes)
    base_model = _get_mobilenet(image_shape, num_classes)
 
    # Add the preprocessing layers
    augmented_model = _create_augmented_model(base_model, rescaling, pixels_range)

    # Compile the model
    augmented_model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    # Set up callback to save the best model weights obtained during training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'augmented_model.weights.h5',
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max"
    )

    # Train the augmented model
    augmented_model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_callback])

    # Load the best weights obtained during training
    augmented_model.load_weights('augmented_model.weights.h5')

    # Save the best model without the preprocessing layers
    best_model = augmented_model.layers[-1]
    best_model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    best_model.save('model.h5')

    # Evaluate the best model
    print('Evaluating best model')
    best_model.evaluate(val_ds, batch_size=32)


if __name__ == '__main__':
    train()
