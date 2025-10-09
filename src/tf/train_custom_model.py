# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import RandomContrast, RandomBrightness, RandomFlip, RandomRotation

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

tf.config.run_functions_eagerly(True)

class CustomModel(tf.keras.Model):

    def __init__(self, base_model, pixels_range, **kwargs):
        super().__init__(**kwargs)

        self.base_model = base_model

        # Initialize augmentation layers
        self.random_contrast = RandomContrast(factor=0.2)
        self.random_brightness = RandomBrightness(factor=0.2)
        self.random_flip = RandomFlip(mode='horizontal')
        self.random_rotation = RandomRotation(factor=0.05)

        self.random_cutout = RandomCutout(
            patch_area=0.3,
            fill_method='black',
            pixels_range=pixels_range,
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_erasing = RandomErasing(
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            fill_method='black',
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_hide_and_seek = RandomHideAndSeek(
            grid_size=(4, 4),
            erased_patches=(1, 5),
            fill_method='black',
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_grid_mask = RandomGridMask(
            unit_length=(0.2, 0.4),
            masked_ratio=0.5,
            fill_method='black',
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_cutblur = RandomCutBlur(
            patch_area=(0.2, 0.4),
            patch_aspect_ratio=(0.3, 0.4),
            blur_factor=0.2,
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_cut_thumbnail = RandomCutThumbnail(
            thumbnail_area=0.1,
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_cutpaste = RandomCutPaste(
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0),
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )
        self.random_cutswap = RandomCutSwap(
            patch_area=(0.1, 0.3),
            patch_aspect_ratio=(0.3, 2.0),
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        images, labels = data

        # Apply augmentations during training
        images = self.random_contrast(images, training=True)
        images = self.random_brightness(images, training=True)
        images = self.random_flip(images, training=True)
        images = self.random_rotation(images, training=True)

        images = self.random_cutout(images, training=True)
        images = self.random_erasing(images, training=True)
        images = self.random_hide_and_seek(images, training=True)
        images = self.random_grid_mask(images, training=True)
        images = self.random_cutblur(images, training=True)
        images = self.random_cut_thumbnail(images, training=True)
        images = self.random_cutpaste(images, training=True)
        images = self.random_cutswap(images, training=True)

        images, labels = random_cutmix(
            images,
            labels,
            patch_area=(0.05, 0.3),
            patch_aspect_ratio=(0.3, 3.0),
            alpha=1.0,
            augmentation_ratio=0.1,    # Augment 10% of images
            bernoulli_mix=False
        )
        images, labels = random_mixup(
            images,
            labels,
            alpha=1.0,
            augmentation_ratio=0.1,
            bernoulli_mix=False
        )

        with tf.GradientTape() as tape:
            # Make a prediction and compute the loss value
            pred_labels = self(images, training=True)
            loss = self.compute_loss(y=labels, y_pred=pred_labels)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (fixed variable names)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(labels, pred_labels)

        # Return metrics dict
        return {m.name: m.result() for m in self.metrics}


def _get_data_loaders(image_size, num_classes, batch_size, rescaling):
    """
    Creates training and validation data loaders for the Flowers dataset
    """
    def preprocess_image(image, label):
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        image = rescaling[0] * image + rescaling[1]
        # One-hot labels are required for CutMix and Mixup
        label = tf.one_hot(label, num_classes, on_value=1.0, off_value=0.0, dtype=tf.float32)
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


def get_base_model(input_shape, num_classes):
    # Get a ResNet50 model
    resnet_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the model backbone (transfer learning)
    resnet_model.trainable = False

    # Add classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name='ResNet50')


def train():

    image_shape = (64, 64, 3)
    rescaling = (1/255., 0)
    num_classes = 5
    batch_size = 32
    epochs = 1
    
    pixels_range = (0, 1)

    # Get dataloaders for the Flowers dataset
    train_ds, val_ds = _get_data_loaders(image_shape[:2], num_classes, batch_size, rescaling)

    # Create custom model
    base_model = get_base_model(image_shape, num_classes)
    model = CustomModel(base_model, pixels_range)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),   # One-hot labels
        metrics=['accuracy']
    )

    # Set up callback to save the best model weights obtained during training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'best_model.weights.h5',
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max"
    )

    # Train the model
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_callback])

    # Load the best weights after training
    print("Loading best weights")
    model.load_weights('best_model.weights.h5')
    
    # Evaluate the model with best weights
    final_loss, final_accuracy = model.evaluate(val_ds)
    print(f"Final validation accuracy with best weights: {final_accuracy:.4f}")
 

if __name__ == '__main__':
    train()
