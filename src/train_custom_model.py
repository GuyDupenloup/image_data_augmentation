# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import RandomContrast, RandomBrightness, RandomFlip, RandomRotation
from cutmix import random_cutmix


class CustomModel(tf.keras.Model):

    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        
        # Initialize Tensorflow augmentation layers
        self.contrast_aug = RandomContrast(factor=0.2)
        self.brightness_aug = RandomBrightness(factor=0.2)
        self.flip_aug = RandomFlip(mode='horizontal')
        self.rotation_aug = RandomRotation(factor=0.05)
    
    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        images, labels = data

        # Apply augmentations during training
        images = self.contrast_aug(images, training=True)
        images = self.brightness_aug(images, training=True)
        images, labels = random_cutmix(
            images,
            labels,
            alpha=1.0,
            patch_aspect_ratio=(0.3, 3.0),
            augmentation_ratio=0.1,   # Use 10% augmented images
            bernoulli_mix=False
        )
        images = self.flip_aug(images, training=True)
        images = self.rotation_aug(images, training=True)

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


def train():

    image_shape = (128, 128, 3)
    rescaling = (1/255., 0)
    num_classes = 5
    batch_size = 32
    epochs = 5
    
    # Get dataloaders for the Flowers dataset
    train_ds, val_ds = _get_data_loaders(image_shape[:2], num_classes, batch_size, rescaling)

    # Get a ResNet50 model
    resnet_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )
    
    # Freeze the model backbone (transfer learning)
    resnet_model.trainable = False

    # Add classification head
    inputs = tf.keras.Input(shape=image_shape)
    x = resnet_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    base_model = tf.keras.Model(inputs, outputs)

    # Create custom model
    model = CustomModel(base_model)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),   # One-hot encoded labels
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
