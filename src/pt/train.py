# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.models import mobilenet_v3_small

from cutout import RandomCutout
from erasing import RandomErasing
from hide_and_seek import RandomHideAndSeek
from grid_mask import RandomGridMask
from cutmix import RandomCutMix
from mixup import RandomMixup
from cutblur import RandomCutBlur
from cutpaste import RandomCutPaste
from cutswap import RandomCutSwap
from cut_thumbnail import RandomCutThumbnail


def get_data_loaders(image_size, rescaling, num_classes, batch_size, device):

    # Resize and rescale the images (using v2 transforms to run on GPU)
    transform = v2.Compose([
        v2.ToImage(),       # PIL -> Tensor
        v2.Resize(image_size),
        v2.ToDtype(torch.float32, scale=False),
        v2.Lambda(lambda t: t * rescaling[0] + rescaling[1]) 
    ])

    # Datasets with resizing and rescaling
    train_dataset = Flowers102(root='data', split='train', download=True, transform=transform)
    val_dataset   = Flowers102(root='data', split='val', download=True, transform=transform)

    # Stack images and labels, convert the labels to one-hot
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images).to(device)  # [B, C, H, W]
        labels = torch.tensor(labels, device=device)  # [B]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # [B, num_classes]
        return images, one_hot_labels

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def augment_data(images, labels, pixels_range):
 
    images = v2.ColorJitter(brightness=0.2, contrast=0.2)(images)
    images = v2.RandomHorizontalFlip(p=0.5)(images)

    images = RandomCutout(
        patch_area=0.1,
        fill_method='mean_per_channel',
        pixels_range=pixels_range,
        augmentation_ratio=1.0,
        bernoulli_mix=False
    )(images)

    images = RandomErasing(
        patch_area=(0.05, 0.3),
        patch_aspect_ratio=(0.3, 3.0),
        fill_method='black',
        pixels_range=pixels_range,
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)

    images = RandomHideAndSeek(
        grid_size=(4, 4),
        erased_patches=(1, 5),
        fill_method='black',
        pixels_range=pixels_range,
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)

    images = RandomGridMask(
        unit_length=(0.2, 0.4),
        masked_ratio=0.5,
        fill_method='black',
        pixels_range=pixels_range,
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)
	
    images, labels = RandomCutMix(
        patch_area=(0.05, 0.3),
        patch_aspect_ratio=(0.3, 3.0),
        alpha=1.0,
        augmentation_ratio=1.0,
        bernoulli_mix=False
    )((images, labels))

    images, labels = RandomMixup(
        alpha=1.0,
        augmentation_ratio=1.0,
        bernoulli_mix=False
    )((images, labels))
	
    images = RandomCutBlur(
        patch_area=(0.2, 0.4),
        patch_aspect_ratio=(0.3, 0.4),
        blur_factor=0.2,
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)

    images = RandomCutPaste(
        patch_area=(0.1, 0.3),
        patch_aspect_ratio=(0.3, 2.0),
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)

    images = RandomCutSwap(
        patch_area=(0.1, 0.3),
        patch_aspect_ratio=(0.3, 2.0),
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)

    images = RandomCutThumbnail(
        thumbnail_area=0.1,
        augmentation_ratio=0.1,
        bernoulli_mix=False
    )(images)

    return images, labels


def evaluate_model(model, val_loader, criterion):

    model.eval()

    # Track metrics
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            predicted_class = outputs.argmax(dim=1)
            true_class = labels.argmax(dim=1)
            correct += (predicted_class == true_class).sum().item()
            total += labels.size(0)

        # Compute average loss and accuracy
        val_loss = running_loss / total
        val_accuracy = correct / total

    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy*100:.2f}%")


def train():

    image_size = (224, 224)
    rescaling = (1/255., 0)
    pixels_range = (0, 1)

    num_classes = 102
    batch_size = 32
    num_epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(image_size, rescaling, num_classes, batch_size, device)

    # Get a small model for testing purposes
    model = mobilenet_v3_small(weights=None, num_classes=num_classes, width_mult=0.1)

    # Loss function for one-hot encoded labels
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Set model to training mode
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            images, labels = augment_data(images, labels, pixels_range)

            # Forward pass and loss calculation
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and weights update
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * images.size(0)
            predicted_class = outputs.argmax(dim=1)
            true_class = labels.argmax(dim=1)
            correct += (predicted_class == true_class).sum().item()
            total += labels.size(0)

        # Epoch summary
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

    # Evaluate the train model
    evaluate_model(model, val_loader, criterion)


if __name__ == '__main__':
    train()
