
import math
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_dataaug_function_arg, check_augment_mix_args, check_fill_method_arg, check_pixels_range_args
from dataaug_utils import sample_patch_dims, sample_patch_locations, gen_patch_mask, mix_augmented_images


class RandomCutMix(v2.Transform):
    """
    Applies the "Cutout" data augmentation technique to a batch of images.

    Reference paper:
        Terrance DeVries, Graham W. Taylor (2017). “Improved Regularization 
        of Convolutional Neural Networks with Cutout”.

    For each image in the batch, a square patch centered at a random location
    is erased and filled with solid color or noise. All the patches have 
    the same size and are entirely contained inside the images.

    Patch locations are sampled independently for each image, ensuring variety 
    across the batch.

    Args:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images
            Pixel values are expected to be in the range specified by 
            the `pixels_range` argument.

        patch_area:
            A float specifying the patch area as a fraction of the image area.
            Values must be > 0 and < 1.

        fill_method:
            A string specifying how to fill the erased patches.  
            Options:  
            - 'black': filled with black  
            - 'gray': filled with mid-gray (128)  
            - 'white': filled with white  
            - 'mean_per_channel': filled with the mean color of the image channels  
            - 'random': filled with random solid colors  
            - 'noise': filled with random pixel values

        pixels_range:
            Tuple or list of two numbers specifying the expected min and max
            pixel values in the input images, e.g. (0, 255), (0, 1), (-1, 1).
            This ensures the fill values are scaled correctly.

        augmentation_ratio:
            A float in the interval [0, 1] specifying the augmented/original
            images ratio in the output mix. If set to 0, no images are augmented.
            If set to 1, all the images are augmented.

        bernoulli_mix:
            A boolean specifying the method to use to mix the original and augmented
            images in the output images:
              - False: the augmented/original ratio is equal to `augmentation_ratio`
                for every batch.
              - True: the augmented/original ratio varies stochastically from batch
                to batch with an expectation equal to `augmentation_ratio`.
            Augmented images are at random positions in the output mix.

    Returns:
        A tensor of the same shape and dtype as the input images, containing a mix
        of original and Cutout-augmented images. Pixel values are in the same range
        as the input images.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.alpha = alpha
        self.patch_aspect_ratio = patch_aspect_ratio
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        # Check that all the arguments are valid
        self._check_arguments()


    def _check_arguments(self):
        """
        Checks the arguments passed to `RandomCutMix`
        """
        check_dataaug_function_arg(
            self.alpha,
            context={'arg_name': 'alpha', 'function_name' : 'RandomCutMix'},
            constraints={'format': 'number'}
        )

        check_dataaug_function_arg(
            self.patch_aspect_ratio,
            context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'RandomCutMix'},
            constraints={'format': 'tuple', 'min_val': ('>', 0)}
        )

        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, 'RandomCutMix')


    def _get_patch_dims(self, images, lambda_vals, patch_aspect_ratio):

        batch_size, _, img_height, img_width = images.shape

        # Sample aspect ratios for the patches
        aspect_ratio = torch.empty(batch_size, device=images.device).uniform_(
            patch_aspect_ratio[0], patch_aspect_ratio[1]
        )

        # Calculate patch areas
        area = torch.sqrt(1.0 - lambda_vals) * float(img_height * img_width)
        patch_w = torch.sqrt(area / aspect_ratio)
        patch_h = patch_w * aspect_ratio

        # Round and convert to integers
        patch_h = torch.round(patch_h).to(torch.int32)
        patch_w = torch.round(patch_w).to(torch.int32)

        # Clip patch dimensions to image size
        patch_h = torch.clamp(patch_h, 0, img_height)
        patch_w = torch.clamp(patch_w, 0, img_width)

        return patch_h, patch_w


    def forward(self, data) -> torch.Tensor:

        images, labels = data

        original_shape = images.shape

        # Reshape grayscale images [B, H, W] -> [B, H, W, 1]
        if images.ndim == 3:
            images = images.unsqueeze(-1)  # [B, H, W, 1]

        batch_size = images.shape[0]

        # Sample lambda values from Beta distribution
        gamma1 = torch.distributions.Gamma(self.alpha, 1.0).sample([batch_size])
        gamma2 = torch.distributions.Gamma(self.alpha, 1.0).sample([batch_size])
        lambda_vals = gamma1 / (gamma1 + gamma2)

        # Get patch sizes based on lambda and aspect ratio
        patch_size = self._get_patch_dims(images, lambda_vals, self.patch_aspect_ratio)

        # Sample patch locations and generate boolean mask
        patch_corners = sample_patch_locations(images, patch_size)
        patch_mask = gen_patch_mask(images, patch_corners)  # [B, H, W]

        # Randomly select other images in the batch
        shuffle_indices = torch.randperm(batch_size, device=images.device)
        shuffled_images = images[shuffle_indices]

        # Paste the patches from the shuffled images
        images_aug = torch.where(patch_mask[:, None, :, :], shuffled_images, images)

        # Recompute lambda to reflect actual patch size
        img_area = images.shape[1] * images.shape[2]
        patch_area = patch_size[0].float() * patch_size[1].float()
        lambda_vals = 1.0 - (patch_area / img_area)
        lambda_vals = lambda_vals[:, None]  # [B, 1]

        # Update labels
        labels = labels.float()
        shuffled_labels = labels[shuffle_indices]
        labels_aug = lambda_vals * labels + (1.0 - lambda_vals) * shuffled_labels

        # Mix the original and augmented images/labels
        output_images, augment_mask = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)
        output_labels = torch.where(augment_mask[:, None], labels_aug, labels)

        # Restore original image shape
        output_images = output_images.reshape(original_shape)

        return output_images, output_labels
    