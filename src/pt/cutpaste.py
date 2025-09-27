
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_patch_sampling_args, check_augment_mix_args
from dataaug_utils import sample_patch_sizes, sample_patch_locations, gen_patch_mask, mix_augmented_images


class RandomCutPaste(v2.Transform):
    """
    Applies the "CutPaste" data augmentation technique to a batch of images.

    Reference paper:
        Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister (2021).
        "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization."

    For each image in the batch, the function:
    1. Samples a patch size based on the specified area and aspect ratio ranges.
    2. Chooses two random locations in the image for a source patch and a target
       patch, both with the sampled size. The two patches may overlap.
    3. Copies the contents of the source patch into the target patch, leaving 
       the source region unchanged.

    Patch sizes and locations are sampled independently for each image, ensuring 
    variety across the batch.

    By default, the augmented/original image ratio in the output mix is `1.0`. 
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images

        patch_area:
            A tuple of two floats specifying the range from which patch areas 
            are sampled. Values must be > 0 and < 1, representing fractions 
            of the image area.
            Patch areas are sampled from a Beta distribution. See `alpha` argument.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Minimum value must be > 0.
            Patch aspect ratios are sampled from a uniform distribution.

        alpha:
            A float specifying the alpha parameter of the Beta distribution used
            to sample patch areas. Set to 0 by default, making the distribution
            uniform.

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
        of original and CutPaste-augmented images.
    """

    def __init__(
        self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        alpha=1.0,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.transform_name = 'RandomCutPaste'
        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.alpha = alpha
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_transform_args()


    def _check_transform_args(self):
        """
        Checks that the arguments passed to the transform are valid
        """
        check_patch_sampling_args(self.patch_area, self.patch_aspect_ratio, self.alpha, self.transform_name)
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, self.transform_name)


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        # Preserve original shape
        original_image_shape = images.shape
        device = images.device  # ensure we keep track of the device

        # Reshape [B,H,W] -> [B,1,H,W] if needed
        if images.ndim == 3:
            images = images.unsqueeze(1)  # [B,1,H,W]

        # Sample patches sizes
        patch_sizes = sample_patch_sizes(images, self.patch_area, self.patch_aspect_ratio, self.alpha)

        # Sample locations of the source patches, then generate
        # a boolean mask (True inside patches, False outside)
        source_corners = sample_patch_locations(images, patch_sizes)
        source_mask = gen_patch_mask(images, source_corners)
        source_mask = source_mask.unsqueeze(1).to(device)    # Add channel dim

        # Sample locations of the target patches, then generate
        # a boolean mask (True inside patches, False outside)
        target_corners = sample_patch_locations(images, patch_sizes)
        target_mask = gen_patch_mask(images, target_corners)
        target_mask = target_mask.unsqueeze(1).to(device)    # Add channel dim

        # Gather source patch contents (all channels)
        # use mask.nonzero so indices are on the same device as the mask
        source_indices = source_mask.nonzero(as_tuple=False)
        source_contents = images[source_indices[:, 0], :,  # all channels
                                source_indices[:, 2], source_indices[:, 3]]  # [num_pixels, C]

        # Scatter into target patches (all channels)
        target_indices = target_mask.nonzero(as_tuple=False)
        images_aug = images.clone()
        images_aug[target_indices[:, 0], :,  # all channels
                target_indices[:, 2], target_indices[:, 3]] = source_contents

        # Mix original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)
        output_images = output_images.to(device)  # ensure result on same device

        # Restore shape of input images
        output_images = output_images.reshape(original_image_shape)

        return output_images
