import math
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_dataaug_function_arg, check_augment_mix_args
from dataaug_utils import sample_patch_dims, sample_patch_locations, gen_patch_mask, mix_augmented_images


class RandomCutblur(v2.Transform):
    """
    Applies the "CutBlur" data augmentation technique to a batch of images.

    Reference paper:
        Jaejun Yoo, Namhyuk Ahn, and Kyung-Ah Sohn (2020). "Rethinking Data Augmentation 
        for Image Super-Resolution: : A comprehensive analysis and a new strategy".

    For each image in the batch:
    1. A low-resolution version of the image is created by downscaling and then
       upscaling it back to the original size.
    2. A random rectangular patch is cropped from the low-resolution image.
    3. The patch is pasted at the same location in the original image, producing
       an image identical to the original except for the blurred region.

    Patch sizes and locations are sampled independently for each image, ensuring 
    variety across the batch.

    By default, the augmented/original image ratio in the output mix is `1.0`. 
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.  
            Supported shapes:  
                [B, H, W, 3]  -->  Color  
                [B, H, W, 1]  -->  Grayscale  
                [B, H, W]     -->  Grayscale  

        patch_area:
            A tuple of two floats specifying the range from which patch areas 
            are sampled. Values must be > 0 and < 1, representing fractions 
            of the image area.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch height/width
            aspect ratios are sampled. Values must be > 0.

        blur_factor:
            A float specifying the size of the low-resolution images (they all 
            have the same size). Values must be > 0 and < 1, representing fractions 
            of the image size. Smaller values produce stronger blur.

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
        of original and CutBlur-augmented images. Pixel values are in the same
        range as the input images.
    """

    def __init__(
        self,
        patch_area: tuple[float, float] = (0.05, 0.3),
        patch_aspect_ratio: tuple[float, float] = (0.3, 3.0),
        blur_factor: float = 0.1,
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.patch_area = patch_area
        self.patch_aspect_ratio = patch_aspect_ratio
        self.blur_factor = blur_factor
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_arguments()
		

    def _check_arguments(self):
        """
        Checks the arguments passed to `RandomCutBlur`
        """

        check_dataaug_function_arg(
            self.patch_area,
            context={'arg_name': 'patch_area', 'function_name' : 'random_cutblur'},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )
        check_dataaug_function_arg(
            self.patch_aspect_ratio,
            context={'arg_name': 'patch_aspect_ratio', 'function_name' : 'random_cutblur'},
            constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
        )
        check_dataaug_function_arg(
            self.blur_factor,
            context={'arg_name': 'blur_factor', 'function_name' : 'random_cutblur'},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )

        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, 'RandomCutBlur')


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        device = images.device

        # Reshape images with shape [batch_size, height, width]
        # to shape [batch_size, height, width, 1]
        original_image_shape = images.shape
        if images.ndim == 3:  # i.e., [B, H, W]
            images = images.unsqueeze(1)  # insert a channel dimension at index 1

        # Original image size as ints for interpolate
        original_size = images.shape[2:]

        # Low-res size
        image_size = torch.tensor(images.shape[2:], dtype=torch.float32, device=device)
        low_res_size = self.blur_factor * image_size
        low_res_size = torch.round(low_res_size).to(torch.int64)

        # Downsize then up-size images
        smaller_images = F.interpolate(images, size=tuple(low_res_size.tolist()), mode='bilinear', align_corners=False)
        low_res_images = F.interpolate(smaller_images, size=original_size, mode='bilinear', align_corners=False)

        # Match dtype
        low_res_images = low_res_images.to(images.dtype)

        # Generate random patches
        patch_dims = sample_patch_dims(images, self.patch_area, self.patch_aspect_ratio)
        patch_corners = sample_patch_locations(images, patch_dims)
        patch_mask = gen_patch_mask(images, patch_corners)

        # Erase the patches from the images and fill them with the low-res images
        images_aug = torch.where(patch_mask[:, None, :, :], low_res_images, images)

        # Mix the original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore input images shape
        output_images = output_images.reshape(original_image_shape)

        return output_images
    