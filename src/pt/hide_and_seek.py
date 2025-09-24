
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from dataaug_utils import check_dataaug_function_arg, rescale_pixel_values, mix_augmented_images


class RandomHideAndSeek(v2.Transform):
    """
    Applies the "Hide-and-Seek" data augmentation technique to a batch of images.

    Reference paper:
        Krishna Kumar Singh, Hao Yu, Aron Sarmasi, Gautam Pradeep, Yong Jae Lee (2018).
        “Hide-and-Seek: A Data Augmentation Technique for Weakly-Supervised Localization 
        and Beyond”.

    For each image in the batch:
    - The image is divided into a regular grid of patches.
    - A random number of patches is sampled from the specified range.
    - That number of patches are erased from the image at random 
      locations in the grid.
    - The erased patches are filled with solid color or noise.
    
    The number of patches to erase and their locations in the grid are sampled
    independently for each image, ensuring variety across the batch.

    Args:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images
            Pixel values are expected to be in the range specified by 
            the `pixels_range` argument.

        grid_size:
            A tuple of two positive integers specifying the number of patches
            of the grid in each direction. Columns are first, rows second.

        erased_patches:
            A tuple of two integers specifying the range from which to sample 
            the number of patches to erase.
            The minimum value can be 0. For example, (0, 5) means that 0 to 5
            patches can be erased.

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
        of original and Hide-and-Seek-augmented images. Pixel values are in the same 
        range as the input images.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 8),
        erased_patches: tuple[int, int] = (0, 5),
        fill_method: str = 'black',
        pixels_range: tuple[float, float] = (0, 1),
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        # Check that all arguments are valid
        self._check_arguments(grid_size, erased_patches, fill_method, pixels_range, augmentation_ratio, bernoulli_mix)

        self.grid_size = grid_size
        self.erased_patches = erased_patches
        self.fill_method = fill_method
        self.pixels_range = pixels_range
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix


    def _check_arguments(self, grid_size, erased_patches, fill_method, pixels_range, augmentation_ratio, bernoulli_mix):

        check_dataaug_function_arg(
            grid_size,
            context={'arg_name': 'grid_size', 'function_name' : 'random_hide_and_seek'},
            constraints={'format': 'tuple', 'tuple_ordering': 'None', 'data_type': 'int', 'min_val': ('>', 0)}
       )

        check_dataaug_function_arg(
            erased_patches,
            context={'arg_name': 'erased_patches', 'function_name' : 'random_hide_and_seek'},
            constraints={'format': 'tuple', 'data_type': 'int', 'min_val': ('>=', 0)}
        )

        supported_fill_methods = ('black', 'gray', 'white', 'mean_per_channel', 'random', 'noise')
        if fill_method not in supported_fill_methods:
            raise ValueError(
                '\nArgument `fill_method` of function `random_hide_and_seek`: '
                f'expecting one of {supported_fill_methods}\n'
                f'Received: {fill_method}'
            )

        check_dataaug_function_arg(
            pixels_range,
            context={'arg_name': 'pixels_range', 'function_name' : 'random_hide_and_seek'},
            constraints={'format': 'tuple'}
        )

        check_dataaug_function_arg(
            augmentation_ratio,
            context={'arg_name': 'augmentation_ratio', 'function_name' : 'random_hide_and_seek'},
            constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
        )

        if not isinstance(bernoulli_mix, bool):
            raise ValueError(
                'Argument `bernoulli_mix` of function `random_hide_and_seek`: '
                f'expecting a boolean value\nReceived: {bernoulli_mix}'
            )


    def _gen_patch_mask(self, image_shape, erased_patches, grid_size, device):
        """
        Generates a boolean mask that will be used to erase the patches from 
        the images. Value is True inside areas to erase, False outside.
        """

        batch_size, _, img_height, img_width = image_shape

        # Sample the coordinates (x, y) of the grid points where
        # the corresponding patches in the image will be erased
        num_patches = torch.randint(
                low=erased_patches[0], 
                high=erased_patches[1] + 1, 
                size=(batch_size,), 
                dtype=torch.int32,
                device=device
            )

        # Don't erase more patches than total available (for robustness)
        num_patches = torch.minimum(num_patches, torch.tensor(grid_size[0] * grid_size[1]))

        indices = torch.argsort(torch.rand(batch_size, grid_size[0] * grid_size[1], device=device))

        grid_mask = torch.where(indices < num_patches[:, None], True, False)
        grid_mask = grid_mask.reshape(batch_size, grid_size[0], grid_size[1])

        # Calculate width and height of patches
        patch_h = torch.div(img_height, grid_size[0], rounding_mode='floor').to(torch.int32)
        patch_w = torch.div(img_width, grid_size[1], rounding_mode='floor').to(torch.int32)
        patch_size = torch.stack([patch_h, patch_w], dim=-1)

        # Fill patches with the mask values of the corresponding grid points
        patch_mask = torch.repeat_interleave(grid_mask, repeats=patch_h + 1, dim=1)
        patch_mask = torch.repeat_interleave(patch_mask, repeats=patch_w + 1, dim=2)

        # Truncate mask to image size
        patch_mask = patch_mask[:, :img_height, :img_width]

        return patch_mask, patch_size


    def _gen_patch_contents(self, images, grid_size, patch_size, fill_method, device):

        """
        This function generates the color contents of the erased patches,
        accordingly to the specified fill method: random color, uniform color,
        or noise.
        It outputs a tensor with shape [batch_size, img_width, img_height, 3].
        """

        image_shape = images.shape
        batch_size, img_channels, img_height, img_width = image_shape

        if fill_method == 'black':
            contents = torch.zeros(image_shape, dtype=torch.int32, device=device)

        elif fill_method == 'gray':
            contents = torch.full(image_shape, 128, dtype=torch.int32, device=device)

        elif fill_method == 'white':
            contents = torch.full(image_shape, 255, dtype=torch.int32, device=device)

        elif fill_method == 'mean_per_channel':
            channel_means = torch.mean(images.float(), dim=[2, 3]).to(torch.int32)
            contents = channel_means[:, :, None, None].expand(image_shape)


        elif fill_method == 'random':
            color_grid = torch.randint(
                low=0, high=256,
                size=(batch_size, img_channels, grid_size[0], grid_size[1]),
                dtype=torch.int32,
                device=device
            )
            # Repeat along spatial dimensions (H, W), not channels
            contents = torch.repeat_interleave(color_grid, repeats=patch_size[0] + 1, dim=2)
            contents = torch.repeat_interleave(contents, repeats=patch_size[1] + 1, dim=3)
            contents = contents[:, :, :img_height, :img_width]

        elif fill_method == 'noise':
            contents = torch.randint(
                low=0, high=256,
                size=image_shape,
                dtype=torch.int32,
                device=device
            )

        return contents


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        device = images.device

        original_image_shape = images.shape
        if images.ndim == 3:  # i.e., [B, H, W]
            images = images.unsqueeze(1)  # insert a channel dimension at index 1
        image_shape = images.shape

        pixels_dtype = images.dtype
        images = rescale_pixel_values(images, self.pixels_range, (0, 255), dtype=torch.int32)

        # Generate a boolean mask with value True inside the areas to erase, False outside
        patch_mask, patch_size = self._gen_patch_mask(image_shape, self.erased_patches, self.grid_size, device)

        # Generate the color contents of the erased patches
        patch_contents = self._gen_patch_contents(images, self.grid_size, patch_size, self.fill_method, device)

        # Erase the patches from the images and fill them
        images_aug = torch.where(patch_mask[:, None, :, :], patch_contents, images)

        # Mix the original and augmented images
        output_images, _ = mix_augmented_images(images, images_aug, self.augmentation_ratio, self.bernoulli_mix)

        # Restore shape, data type and pixels range of input images
        output_images = output_images.reshape(original_image_shape)
        output_images = rescale_pixel_values(output_images, (0, 255), self.pixels_range, dtype=pixels_dtype)

        return images_aug
