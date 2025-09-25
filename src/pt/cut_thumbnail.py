
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple, Union

from argument_utils import check_dataaug_function_arg, check_augment_mix_args
from dataaug_utils import sample_patch_locations, gen_patch_mask, mix_augmented_images


class RandomCutThumbnail(v2.Transform):
    """
   Applies the "Cut-Thumbnail" data augmentation technique to a batch of images.

    Reference paper:
        Tianshu Xie, Xuan Cheng, Xiaomin Wang, Minghui Liu, Jiali Deng, Tao Zhou, 
        Ming Liu (2021). "Cut-thumbnail: A novel data augmentation for convolutional
        neural network".

    For each image in the batch, the function:
      1. Resizes the image to a smaller size to create a thumbnail.
      2. Pastes the thumbnail into the original image at a random location.
    All the thumbnails have the same area, specified as a fraction of the image area,
    and have the same aspect ratio as the images.

    Thumbnail locations are sampled independently for each image, ensuring variety 
    across the batch.

    By default, the augmented/original image ratio in the output mix is 1.0.
    This may be too aggressive depending on the use case, so you may want to lower it.

    Arguments:
        images:
            Input RGB or grayscale images.
            Supported shapes:
                [B, H, W, 3]  --> Color images
                [B, H, W, 1]  --> Grayscale images
                [B, H, W,]    --> Grayscale images

        thumbnail_area:
            A float specifying the area of the thumbnail as a fraction of the 
            image area. Values must be > 0 and < 1.

        resize_method:
            A string specifying the interpolation method used by tf.image.resize() to
            create the thumbnail. Supported methods include:
            {'bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'nearest', 
            'area', 'mitchellcubic'}

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
        of original and Cut-Thumbnail-augmented images.
    """

    def __init__(
        self,
        thumbnail_area: float = 0.1,
        resize_method: str = 'bilinear',
        augmentation_ratio: float = 1.0,
        bernoulli_mix: bool = False
    ):
        super().__init__()

        self.thumbnail_area = thumbnail_area
        self.resize_method = resize_method
        self.augmentation_ratio = augmentation_ratio
        self.bernoulli_mix = bernoulli_mix

        self._check_arguments()


    def _check_arguments(self):
        """
        Checks the arguments passed to `RandomCutThumbnail`
        """

        check_dataaug_function_arg(
            self.thumbnail_area,
            context={'arg_name': 'thumbnail_area', 'function_name' : 'random_cut_thumbnail'},
            constraints={'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
        )

        supported_resize_methods = (
            'bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'nearest', 'area', 'mitchellcubic')
        if self.resize_method not in supported_resize_methods:
            raise ValueError(
                '\nArgument `resize_method` of function `random_cut_thumbnail`: '
                f'expecting one of {supported_resize_methods}\n'
                f'Received: {self.resize_method}'
            )
        
        check_augment_mix_args(self.augmentation_ratio, self.bernoulli_mix, 'RandomCutThumbnail')


    def _calculate_thumbnail_size(self, image_size, thumbnail_area):
        """
        Calculates height and width of the thumbnails.
        The thumbnail area is specified as a fraction of the image's area.
        The aspect ratio is preserved.
        
        Args:
            image_size: tuple or list (height, width)
            thumbnail_area: float, fraction of the image area

        Returns:
            thumb_h, thumb_w: integers, height and width of the thumbnail
        """

        img_height = float(image_size[0])
        img_width = float(image_size[1])

        area = thumbnail_area * img_height * img_width
        aspect_ratio = img_height / img_width

        thumb_w = (area / aspect_ratio) ** 0.5
        thumb_h = thumb_w * aspect_ratio

        # Round and convert to int
        thumb_h = int(round(thumb_h))
        thumb_w = int(round(thumb_w))

        # Clip to original image size
        thumb_h = max(0, min(thumb_h, image_size[0]))
        thumb_w = max(0, min(thumb_w, image_size[1]))

        return thumb_h, thumb_w


    def forward(self, images: torch.Tensor) -> torch.Tensor:

        device = images.device

        # ---- Preserve original shape
        original_image_shape = images.shape

        # ---- Reshape [B,H,W] -> [B,1,H,W] if grayscale
        if images.ndim == 3:
            images = images.unsqueeze(1)

        image_shape = images.shape
        batch_size, channels, img_height, img_width = image_shape[:2]

        # ---- Calculate thumbnail size and resize
        thumb_h, thumb_w = self._calculate_thumbnail_size(images.shape[2:], self.thumbnail_area)
        thumbnails = F.interpolate(images, size=(thumb_h, thumb_w),
                                mode=self.resize_method, align_corners=False)
        thumbnails = thumbnails.to(images.dtype)

        # ---- Create patch mask
        batched_thumbnail_size = (
            torch.full((batch_size,), thumb_h, device=device, dtype=torch.int64),
            torch.full((batch_size,), thumb_w, device=device, dtype=torch.int64)
        )
        patch_corners = sample_patch_locations(images, batched_thumbnail_size)
        patch_mask = gen_patch_mask(images, patch_corners)
        patch_mask = patch_mask.unsqueeze(1)  # [B,1,H,W]

        # Get patch indices
        patch_indices = torch.nonzero(patch_mask, as_tuple=False)  # [num_pixels, 4]
        batch_indices = patch_indices[:, 0]  # batch index for each pixel

        # ---- Compute relative coordinates inside the patch and scale to thumbnail
        patch_top = patch_corners[batch_indices, 0].long()    # y1
        patch_left = patch_corners[batch_indices, 1].long()   # x1
        patch_bottom = patch_corners[batch_indices, 2].long() # y2  
        patch_right = patch_corners[batch_indices, 3].long()  # x2

        # Get current pixel coordinates (H, W indexing)
        pixel_y = patch_indices[:, 2]  # H dimension (row)
        pixel_x = patch_indices[:, 3]  # W dimension (col)

        # Compute relative position within patch [0, 1]
        patch_h = (patch_bottom - patch_top).float()
        patch_w = (patch_right - patch_left).float()

        # Avoid division by zero
        patch_h = torch.clamp(patch_h, min=1.0)
        patch_w = torch.clamp(patch_w, min=1.0)

        y_rel_norm = (pixel_y.float() - patch_top.float()) / patch_h
        x_rel_norm = (pixel_x.float() - patch_left.float()) / patch_w

        # Map to thumbnail coordinates
        thumb_y = (y_rel_norm * (thumb_h - 1)).clamp(0, thumb_h - 1).long()
        thumb_x = (x_rel_norm * (thumb_w - 1)).clamp(0, thumb_w - 1).long()

        # ---- Gather pixel values from thumbnails across channels
        num_pixels = patch_indices.shape[0]
        channel_idx = torch.arange(channels, device=device).view(1, channels).expand(num_pixels, channels)
        thumbnail_contents = thumbnails[batch_indices.view(-1, 1), channel_idx, thumb_y.view(-1, 1), thumb_x.view(-1, 1)]
        thumbnail_contents = thumbnail_contents.view(num_pixels, channels)

        # ---- Paste thumbnails into images
        images_aug = images.clone()
        images_aug[batch_indices, :, pixel_y, pixel_x] = thumbnail_contents


        # ---- Mix original and augmented images
        output_images, _ = mix_augmented_images(
            images, images_aug, self.augmentation_ratio, self.bernoulli_mix
                )

        # ---- Restore original shape
        output_images = output_images.reshape(original_image_shape)

        return output_images
