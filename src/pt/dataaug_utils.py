import torch


def sample_patch_dims(
    image_shape: tuple[int, int, int, int] | torch.Size,
    patch_area: tuple[float, float],
    patch_aspect_ratio: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Samples heights and widths of patches

    Arguments:
        image_shape:
            The shape of the images (4D tensor).

        patch_area:
            A tuple of two floats specifying the range from which patch areas
            are sampled. Values must be > 0 and < 1, representing fractions 
            of the image area.

        patch_aspect_ratio:
            A tuple of two floats specifying the range from which patch 
            height/width aspect ratios are sampled. Values must be > 0.

    Returns:
        A tuple of 2 tensors with shape [batch_size]:
            `(patch_height, patch_width)`. 
    """

    batch_size, _, img_height, img_width = image_shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample patch areas and aspect ratios uniformly
    area_fraction = torch.empty(batch_size, device=device).uniform_(*patch_area)
    aspect_ratio = torch.empty(batch_size, device=device).uniform_(*patch_aspect_ratio)

    # Compute patch area in absolute pixels
    img_area = float(img_width * img_height)
    area = area_fraction * img_area

    # Compute width/height from area + aspect ratio
    patch_w = torch.sqrt(area / aspect_ratio)
    patch_h = patch_w * aspect_ratio

    # Round to nearest int, cast to int32
    patch_h = torch.round(patch_h).to(torch.int32)
    patch_w = torch.round(patch_w).to(torch.int32)

    # Clip to stay within image bounds
    patch_h = torch.clamp(patch_h, min=0, max=img_height)
    patch_w = torch.clamp(patch_w, min=0, max=img_width)

    return patch_h, patch_w


def sample_patch_locations(
    image_shape: tuple[int, int, int, int] | torch.Size,
    patch_size: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    
    """
    Samples patch locations given their heights and widths.
    Locations are constrained in such a way that the patches 
    are entirely contained inside the images.

    Arguments:
        image_shape:
            The shape of the images. Either [B, H, W, C] or
            [B, H, W] (the channel is not used).

        patch_size:
            A tuple of 2 tensors with shape [batch_size]:
                `(patch_height, patch_width)` 

    Returns:
        A tensor with shape [batch_size, 4].
        Contains the opposite corners coordinates (y1, x1, y2, x2)
        of the patches.
    """

    batch_size, _, img_height, img_width = image_shape
    patch_h, patch_w = patch_size

    device = patch_h.device

    # Sample random positions uniformly in [0, 1]
    x_rand = torch.rand(batch_size, device=device)
    y_rand = torch.rand(batch_size, device=device)

    # Compute valid start positions so patches stay inside image
    max_x1 = (img_width - patch_w).to(torch.float32)
    max_y1 = (img_height - patch_h).to(torch.float32)

    # Scale random positions to valid range & round
    x1 = torch.round(x_rand * max_x1).to(torch.int32)
    y1 = torch.round(y_rand * max_y1).to(torch.int32)

    # Compute opposite corners
    x2 = x1 + patch_w
    y2 = y1 + patch_h

    return torch.stack([y1, x1, y2, x2], dim=-1)


def gen_patch_mask(
    image_shape: tuple[int, int, int, int] | torch.Size,
    patch_corners: torch.Tensor
) -> torch.Tensor:
    """
    Given opposite corners coordinates of patches, generates
    a boolean mask with value True inside patches.

    Arguments:
        image_shape:
            The shape of the images (4D tensor).

        patch_corners:
            The opposite corners coordinates (y1, x1, y2, x2) 
            of the patches.
            Shape: [batch_size, 4]

    Returns:
        A boolean mask with value True inside the patches, False outside
        Shape: [batch_size, img_height, img_width]
    """

    device = patch_corners.device

    img_height, img_width = image_shape[2:]

    # Unpack corner coordinates
    y1, x1, y2, x2 = patch_corners.unbind(dim=-1)  # each is [B]

    # Create coordinate grids (on GPU)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(img_width, device=device),
        torch.arange(img_height, device=device),
        indexing='xy'  # Added indexing parameter for clarity
    )

    # Broadcast to [B, H, W] and apply patch logic
    mask = (
        (grid_x.unsqueeze(0) >= x1.view(-1, 1, 1)) &
        (grid_x.unsqueeze(0) <  x2.view(-1, 1, 1)) &
        (grid_y.unsqueeze(0) >= y1.view(-1, 1, 1)) &
        (grid_y.unsqueeze(0) <  y2.view(-1, 1, 1))
    )

    return mask


def gen_patch_contents(
    images: torch.Tensor,
    fill_method: str
) -> torch.Tensor:

    """
    Generates color contents for erased cells (images filled 
    with solid color or noise).

    Arguments:
        images:
            The images being augmented (4D tensor).
            Pixels must be in range [0, 255] with tf.int32 data type.

        fill_method:
            A string, the method to use to generate the contents.
            One of: {'black', 'gray', 'white', 'mean_per_channel', 'random', 'noise'}

    Returns:
        A tensor with the same shape as the images.
    """

    image_shape = images.shape
    batch_size, channels = image_shape[:2]
    device = images.device

    # Ensure integer math stays in 32-bit range
    images = images.to(torch.int32)

    if fill_method == "black":
        contents = torch.zeros(image_shape, dtype=torch.int32, device=device)

    elif fill_method == "gray":
        contents = torch.full(image_shape, 128, dtype=torch.int32, device=device)

    elif fill_method == "white":
        contents = torch.full(image_shape, 255, dtype=torch.int32, device=device)

    elif fill_method == "mean_per_channel":
        # Compute mean per channel per image, keep int precision
        channel_means = images.to(torch.float32).mean(dim=(2, 3))
        channel_means = channel_means.round().to(torch.int32)
        contents = channel_means[:, :, None, None].expand(image_shape)

    elif fill_method == "random":
        # Random color per image per channel
        color = torch.randint(0, 256, (batch_size, channels), dtype=torch.int32, device=device)
        contents = color[:, :, None, None].expand(image_shape)

    elif fill_method == "noise":
        contents = torch.randint(0, 256, image_shape, dtype=torch.int32, device=device)

    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    return contents


def rescale_pixel_values(
    images: torch.Tensor,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Linearly rescales pixel values of images from one range to another.

    A linear transformation is applied so that values in `input_range`
    are mapped to `output_range`. Output is clipped and cast to `dtype`.

    Example:
        # Convert uint8 images [0, 255] to float32 [0.0, 1.0]
        images = rescale_pixel_values(images, (0, 255), (0.0, 1.0), torch.float32)
    """
    if input_range != output_range:
        input_min, input_max = input_range
        output_min, output_max = output_range

        # Ensure float math on GPU
        images = images.to(torch.float32)

        # Fully vectorized linear transform
        images = ((output_max - output_min) * images +
                  output_min * input_max - output_max * input_min) / (input_max - input_min)

        # Clip to output range on GPU
        images = torch.clamp(images, min=output_min, max=output_max)

    return images.to(dtype)


def mix_augmented_images(
    original_images: torch.Tensor,
    augmented_images: torch.Tensor,
    augmentation_ratio: float = 1.0,
    bernoulli_mix: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixes original and augmented images according to a specified
    augmented/original ratio and method. Fully vectorized and GPU-friendly.
    """

    # Ensure inputs have the same shape
    if original_images.shape != augmented_images.shape:
        raise ValueError("original_images and augmented_images must have the same shape")

    batch_size = original_images.shape[0]
    device = original_images.device

    if augmentation_ratio == 0.0:
        mixed_images = original_images
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    elif augmentation_ratio == 1.0:
        mixed_images = augmented_images
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    else:
        # Ensure shape is [B, H, W, C] if input was [B, H, W]
        if original_images.ndim == 3:
            original_images = original_images.unsqueeze(-1)
            augmented_images = augmented_images.unsqueeze(-1)

        if bernoulli_mix:
            # Bernoulli sampling: each position independent
            mask = torch.rand(batch_size, device=device) < augmentation_ratio
        else:
            # Deterministic number of augmented images
            num_augmented = int(torch.round(torch.tensor(batch_size * augmentation_ratio, device=device)))
            mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            mask[:num_augmented] = True
            # Shuffle mask on GPU
            perm = torch.randperm(batch_size, device=device)
            mask = mask[perm]

        # Apply mask: broadcasting over H, W, C
        mixed_images = torch.where(mask.view(-1, 1, 1, 1), augmented_images, original_images)

        # Restore to original input shape
        mixed_images = mixed_images.view_as(original_images)

    return mixed_images, mask


def check_dataaug_function_arg(
    arg: int | float | tuple | list,
    context: dict,
    constraints: dict | None = None
) -> None:

    """
    Checks that an argument passed to a data augmentation function 
    meets specific constraints. Value errors are raised if the argument
    does not meet them.

    Arguments:
    ---------
        arg :
            The argument value to validate.

        context:
            A dictionary providing context information for error messages.
            Must contain:
            - 'arg_name': a string, the name of the argument being validated
            - 'function_name': a string, the name of the calling function
              containing the argument

        constraints:
            A dictionary specifying the constraints to verify with the
            following keys:
                'format':
                    A string specifying the argument format:
                    - 'number': single number (default)
                    - 'tuple': tuple of exactly 2 numbers
                    - 'number_or_tuple': either single number or tuple of 2 numbers

                'tuple_ordering':
                    Usable only when `format` is set to 'tuple' or 'number_or_tuple'.
                    Specifies a relative value constraint between the 1st and 2nd
                    elements of the tuple passed in argument.
                    Options:
                    - '>=' : 2nd value >= 1st value.
                    - '>' : 2nd value > 1st value (default).
                    - 'None' : no constraints.

                'data_type':
                    A string specifying the argument data type:
                    - 'int': integers only
                    - 'float': floats only
                    - 'int_or_float': integers or floats (default)

                'min_val':
                    A tuple specifying a minimum value constraint for the argument.
                    The tuple 1st element is a string specifying the relational
                    operator to use: '>=', '>', '<=', or '<'.
                    The 2nd element is a number specifying the threshold.

                'max_val':
                    Same as 'min_val' but for maximum values.

    Raises
    ------
        ValueError
            If the argument fails any validation constraint.

    Examples
    --------
    >>> # Validate a single integer or float in the interval [0, 1]
    >>> check_dataaug_argument(
    ...     0.5,
    ...     context={'arg_name': 'augmentation_ratio', 'function_name': 'cutout'},
    ...     constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    ... )

    >>> # Validate a tuple of 2 floats greater than 0,
    >>> # 2nd tuple value greater than the 1st one
    >>> check_dataaug_argument(
    ...     (0.3, 0.7),
    ...     context={'area_ratio_range': 'arg', 'function_name': 'random_erasing'},
    ...     constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
    ... )

    >>> # Validate a tuple of 2 integers greater than 0, no tuple ordering constraint
    >>> check_dataaug_argument(
    ...     (8, 8),
    ...     context={'arg_name': 'grid_size', 'function_name': 'hide_and_seek'},
    ...     constraints={'format': 'tuple', 'tuple_ordering': 'None', 'data_type: 'int', min_val': ('>', 0)}
    ... )

    """

    def is_number(x):
        return isinstance(x, (int, float))

    def is_valid_tuple(x, ordering):
        if (isinstance(x, (tuple, list)) and len(x) == 2 and
            is_number(x[0]) and is_number(x[1])):
                if ordering is None:
                    return True
                elif ordering == '>=':
                    if x[1] >= x[0]:
                        return True
                elif ordering == '>':
                    if x[1] > x[0]:
                        return True
        return False


    def get_error_message(arg, format, data_type, tuple_ordering, context_msg):
        message_dict = {
            'number': {
                'int': 'an integer',
                'float': 'a float',
                'int_or_float': 'a number'
            },
            'tuple': {
                'int': 'a tuple of 2 integers',
                'float': 'a tuple of 2 floats',
                'int_or_float': 'a tuple of 2 numbers'
            },
            'number_or_tuple': {
                'int': 'an integer or a tuple of 2 integers',
                'float': 'a float or a tuple of 2 floats',
                'int_or_float': 'a number or a tuple of 2 numbers'
            }
        }

        message = context_msg + ': expecting ' + message_dict[format][data_type]

        if tuple_ordering is not None:
            if tuple_ordering == '>=':
                ordering_msg = 'greater than or equal to'
            elif tuple_ordering == '>':
                ordering_msg = 'greater than'
            if format == 'tuple':
                message += f'\nThe 2nd element of the tuple must be {ordering_msg} the 1st element.'
            elif format == 'number_or_tuple':
                message += f'\nIf a tuple is used, the 2nd element must be {ordering_msg} the 1st element.'

        message += f'\nReceived: {arg}'

        return message


    def check_value_constraint(arg, value_constraint, min_or_max, context_msg):
        operator_dict = {
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '>':  lambda x, y: x > y,
            '<':  lambda x, y: x < y
        }

        if ( not isinstance(value_constraint, (tuple, list)) or
             len(value_constraint) != 2 or
             value_constraint[0] not in operator_dict or
             not is_number(value_constraint[1])
           ):
            raise ValueError(f'{context_msg}: invalid value constraint `{value_constraint}`')

        # Get the bound
        if isinstance(arg, tuple):
            arg_bound = min(arg) if min_or_max == 'min' else max(arg)
        else:
            arg_bound = arg

        # Check value versus bound
        operator, value = value_constraint
        if not operator_dict[operator](arg_bound, value):
            raise ValueError(f'{context_msg}: expecting {min_or_max}imum '
                             f'value {operator} {value}\nReceived: {arg}')


    # Set default values
    if constraints is None:
        constraints = {}
    format = constraints.setdefault('format', 'number')
    data_type = constraints.setdefault('data_type', 'int_or_float')

    # Context message to include in error messages
    context_msg = f"\nArgument `{context['arg_name']}` of function `{context['function_name']}`"

    # Check format usage
    if format not in ('number', 'tuple', 'number_or_tuple'):
        raise ValueError(f'{context_msg}: invalid format constraint `{format}`')

    # Check data type usage
    if data_type not in ('int', 'float', 'int_or_float'):
        raise ValueError(f'{context_msg}: invalid data type constraint `{data_type}`')

    # Check tuple ordering usage
    tuple_ordering = constraints.get('tuple_ordering')
    if tuple_ordering is not None:
        if tuple_ordering not in ('None', '>=', '>'):
            raise ValueError(f'{context_msg}: invalid tuple ordering constraint `{format}`')
        if format not in ('tuple', 'number_or_tuple'):
            raise ValueError(
                f"{context_msg}: tuple ordering can only be used"
                "with 'tuple' and 'number_or_tuple' formats"
            )
        if tuple_ordering == 'None':
            tuple_ordering = None
    else:
        # Default value
        tuple_ordering = '>'

    # Prepare error message
    error_msg = get_error_message(arg, format, data_type, tuple_ordering, context_msg)

    # Check format
    format_dict = {
        'number': is_number(arg),
        'tuple': is_valid_tuple(arg, tuple_ordering),
        'number_or_tuple': is_number(arg) or is_valid_tuple(arg, tuple_ordering),
    }
    if not format_dict[format]:
        raise ValueError(error_msg)

    # Check data type
    data_type_dict = {'int': int, 'float': float, 'int_or_float': (int, float)}
    arg_tuple = (arg,) if is_number(arg) else arg
    if not all(isinstance(v, data_type_dict[data_type]) for v in arg_tuple):
        raise ValueError(error_msg)

    # Check min/max value constraints
    if 'min_val' in constraints:
        check_value_constraint(arg, constraints['min_val'], 'min', context_msg)
    if 'max_val' in constraints:
        check_value_constraint(arg, constraints['max_val'], 'max', context_msg)

