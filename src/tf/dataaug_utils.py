# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf


def sample_patch_dims(
    image_shape: tf.Tensor,
    patch_area: tuple[float, float],
    patch_aspect_ratio: tuple[float, float]
) -> tf.Tensor:
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

    batch_size, img_height, img_width = tf.unstack(image_shape[:3])
    
    # Sample patch areas and aspect ratios
    area_fraction = tf.random.uniform([batch_size], minval=patch_area[0], maxval=patch_area[1], dtype=tf.float32)
    aspect_ratio = tf.random.uniform([batch_size], minval=patch_aspect_ratio[0], maxval=patch_aspect_ratio[1], dtype=tf.float32)

    # Calculate width and height of patches
    area = area_fraction  * tf.cast(img_width, tf.float32) * tf.cast(img_height, tf.float32)
    patch_w = tf.math.sqrt(area / aspect_ratio)
    patch_h = patch_w * aspect_ratio

    patch_h = tf.cast(tf.round(patch_h), tf.int32)
    patch_w = tf.cast(tf.round(patch_w), tf.int32)

    # Clip oversized patches to image size
    patch_h = tf.clip_by_value(patch_h, 0, img_height)
    patch_w = tf.clip_by_value(patch_w, 0, img_width)

    return patch_h, patch_w


def sample_patch_locations(
    image_shape: tf.Tensor,
    patch_size: tuple[tf.Tensor, tf.Tensor]
) -> tf.Tensor:

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

    batch_size, img_height, img_width = tf.unstack(image_shape[:3])
    patch_h, patch_w = patch_size

    # Sample uniformly between 0 and 1
    x_rand = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)
    y_rand = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Calculate valid ranges for each patch
    max_x1 = tf.cast(img_width - patch_w, tf.float32)
    max_y1 = tf.cast(img_height - patch_h, tf.float32)
    
    # Scale linearly to valid ranges (distributions remain uniform)
    x1 = tf.cast(tf.round(x_rand * max_x1), tf.int32)
    y1 = tf.cast(tf.round(y_rand * max_y1), tf.int32)
    
    # Get coordinates of opposite corners
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    corners = tf.stack([y1, x1, y2, x2], axis=-1)

    return corners


def gen_patch_mask(
    image_shape: tf.Tensor,
    patch_corners: tf.Tensor
) -> tf.Tensor:
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

    img_height, img_width = tf.unstack(image_shape[1:3])
    y1, x1, y2, x2 = tf.unstack(patch_corners, axis=-1)

    # Create coordinate grids
    grid_x, grid_y = tf.meshgrid(tf.range(img_width), tf.range(img_height))
    grid_x = tf.broadcast_to(grid_x, image_shape[:3])
    grid_y = tf.broadcast_to(grid_y, image_shape[:3])

    # Add new axis for broadcasting
    x1 = x1[:, None, None]
    x2 = x2[:, None, None]
    y1 = y1[:, None, None]
    y2 = y2[:, None, None]
    
    # Create the boolean mask
    mask = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y <  y2)
 
    return mask


def gen_patch_contents(
    images: tf.Tensor,
    fill_method: str
) -> tf.Tensor:

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

    image_shape = tf.shape(images)

    if fill_method == 'black':
        contents = tf.zeros(image_shape, dtype=tf.int32)

    elif fill_method == 'gray':
        contents = tf.fill(image_shape, 128)

    elif fill_method == 'white':
        contents = tf.fill(image_shape, 255)

    elif fill_method == 'mean_per_channel':
        channel_means = tf.reduce_mean(images, axis=[1, 2])
        channel_means = tf.cast(channel_means, tf.int32)
        contents = tf.broadcast_to(channel_means[:, None, None, :], image_shape)

    elif fill_method == 'random':
        color = tf.random.uniform([image_shape[0], image_shape[-1]], minval=0, maxval=256, dtype=tf.int32)
        contents = tf.broadcast_to(color[:, None, None, :], image_shape)

    elif fill_method == 'noise':
        contents = tf.random.uniform(image_shape, minval=0, maxval=256, dtype=tf.int32)

    return contents


def mix_augmented_images(
    original_images: tf.Tensor,
    augmented_images: tf.Tensor,
    augmentation_ratio: int | float = 1.0,
    bernoulli_mix: bool = False
) -> tf.Tensor:

    """
    Mixes original images and augmented images according to a specified
    augmented/original images ratio and method.
    The augmented images are at random positions in the output mix.

    The original and augmented images must have the same shape, one of:
        [B, H, W, 3]  -->  RGB
        [B, H, W, 1]  -->  Grayscale
        [B, H, W]     -->  Grayscale

    Arguments:
    ---------
        original_images:
            The original images.

        augmented_images:
            The augmented images to mix with the original images.

        augmentation_ratio:
            A float in the interval [0, 1] specifying the augmented/original
            images ratio in the output mix. If set to 0, no images are
            augmented. If set to 1, all the images are augmented.

        bernoulli_mix:
            A boolean specifying the method to use to mix the images:
            - False:
                The fraction of augmented images in the mix is equal
                to `augmentation_ratio` for every batch.
            - True:
                The fraction of augmented images in the mix varies from batch
                to batch. Because Bernoulli experiments are used, the expectation
                of the fraction is equal to `augmentation_ratio`.

    Returns:
    -------
        A tensor of the same shape as the input images containing
        a mix of original and augmented images.
    """

    # Ensure original and augmented images have the same shape
    tf.debugging.assert_equal(
        tf.shape(original_images),
        tf.shape(augmented_images),
        message=('Function `mix_augmented_images`: original '
                 'and augmented images must have the same shape')
    )

    image_shape = tf.shape(original_images)
    batch_size = image_shape[0]

    if augmentation_ratio == 0.0:
        mixed_images = original_images
        mask = tf.zeros([batch_size], dtype=tf.bool)

    elif augmentation_ratio == 1.0:
        mixed_images = augmented_images
        mask = tf.ones([batch_size], dtype=tf.bool)
    else:

        # Reshape images with shape [B, H, W] to [B, H, W, 1]
        if original_images.shape.rank == 3:
            original_images = tf.expand_dims(original_images, axis=-1)
            augmented_images = tf.expand_dims(augmented_images, axis=-1)

        if bernoulli_mix:
            # For each image position in the output mix, make a Bernoulli
            # experiment to decide if the augmented image takes it.
            probs = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)
            mask = tf.where(probs < augmentation_ratio, True, False)
        else:
            # Calculate the number of augmented images in the mix
            num_augmented = augmentation_ratio * tf.cast(batch_size, tf.float32)
            num_augmented = tf.cast(tf.round(num_augmented), tf.int32)

            # Generate a mask set to True for positions in the mix
            # occupied by augmented images, False by original images
            grid = tf.range(batch_size)
            mask = tf.where(grid < num_augmented, True, False)

            # Shuffle the mask so that the augmented images
            # are at random positions in the output mix
            mask = tf.random.shuffle(mask)

        # Apply the mask to images to generate the output mix
        mixed_images = tf.where(mask[:, None, None, None], augmented_images, original_images)

        # Restore the image input shape
        mixed_images = tf.reshape(mixed_images, image_shape)

    return mixed_images, mask


def rescale_pixel_values(
    images: tf.Tensor,
    input_range: tuple[int | float, int | float],
    output_range: tuple[int | float, int | float],
    dtype: tf.DType = tf.float32
) -> tf.Tensor:

    """
    Linearly rescales pixel values of images from one range to another.

    A linear transformation is applied to each pixel so that values
    originally in `input_range` are mapped to `output_range`. Output
    values are clipped to the target range and cast to the specified 
    data type `dtype`.

    Example:
        # Convert uint8 images [0, 255] to float32 [0.0, 1.0]
        images_mapped = remap_pixel_values_range(images, (0, 255), (0.0, 1.0), tf.float32)

    Args:
        images:
            Input images.
        input_range:
            (min, max) range of input pixel values.
        output_range:
            (min, max) target range for output pixel values.
        dtype:
            Desired output data type.

    Returns:
        Images with pixel values rescaled to `output_range` and cast to `dtype`.
    """

    if input_range != output_range:
        input_min, input_max = input_range
        output_min, output_max = output_range

        images = tf.cast(images, tf.float32)
        images = ((output_max - output_min) * images +
                   output_min * input_max - output_max * input_min) / (input_max - input_min)
        
        # Clip to the output range
        images = tf.clip_by_value(images, output_min, output_max)

    return tf.cast(images, dtype)


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


def check_pixels_range_args(arg, function_name):
    check_dataaug_function_arg(
        arg,
        context={'arg_name': 'pixels_range', 'function_name': function_name},
        constraints={'format': 'tuple'}
    )


def check_augment_mix_args(ratio_arg, bernoulli_arg, function_name):

    check_dataaug_function_arg(
        ratio_arg,
            context={'arg_name': 'augmentation_ratio', 'function_name' : function_name},
            constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
        )

    if not isinstance(bernoulli_arg, bool):
        raise ValueError(
            f'Argument `bernoulli_mix` of function `{function_name}`: '
            f'expecting a boolean value\nReceived: {bernoulli_arg}'
            )
        

def check_fill_method_arg(arg, function_name):
    supported_fill_methods = ('black', 'gray', 'white', 'mean_per_channel', 'random', 'noise')
    if arg not in supported_fill_methods:
        raise ValueError(
            f'\nArgument `fill_method` of function `{function_name}`: '
            f'expecting one of {supported_fill_methods}\n'
            f'Received: {arg}'
        )
