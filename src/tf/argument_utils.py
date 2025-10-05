# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.


def check_argument(
    arg: int | float | tuple | list,
    context: dict,
    constraints: dict | None = None
) -> None:

    """
    Checks that an argument passed from a caller (function, layer, transform) 
    meets specific constraints. Value errors are raised if the argument 
    does not meet them.

    Arguments:
    ---------
        arg :
            The argument value to validate.

        context:
            A dictionary providing context information for error messages.
            Must contain:
            - 'arg_name': a string, the name of the argument.
            - 'caller_name': a string, the name of the caller that passed
              the argument.

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
                    - '>' : 2nd value > 1st value (default)
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
    >>> check_argument(
    ...     0.5,
    ...     context={'arg_name': 'augmentation_ratio', 'caller_name': 'cutout'},
    ...     constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
    ... )

    >>> # Validate a tuple of 2 floats greater than 0,
    >>> # 2nd tuple value greater than the 1st one
    >>> check_argument(
    ...     (0.3, 0.7),
    ...     context={'area_ratio_range': 'arg', 'caller_name': 'random_erasing'},
    ...     constraints={'format': 'tuple', 'data_type': 'float', 'min_val': ('>', 0)}
    ... )

    >>> # Validate a tuple of 2 integers greater than 0, no tuple ordering constraint
    >>> check_argument(
    ...     (8, 8),
    ...     context={'arg_name': 'grid_size', 'caller_name': 'hide_and_seek'},
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
    context_msg = f"\nArgument `{context['arg_name']}` of transform `{context['caller_name']}`"

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


# Check range of pixel values
def check_pixels_range_args(arg, caller_name):
    check_argument(
        arg,
        context={'arg_name': 'pixels_range', 'caller_name': caller_name},
        constraints={'format': 'tuple'}
    )

# Check original/augmented images mixing ratio and method
def check_augment_mix_args(ratio_arg, bernoulli_arg, caller_name):
    check_argument(
        ratio_arg,
            context={'arg_name': 'augmentation_ratio', 'caller_name' : caller_name},
            constraints={'min_val': ('>=', 0), 'max_val': ('<=', 1)}
        )
    if not isinstance(bernoulli_arg, bool):
        raise ValueError(
            f'Argument `bernoulli_mix` of function `{caller_name}`: '
            f'expecting a boolean value\nReceived: {bernoulli_arg}'
            )

# Check the arguments used in patch sampling
def check_patch_sampling_args(patch_area, patch_aspect_ratio, caller_name):
    check_argument(
        patch_area,
        context={'arg_name': 'patch_area', 'caller_name': caller_name},
        constraints={'format': 'number_or_tuple', 'data_type': 'float', 'min_val': ('>', 0), 'max_val': ('<', 1)}
    )
    check_argument(
        patch_aspect_ratio,
        context={'arg_name': 'patch_aspect_ratio', 'caller_name': caller_name},
        constraints={'format': 'number_or_tuple', 'min_val': ('>', 0)}
    )


# Check patch fill method
def check_fill_method_arg(arg, caller_name):
    supported_fill_methods = ('black', 'gray', 'white', 'mean_per_channel', 'random', 'noise')
    if arg not in supported_fill_methods:
        raise ValueError(
            f'\nArgument `fill_method` of function `{caller_name}`: '
            f'expecting one of {supported_fill_methods}\n'
            f'Received: {arg}'
        )
