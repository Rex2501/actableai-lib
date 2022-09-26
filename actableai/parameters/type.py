from enum import Enum


class ParameterType(str, Enum):
    """
    TODO write documentation
    """

    BOOL = "bool"
    INT = "int"
    INT_RANGE = "int_range"
    FLOAT = "float"
    FLOAT_RANGE = "float_range"
    OPTIONS = "options"
