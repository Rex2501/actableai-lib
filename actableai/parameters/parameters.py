from enum import Enum
from typing import TypeVar, Generic, Dict, Any, Optional, List

from pydantic import BaseModel, root_validator
from pydantic.generics import GenericModel


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


OptionT = TypeVar("OptionT")


class BaseParameter(BaseModel):
    """
    TODO write documentation
    """

    name: str
    display_name: str
    description: Optional[str]
    parameter_type: ParameterType


class Option(BaseModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    display_name: str
    value: OptionT


class OptionsParameter(BaseParameter, GenericModel, Generic[OptionT]):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.OPTIONS
    is_multi: bool
    default: List[OptionT]
    options: Dict[OptionT, Option[OptionT]]


class FloatParameter(BaseParameter):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.FLOAT
    default: float
    min: Optional[float]
    max: Optional[float]


class IntegerParameter(BaseParameter):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.INT
    default: int
    min: Optional[int]
    max: Optional[int]


class BooleanParameter(BaseParameter):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.BOOL
    default: bool


class Parameters(BaseModel):
    """
    TODO write documentation
    """

    parameters: Dict[str, BaseParameter]
