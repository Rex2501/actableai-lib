from typing import TypeVar, Generic, Optional, Any, Union, Tuple, Dict, List

from pydantic import root_validator
from pydantic.generics import GenericModel

from actableai.parameters.list import ListParameter
from actableai.parameters.validation import OutOfRangeError, ParameterValidationErrors
from actableai.parameters.value import ValueParameter

NumericT = TypeVar("NumericT", float, int)


class _NumericParameter(ValueParameter[NumericT], GenericModel, Generic[NumericT]):
    """
    TODO write documentation
    """

    min: Optional[NumericT]
    max: Optional[NumericT]

    @root_validator(skip_on_failure=True)
    def check_min_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO write documentation
        """
        if (
            values["min"] is not None
            and values["max"] is not None
            and values["max"] <= values["min"]
        ):
            raise ValueError("`max` must be strictly greater than `min`.")
        return values

    @root_validator(skip_on_failure=True)
    def check_default(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check default value using the current parameter.

        Args:
            values: Values used to create the current parameter.

        Returns:
            The validated dictionary containing the values used to create the current
                parameters.
        """
        default = values["default"]
        min_val = values["min"]
        max_val = values["max"]

        if (min_val is not None and default < min_val) or (
            max_val is not None and default >= max_val
        ):
            raise ValueError(f"Default value {default} is out of range.")

        return values

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        errors = super().validate_parameter(value)
        if len(errors) > 0:
            return errors

        if (self.min is not None and value < self.min) or (
            self.max is not None and value > self.max
        ):
            errors.add_error(
                OutOfRangeError(
                    parameter_name=self.name,
                    min=self.min,
                    max=self.max,
                    given=value,
                )
            )

        return errors


class _NumericListParameter(ListParameter[NumericT], GenericModel, Generic[NumericT]):
    """
    TODO write documentation
    """

    min: Optional[NumericT]
    max: Optional[NumericT]

    @root_validator(skip_on_failure=True)
    def check_min_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO write documentation
        """
        if (
            values["min"] is not None
            and values["max"] is not None
            and values["max"] <= values["min"]
        ):
            raise ValueError("`max` must be strictly greater than `min`.")
        return values

    @root_validator(skip_on_failure=True)
    def check_default_val(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check default value using the current parameter.

        Args:
            values: Values used to create the current parameter.

        Returns:
            The validated dictionary containing the values used to create the current
                parameters.
        """
        default = values["default"]
        min_val = values["min"]
        max_val = values["max"]

        if not isinstance(default, (tuple, list)):
            default = [default]

        for val in default:
            if (min_val is not None and val < min_val) or (
                max_val is not None and val >= max_val
            ):
                raise ValueError(f"Default value {val} is out of range.")

        return values

    def validate_parameter(self, value: Any) -> ParameterValidationErrors:
        """Validate value using the current parameter.

        Args:
            value: Value to validate.

        Returns:
            ParameterValidationErrors object containing the validation errors.
        """
        errors = super().validate_parameter(value)
        if len(errors) > 0:
            return errors

        if not isinstance(value, (list, tuple)):
            value = [value]

        for val in value:
            if (self.min is not None and val < self.min) or (
                self.max is not None and val > self.max
            ):
                errors.add_error(
                    OutOfRangeError(
                        parameter_name=self.name,
                        min=self.min,
                        max=self.max,
                        given=val,
                    )
                )

        return errors


class _NumericRangeSpace(
    _NumericListParameter[NumericT], GenericModel, Generic[NumericT]
):
    """Simple Numeric range space (either integer or float)."""

    default: Optional[Union[NumericT, Tuple[NumericT, NumericT], List[NumericT]]] = None
    min_len: int = 1
    max_len: int = 3
    _log: bool = False

    def process_parameter(self, value: Any) -> Any:
        """Process a value using the current parameter.

        Args:
            value: Value to process.

        Returns:
            Processed value.
        """
        if isinstance(value, list):
            value = tuple(value)

        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]

        return value

    def is_log(self) -> bool:
        return self._log


FloatParameter = _NumericParameter[float]
IntegerParameter = _NumericParameter[int]

FloatRangeSpace = _NumericRangeSpace[float]
IntegerRangeSpace = _NumericRangeSpace[int]

FloatListParameter = _NumericListParameter[float]
IntegerListParameter = _NumericListParameter[int]
