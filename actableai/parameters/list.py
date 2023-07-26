from typing import (
    Any,
    Generic,
    TypeVar,
    Type,
    get_type_hints,
    Union,
    Tuple,
    List,
    Dict,
    Optional,
)

from pydantic import validator, root_validator
from pydantic.generics import GenericModel
from pydantic.typing import get_args

from actableai.parameters.base import BaseParameter
from actableai.parameters.type import ValueType, ParameterType
from actableai.parameters.validation import (
    ParameterValidationErrors,
    ParameterTypeError,
    OutOfRangeError,
)

ListT = TypeVar("ListT", bool, int, float, str)


class ListParameter(BaseParameter, GenericModel, Generic[ListT]):
    """
    TODO write documentation
    """

    parameter_type: ParameterType = ParameterType.LIST
    default: Union[ListT, Tuple[ListT, ...], List[ListT]] = []
    # Automatic dynamic field
    value_type: ValueType = None
    min_len: int = 0
    max_len: Optional[int]

    @validator("default")
    def set_default(
        cls, value: Union[ListT, Tuple[ListT, ...], List[ListT]]
    ) -> List[str]:
        """Set `default` value.

        Args:
            value: Current value of `default`.

        Returns:
            New `default` value.
        """
        if not isinstance(value, (tuple, list)):
            value = [value]
        return list(value)

    @validator("value_type", always=True)
    def set_value_type(cls, value: ValueType) -> ValueType:
        """
        TODO write documentation
        """

        val_type = cls._get_value_type()
        if val_type == bool:
            return ValueType.BOOL
        if val_type == int:
            return ValueType.INT
        if val_type == float:
            return ValueType.FLOAT
        if val_type == str:
            return ValueType.STR

        raise ValueError("Invalid generic type.")

    @validator("min_len")
    def check_min_len(cls, value: int) -> int:
        """
        TODO write documentation
        """
        if value < 0:
            raise ValueError("`min_len` must be a positive integer.")
        return value

    @root_validator(skip_on_failure=True)
    def check_len(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO write documentation
        """
        if values.get("max_len") is not None and values["max_len"] <= values["min_len"]:
            raise ValueError("`max_len` must be strictly greater than `min_len`.")
        return values

    @root_validator(skip_on_failure=True)
    def check_default_len(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check default value using the current parameter.

        Args:
            values: Values used to create the current parameter.

        Returns:
            The validated dictionary containing the values used to create the current
                parameters.
        """
        default = values["default"]

        if not isinstance(default, (tuple, list)):
            default = [default]

        default_len = len(default)
        if default_len < values["min_len"] or (
            values.get("max_len") is not None and default_len >= values["max_len"]
        ):
            raise ValueError(f"Default len {default_len} is out of range.")

        return values

    @classmethod
    def _get_value_type(cls) -> Type:
        """Returns the type of the `ValueT`.

        Returns:
            `ValueT` type.
        """
        # Trick to get the generic type, impossible to use traditional method, see:
        # https://github.com/pydantic/pydantic/issues/3559
        return get_args(get_type_hints(cls)["default"])[0]

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

        if not isinstance(value, (tuple, list)):
            value = [value]

        value_len = len(value)
        if value_len < self.min_len or (
            self.max_len is not None and value_len > self.max_len
        ):
            errors.add_error(
                OutOfRangeError(
                    parameter_name=self.name,
                    min=self.min_len,
                    max=self.max_len,
                    given=value_len,
                )
            )

        if len(errors) > 0:
            return errors

        for val in value:
            type_valid = True

            if self.value_type == ValueType.BOOL:
                type_valid = isinstance(val, bool)
            elif self.value_type == ValueType.INT:
                type_valid = isinstance(val, int)
            elif self.value_type == ValueType.FLOAT:
                type_valid = isinstance(val, (int, float))
            elif self.value_type == ValueType.STR:
                type_valid = isinstance(val, str)

            if not type_valid:
                errors.add_error(
                    ParameterTypeError(
                        parameter_name=self.name,
                        expected_type=self.value_type,
                        given_type=str(type(val)),
                    )
                )

        return errors

    def process_parameter(self, value: Any) -> Any:
        """Process a value using the current parameter.

        Args:
            value: Value to process.

        Returns:
            Processed value.
        """
        if not isinstance(value, (tuple, list)):
            value = [value]
        return list(value)
