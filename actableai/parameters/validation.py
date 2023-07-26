from typing import Optional, Union, List, Dict, Any, Type

from pydantic import BaseModel, root_validator

from actableai.data_validation.base import CheckResult, CheckLevels
from actableai.parameters.type import ParameterType, ValueType


class ParameterValidationError(BaseModel):
    """Class representing a parameter validation error."""

    parameter_name: str
    message: Optional[str]
    parent_parameter_list: List[str] = []

    @property
    def path(self) -> str:
        """Compute the path of the error.

        Returns:
            Error path.
        """
        path = ""
        for parent in reversed(self.parent_parameter_list):
            path = f"{path}{parent}."

        path = f"{path}{self.parameter_name}"

        return path

    def __str__(self) -> str:
        """Convert the error to a readable string.

        Returns:
            Readable string.
        """
        error = self.path
        if self.message is not None:
            error = f"{error}: {self.message}"
        return error

    def add_parent(self, parameter_name: str):
        """Add a new parent to the error.

        Args:
            parameter_name: Name of the parent.
        """
        self.parent_parameter_list.append(parameter_name)

    @staticmethod
    def _process_message(message: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Process (format) message.

        Args:
            message: Message to process.
            values: Values containing the original message.

        Returns:
            Values containing the processed message.
        """
        if values.get("message") is not None:
            message = f"{message} (values['message'])"

        values["message"] = f"{message}."
        return values


class ParameterValidationErrors(BaseModel):
    """Class representing a multiple validation errors."""

    parameter_name: str
    validation_error_list: List[ParameterValidationError] = []

    def __len__(self) -> int:
        """Get the number of errors.

        Returns:
             Number of errors.
        """
        return len(self.validation_error_list)

    def has_error(self, error_type: Type[ParameterValidationError]) -> bool:
        """Check whether the current object contains any error of a specific type.

        Args:
            error_type: Type of the error to look for.

        Returns:
            True if the `error_type` has been found in the current errors.
        """
        for error in self.validation_error_list:
            if isinstance(error, error_type):
                return True
        return False

    def add_error(self, error: ParameterValidationError):
        """Add a new error.

        Args:
            error: New error to add.
        """
        self.validation_error_list.append(error)

    def add_errors(self, errors: "ParameterValidationErrors"):
        """Concatenate two `ParameterValidationErrors`.

        Args:
            errors: Other `ParameterValidationErrors` to concatenate.
        """
        for error in errors.validation_error_list:
            error.add_parent(self.parameter_name)
            self.validation_error_list.append(error)

    def to_check_results(self, name: str) -> List[CheckResult]:
        """Convert to `CheckResult` list.

        Args:
            name: Name of the `CheckResult`.

        Returns:
            List of `CheckResult` objects.
        """
        check_result_list = []
        for error in self.validation_error_list:
            check_result_list.append(
                CheckResult(name=name, message=str(error), level=CheckLevels.CRITICAL)
            )

        return check_result_list


class ParameterTypeError(ParameterValidationError):
    """Class representing a parameter type error."""

    expected_type: Union[ParameterType, ValueType, str]
    given_type: str

    @root_validator
    def set_message(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set `message` value.

        Args:
            values: Dictionary containing the current value of the `message`.

        Returns:
            Updated dictionary containing the correct `message`.
        """
        message = f"Incorrect parameter type, given: `{values['given_type']}`, expected: `{values['expected_type']}`"
        return cls._process_message(message, values)


class OutOfRangeError(ParameterValidationError):
    """Class representing a parameter out of range error."""

    min: Optional[Union[int, float]]
    max: Optional[Union[int, float]]
    given: Union[int, float]

    @root_validator
    def set_message(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set `message` value.

        Args:
            values: Dictionary containing the current value of the `message`.

        Returns:
            Updated dictionary containing the correct `message`.
        """
        message = f"Out of range value, given: `{values['given']}`"

        if values.get("min") is not None:
            message = f"{message}, minimum: `{values['min']}`"
        if values.get("max") is not None:
            message = f"{message}, maximum: `{values['max']}`"

        return cls._process_message(message, values)


class InvalidKeyError(ParameterValidationError):
    """Class representing a parameter invalid key error."""

    key: str

    @root_validator
    def set_message(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set `message` value.

        Args:
            values: Dictionary containing the current value of the `message`.

        Returns:
            Updated dictionary containing the correct `message`.
        """
        message = f"The key `{values['key']}` is invalid"

        return cls._process_message(message, values)
