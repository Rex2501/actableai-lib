from __future__ import annotations

from abc import ABC
from typing import Dict, List, Any, Union, TYPE_CHECKING, Type, Optional

from actableai.parameters.models import ModelSpace
from actableai.parameters.parameters import Parameters

if TYPE_CHECKING:
    from autogluon.core.models import AbstractModel

from actableai.tasks.base import AAITunableTask
from actableai.models.autogluon import Model

import pandas as pd


class AAIAutogluonTask(AAITunableTask, ABC):
    @staticmethod
    def get_available_models(
        problem_type: str,
        explain_samples: bool,
        gpu: bool = False,
        ag_automm_enabled: bool = False,
        tabpfn_enabled: bool = False,
    ) -> List[Model]:
        """Get list of available models for the given problem type.

        Args:
            problem_type: The type of the problem ('regression' or 'quantile')
            explain_samples: Boolean indicating if explanations for predictions
                in test and validation will be generated.
            gpu: If GPU is available. If False, 'CPU' is used, otherwise 'GPU'
                is used. Used to filter out any models which can only run on the
                GPU and the GPU is unavailable.
            ag_automm_enabled: Boolean indicating if AG_AUTOMM model should be used
            tabpfn_enabled: Boolean indicating if TabPFN model should be used

        Returns:
            List of available models

        """
        from actableai.models.autogluon import model_params_dict

        available_models = []
        for model, model_params_class in model_params_dict.items():
            # Check if model is supported for given problem type and for
            # explanations, if enabled
            if problem_type not in model_params_class.supported_problem_types:
                continue

            if explain_samples and not model_params_class.explain_samples_supported:
                continue

            # If GPU available, run all models; if do not have GPU, run only
            # models which do not need the GPU
            if not gpu and model_params_class.gpu_required:
                continue

            # Do not include AG_AUTOMM or TabPFN if they have not been
            # enabled
            if model == Model.ag_automm and not ag_automm_enabled:
                continue
            if model == Model.tabpfn and not tabpfn_enabled:
                continue

            available_models.append(model)

        return available_models

    @staticmethod
    def _hyperparameters_to_model_params(
        hyperparameters: Dict[str, Any], hyperparameters_space: ModelSpace
    ) -> Dict[
        Union[str, Type[AbstractModel]], Union[List[Dict[str, Any]], Dict[str, Any]]
    ]:
        """Convert the hyperparameters into a list of model parameters
            compatible with AutoGluon.
        Args:
            hyperparameters: Hyperparameters to convert.
            hyperparameters_space: The full hyperparameters space of the model.

        Returns:
            List of model parameters.
        """
        from actableai.models.autogluon import model_params_dict

        model_params = dict()

        # Get list of model names
        model_names = set(model.value for model in Model)

        for model_name, model_parameters in hyperparameters.items():
            if model_name in model_names:
                model_params_class = model_params_dict[Model(model_name)]

                # Convert all parameters to AutoGluon format
                params = model_params_class.get_autogluon_parameters(
                    model_parameters,
                    hyperparameters_space.options[model_name].value,
                    process_hyperparameters=False,
                )

                model_name_ag = model_params_class.get_autogluon_name()

            # For any model which is not yet implemented, simply use as is
            else:
                model_name_ag = model_name
                params = model_parameters

            # Create new entry if it doesn't exist; otherwise, append to the
            # existing values (enabling a model to have multiple configurations)
            if model_name_ag not in model_params:
                model_params[model_name_ag] = [params]
            else:
                model_params[model_name_ag].append(params)

        return model_params

    @classmethod
    def get_base_hyperparameters_space(
        cls,
        df: pd.DataFrame,
        num_class: int,
        problem_type: str,
        device: str = "cpu",
        explain_samples: bool = False,
        ag_automm_enabled: bool = False,
        tabpfn_enabled: bool = False,
    ) -> ModelSpace:
        """Return the hyperparameters space of the task.

        Args:
            df: DataFrame containing the features
            num_class: The number of classes in the target column ('-1' can be
                used for regression which does not use classes)
            problem_type: The type of the problem
                ('regression'/'quantile'/'multiclass'/'binary')
            device: Which device is being used, can be one of 'cpu' or 'gpu'.
            explain_samples: Boolean indicating if explanations for predictions
                in test and validation will be generated.
            ag_automm_enabled: Boolean indicating if AG_AUTOMM model should be used.
            tabpfn_enabled: Boolean indicating if TabPFN model should be used.

        Returns:
            default: Default models and settings.
            options: Display name and hyperparameters of the available models
        """

        from actableai.models.autogluon.base import Model
        from actableai.models.autogluon import model_params_dict

        # Get list of available models for the given problem type
        available_models = cls.get_available_models(
            problem_type=problem_type,
            explain_samples=explain_samples,
            gpu=device == "gpu",
            ag_automm_enabled=ag_automm_enabled,
            tabpfn_enabled=tabpfn_enabled,
        )

        dataset_len = df.shape[0]

        # TODO: Consider raising error if problem type is not supported, instead
        # of defaulting to classification
        if problem_type == "quantile":
            # TODO: Check list of default models
            default_models = []
            if Model.nn_torch in available_models:
                default_models.append(Model.nn_torch)

            if Model.nn_fastainn in available_models:
                default_models.append(Model.nn_fastainn)

            # TODO: Check if enable any models if dataset exceeds a certain size
            if dataset_len <= 10000:
                if Model.rf in available_models:
                    default_models.append(Model.rf)

        elif problem_type == "regression":
            # TODO: Check list of default models
            default_models = [
                Model.cat,
                Model.xgb_tree,
                Model.rf,
                Model.gbm,
                Model.xt,
            ]
            if not explain_samples:
                if Model.nn_fastainn in available_models:
                    default_models.append(Model.nn_fastainn)

                if Model.knn in available_models:
                    default_models.append(Model.knn)

            # TODO: Check if enable any models if dataset exceeds a certain size
            # Use GBM if dataset >= 10000 (see https://neptune.ai/blog/lightgbm-parameters-guide)
            # if dataset_len >= 10000:
            #     default_models.append(Model.gbm)

        else:  # Multi-class/binary/soft-class classification
            # TODO: Check list of default models
            default_models = [
                Model.cat,
                Model.xgb_tree,
                Model.rf,
                Model.gbm,
                Model.xt,
            ]

            if not explain_samples:
                if Model.nn_fastainn in available_models:
                    default_models.append(Model.nn_fastainn)

                if Model.knn in available_models:
                    default_models.append(Model.knn)

                if Model.fasttext in available_models:
                    default_models.append(Model.fasttext)

            # TODO: Check if enable any models if dataset exceeds a certain size
            # Use GBM if dataset >= 10000 (see https://neptune.ai/blog/lightgbm-parameters-guide)
            # if dataset_len >= 10000:
            #     default_models.append(Model.gbm)

        if ag_automm_enabled and (Model.ag_automm in available_models):
            default_models += [Model.ag_automm]

        if tabpfn_enabled and (Model.tabpfn in available_models):
            default_models += [Model.tabpfn]

        options = {}
        for model in available_models:
            model_hyperparameters = model_params_dict[model].get_hyperparameters(
                problem_type=problem_type, device=device, num_class=num_class
            )
            options[model] = {
                "display_name": model_hyperparameters.display_name,
                "value": model_hyperparameters,
            }

        return default_models, options
