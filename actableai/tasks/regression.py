import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask
from actableai.tasks.autogluon import AAIAutogluonTask
from actableai.classification.utils import split_validation_by_datetime
from actableai.parameters.models import ModelSpace


class _AAIRegressionTrainTask(AAITask):
    """
    TODO write documentation
    """

    @AAITask.run_with_ray_remote(TaskType.REGRESSION_TRAIN)
    def run(
        self,
        explain_samples: bool,
        presets: str,
        hyperparameters: Dict,
        model_directory: str,
        target: str,
        features: List[str],
        run_model: bool,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame],
        df_test: Optional[pd.DataFrame],
        prediction_quantiles: Optional[List[float]],
        drop_duplicates: bool,
        run_debiasing: bool,
        biased_groups: List[str],
        debiased_features: List[str],
        residuals_hyperparameters: Optional[dict],
        num_gpus: Union[str, int],
        eval_metric: str,
        time_limit: Optional[int],
        drop_unique: bool,
        drop_useless_features: bool,
        feature_prune: bool,
        feature_prune_time_limit: Optional[float],
        num_trials: int,
        problem_type: str,
        infer_limit: float,
        infer_limit_batch_size: int,
    ) -> Tuple[
        Any,
        List,
        Optional[Dict],
        Any,
        Any,
        Any,
        Optional[pd.DataFrame],
        pd.DataFrame,
    ]:
        """Sub class for running a regression without cross validation

        Args:
            explain_samples: Whether we explain the samples
            presets: Presets for AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            hyperparameters: Hyperparameters for AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            model_directory: Directory for model
            target: Target for regression
            features: Features used in DataFrame for regression
            run_model: Whether the model should be run on df_test
            df_train: DataFrame for training
            df_val: DataFrame for validation
            df_test: DataFrame for testing
            prediction_quantiles: Prediction quantiles for prediction and validation
            drop_duplicates: Whether we should drop duplicated rows
            run_debiasing: Whether we should debias the data
            biased_groups: Features that creates a bias in prediction
            debiased_features: Features debiased w.r.t to biased_groups
            residuals_hyperparameters: Hyperparameters for debiasing with AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            num_gpus: Number of GPUs used by AutoGluon
            eval_metric: Evaluation metric for validation
            time_limit: Time limit of training
            feature_prune_time_limit: Time limit for feature pruning (in seconds)
                if feature_prune is True
            num_trials: The number of trials for hyperparameter optimization
            problem_type: The type of the problem ('regression' or 'quantile')
            infer_limit: The time in seconds to predict 1 row of data. For
                example, infer_limit=0.05 means 50 ms per row of data, or 20 rows /
                second throughput.
            infer_limit_batch_size: The amount of rows passed at once to be
                predicted when calculating per-row speed. This is very important
                because infer_limit_batch_size=1 (online-inference) is highly
                suboptimal as various operations have a fixed cost overhead
                regardless of data size. If you can pass your test data in bulk,
                you should specify infer_limit_batch_size=10000. Must be an
                integer greater than 0.

        Returns:
            Tuple:
                - AutoGluon's predictor
                - List of feature importance
                - Dictionary of evaluation metrics
                - Predicted values for df_val
                - Explainer for explaining the prediction and validation
                - Predicted values for df_test if run_model is true, if quantiles then
                    dataframe containing all the values for each quantile
                - Predicted shap values if explain_samples is true
                - Leaderboard of the best model ran by AutoGluon
        """
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from autogluon.core.metrics import (
            pinball_loss,
            root_mean_squared_error,
            r2,
            mean_absolute_error,
            mean_squared_error,
            median_absolute_error,
        )
        from gluonts.evaluation.metrics import quantile_loss
        from actableai.utils import debiasing_feature_generator_args
        from actableai.debiasing.debiasing_model import DebiasingModel
        from actableai.explanation.autogluon_explainer import AutoGluonShapTreeExplainer

        # TODO: To finalise parameters
        hyperparameter_tune_kwargs = (
            {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
                "num_trials": num_trials,
                "scheduler": "local",
                "searcher": "auto",
            }
        )

        ag_args_fit = {"drop_unique": drop_unique}
        feature_generator_args = {}
        if "AG_AUTOMM" in hyperparameters:
            feature_generator_args["enable_raw_text_features"] = True

        if not drop_useless_features:
            feature_generator_args["pre_drop_useless"] = False
            feature_generator_args["post_generators"] = []

        if run_debiasing:
            ag_args_fit["drop_duplicates"] = drop_duplicates
            ag_args_fit["label"] = target
            ag_args_fit["features"] = features
            ag_args_fit["biased_groups"] = biased_groups
            ag_args_fit["debiased_features"] = debiased_features
            ag_args_fit["hyperparameters_residuals"] = residuals_hyperparameters
            ag_args_fit["presets_residuals"] = presets
            ag_args_fit["hyperparameters_non_residuals"] = hyperparameters
            ag_args_fit["presets_non_residuals"] = presets
            ag_args_fit["drop_useless_features"] = drop_useless_features
            ag_args_fit["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs

            feature_generator_args = {
                **feature_generator_args,
                **debiasing_feature_generator_args(),
            }

            hyperparameters = {DebiasingModel: {}}

        df_train = df_train[features + biased_groups + [target]]
        if df_val is not None:
            df_val = df_val[features + biased_groups + [target]]
        if df_test is not None:
            df_test = df_test[features + biased_groups]

        # Train
        predictor = TabularPredictor(
            label=target,
            path=model_directory,
            problem_type=problem_type,
            eval_metric=eval_metric,
            quantile_levels=prediction_quantiles,
        )

        feature_prune_kwargs = None
        if feature_prune:
            feature_prune_kwargs = {}
            if feature_prune_time_limit is not None:
                feature_prune_kwargs[
                    "feature_prune_time_limit"
                ] = feature_prune_time_limit

        predictor = predictor.fit(
            train_data=df_train,
            presets=presets,
            hyperparameters=hyperparameters,
            ag_args_fit=ag_args_fit,
            feature_generator=AutoMLPipelineFeatureGenerator(**feature_generator_args),
            time_limit=time_limit,
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            feature_prune_kwargs=feature_prune_kwargs,
            num_cpus=1,
            num_gpus=num_gpus,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
        )

        explainer = None
        if explain_samples:
            # Filter out models which are not compatible with explanations
            while not AutoGluonShapTreeExplainer.is_predictor_compatible(predictor):
                predictor.delete_models(
                    models_to_delete=predictor.get_model_best(),
                    dry_run=False,
                    allow_delete_cascade=True,
                )

            explainer = AutoGluonShapTreeExplainer(predictor)

        predictor.persist_models()
        leaderboard = predictor.leaderboard(extra_info=True)
        pd.set_option("chained_assignment", "warn")

        important_features = []
        metrics = None
        if prediction_quantiles is None and df_val is not None:
            feature_importance = predictor.feature_importance(df_val)
            for i in range(len(feature_importance)):
                if feature_importance.index[i] in biased_groups:
                    continue
                important_features.append(
                    {
                        "feature": feature_importance.index[i],
                        "importance": feature_importance["importance"][i],
                        "p_value": feature_importance["p_value"][i],
                    }
                )
        y_pred = None
        if df_val is not None:
            y_true = df_val[target]
            y_pred = predictor.predict(df_val)

            y_pred_metrics = y_pred
            if prediction_quantiles is not None:
                y_pred_metrics = y_pred_metrics[0.5]

            metrics = {
                "root_mean_squared_error": root_mean_squared_error(
                    y_true, y_pred_metrics
                ),
                "r2": r2(y_true, y_pred_metrics),
                "mean_absolute_error": mean_absolute_error(y_true, y_pred_metrics),
                "mean_squared_error": mean_squared_error(y_true, y_pred_metrics),
                "median_absolute_error": median_absolute_error(y_true, y_pred_metrics),
            }

            if prediction_quantiles is not None:
                metrics["pinball_loss"] = pinball_loss(
                    df_val[target], y_pred, quantile_levels=prediction_quantiles
                )

                for quantile_level in prediction_quantiles:
                    metrics[f"quantile_loss-{quantile_level}"] = quantile_loss(
                        y_true, y_pred[quantile_level], q=quantile_level
                    )

        evaluate = None
        if metrics is not None:
            # Legacy (TODO: to be removed)
            evaluate = {
                "RMSE": abs(metrics["root_mean_squared_error"]),
                "R2": metrics["r2"],
                "MAE": abs(metrics["mean_absolute_error"]),
                "MSE": abs(metrics["mean_squared_error"]),
                "MEDIAN_ABSOLUTE_ERROR": abs(metrics["median_absolute_error"]),
            }

            metric_list = [
                "Root Mean Squared Error",
                "R2",
                "Mean Absolute Error",
                "Median Absolute Error",
            ]

            metric_value_list = [
                abs(metrics["root_mean_squared_error"]),
                metrics["r2"],
                abs(metrics["mean_absolute_error"]),
                abs(metrics["median_absolute_error"]),
            ]

            if prediction_quantiles is not None:
                # Legacy (TODO: to be removed)
                evaluate["PINBALL_LOSS"] = abs(metrics["pinball_loss"])

                metric_list.append("Pinball Loss")
                metric_value_list.append(abs(metrics["pinball_loss"]))

                for quantile_level in prediction_quantiles:
                    # Legacy (TODO: to be removed)
                    evaluate[f"QUANTILE_LOSS-{quantile_level}"] = abs(
                        metrics[f"quantile_loss-{quantile_level}"]
                    )

                    metric_list.append(f"Quantile Loss {quantile_level}")
                    metric_value_list.append(
                        abs(metrics[f"quantile_loss-{quantile_level}"])
                    )

            evaluate["metrics"] = pd.DataFrame(
                {"metric": metric_list, "value": metric_value_list}
            )

        predictions = None
        predict_shap_values = None
        if run_model and df_test is not None:
            if explain_samples:
                predict_shap_values = explainer.shap_values(df_test)

            predictions = predictor.predict(df_test)

        return (
            predictor,
            important_features,
            evaluate,
            y_pred,
            explainer,
            predictions,
            predict_shap_values,
            leaderboard,
        )


class AAIRegressionTask(AAIAutogluonTask):
    """Regression task."""

    @classmethod
    def get_hyperparameters_space(
        cls,
        dataset_len: int,
        prediction_quantiles: Optional[List[float]] = None,
        device: str = "cpu",
        explain_samples: bool = False,
        ag_automm_enabled: bool = False,
        tabpfn_enabled: bool = False,
    ) -> ModelSpace:
        """Return the hyperparameters space of the task.

        Args:
            dataset_len: Length of the dataset.
            prediction_quantiles: List of quantiles (for regression task only),
                as a percentage
            device: Which device is being used, can be one of 'cpu' or 'gpu'.
            explain_samples: Boolean indicating if explanations for predictions
                in test and validation will be generated.
            ag_automm_enabled: Boolean indicating if AG_AUTOMM model should be used.
            tabpfn_enabled: Boolean indicating if TabPFN model should be used.

        Returns:
            Hyperparameters space represented as a ModelSpace.
        """

        problem_type = cls.compute_problem_type(prediction_quantiles)

        default_models, options = AAIAutogluonTask.get_base_hyperparameters_space(
            num_class=-1,
            dataset_len=dataset_len,
            problem_type=problem_type,
            device=device,
            explain_samples=explain_samples,
            ag_automm_enabled=ag_automm_enabled,
            tabpfn_enabled=tabpfn_enabled,
        )

        return ModelSpace(
            name="regression_model_space",
            display_name="Regression Model Space",
            description="The space of available and default regression models and parameters.",
            default=default_models,
            options=options,
        )

    @staticmethod
    def compute_problem_type(prediction_quantiles: Optional[List[int]]) -> str:
        """Determine the problem type ('regression' or 'quantile')

        Args:
            prediction_quantiles: List of quantiles. (in percentage)

        Returns:
            problem_type: String representation of the problem type: 'regression' or 'quantile'
        """

        if prediction_quantiles is None:
            problem_type = "regression"
        else:
            problem_type = "quantile"

        return problem_type

    @AAIAutogluonTask.run_with_ray_remote(TaskType.REGRESSION)
    def run(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        biased_groups: Optional[List[str]] = None,
        debiased_features: Optional[List[str]] = None,
        eval_metric: str = "r2",
        validation_ratio: float = 0.2,
        prediction_quantiles: Optional[List[float]] = None,
        explain_samples: bool = False,
        model_directory: Optional[str] = None,
        presets: str = "medium_quality_faster_train",
        hyperparameters: Optional[dict] = None,
        train_task_params: Optional[dict] = None,
        kfolds: int = 1,
        cross_validation_max_concurrency: int = 1,
        residuals_hyperparameters: Optional[dict] = None,
        drop_duplicates: bool = True,
        return_residuals: bool = False,
        kde_steps: int = 10,
        num_gpus: Union[int, str] = 0,
        time_limit: Optional[int] = None,
        drop_unique: bool = True,
        drop_useless_features: bool = True,
        split_by_datetime: bool = False,
        datetime_column: Optional[str] = None,
        ag_automm_enabled: bool = False,
        refit_full: bool = False,
        feature_prune: bool = True,
        feature_prune_time_limit: Optional[float] = None,
        intervention_run_params: Optional[Dict] = None,
        causal_feature_selection: bool = False,
        causal_feature_selection_max_concurrent_tasks: int = 20,
        ci_for_causal_feature_selection_task_params: Optional[dict] = None,
        ci_for_causal_feature_selection_run_params: Optional[dict] = None,
        run_pdp: bool = True,
        run_ice: bool = True,
        pdp_ice_grid_resolution: Optional[int] = 100,
        pdp_ice_n_samples: Optional[int] = 100,
        num_trials: int = 1,
        infer_limit: float = 60,
        infer_limit_batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Run this regression task and return results.

        Args:
            df: Input data frame
            target: Target columns in df. If there are empty values in this columns,
                predictions will be generated for these rows.
            features: A list of features to be used for prediction. If None, all columns
                except target are used as features
            biased_groups: A list of columns of groups that should be protected from
                biases (e.g. gender, race, age)
            debiased_features: A list of proxy features that need to be debiased for
                protection of sensitive groups
            eval_metric: Metric to be optimized during training. Possible values include
                'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error',
                'median_absolute_error', 'r2'
            validation_ratio: The ratio to randomly split data for training and
                validation
            prediction_quantiles: List of quantiles. (in percentage)
            explain_samples: If true, explanations for predictions in test and
                validation will be generated. It takes significantly longer time to run.
            model_directory: Destination to store trained model. If not set, a temporary
                folder will be created
            presets: Autogluon's presets for training model.
                See
                https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit.
            hyperparameters: Autogluon's hyperparameters for training model.
                See
                https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit.
            train_task_params: Parameters for _AAITrainTask constructor.
            kfolds: Number of folds for cross validation. If 1, train test split is used
                instead.
            cross_validation_max_concurrency: Maximum number of Ray actors used for
                cross validation (each actor execute for one split)
            residuals_hyperparameters: Autogluon's hyperparameteres used in final model
                of counterfactual predictions
            drop_duplicates: Whether duplicate values should be dropped before training.
            return_residuals: Whether residual values should be returned in
                counterfactual prediction
            kde_steps: Steps used to generate KDE plots with debiasing
            num_gpus: Number of GPUs used in nuisnace models in counterfactual
                prediction
            time_limit: time limit (in seconds) of training. None means no time limit
            drop_unique: Wether to drop columns with only unique values as preprocessing step
            drop_useless_features: Whether to drop columns with only unique values at fit time
            split_by_datetime: Wether train/validation sets are split using datetime.
                Training will be the most recent data and validation the latest.
            datetime_column: The specified datetime column if split_by_datetime is enabled
            ag_automm_enabled: Whether to use autogluon multimodal model on text
                columns.
            refit_full: Wether the model is completely refitted on validation at
                the end of the task. Training time is divided by 2 to allow refitting
                for the other half of the time
            feature_prune: Whether feature pruning is enabled. Can increase accuracy
                by removing harmful features for the model (features that are
                detrimental to the performance). If no training time left, this step
                is skipped
            feature_prune_time_limit: Time limit for feature pruning.
            intervention_run_params: Parameters for running an intervention task.
                Check actableai/tasks/intervention.py for more details.
            causal_feature_selection: if True, it will search for direct causal
                features and use only these features for the prediction
            causal_feature_selection_max_concurrent_tasks: maximum number of concurrent
                tasks for selecting causal features
            ci_for_causal_feature_selection_task_params: Parameters for AAIDirectCausalFeatureSelectionTask
            ci_for_causal_feature_selection_run_params: Kwargs for AAIDirectCausalFeatureSelectionTask's run
            run_pdp: Run Partial Dependency to get Partial Dependency Plot (PDP)
            run_ice: Run Independent Conditional Expectation (ICE)
            pdp_ice_grid_resolution: Maximum resolution of the grid to use for
                computation of the PDP and/or ICE
            pdp_ice_n_samples: The number of rows to sample in df_train. If 'None,
                no sampling is performed.
            num_trials: The number of trials for hyperparameter optimization
            infer_limit: The time in seconds to predict 1 row of data. For
                example, infer_limit=0.05 means 50 ms per row of data, or 20 rows /
                second throughput.
            infer_limit_batch_size: The amount of rows passed at once to be
                predicted when calculating per-row speed. This is very important
                because infer_limit_batch_size=1 (online-inference) is highly
                suboptimal as various operations have a fixed cost overhead
                regardless of data size. If you can pass your test data in bulk,
                you should specify infer_limit_batch_size=10000. Must be an
                integer greater than 0.

        Examples:
            >>> import pandas as pd
            >>> from actableai.tasks.regression import AAIRegressionTask
            >>> df = pd.read_csv("path/to/csv")
            >>> result = AAIRegressionTask().run(
            ...     df,
            ...     'target_column',
            ... )

        Returns:
            Dict: Dictionary containing the results for this task
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "messenger": Message returned with the task
                - "validations": List of validations on the data.
                    non-empty if the data presents a problem for the task
                - "runtime": Execution time of the task
                - "data": Dictionary containing the data for the task
                    - "validation_table": Validation table
                    - "prediction_table": Prediction table
                    - "predict_shaps": Shapley values for prediction table
                    - "evaluate": Evaluation metrics for the task
                    - "validation_shaps": Shapley values for the validation table
                    - "importantFeatures": Feature importance for the validation table
                    - "debiasing_charts": If debiasing enabled, charts to display debiasing
                    - "leaderboard": Leaderboard of the best trained models
            - "model": AAIModel to redeploy the model
        """
        import time
        from decimal import Decimal
        from tempfile import mkdtemp
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KernelDensity
        from autogluon.common.features.infer_types import check_if_nlp_feature

        from actableai.data_validation.params import RegressionDataValidator
        from actableai.data_validation.base import (
            CheckLevels,
            UNIQUE_CATEGORY_THRESHOLD,
        )
        from actableai import AAIInterventionTask
        from actableai.models.aai_predictor import (
            AAITabularModel,
            AAITabularModelInterventional,
        )
        from actableai.regression.cross_validation import run_cross_validation
        from actableai.utils.sanitize import sanitize_timezone
        from actableai.tasks.direct_causal import AAIDirectCausalFeatureSelection

        from actableai.utils.pdp_ice import get_pdp_and_ice
        from actableai.utils import is_gpu_available

        # TODO: Add check for num_trials (min and max values)

        start = time.time()
        # To resolve any issues of access rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        # Handle default parameters
        if features is None:
            features = list(df.columns.drop(target, errors="ignore"))
        if biased_groups is None:
            biased_groups = []
        if debiased_features is None:
            debiased_features = []
        if model_directory is None:
            model_directory = mkdtemp(prefix="autogluon_model")
        if train_task_params is None:
            train_task_params = {}
        if refit_full and time_limit is not None:
            # Half the time limit for train and half the time for refit
            time_limit = time_limit // 2
        if prediction_quantiles is not None and len(prediction_quantiles) == 0:
            prediction_quantiles = None

        run_debiasing = len(biased_groups) > 0 and len(debiased_features) > 0

        if run_debiasing and drop_useless_features:
            drop_useless_features = False
            logging.warning(
                "`drop_useless_features` is set to False: `run_debiasing` is True"
            )

        # Pre process data
        df = df.fillna(np.nan)

        # Pre-process prediction quantiles
        quantile_original_values = {}
        if prediction_quantiles is not None:
            processed_prediction_quantiles = []
            add_median = True

            # Use set to remove duplicates
            for quantile in set(prediction_quantiles):
                if quantile == 50:
                    add_median = False

                quantile_value = float(Decimal(str(quantile)) / Decimal(100))
                processed_prediction_quantiles.append(quantile_value)

                quantile_original_values[quantile_value] = quantile

            if add_median:
                processed_prediction_quantiles.append(0.5)

            prediction_quantiles = processed_prediction_quantiles

        # Validate parameters
        data_validation_results = RegressionDataValidator().validate(
            target,
            features,
            df,
            biased_groups,
            debiased_features,
            eval_metric,
            prediction_quantiles,
            presets,
            explain_samples,
            drop_duplicates,
            drop_unique=drop_unique,
            kfolds=kfolds,
        )

        # Determine GPU type
        device = "gpu" if is_gpu_available() else "cpu"

        any_text_cols = df.apply(check_if_nlp_feature).any(axis=None)
        hyperparameters_space = self.get_hyperparameters_space(
            dataset_len=df.shape[0],
            prediction_quantiles=prediction_quantiles,
            device=device,
            explain_samples=explain_samples,
            ag_automm_enabled=ag_automm_enabled and any_text_cols,
            tabpfn_enabled=False,
        )
        hyperparameters_validation = None
        if hyperparameters is None or len(hyperparameters) <= 0:
            hyperparameters = hyperparameters_space.get_default()
        else:
            (
                hyperparameters_validation,
                hyperparameters,
            ) = hyperparameters_space.validate_process_parameter(hyperparameters)

        if hyperparameters_validation is not None:
            data_validation_results += hyperparameters_validation.to_check_results(
                name="HyperparametersChecker"
            )

        failed_checks = [
            check for check in data_validation_results if check is not None
        ]
        validations = [
            {"name": check.name, "level": check.level, "message": check.message}
            for check in failed_checks
        ]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": validations,
                "runtime": time.time() - start,
            }

        # Convert hyperparameters to AutoGluon format
        hyperparameters = self._hyperparameters_to_model_params(
            hyperparameters, hyperparameters_space
        )

        if prediction_quantiles is not None:
            eval_metric = "pinball_loss"

        # Split data
        df_train = df[pd.notnull(df[target])]
        if drop_duplicates:
            df_train = df_train.drop_duplicates(subset=features + [target])
        df_val = None
        if kfolds <= 1:
            if split_by_datetime and datetime_column is not None:
                df_train, df_val = split_validation_by_datetime(
                    df_train, datetime_column, validation_ratio
                )
            else:
                df_train, df_val = train_test_split(
                    df_train, test_size=validation_ratio
                )

        df_test = df[pd.isnull(df[target])].drop(columns=[target])

        # If true it means that the regression needs to be run on test data
        run_model = df_test.shape[0] > 0

        df_predict = df_test.copy()

        if causal_feature_selection:
            causal_feature_selection_task = AAIDirectCausalFeatureSelection()
            causal_feature_selection = causal_feature_selection_task.run(
                df_train,
                target,
                features,
                max_concurrent_ci_tasks=causal_feature_selection_max_concurrent_tasks,
                causal_inference_task_params=ci_for_causal_feature_selection_task_params,
                causal_inference_run_params=ci_for_causal_feature_selection_run_params,
            )

            if causal_feature_selection["status"] == "FAILURE":
                return causal_feature_selection

            validations.extend(causal_feature_selection["validations"])

            causal_features = set()
            for f, v in causal_feature_selection["data"].items():
                if v["is_direct_cause"]:
                    causal_features.add(f.split(":::")[0])
            features = [f for f in features if f in causal_features]
            debiased_features = [f for f in debiased_features if f in causal_features]

            if len(features) == 0:
                return {
                    "status": "FAILURE",
                    "messenger": "No causal feature is found",
                    "data": {},
                    "validations": validations,
                    "runtime": time.time() - start,
                }

        # Train
        regression_train_task = _AAIRegressionTrainTask(**train_task_params)

        problem_type = self.compute_problem_type(prediction_quantiles)

        y_pred = None
        predictor = None
        explainer = None
        leaderboard = None
        if kfolds > 1:
            (
                important_features,
                evaluate,
                predictions,
                predict_shap_values,
                df_val,
                leaderboard,
            ) = run_cross_validation(
                regression_train_task=regression_train_task,
                kfolds=kfolds,
                cross_validation_max_concurrency=cross_validation_max_concurrency,
                explain_samples=explain_samples,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=run_model,
                df_train=df_train,
                df_test=df_test,
                prediction_quantiles=prediction_quantiles,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
                eval_metric=eval_metric,
                time_limit=time_limit,
                drop_unique=drop_unique,
                drop_useless_features=drop_useless_features,
                feature_prune=feature_prune,
                feature_prune_time_limit=feature_prune_time_limit,
                num_trials=num_trials,
                problem_type=problem_type,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
            )
        else:
            (
                predictor,
                important_features,
                evaluate,
                y_pred,
                explainer,
                predictions,
                predict_shap_values,
                leaderboard,
            ) = regression_train_task.run(
                explain_samples=explain_samples,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=run_model,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                prediction_quantiles=prediction_quantiles,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
                eval_metric=eval_metric,
                time_limit=time_limit,
                drop_unique=drop_unique,
                drop_useless_features=drop_useless_features,
                feature_prune=feature_prune,
                feature_prune_time_limit=feature_prune_time_limit,
                num_trials=num_trials,
                problem_type=problem_type,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
            )

        # Validation
        eval_shap_values = []
        if kfolds <= 1:
            if prediction_quantiles is None:
                df_val[target + "_predicted"] = y_pred
            else:
                for quantile in prediction_quantiles:
                    str_quantile = str(quantile_original_values.get(quantile))
                    df_val[f"{target}_predicted_{str_quantile}"] = y_pred[quantile]

                df_val[target + "_predicted"] = df_val[f"{target}_predicted_50"]

            # FIXME this should not be done, but we need this for now so we can return the models
            predictor.persist_models()
            # predictor.unpersist_models()

            if explain_samples:
                eval_shap_values = explainer.shap_values(
                    df_val[features + biased_groups]
                )

        # Prediction
        if run_model:
            if prediction_quantiles is None:
                df_predict[target + "_predicted"] = predictions
            else:
                for quantile in prediction_quantiles:
                    str_quantile = str(quantile_original_values.get(quantile))
                    df_predict[f"{target}_predicted_{str_quantile}"] = predictions[
                        quantile
                    ]

                df_predict[target + "_predicted"] = df_predict[f"{target}_predicted_50"]

        debiasing_charts = []
        # Generate debiasing charts
        if run_debiasing:

            def kde_bandwidth(x):
                return max(0.5 * x.std() * (x.size ** (-0.2)), 1e-2)

            kde_x_axis_gt = np.linspace(
                df_val[target].min(), df_val[target].max(), kde_steps
            )
            kde_x_axis_predicted = np.linspace(
                df_val[f"{target}_predicted"].min(),
                df_val[f"{target}_predicted"].max(),
                kde_steps,
            )

            plot_targets = [target, f"{target}_predicted"]

            for biased_group in biased_groups:
                group_charts = []
                group_chart_type = None

                df_val_biased_group = df_val[biased_group].fillna("NaN").astype(str)
                biased_classes = sorted(df_val_biased_group.unique())

                if len(biased_classes) <= UNIQUE_CATEGORY_THRESHOLD:
                    # Categorical Biased Group => KDE plot
                    group_chart_type = "kde"

                    for plot_target, x_axis in zip(
                        plot_targets, [kde_x_axis_gt, kde_x_axis_predicted]
                    ):
                        y_prob = {}

                        for biased_class in biased_classes:
                            values = df_val[df_val_biased_group == biased_class][
                                plot_target
                            ]
                            values = values[values.notna()]

                            kde = KernelDensity(bandwidth=kde_bandwidth(values))
                            kde = kde.fit(values.values.reshape(-1, 1))

                            y_prob[biased_class] = kde.score_samples(
                                x_axis.reshape(-1, 1)
                            )

                        corr, pvalue = spearmanr(
                            df_val[biased_group], df_val[plot_target]
                        )
                        group_charts.append(
                            {
                                "x_label": plot_target,
                                "x": x_axis.tolist(),
                                "lines": [
                                    {"y": np.exp(y).tolist(), "name": biased_class}
                                    for biased_class, y in y_prob.items()
                                ],
                                "corr": corr,
                                "pvalue": pvalue,
                            }
                        )
                else:
                    # Non-Categorical Biased Group => Scatter plot
                    group_chart_type = "scatter"

                    for plot_target in plot_targets:
                        X = df_val[plot_target]
                        y = df_val[biased_group]

                        notna_mask = X.notna() & y.notna()
                        X = X[notna_mask]
                        y = y[notna_mask]

                        corr, pvalue = spearmanr(
                            df_val[biased_group], df_val[plot_target]
                        )
                        group_charts.append(
                            {
                                "x_label": plot_target,
                                "x": X.tolist(),
                                "y": y.tolist(),
                                "corr": corr,
                                "pvalue": pvalue,
                            }
                        )

                debiasing_charts.append(
                    {
                        "type": group_chart_type,
                        "group": biased_group,
                        "target": target,
                        "charts": group_charts,
                    }
                )

        leaderboard_obj_cols = leaderboard.select_dtypes(include=["object"]).columns
        leaderboard[leaderboard_obj_cols] = leaderboard[leaderboard_obj_cols].astype(
            str
        )

        data = {
            "validation_table": df_val if kfolds <= 1 else None,
            "prediction_table": df_predict,
            "predict_shaps": predict_shap_values,
            "evaluate": evaluate,
            "validation_shaps": eval_shap_values,
            "importantFeatures": important_features,
            "debiasing_charts": debiasing_charts,
            "leaderboard": leaderboard,
        }

        aai_intervention_model = None
        if intervention_run_params is not None:
            intervention_task_result = AAIInterventionTask(
                return_model=True, upload_model=False
            ).run(**intervention_run_params)
            if intervention_task_result["status"] == "SUCCESS":
                aai_intervention_model = intervention_task_result["model"]
            else:
                intervention_validation = [
                    {
                        "name": "Intervention: " + x["name"],
                        "level": CheckLevels.WARNING,
                        "message": x["message"],
                    }
                    for x in intervention_task_result["validations"]
                ]
                validations.extend(intervention_validation)

        if refit_full:
            df_only_full_training = df.loc[df[target].notnull()]
            predictor, _, _, _, _, _, _, _ = _AAIRegressionTrainTask(
                **train_task_params
            ).run(
                explain_samples=False,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=False,
                df_train=df_only_full_training,
                df_val=None,
                df_test=None,
                prediction_quantiles=prediction_quantiles,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
                eval_metric=eval_metric,
                time_limit=time_limit,
                drop_unique=drop_unique,
                drop_useless_features=drop_useless_features,
                feature_prune=feature_prune,
                feature_prune_time_limit=feature_prune_time_limit,
                num_trials=1,
                problem_type=problem_type,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
            )
            predictor.refit_full(model="best", set_best_to_refit_full=True)

        model = None
        if (kfolds <= 1 or refit_full) and predictor:
            model = AAITabularModel(
                predictor=predictor, df_training=df_train, explainer=explainer
            )
            if aai_intervention_model:
                model = AAITabularModelInterventional(
                    predictor=predictor,
                    intervention_model=aai_intervention_model,
                    df_training=df_train,
                )

        runtime = time.time() - start

        data["pdp_ice"] = dict()
        if (run_pdp or run_ice) and (model is not None):
            data["pdp_ice"] = get_pdp_and_ice(
                model,
                df_train,
                features="all",
                pdp=run_pdp,
                ice=run_ice,
                grid_resolution=pdp_ice_grid_resolution,
                verbosity=0,
                n_samples=pdp_ice_n_samples,
            )

        return {
            "status": "SUCCESS",
            "messenger": "",
            "validations": validations,
            "runtime": runtime,
            "data": data,
            "model": model,
            # FIXME this predictor is not really usable as is for now
        }
