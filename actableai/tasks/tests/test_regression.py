from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from actableai.data_validation.base import CheckLevels
from actableai.tasks.regression import (
    AAIRegressionTask,
)
from actableai.utils.dataset_generator import DatasetGenerator
from actableai.utils.testing import (
    unittest_hyperparameters,
    unittest_autogluon_hyperparameters,
)

from actableai.models.autogluon import model_params_dict


@pytest.fixture(scope="function")
def regression_task():
    yield AAIRegressionTask(use_ray=False)


@pytest.fixture(scope="function")
def regression_task_seed(seed=0):
    yield AAIRegressionTask(use_ray=False, seed=seed)


@pytest.fixture(scope="function")
def data():
    yield pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
        }
    )


def run_regression_task(
    regression_task: AAIRegressionTask, tmp_path, *args, **kwargs
) -> Dict[str, Any]:
    if "hyperparameters" not in kwargs:
        kwargs["hyperparameters"] = unittest_hyperparameters()

    if "drop_duplicates" not in kwargs:
        kwargs["drop_duplicates"] = False

    if "num_trials" not in kwargs:
        kwargs["num_trials"] = 1

    return regression_task.run(
        *args,
        **kwargs,
        presets="medium_quality_faster_train",
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        drop_unique=False,
        drop_useless_features=False,
    )


def run_regression_task_seed(
    regression_task_seed: AAIRegressionTask, tmp_path, *args, **kwargs
) -> Dict[str, Any]:
    if "hyperparameters" not in kwargs:
        kwargs["hyperparameters"] = unittest_hyperparameters()

    if "drop_duplicates" not in kwargs:
        kwargs["drop_duplicates"] = False

    if "num_trials" not in kwargs:
        kwargs["num_trials"] = 1

    return regression_task_seed.run(
        *args,
        **kwargs,
        presets="medium_quality_faster_train",
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        drop_unique=False,
        drop_useless_features=False,
    )


def available_models(
    problem_type,
    gpu,
    explain_samples=False,
    ag_automm_enabled=False,
    tabpfn_enabled=False,
):
    _available_models = AAIRegressionTask.get_available_models(
        problem_type=problem_type,
        explain_samples=explain_samples,
        gpu=gpu,
        ag_automm_enabled=ag_automm_enabled,
        tabpfn_enabled=tabpfn_enabled,
    )
    return [m.value for m in _available_models]


class TestRemoteRegression:
    def test_num_vs_num(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_datetime(self, regression_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {"x": [now()] * 20, "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2}
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "y", features=["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_mixed_datetime(self, regression_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {
                "x": [now()] * 20,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "z",
            features=["x", "y"],
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_datetime_target(self, regression_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {"x": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2, "y": [now()] * 20}
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "y", features=["x"], validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["data"]) == 0

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}
        assert "IsNumericalChecker" in validations_dict

    def test_num_vs_categorical(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_num_vs_mix(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y", "z"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_complex_text(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, None, 3, 4, 4] * 5,
                "z": ["good", "good", "bad", "bad", "freaking bad", "freaking bad"] * 5,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["z"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_feature_missing_value(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, None, 3, 4, 4] * 5,
                "z": ["good", "good", np.nan, np.nan, "freaking bad", "freaking bad"]
                * 5,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["z"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_hyperparam_multimodal(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 2, 2, 3, 3, 3, 7, 7, 7] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_autogluon_multiclass_case(self, regression_task, tmp_path):
        """
        AutoGluon infers your prediction problem is: 'multiclass' (because dtype of label-column == int, but few unique label-values observed).
        """
        df = pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3] * 100,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100,
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "x")

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_mixed_feature_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, "abc xyz", 7.1, 8, 9, 10] * 2,
                "x2": [1, 2, 3, 4, 5, "6", 7, 8, 9, 10] * 2,
                "y": [1, 2, None, None, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "y")

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "leaderboard" in r["data"]

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "DoNotContainMixedChecker" in validations_dict
        assert validations_dict["DoNotContainMixedChecker"] == CheckLevels.WARNING

    def test_mixed_target_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, "5", None, None, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "y")

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "IsNumericalChecker" in validations_dict
        assert validations_dict["IsNumericalChecker"] == CheckLevels.CRITICAL

    def test_insufficient_data(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [1, 2, 3, 4, "5", None, None, 8, 9, 10],
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "y")

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    def test_invalid_eval_metric(self, regression_task, tmp_path, data):
        r = run_regression_task(regression_task, tmp_path, data, "y", eval_metric="abc")
        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "RegressionEvalMetricChecker" in validations_dict
        assert validations_dict["RegressionEvalMetricChecker"] == CheckLevels.CRITICAL

    def test_validation_has_prediction(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 2, 2, 3, 3, 3, 7, 7, 7] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            prediction_quantiles=[5, 50, 95, 99.9],
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "x_predicted" in r["data"]["validation_table"].columns
        assert "x_predicted_5" in r["data"]["validation_table"].columns
        assert "x_predicted_50" in r["data"]["validation_table"].columns
        assert "x_predicted_95" in r["data"]["validation_table"].columns
        assert "x_predicted_99.9" in r["data"]["validation_table"].columns
        assert "leaderboard" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_invalid_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "z", validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL

    def test_empty_feature_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [None, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "y", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "z"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "prediction_table" in r["data"]
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "DoNotContainEmptyColumnsChecker" in validations_dict
        assert (
            validations_dict["DoNotContainEmptyColumnsChecker"] == CheckLevels.WARNING
        )

    def test_empty_target_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [None, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "DoNotContainEmptyColumnsChecker" in validations_dict
        assert (
            validations_dict["DoNotContainEmptyColumnsChecker"] == CheckLevels.CRITICAL
        )

    def test_suggest_analytic(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "prediction_table" in r["data"]
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "CorrectAnalyticChecker" in validations_dict
        assert validations_dict["CorrectAnalyticChecker"] == CheckLevels.WARNING

    def test_explain_samples(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            validation_ratio=0.2,
            explain_samples=True,
        )

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["validation_shaps"]) > 0
        assert "leaderboard" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_explain_samples_quantiles(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            validation_ratio=0.2,
            explain_samples=True,
            prediction_quantiles=[5, 50, 95, 99.9],
        )

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "ExplanationChecker" in validations_dict
        assert validations_dict["ExplanationChecker"] == CheckLevels.CRITICAL

    def test_explain_samples_with_datetime(self, regression_task, tmp_path):
        rng = pd.date_range("2015-02-24", periods=20, freq="T")
        drop_indices = np.random.randint(0, 20, 5)
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": rng,
            }
        )
        df.iloc[drop_indices, :] = None
        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            validation_ratio=0.2,
            explain_samples=True,
        )

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["validation_shaps"]) > 0
        assert "leaderboard" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y", "z"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_drop_duplicates(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ]
                * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 4,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=True,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_drop_duplicates_insufficient(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=True,
        )

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    def test_run_temporal_split_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "temporal_split": [1, 1, 2, 2, 1, 2, 1, 1, 2, 2] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=False,
            split_by_datetime=True,
            datetime_column="temporal_split",
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None
        validation_table = r["data"]["validation_table"]
        sorted_validation_table = validation_table.sort_values(
            by="temporal_split", ascending=True
        )
        # check that the validation table is sorted by temporal split
        assert (validation_table == sorted_validation_table).all(axis=None)

    def test_num_vs_num_refit_full(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2, refit_full=True
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_available_models_regression_gpu(self):
        _available_models = available_models(
            problem_type="regression",
            gpu=True,
            explain_samples=False,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(
            [
                "gbm",
                "cat",
                "xgb_tree",
                "rf",
                "xt",
                "knn",
                "lr",
                "nn_fastainn",
                "nn_mxnet",
                "ag_automm",
            ]
        )

    def test_available_models_regression_gpu_noautomm(self):
        _available_models = available_models(
            problem_type="regression",
            gpu=True,
            explain_samples=False,
            ag_automm_enabled=False,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(
            [
                "gbm",
                "cat",
                "xgb_tree",
                "rf",
                "xt",
                "knn",
                "lr",
                "nn_fastainn",
                "nn_mxnet",
            ]
        )

    def test_available_models_regression_nogpu(self):
        _available_models = available_models(
            problem_type="regression",
            gpu=False,
            explain_samples=False,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(
            [
                "gbm",
                "cat",
                "xgb_tree",
                "rf",
                "xt",
                "knn",
                "lr",
                "nn_fastainn",
                "nn_mxnet",
            ]
        )

    def test_available_models_regression_explain(self):
        _available_models = available_models(
            problem_type="regression",
            gpu=True,
            explain_samples=True,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["gbm", "cat", "xgb_tree", "rf", "xt"])

    def test_available_models_regression_nogpu_explain(self):
        _available_models = available_models(
            problem_type="regression",
            gpu=False,
            explain_samples=True,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["gbm", "cat", "xgb_tree", "rf", "xt"])

    def test_available_models_quantile_gpu(self):
        _available_models = available_models(
            problem_type="quantile",
            gpu=True,
            explain_samples=False,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["nn_torch", "nn_fastainn", "rf"])

    def test_available_models_quantile_nogpu(self):
        _available_models = available_models(
            problem_type="quantile",
            gpu=False,
            explain_samples=False,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["nn_torch", "nn_fastainn", "rf"])

    # TODO: Use is_gpu_available to set gpu
    @pytest.mark.parametrize(
        "model_type",
        available_models("regression", gpu=True, ag_automm_enabled=True),
    )
    def test_hpo_default_regression(
        self, regression_task, tmp_path, model_type, is_gpu_available
    ):
        """Test default settings for HPO models for regression"""

        # Skip testing the model if GPU is not available but the model requires the GPU
        if not is_gpu_available and model_params_dict[model_type].gpu_required:
            pytest.skip(
                "Skipping test where GPU is required but no GPU is available",
                allow_module_level=False,
            )

        # Skip testing of fastai models since they can be unstable during unit
        # tests
        # TODO: Re-enable fastai tests if issue is resolved
        if model_type == "nn_fastainn":
            pytest.skip(
                "Skipping test where GPU is required but no GPU is available",
                allow_module_level=False,
            )
        # Skip testing of ag_automm due to potentially high resource consumption
        if model_type == "ag_automm":
            pytest.skip(
                "Skipping test due to potentially high resource consumption",
                allow_module_level=False,
            )

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "b": [
                    "a a a",
                    "b b b",
                    "c c c",
                    "d d d",
                    "b b b",
                    "c c c",
                    "a a a",
                    "a a a",
                    "c c c",
                    "e e e",
                ]
                * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        hyperparameters = dict()
        hyperparameters = {model_type: {}}

        # Set n_neighbors to avoid error that it is larger than n_samples
        # TODO: This should be handled internally by the function
        if model_type == "knn":
            hyperparameters[model_type] = {"n_neighbors": 5}

        features = ["y", "b"]
        if model_type == "nn_fastainn":
            features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target="x",
            features=features,
            hyperparameters=hyperparameters,
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
        )

        assert r["status"] == "SUCCESS"

    # TODO: Use is_gpu_available to set gpu
    @pytest.mark.parametrize(
        "model_type",
        available_models("quantile", gpu=True, ag_automm_enabled=True),
    )
    def test_hpo_default_quantile(
        self, regression_task, tmp_path, model_type, is_gpu_available
    ):
        """Test default settings for HPO models for quantile regression"""

        # Skip testing the model if GPU is not available but the model requires the GPU
        if not is_gpu_available and model_params_dict[model_type].gpu_required:
            pytest.skip(
                "Skipping test where GPU is required but no GPU is available",
                allow_module_level=False,
            )

        # Skip testing of fastai models since they can be unstable during unit
        # tests
        # TODO: Re-enable fastai tests if issue is resolved
        if model_type == "nn_fastainn":
            pytest.skip(
                "Skipping test using fastai due to potential errors during unit tests",
                allow_module_level=False,
            )
        # Skip testing of ag_automm due to potentially high resource consumption
        if model_type == "ag_automm":
            pytest.skip(
                "Skipping test due to potentially high resource consumption",
                allow_module_level=False,
            )

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "b": [
                    "a a a",
                    "b b b",
                    "c c c",
                    "d d d",
                    "b b b",
                    "c c c",
                    "a a a",
                    "a a a",
                    "c c c",
                    "e e e",
                ]
                * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        hyperparameters = dict()
        hyperparameters = {model_type: {}}

        features = ["y", "b"]
        if model_type == "nn_fastainn":
            features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target="x",
            features=features,
            prediction_quantiles=[5, 50, 95],
            hyperparameters=hyperparameters,
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
        )

        assert r["status"] == "SUCCESS"

    def test_reproducible_results(
        self, regression_task_seed, tmp_path, is_gpu_available
    ):
        """Check if two tasks run with the same seed yield the same results"""

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "b": [
                    "a a a",
                    "b b b",
                    "c c c",
                    "d d d",
                    "b b b",
                    "c c c",
                    "a a a",
                    "a a a",
                    "c c c",
                    "e e e",
                ]
                * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        features = ["y", "b"]

        r1 = run_regression_task_seed(
            regression_task_seed,
            tmp_path,
            df,
            target="x",
            features=features,
            prediction_quantiles=[5, 50, 95],
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
        )

        r2 = run_regression_task_seed(
            regression_task_seed,
            tmp_path,
            df,
            target="x",
            features=features,
            prediction_quantiles=[5, 50, 95],
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
        )

        r1 = r1["data"]["leaderboard"].drop(
            [
                "pred_time_val",
                "fit_time",
                "pred_time_val_marginal",
                "fit_time_marginal",
                "compile_time",
            ],
            axis=1,
        )
        r2 = r2["data"]["leaderboard"].drop(
            [
                "pred_time_val",
                "fit_time",
                "pred_time_val_marginal",
                "fit_time_marginal",
                "compile_time",
            ],
            axis=1,
        )

        assert r1.sort_values(by=["score_val", "model"], ignore_index=True).equals(
            r2.sort_values(by=["score_val", "model"], ignore_index=True)
        )


class TestRemoteRegressionCrossValidation:
    def test_cross_val(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            cross_validation_max_concurrency=4,
        )

        evaluate = r["data"]["evaluate"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["prediction_table"]) > 0
        assert "RMSE" in evaluate
        assert "RMSE_std_err" in evaluate
        assert "R2" in evaluate
        assert "R2_std_err" in evaluate
        assert "MAE" in evaluate
        assert "MAE_std_err" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y", "z"]
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is None

    def test_cross_val_with_explain(self, regression_task, tmp_path):
        rng = pd.date_range("2015-02-24", periods=20, freq="T")
        drop_indices = np.random.randint(0, 20, 5)
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": rng,
            }
        )
        df.iloc[drop_indices, :] = None
        r = run_regression_task(
            regression_task, tmp_path, df, "x", kfolds=4, explain_samples=True
        )

        evaluate = r["data"]["evaluate"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["prediction_table"]) > 0
        assert "RMSE" in evaluate
        assert "RMSE_std_err" in evaluate
        assert "R2" in evaluate
        assert "R2_std_err" in evaluate
        assert "MAE" in evaluate
        assert "MAE_std_err" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y", "z"]
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is None

    def test_with_quantile(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=1,
            prediction_quantiles=[5, 50, 95, 99.9],
        )

        evaluate = r["data"]["evaluate"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["prediction_table"]) > 0
        assert "x_predicted" in r["data"]["prediction_table"].columns
        assert "x_predicted_5" in r["data"]["prediction_table"].columns
        assert "x_predicted_50" in r["data"]["prediction_table"].columns
        assert "x_predicted_95" in r["data"]["prediction_table"].columns
        assert "x_predicted_99.9" in r["data"]["prediction_table"].columns
        assert "x_predicted" in r["data"]["validation_table"].columns
        assert "x_predicted_5" in r["data"]["validation_table"].columns
        assert "x_predicted_50" in r["data"]["validation_table"].columns
        assert "x_predicted_95" in r["data"]["validation_table"].columns
        assert "x_predicted_99.9" in r["data"]["validation_table"].columns
        assert "PINBALL_LOSS" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

    def test_cross_val_with_quantiles(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            prediction_quantiles=[5, 50, 95, 99.9],
        )

        evaluate = r["data"]["evaluate"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["prediction_table"]) > 0
        assert "x_predicted" in r["data"]["prediction_table"].columns
        assert "x_predicted_5" in r["data"]["prediction_table"].columns
        assert "x_predicted_50" in r["data"]["prediction_table"].columns
        assert "x_predicted_95" in r["data"]["prediction_table"].columns
        assert "x_predicted_99.9" in r["data"]["prediction_table"].columns
        assert "PINBALL_LOSS" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert "leaderboard" in r["data"]
        assert r["model"] is None

    def test_debiasing_feature(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            validation_ratio=0.2,
            kfolds=2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "y"]
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list
        assert r["model"] is None

    def test_cross_val_refit_full(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            cross_validation_max_concurrency=4,
            refit_full=True,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_causal_feature_selection(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 5,
                "t": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 5,
            }
        )
        target = "t"
        features = ["x", "y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            causal_feature_selection=True,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_causal_feature_selection_no_causal_feature(
        self, regression_task, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 5,
                "t": [3, 2, 1, 1, 5, None, None, 2, 1, 3] * 5,
            }
        )
        target = "t"
        features = ["x", "y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            causal_feature_selection=True,
        )

        assert r["status"] == "FAILURE"
        assert "No causal feature" in r["messenger"]


class TestDebiasing:
    def test_simple_debiasing_feature(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 3,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list

    def test_debiasing_text_column(self, regression_task, tmp_path):
        df = DatasetGenerator.generate(
            columns_parameters=[
                {"name": "x", "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2},
                {"name": "y", "values": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2},
                {"name": "z", "type": "text", "word_range": (5, 10)},
                {"name": "t", "values": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 2},
            ],
            rows=20,
            random_state=0,
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "FAILURE"

        validations_dict = {val["name"]: val["level"] for val in r["validations"]}

        assert "DoNotContainTextChecker" in validations_dict
        assert validations_dict["DoNotContainTextChecker"] == CheckLevels.CRITICAL

    def test_mixed_debiasing_feature_cat(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": ["a", "a", "a", "a", "a", None, "b", "b", "b", "b"] * 3,
                "z": [1, 2, 3, None, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list

    def test_mixed_debiasing_feature_num(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": ["a", "a", "a", "a", "a", None, "b", "b", "b", "b"] * 3,
                "z": [1, 2, None, 4, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["x"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]
        assert r["model"] is not None

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list

    def test_evaluate_has_data(self, regression_task, data, tmp_path):
        r = run_regression_task(regression_task, tmp_path, data, target="y")
        assert r["status"] == "SUCCESS"
        assert "evaluate" in r["data"]
        assert "RMSE" in r["data"]["evaluate"]
        assert "R2" in r["data"]["evaluate"]
        assert "MAE" in r["data"]["evaluate"]
        assert "MSE" in r["data"]["evaluate"]
        assert "MEDIAN_ABSOLUTE_ERROR" in r["data"]["evaluate"]
        assert "metrics" in r["data"]["evaluate"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_simple_debiasing_feature_refit_full(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 3,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            refit_full=True,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list
        assert r["model"] is not None
