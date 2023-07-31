from unittest.mock import patch

import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor

from actableai.classification.models import TabPFNModel
from actableai.data_validation.checkers import CheckLevels
from actableai.tasks.classification import (
    AAIClassificationTask,
)
from actableai.utils.dataset_generator import DatasetGenerator
from actableai.utils.testing import (
    unittest_autogluon_hyperparameters,
    unittest_hyperparameters,
)

from actableai.models.autogluon import model_params_dict


@pytest.fixture(scope="function")
def classification_task():
    yield AAIClassificationTask(use_ray=False)


@pytest.fixture(scope="function")
def classification_task_seed(seed=0):
    yield AAIClassificationTask(use_ray=False, seed=seed)


def available_models(
    problem_type,
    gpu,
    explain_samples=False,
    ag_automm_enabled=False,
    tabpfn_enabled=False,
):
    _available_models = AAIClassificationTask.get_available_models(
        problem_type=problem_type,
        explain_samples=explain_samples,
        gpu=gpu,
        ag_automm_enabled=ag_automm_enabled,
        tabpfn_enabled=tabpfn_enabled,
    )
    return [m.value for m in _available_models]


def run_classification_task(
    classification_task: AAIClassificationTask,
    tmp_path,
    *args,
    drop_duplicates=False,
    **kwargs,
):
    if "hyperparameters" not in kwargs:
        kwargs["hyperparameters"] = unittest_hyperparameters()

    return classification_task.run(
        *args,
        **kwargs,
        presets="medium_quality_faster_train",
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        drop_duplicates=drop_duplicates,
        drop_unique=False,
        drop_useless_features=False,
    )


def run_classification_task_seed(
    classification_task_seed: AAIClassificationTask,
    tmp_path,
    *args,
    drop_duplicates=False,
    **kwargs,
):
    if "hyperparameters" not in kwargs:
        kwargs["hyperparameters"] = unittest_hyperparameters()

    return classification_task_seed.run(
        *args,
        **kwargs,
        presets="medium_quality_faster_train",
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_autogluon_hyperparameters(),
        drop_duplicates=drop_duplicates,
        drop_unique=False,
        drop_useless_features=False,
    )


class TestRemoteClassification:
    def test_numeric(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert r["model"] is not None

    def test_ray(self, tmp_path, init_ray):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        task = AAIClassificationTask(use_ray=True)
        r = run_classification_task(
            task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert r["model"] is not None

    def test_categorical(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 2,
                "y": [1, 2, 1, 2, 1, 2, 1, None, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_datetime(self, classification_task, tmp_path):
        from datetime import datetime

        now = datetime.now()
        df = pd.DataFrame(
            {
                "x": [now, now, now, now, now, now, now, now, now, now] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_extra_columns(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
                "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
                "y": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x1"])
        assert r["status"] == "SUCCESS"
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x1"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["importantFeatures"]) == 1
        assert r["model"] is not None

    def test_numeric_and_categorical_and_datetime(self, classification_task, tmp_path):
        from datetime import datetime

        now = datetime.now()
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
                "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
                "x3": [now, now, None, now, now, now, now, now, now, now] * 2,
                "y": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x1", "x2", "x3"],
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x1", "x2", "x3"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_multiclass_num(self, classification_task, tmp_path):
        """
        AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == float and many unique label-values observed).
        """
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": [1, 2, 3, 2, 1, None, 1, 2, 1, 3] * 3,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_multiclass_cat(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8] * 5,
                "y": ["a", "a", "a", "b", "b", None, "c", "c"] * 5,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_mix_target_column(self, classification_task, tmp_path):
        from datetime import datetime

        now = datetime.now()
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 5,
                "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 5,
                "x3": [now, now, None, now, now, now, now, now, now, now] * 5,
                "y": [1, 2, 1, 2, 1, 2, None, 2, 1, "a"] * 5,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x1", "x2", "x3"],
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x1", "x2", "x3"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_mix_feature_column(self, classification_task, tmp_path):
        from datetime import datetime

        now = datetime.now()
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", 1, "b", "b", "b"] * 2,
                "x2": [1, 2, 1, 2, 1, 2, 2.1, 2, 1, 2] * 2,
                "x3": [now, now, None, now, now, now, now, now, now, now] * 2,
                "y": ["1", "2", "1", "2", "1", "2", None, "2", "1", "2"] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainMixedChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_datetime_target(self, classification_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {"x": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2, "y": [now()] * 20}
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", features=["x"], validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["data"]) == 0
        assert r["validations"][0]["name"] == "IsCategoricalChecker"

    def test_insufficient_data(self, classification_task, tmp_path):
        from datetime import datetime

        now = datetime.now()
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"],
                "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2],
                "x3": [now, now, None, now, now, now, now, now, now, now],
                "y": [1, 2, 1, 2, 1, 2, None, 2, 1, "a"],
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_column(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "z", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "ColumnsExistChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_empty_column(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": [None, None, None, None, None, None, None, None, None, None] * 2,
                "x2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x1", "x2"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x1", "x2"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainEmptyColumnsChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING
        assert r["model"] is not None

    def test_insufficient_valid_data(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 10,
                "y": [1, 2, 1, 3, 1, 2, 3, 2, 1, 2] * 10,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=0.01
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][1]["name"] == "IsSufficientValidationSampleChecker"
        assert r["validations"][1]["level"] == CheckLevels.CRITICAL

    def test_insufficient_cls_sample(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "c"]
                + ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 2,
                "y": [4, 4, 5] + [1, 2, 1, 2, 1, 2, None, 2, 1, 3] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x1"], validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert (
            r["validations"][0]["name"]
            == "IsSufficientDataClassificationStratification"
        )

    def test_insufficient_class(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
                "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x1"], validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "IsSufficientNumberOfClassChecker"

    def test_unencoded_binary(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": ["1", "2", "1", "2", "1", None, "1", "2", "1", "2"] * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_suggest_analytic(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [
                    0,
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
                    21,
                ]
                * 10,
                "y": [1, 2, 3, 2, 1, None, 1, 2, 1, 3, 1] * 20,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "x", ["y"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["y"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "CorrectAnalyticChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING
        assert r["model"] is not None

    def test_explain_bool_sample_with_nan(
        self, classification_task, tmp_path, init_ray
    ):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4,
                "y": [1, 2, 1, 3, 1, None, 3, 2, None, 2] * 4,
                "z": [True, False, None, True, False] * 8,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            validation_ratio=0.2,
            explain_samples=True,
        )

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["validation_shaps"]) > 0
        assert len(r["data"]["predict_shaps"]) > 0
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x", "z"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_boolean_str_target_column(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [
                    "True",
                    "False",
                    "True",
                    "False",
                    "True",
                    None,
                    "True",
                    "False",
                    "True",
                    "False",
                ]
                * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_boolean_target_column(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [True, False, True, False, True, None, True, False, True, False]
                * 2,
            }
        )

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None

    def test_drop_duplicates(self, classification_task, tmp_path):
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

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=True,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert r["model"] is not None

    def test_drop_duplicates_insufficient(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=True,
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_run_eval_metric_roc_auc_task_binary(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=False,
            eval_metric="roc_auc",
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert r["model"] is not None

    def test_run_temporal_split_column(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "temporal_split": [1, 1, 2, 2, 1, 2, 1, 1, 2, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task,
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
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert r["model"] is not None
        validation_table = r["data"]["validation_table"]
        sorted_validation_table = validation_table.sort_values(
            by="temporal_split", ascending=True
        )
        # check that the validation table is sorted by temporal split
        assert (validation_table == sorted_validation_table).all(axis=None)

    def test_numeric_refit_full(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            refit_full=True,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert feat["feature"] in ["x"]
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert r["model"] is not None

    def test_tabpfn_activated(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 50,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 50,
            }
        )

        original_fit = TabularPredictor.fit

        with patch.object(TabularPredictor, "fit", autospec=True) as mock_function:

            def side_effect_tabular_predictor_fit(self_, *args, **kwargs):
                kwargs["hyperparameters"] = unittest_autogluon_hyperparameters()

                return original_fit(self_, *args, **kwargs)

            mock_function.side_effect = side_effect_tabular_predictor_fit

            r = run_classification_task(
                classification_task,
                tmp_path,
                df,
                "y",
                ["x"],
                validation_ratio=0.2,
                hyperparameters={"tabpfn": {}},
            )

        _, function_kwargs = mock_function.call_args
        assert TabPFNModel in function_kwargs.get("hyperparameters", {})

    def test_tabpfn_not_activated(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 200,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 200,
            }
        )

        original_fit = TabularPredictor.fit

        with patch.object(TabularPredictor, "fit", autospec=True) as mock_function:

            def side_effect_tabular_predictor_fit(self_, *args, **kwargs):
                kwargs["hyperparameters"] = unittest_autogluon_hyperparameters()

                return original_fit(self_, *args, **kwargs)

            mock_function.side_effect = side_effect_tabular_predictor_fit

            r = run_classification_task(
                classification_task,
                tmp_path,
                df,
                "y",
                ["x"],
                validation_ratio=0.2,
                hyperparameters={"tabpfn": {}},
            )

        _, function_kwargs = mock_function.call_args
        assert TabPFNModel not in function_kwargs.get("hyperparameters", {})

    def test_available_models_multiclass_gpu(self):
        _available_models = available_models(
            problem_type="multiclass",
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
                "nn_mxnet",
                "nn_fastainn",
                "ag_automm",
                "fasttext",
                "tabpfn",
            ]
        )

    def test_available_models_multiclass_gpu_noautomm(self):
        _available_models = available_models(
            problem_type="multiclass",
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
                "nn_mxnet",
                "nn_fastainn",
                "fasttext",
                "tabpfn",
            ]
        )

    def test_available_models_multiclass_nogpu(self):
        _available_models = available_models(
            problem_type="multiclass",
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
                "nn_mxnet",
                "nn_fastainn",
                "fasttext",
                "tabpfn",
            ]
        )

    def test_available_models_multiclass_explain(self):
        _available_models = available_models(
            problem_type="multiclass",
            gpu=True,
            explain_samples=True,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["gbm", "cat", "xgb_tree", "rf", "xt"])

    def test_available_models_multiclass_nogpu_explain(self):
        _available_models = available_models(
            problem_type="multiclass",
            gpu=False,
            explain_samples=True,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["gbm", "cat", "xgb_tree", "rf", "xt"])

    def test_available_models_binary_gpu(self):
        _available_models = available_models(
            problem_type="binary",
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
                "nn_mxnet",
                "nn_fastainn",
                "ag_automm",
                "fasttext",
                "tabpfn",
            ]
        )

    def test_available_models_binary_nogpu(self):
        _available_models = available_models(
            problem_type="binary",
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
                "nn_mxnet",
                "nn_fastainn",
                "fasttext",
                "tabpfn",
            ]
        )

    def test_available_models_binary_gpu_noautomm(self):
        _available_models = available_models(
            problem_type="binary",
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
                "nn_mxnet",
                "nn_fastainn",
                "fasttext",
                "tabpfn",
            ]
        )

    def test_available_models_binary_explain(self):
        _available_models = available_models(
            problem_type="binary",
            gpu=True,
            explain_samples=True,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["gbm", "cat", "xgb_tree", "rf", "xt"])

    def test_available_models_binary_nogpu_explain(self):
        _available_models = available_models(
            problem_type="binary",
            gpu=False,
            explain_samples=True,
            ag_automm_enabled=True,
            tabpfn_enabled=True,
        )
        assert set(_available_models) == set(["gbm", "cat", "xgb_tree", "rf", "xt"])

    # TODO: Use is_gpu_available to set gpu
    @pytest.mark.parametrize(
        "model_type",
        available_models(
            "multiclass", gpu=True, ag_automm_enabled=True, tabpfn_enabled=True
        ),
    )
    def test_hpo_default_multiclass(
        self, classification_task, tmp_path, model_type, is_gpu_available
    ):
        """Test default settings for HPO models for multiclass classification"""

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
        # Skip testing of ag_automm and fasttext due to potentially high
        #   resource consumption
        if model_type in ["ag_automm", "fasttext"]:
            pytest.skip(
                "Skipping test due to potentially high resource consumption",
                allow_module_level=False,
            )

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 3,
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
                    "a a a",
                ]
                * 3,
                "y": [1, 2, 3, 3, 1, 2, 1, 1, 3, 2] * 3,
            }
        )

        hyperparameters = dict()
        hyperparameters = {model_type: {}}

        # Set n_neighbors to avoid error that it is larger than n_samples
        # TODO: This should be handled internally by the function
        if model_type == "knn":
            hyperparameters[model_type] = {"n_neighbors": 2}

        features = ["x", "b"]
        if model_type == "nn_fastainn":
            features = ["x"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target="y",
            features=features,
            hyperparameters=hyperparameters,
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
            validation_ratio=0.5,
        )

        assert r["status"] == "SUCCESS"

    # TODO: Use is_gpu_available to set gpu
    @pytest.mark.parametrize(
        "model_type",
        available_models(
            "binary", gpu=True, ag_automm_enabled=True, tabpfn_enabled=True
        ),
    )
    def test_hpo_default_binary(
        self, classification_task, tmp_path, model_type, is_gpu_available
    ):
        """Test default settings for HPO models for binary classification"""

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
        # Skip testing of ag_automm and fasttext due to potentially high
        #   resource consumption
        if model_type in ["ag_automm", "fasttext"]:
            pytest.skip(
                "Skipping test due to potentially high resource consumption",
                allow_module_level=False,
            )

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 3,
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
                    "a a a",
                ]
                * 3,
                "y": [1, 2, 1, 2, 1, 1, 2, 2, 2, 1] * 3,
            }
        )

        hyperparameters = dict()
        hyperparameters = {model_type: {}}

        # Set n_neighbors to avoid error that it is larger than n_samples
        # TODO: This should be handled internally by the function
        if model_type == "knn":
            hyperparameters[model_type] = {"n_neighbors": 2}

        features = ["x", "b"]
        if model_type == "nn_fastainn":
            features = ["x"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target="y",
            features=features,
            hyperparameters=hyperparameters,
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
            validation_ratio=0.5,
        )

        assert r["status"] == "SUCCESS"

    def test_reproducible_results(
        self, classification_task_seed, tmp_path, is_gpu_available
    ):
        """Check if two tasks run with the same seed yield the same results"""

        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 3,
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
                    "a a a",
                ]
                * 3,
                "y": [1, 2, 3, 3, 1, 2, 1, 1, 3, 2] * 3,
            }
        )

        features = ["x", "b"]

        r1 = run_classification_task_seed(
            classification_task_seed,
            tmp_path,
            df,
            target="y",
            features=features,
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
            validation_ratio=0.5,
        )
        r2 = run_classification_task_seed(
            classification_task_seed,
            tmp_path,
            df,
            target="y",
            features=features,
            ag_automm_enabled=True,
            num_gpus=1 if is_gpu_available else 0,
            num_trials=2,
            validation_ratio=0.5,
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


class TestRemoteClassificationCrossValidation:
    def test_cross_val(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            cross_validation_max_concurrency=4,
        )

        assert r["status"] == "SUCCESS"
        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert len(r["data"]["predictData"]) > 0
        assert "accuracy" in evaluate
        assert "accuracy_std_err" in evaluate
        assert "confusion_matrix" in evaluate
        assert "confusion_matrix_std_err" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert r["model"] is None

    def test_cross_val_with_text(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "z": ["2", "2", "2", "2", "2", "3", "3", "3", "4", "4"] * 3,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "x",
            ["y", "z"],
            kfolds=4,
            cross_validation_max_concurrency=4,
        )

        assert r["status"] == "SUCCESS"
        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert len(r["data"]["predictData"]) > 0
        assert "accuracy" in evaluate
        assert "accuracy_std_err" in evaluate
        assert "confusion_matrix" in evaluate
        assert "confusion_matrix_std_err" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert r["model"] is None

    def test_cross_val_with_text_fail(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": ["a", "a", "c"]
                + ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 3,
                "y": [4, 4, 5] + [1, 2, 1, 2, 1, 2, None, 2, 1, 3] * 3,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x1"],
            kfolds=4,
            cross_validation_max_concurrency=4,
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert (
            r["validations"][0]["name"]
            == "IsSufficientDataClassificationStratification"
        )

    def test_debiasing_feature(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            validation_ratio=0.2,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            kfolds=2,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert r["model"] is None
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list

    def test_cross_val_refit_full(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
            }
        )

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            cross_validation_max_concurrency=4,
            refit_full=True,
        )

        assert r["status"] == "SUCCESS"
        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert len(r["data"]["predictData"]) > 0
        assert "accuracy" in evaluate
        assert "accuracy_std_err" in evaluate
        assert "confusion_matrix" in evaluate
        assert "confusion_matrix_std_err" in evaluate
        assert "importantFeatures" in r["data"]
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
            assert "p_value" in feat
            assert "p_value_std_err" in feat
        assert r["model"] is not None


class TestDebiasing:
    def test_simple_debiasing_feature(self, classification_task, tmp_path):
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

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert "feature" in feat
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list

    def test_debiasing_text_column(self, classification_task, tmp_path):
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

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "DoNotContainTextChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_mixed_debiasing_feature_num(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": ["a", "a", "a", "a", "b", None, "b", "b", "b", "b"] * 3,
                "z": [1, 2, 3, None, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["x"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert "feature" in feat
            assert "importance" in feat
            assert "p_value" in feat
        assert r["model"] is not None
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list

    def test_mixed_debiasing_feature_cat(self, classification_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": ["a", "a", "a", "a", "b", None, "b", "b", "b", "b"] * 3,
                "z": [1, 2, 3, None, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert r["model"] is not None
        for feat in r["data"]["importantFeatures"]:
            assert "feature" in feat
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list

    def test_simple_debiasing_feature_refit_full(self, classification_task, tmp_path):
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

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            refit_full=True,
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        for feat in r["data"]["importantFeatures"]:
            assert "feature" in feat
            assert "importance" in feat
            assert "p_value" in feat
        assert len(r["data"]["validation_shaps"]) == 0
        assert len(r["data"]["predict_shaps"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list
        assert r["model"] is not None
