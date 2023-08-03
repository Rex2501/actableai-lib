import pandas as pd
import pytest

from actableai.tasks.intervention import AAIInterventionTask
from actableai.utils.testing import unittest_autogluon_hyperparameters


@pytest.fixture(scope="function")
def intervention_task():
    yield AAIInterventionTask(use_ray=False)


class TestIntervention:
    def test_intervention_numeric(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        result = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert result["status"] == "SUCCESS"
        assert "df" in result["data"]
        assert result["data"]["df"].shape == (20, 6)

    def test_intervention_numeric_cate(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        result = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            cate_alpha=0.5,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert result["status"] == "SUCCESS"
        assert "df" in result["data"]
        assert result["data"]["df"].shape == (20, 10)

    def test_intervention_categorical(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 6)

    def test_intervention_categorical_cate(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            cate_alpha=0.5,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 10)

    def test_intervention_with_common_causes(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=["y", "z"],
            model_directory=tmp_path,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 7)

    def test_intervention_with_common_causes_cate(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=["y", "z"],
            model_directory=tmp_path,
            cate_alpha=0.5,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 11)

    def test_intervention_with_common_causes_date(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "d": pd.date_range("2020-01-01", periods=20),
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=["y", "z", "d"],
            model_directory=tmp_path,
            cate_alpha=0.5,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 12)

    def test_intervention_no_common_causes_drop_features(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "d": pd.date_range("2020-01-01", periods=20),
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=[],
            model_directory=tmp_path,
            cate_alpha=0.5,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=True,
            drop_useless_features=True,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 12)

    def test_intervention_numeric_treatment_categorical_outcome_not_enough_classes(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["1", "2", "3", "a", "5", "b", "7", "8", "9", "10"] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        result = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert result["status"] == "FAILURE"
        assert "StratifiedKFoldChecker" in [x["name"] for x in result["validations"]]

    def test_intervention_numeric_treatment_categorical_outcome(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        result = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            model_directory=tmp_path,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            cross_validation_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert result["status"] == "SUCCESS"
        assert "df" in result["data"]
        assert result["data"]["df"].shape == (20, 5)

    def test_intervention_categorical_outcome_with_common_causes(
        self, intervention_task: AAIInterventionTask, tmp_path
    ):
        current_intervention = ["a", None, "a", "a", "a", "b", "b", "b", "b", "b"]
        new_intervention = ["b", "b", "b", "b", "b", None, None, "a", "a", "a"]
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
                "z": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"] * 2,
                "current_intervention": current_intervention * 2,
                "new_intervention": new_intervention * 2,
            }
        )

        r = intervention_task.run(
            df,
            "y",
            "current_intervention",
            "new_intervention",
            common_causes=["y", "z"],
            model_directory=tmp_path,
            causal_hyperparameters=unittest_autogluon_hyperparameters(),
            cross_validation_hyperparameters=unittest_autogluon_hyperparameters(),
            drop_unique=False,
            drop_useless_features=False,
        )

        assert r["status"] == "SUCCESS"
        assert "df" in r["data"]
        assert r["data"]["df"].shape == (20, 6)
