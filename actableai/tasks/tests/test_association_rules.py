import pytest
import pandas as pd

from actableai.tasks.association_rules import AAIAssociationRulesTask
from actableai.data_validation.base import CheckLevels


@pytest.fixture(scope="function")
def association_rules_task():
    yield AAIAssociationRulesTask(use_ray=False)


class TestAssociationRules:
    def test_association_rules(self, association_rules_task: AAIAssociationRulesTask):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, 2, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        result = association_rules_task.run(
            df=df,
            group_by=["x"],
            items="y",
            frequent_method="fpgrowth",
            association_metric="confidence",
            min_support=0.01,
            min_association_metric=0.01,
        )

        assert result["status"] == "SUCCESS"
        assert "rules" in result["data"]
        assert result["data"]["rules"].shape[1] == 9
        assert "frequent_itemset" in result["data"]
        assert result["data"]["frequent_itemset"].shape[1] == 2
        assert "df_list" in result["data"]
        assert result["data"]["df_list"].shape[1] == 2

    def test_association_rules_none(
        self, association_rules_task: AAIAssociationRulesTask
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, None, 2, 3, 3, None, 4] * 2,
                "y": [1, None, 3, 4, 5, 6, 7, None, 9, 10] * 2,
            }
        )

        result = association_rules_task.run(
            df=df,
            group_by=["x"],
            items="y",
            frequent_method="fpgrowth",
            association_metric="confidence",
            min_support=0.01,
            min_association_metric=0.01,
        )

        assert result["status"] == "SUCCESS"
        assert "rules" in result["data"]
        assert result["data"]["rules"].shape[1] == 9
        assert "frequent_itemset" in result["data"]
        assert result["data"]["frequent_itemset"].shape[1] == 2
        assert "df_list" in result["data"]
        assert result["data"]["df_list"].shape[1] == len(["x"]) + 1

    def test_association_rules_multi_group_by(
        self, association_rules_task: AAIAssociationRulesTask
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, 2, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        result = association_rules_task.run(
            df=df,
            group_by=["x", "y"],
            items="z",
            frequent_method="fpgrowth",
            association_metric="confidence",
            min_support=0.01,
            min_association_metric=0.01,
        )

        assert result["status"] == "SUCCESS"
        assert "rules" in result["data"]
        assert result["data"]["rules"].shape[1] == 9
        assert "frequent_itemset" in result["data"]
        assert result["data"]["frequent_itemset"].shape[1] == 2
        assert "df_list" in result["data"]
        assert result["data"]["df_list"].shape[1] == len(["x", "y"]) + 1

    def test_association_rules_fpmax(
        self, association_rules_task: AAIAssociationRulesTask
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, 2, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        result = association_rules_task.run(
            df=df,
            group_by=["x"],
            items="y",
            frequent_method="fpmax",
            association_metric="confidence",
            min_support=0.01,
            min_association_metric=0.01,
        )

        assert result["status"] == "SUCCESS"
        assert "rules" in result["data"]
        assert result["data"]["rules"].shape[1] == 9
        assert "frequent_itemset" in result["data"]
        assert result["data"]["frequent_itemset"].shape[1] == 2
        assert "df_list" in result["data"]
        assert result["data"]["df_list"].shape[1] == 2

    def test_association_rules_wrong_frequent_method(
        self, association_rules_task: AAIAssociationRulesTask
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, 2, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        with pytest.raises(Exception):
            association_rules_task.run(
                df=df,
                group_by=["x"],
                items="y",
                frequent_method="jerome",
                association_metric="confidence",
                min_support=0.01,
                min_association_metric=0.01,
            )

    def test_association_rules_wrong_association_metric(
        self, association_rules_task: AAIAssociationRulesTask
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, 2, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        with pytest.raises(Exception):
            association_rules_task.run(
                df=df,
                group_by=["x"],
                items="y",
                frequent_method="fpgrowth",
                association_metric="jerome",
                min_support=0.01,
                min_association_metric=0.01,
            )

    def test_empty_dataframe(self, association_rules_task: AAIAssociationRulesTask):
        df = pd.DataFrame({"x": [], "y": []})

        result = association_rules_task.run(
            df=df,
            group_by=["x"],
            items="y",
            frequent_method="fpgrowth",
            association_metric="confidence",
            min_support=0.01,
            min_association_metric=0.01,
        )

        assert result["status"] == "FAILURE"
        validations_dict = {val["name"]: val["level"] for val in result["validations"]}
        assert "NoFrequentItemSet" in validations_dict
        assert validations_dict["NoFrequentItemSet"] == CheckLevels.CRITICAL
