import numpy as np
import pandas as pd
import pytest

from actableai.data_validation.base import CheckLevels
from actableai.tasks.forecast import AAIForecastTask
from actableai.utils.testing import generate_forecast_df


@pytest.fixture(scope="function")
def forecast_task():
    yield AAIForecastTask(use_ray=False)


class TestTimeSeries:
    @pytest.mark.parametrize("n_group_by", [0, 1, 2])
    @pytest.mark.parametrize("n_targets", [1, 5])
    @pytest.mark.parametrize("use_features", [True, False])
    @pytest.mark.parametrize("sorted_data", [True, False])
    @pytest.mark.parametrize("freq", ["T", "MS", "YS"])
    def test_simple(
        self,
        np_rng,
        forecast_task,
        n_group_by,
        n_targets,
        use_features,
        sorted_data,
        freq,
    ):
        prediction_length = np_rng.integers(1, 3)
        (
            df,
            date_column,
            date_column_str,
            target_columns,
            group_by,
            feature_columns,
            n_groups,
        ) = generate_forecast_df(
            np_rng,
            prediction_length,
            n_group_by=n_group_by,
            n_targets=n_targets,
            freq=freq,
            n_real_static_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_static_features=np_rng.integers(1, 10) if use_features else 0,
            n_real_dynamic_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_dynamic_features=np_rng.integers(1, 10) if use_features else 0,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        df_original = df
        if not sorted_data:
            df = df.sample(frac=1, random_state=0)

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            group_by=group_by,
            feature_columns=feature_columns,
            hyperparameters={
                "constant_value": {},
                "multivariate_constant_value": {},
            },
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "data_v2" in results
        assert "data" in results
        assert "validations" in results
        assert "runtime" in results

        assert results["status"] == "SUCCESS"

        data = results["data_v2"]
        assert "predict" in data
        assert "validation" in data

        validation_data = data["validation"]
        assert "predict" in validation_data
        assert "agg_metrics" in validation_data
        assert "item_metrics" in validation_data

        # Test prediction output
        df_predict = data["predict"]
        assert len(df_predict) == prediction_length * n_groups * n_targets
        assert "target" in df_predict.columns
        assert date_column_str in df_predict.columns
        assert "0.05" in df_predict.columns
        assert "0.5" in df_predict.columns
        assert "0.95" in df_predict.columns
        for col in group_by:
            assert col in df_predict.columns

        for col in target_columns:
            assert col in df_predict["target"].unique()
        # Test date column
        if n_group_by > 0:
            for group_predict, df_group_predict in df_predict.groupby(group_by):
                df_group = dict(df_original.groupby(group_by).__iter__())[group_predict]

                date_list = None
                if use_features:
                    date_list = df_group[date_column].iloc[-prediction_length:]
                else:
                    first_date = df_group[date_column].iloc[-1]
                    date_list = pd.date_range(
                        start=first_date, freq=freq, periods=prediction_length + 1
                    )[1:]

                for target in target_columns:
                    df_group_target_predict = df_group_predict[
                        df_group_predict["target"] == target
                    ]
                    assert (
                        df_group_target_predict[date_column_str].values
                        == date_list.values
                    ).all()
        else:
            if use_features:
                date_list = df_original[date_column].iloc[-prediction_length:]
            else:
                first_date = df_original[date_column].iloc[-1]
                date_list = pd.date_range(
                    start=first_date, freq=freq, periods=prediction_length + 1
                )[1:]

            for target in target_columns:
                df_target_predict = df_predict[df_predict["target"] == target]
                assert (
                    df_target_predict[date_column_str].values == date_list.values
                ).all()

        # Test validation output
        df_val_predict = validation_data["predict"]
        assert len(df_val_predict) == prediction_length * n_groups * n_targets
        assert "target" in df_val_predict.columns
        assert date_column_str in df_val_predict.columns
        assert "0.05" in df_val_predict.columns
        assert "0.5" in df_val_predict.columns
        assert "0.95" in df_val_predict.columns
        for col in group_by:
            assert col in df_val_predict.columns

        for col in target_columns:
            assert col in df_val_predict["target"].unique()
        # Test date column
        if n_group_by > 0:
            for group_predict, df_group_predict in df_val_predict.groupby(group_by):
                df_group = dict(df_original.groupby(group_by).__iter__())[group_predict]

                date_list = None
                if use_features:
                    date_list = df_group[date_column].iloc[
                        -2 * prediction_length : -prediction_length
                    ]
                else:
                    date_list = df_group[date_column].iloc[-prediction_length:]

                for target in target_columns:
                    df_group_target_predict = df_group_predict[
                        df_group_predict["target"] == target
                    ]
                    assert (
                        df_group_target_predict[date_column_str].values
                        == date_list.values
                    ).all()
        else:
            if use_features:
                date_list = df_original[date_column].iloc[
                    -2 * prediction_length : -prediction_length
                ]
            else:
                date_list = df_original[date_column].iloc[-prediction_length:]

            for target in target_columns:
                df_target_predict = df_val_predict[df_val_predict["target"] == target]
                assert (
                    df_target_predict[date_column_str].values == date_list.values
                ).all()

        df_agg_metrics = validation_data["agg_metrics"]
        assert len(df_agg_metrics) == n_targets
        assert "target" in df_agg_metrics.columns
        assert "MAPE" in df_agg_metrics.columns
        assert "MASE" in df_agg_metrics.columns
        assert "RMSE" in df_agg_metrics.columns
        assert "sMAPE" in df_agg_metrics.columns
        assert (
            not df_agg_metrics[["MAPE", "MASE", "RMSE", "sMAPE"]].isna().any(axis=None)
        )
        for col in target_columns:
            assert col in df_agg_metrics["target"].unique()

        df_item_metrics = validation_data["item_metrics"]
        assert len(df_item_metrics) == n_targets * n_groups
        assert "target" in df_item_metrics.columns
        for col in group_by:
            assert col in df_item_metrics.columns
        assert "MAPE" in df_item_metrics.columns
        assert "MASE" in df_item_metrics.columns
        assert "RMSE" in df_item_metrics.columns
        assert "sMAPE" in df_item_metrics.columns
        assert (
            not df_item_metrics[["MAPE", "MASE", "RMSE", "sMAPE"]].isna().any(axis=None)
        )
        for col in target_columns:
            assert col in df_item_metrics["target"].unique()

        # TODO delete legacy
        # Test Legacy output
        legacy_data = results["data"]
        assert "predict" in legacy_data
        assert "evaluate" in legacy_data

        assert len(legacy_data["predict"]) == n_groups
        group_list = []
        for pred_group in legacy_data["predict"]:
            assert len(pred_group) == n_targets

            for pred_target_group in pred_group:
                assert "name" in pred_target_group
                assert "group" in pred_target_group
                assert "value" in pred_target_group

                target = pred_target_group["name"]
                assert target is not None
                group = pred_target_group["group"]
                assert group is not None
                if n_group_by == 1:
                    group = group[0]

                if n_group_by <= 0:
                    df_group_target = df_original
                else:
                    df_group_target = dict(df_original.groupby(group_by).__iter__())[
                        group
                    ]

                df_group_target_train = None
                future_date_list = None
                if use_features:
                    df_group_target_train = df_group_target.iloc[:-prediction_length]
                    future_date_list = df_group_target[date_column].iloc[
                        -prediction_length:
                    ]
                else:
                    df_group_target_train = df_group_target
                    first_date = df_group_target[date_column].iloc[-1]
                    future_date_list = pd.date_range(
                        start=first_date, freq=freq, periods=prediction_length + 1
                    )[1:]

                value = pred_target_group["value"]
                assert "data" in value
                assert "prediction" in value

                data = value["data"]
                assert "date" in data
                assert "value" in data

                assert len(data["date"]) == 4 * prediction_length
                assert len(data["value"]) == 4 * prediction_length

                date_list = df_group_target_train[date_column][-4 * prediction_length :]
                assert (pd.to_datetime(data["date"]).values == date_list.values).all()
                assert (
                    data["value"]
                    == df_group_target_train[target][-4 * prediction_length :]
                ).all()

                prediction = value["prediction"]
                assert "date" in prediction
                assert "min" in prediction
                assert "median" in prediction
                assert "max" in prediction

                assert len(prediction["date"]) == prediction_length
                assert len(prediction["min"]) == prediction_length
                assert len(prediction["median"]) == prediction_length
                assert len(prediction["max"]) == prediction_length

                assert (
                    pd.to_datetime(prediction["date"]).values == future_date_list.values
                ).all()

            group_list.append(group)

        evaluate = legacy_data["evaluate"]
        assert "dates" in evaluate
        assert "values" in evaluate
        assert "item_metrics" in evaluate

        val_dates = None
        if n_group_by == 0:
            val_dates = [evaluate["dates"]]
        else:
            val_dates = evaluate["dates"]
            assert len(val_dates) == n_groups

        for group, val_date in zip(group_list, val_dates):
            assert len(val_date) == prediction_length

            df_group = None
            if n_group_by <= 0:
                df_group = df_original
            else:
                df_group = dict(df_original.groupby(group_by).__iter__())[group]

            date_list = None
            if use_features:
                date_list = df_group[date_column].iloc[
                    -2 * prediction_length : -prediction_length
                ]
            else:
                date_list = df_group[date_column].iloc[-prediction_length:]

            assert (pd.to_datetime(val_date).values == date_list.values).all()

        assert len(evaluate["values"]) == n_groups
        for val_group in evaluate["values"]:
            assert len(val_group) == n_targets

            for val_group_target in val_group:
                assert "q5" in val_group_target
                assert "q50" in val_group_target
                assert "q95" in val_group_target

                assert len(val_group_target["q5"]) == prediction_length
                assert len(val_group_target["q50"]) == prediction_length
                assert len(val_group_target["q95"]) == prediction_length

        item_metrics = evaluate["item_metrics"]
        assert "item_id" in item_metrics
        for col in target_columns:
            assert col in item_metrics["item_id"].values()
            assert col in item_metrics["item_id"].keys()
        assert "MAPE" in item_metrics
        assert "MASE" in item_metrics
        assert "MSE" in item_metrics
        assert "sMAPE" in item_metrics
        assert not np.isnan(list(item_metrics["MAPE"].values())).any()
        assert not np.isnan(list(item_metrics["MASE"].values())).any()
        assert not np.isnan(list(item_metrics["MSE"].values())).any()
        assert not np.isnan(list(item_metrics["sMAPE"].values())).any()
        for metric_dict in item_metrics.values():
            assert len(metric_dict) == n_targets

    @pytest.mark.parametrize("n_targets", [1, 2])
    @pytest.mark.parametrize("freq", ["T"])
    def test_hyperopt(self, np_rng, forecast_task, n_targets, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            n_targets=n_targets,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            trials=10,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "data_v2" in results
        assert "data" in results
        assert "validations" in results
        assert "runtime" in results

        assert results["status"] == "SUCCESS"

    @pytest.mark.parametrize("freq", ["T"])
    def test_ray(self, np_rng, init_ray, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        forecast_task = AAIForecastTask(use_ray=True)
        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=True,
            ray_tune_kwargs={
                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 0,
                },
            },
            max_concurrent=1,
        )

        assert results is not None
        assert "status" in results
        assert "data_v2" in results
        assert "data" in results
        assert "validations" in results
        assert "runtime" in results

        assert results["status"] == "SUCCESS"

    @pytest.mark.parametrize("freq", ["T"])
    def test_mix_target_column(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        df_cat_1 = df.sample(int(len(df) * 0.2), random_state=0)
        df_cat_2 = df.sample(int(len(df) * 0.2), random_state=1)
        df.loc[df_cat_1.index, target_columns] = "a"
        df.loc[df_cat_2.index, target_columns] = "b"

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "DoNotContainMixedChecker" in validations_dict
        assert validations_dict["DoNotContainMixedChecker"] == CheckLevels.CRITICAL

    @pytest.mark.parametrize("freq", ["T"])
    def test_invalid_date_column(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        df[date_column_str] = np_rng.choice(["a", "b", "c"], size=len(df))

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "IsDatetimeChecker" in validations_dict
        assert validations_dict["IsDatetimeChecker"] == CheckLevels.CRITICAL

    @pytest.mark.parametrize("freq", ["T"])
    def test_insufficient_data(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 10, "max_periods": 11},
        )

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "IsSufficientDataChecker" in validations_dict
        assert validations_dict["IsSufficientDataChecker"] == CheckLevels.CRITICAL

    @pytest.mark.parametrize("freq", ["T"])
    def test_invalid_prediction_length(self, np_rng, forecast_task, freq):
        prediction_length = 10
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "IsValidPredictionLengthChecker" in validations_dict
        assert (
            validations_dict["IsValidPredictionLengthChecker"] == CheckLevels.CRITICAL
        )

    @pytest.mark.parametrize("freq", ["T"])
    def test_cat_features(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        df[target_columns] = np_rng.choice(
            ["a", "b", "c"], size=(len(df), len(target_columns))
        )

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "CategoryChecker" in validations_dict
        assert validations_dict["CategoryChecker"] == CheckLevels.CRITICAL

    @pytest.mark.parametrize("freq", ["T"])
    def test_invalid_column(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=["test"],
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "ColumnsExistChecker" in validations_dict
        assert validations_dict["ColumnsExistChecker"] == CheckLevels.CRITICAL

    @pytest.mark.parametrize("freq", ["T"])
    def test_empty_column(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        df[target_columns] = np.nan

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "DoNotContainEmptyColumnsChecker" in validations_dict
        assert (
            validations_dict["DoNotContainEmptyColumnsChecker"] == CheckLevels.CRITICAL
        )

    @pytest.mark.parametrize("freq", ["T", "MS", "YS"])
    def test_invalid_frequency(self, np_rng, forecast_task, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        df = df.append(df).sort_index()

        results = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            hyperparameters={"constant_value": {}},
            trials=1,
            use_ray=False,
        )

        assert results is not None
        assert "status" in results
        assert "validations" in results

        assert results["status"] == "FAILURE"

        assert len(results["validations"]) > 0

        validations_dict = {val["name"]: val["level"] for val in results["validations"]}

        assert "IsValidFrequencyChecker" in validations_dict
        assert validations_dict["IsValidFrequencyChecker"] == CheckLevels.CRITICAL

    @pytest.mark.parametrize("freq", ["T"])
    def test_reproducible_results(self, np_rng, init_ray, freq):
        prediction_length = np_rng.integers(1, 3)
        df, _, date_column_str, target_columns, _, _, _ = generate_forecast_df(
            np_rng,
            prediction_length,
            freq=freq,
            date_range_kwargs={"min_periods": 30, "max_periods": 60},
        )

        forecast_task = AAIForecastTask(use_ray=True)
        results1 = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            trials=1,
            use_ray=False,
            # seed=20,
        )

        results2 = forecast_task.run(
            df,
            prediction_length=prediction_length,
            date_column=date_column_str,
            predicted_columns=target_columns,
            trials=1,
            use_ray=False,
            # seed=20,
        )

        assert results1["data_v2"]["predict"].equals(results2["data_v2"]["predict"])
